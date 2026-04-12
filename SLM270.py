"""
Gemma3 Implementation is sourced from 
https://github.com/rasbt/LLMs-from-scratch/blob/6b9502056fba9388cdbb93c8a7d2d0d9d5deec81/ch05/12_gemma3/standalone-gemma3.ipynb

Copyright 2023-2026 Sebastian Raschka

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
from transformers import PreTrainedTokenizerFast
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc2 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc3 = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False)

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = nn.functional.gelu(x_fc1, approximate="tanh") * x_fc2
        return self.fc3(x)

class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-6, bias=False):
        super().__init__()
        self.eps = eps
        # Gemma3 stores zero-centered weights and uses (1 + weight) during forward
        self.scale = nn.Parameter(torch.zeros(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim)) if bias else None

    def forward(self, x):
        # Match HF Gemma3: compute norm in float32, then scale by (1 + w)
        input_dtype = x.dtype
        x_f = x.float()
        var = x_f.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x_f * torch.rsqrt(var + self.eps)
        out = x_norm * (1.0 + self.scale.float())
         
        if self.shift is not None:
            out = out + self.shift.float()
         
        return out.to(input_dtype)

def compute_rope_params(head_dim, theta_base=10_000, context_length=4096, dtype=torch.float32):
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    # Compute the inverse frequencies
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2, dtype=dtype)[: (head_dim // 2)].float() / head_dim))

    # Generate position indices
    positions = torch.arange(context_length, dtype=dtype)

    # Compute the angles
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)  # Shape: (context_length, head_dim // 2)

    # Expand angles to match the head_dim
    angles = torch.cat([angles, angles], dim=1)  # Shape: (context_length, head_dim)

    # Precompute sine and cosine
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin


def apply_rope(x, cos, sin):
    # x: (batch_size, num_heads, seq_len, head_dim)
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"

    # Split x into first half and second half
    x1 = x[..., : head_dim // 2]  # First half
    x2 = x[..., head_dim // 2 :]  # Second half

    # Adjust sin and cos shapes
    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_len, head_dim)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)

    # Apply the rotary transformation
    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)

    # It's ok to use lower-precision after applying cos and sin rotation
    return x_rotated.to(dtype=x.dtype)

class GroupedQueryAttention(nn.Module):
    def __init__(
        self, d_in, num_heads, num_kv_groups, head_dim=None, qk_norm=False,
        query_pre_attn_scalar=None, dtype=None,
    ):
        super().__init__()
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"

        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups

        if head_dim is None:
            assert d_in % num_heads == 0, "`d_in` must be divisible by `num_heads` if `head_dim` is not set"
            head_dim = d_in // num_heads

        self.head_dim = head_dim
        self.d_out = num_heads * head_dim

        self.W_query = nn.Linear(d_in, self.d_out, bias=False, dtype=dtype)
        self.W_key = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)

        self.out_proj = nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)

        if qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=1e-6)
            self.k_norm = RMSNorm(head_dim, eps=1e-6)
        else:
            self.q_norm = self.k_norm = None

        if query_pre_attn_scalar is not None:
            self.scaling = (query_pre_attn_scalar) ** -0.5
        else:
            self.scaling = (head_dim) ** -0.5


    def forward(self, x, mask, cos, sin):
        b, num_tokens, _ = x.shape

        # Apply projections
        queries = self.W_query(x)  # (b, num_tokens, num_heads * head_dim)
        keys = self.W_key(x)       # (b, num_tokens, num_kv_groups * head_dim)
        values = self.W_value(x)   # (b, num_tokens, num_kv_groups * head_dim)

        # Reshape
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        values = values.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)

        # Optional normalization
        if self.q_norm:
            queries = self.q_norm(queries)
        if self.k_norm:
            keys = self.k_norm(keys)

        # Apply RoPE
        queries = apply_rope(queries, cos, sin)
        keys = apply_rope(keys, cos, sin)

        # Expand K and V to match number of heads
        keys = keys.repeat_interleave(self.group_size, dim=1)
        values = values.repeat_interleave(self.group_size, dim=1)

        # FlashAttention via SDPA.
        # mask=None  → global causal attention: is_causal=True lets SDPA pick the
        #              FlashAttention kernel with no materialised mask at all.
        # mask=Tensor → sliding-window attention: convert the bool block-mask
        #              (True=ignore) to an additive float mask and let SDPA use
        #              the memory-efficient kernel.
        if mask is None:
            context = F.scaled_dot_product_attention(
                queries, keys, values,
                is_causal=True,
                scale=self.scaling,
            )
        else:
            # mask is a precomputed (seq, seq) float tensor (0 / -inf);
            # broadcasts over batch & heads inside SDPA — no allocation here.
            context = F.scaled_dot_product_attention(
                queries, keys, values,
                attn_mask=mask,
                scale=self.scaling,
            )

        context = context.transpose(1, 2).reshape(b, num_tokens, self.d_out)
        return self.out_proj(context)

class TransformerBlock(nn.Module):

    def __init__(self, cfg, attn_type):
        super().__init__()
        self.attn_type = attn_type 

        self.att = GroupedQueryAttention(
            d_in=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            num_kv_groups=cfg["n_kv_groups"],
            head_dim=cfg["head_dim"],
            qk_norm=cfg["qk_norm"],
            query_pre_attn_scalar=cfg["query_pre_attn_scalar"],
            dtype=cfg["dtype"],
        )
        self.ff = FeedForward(cfg)
        self.input_layernorm = RMSNorm(cfg["emb_dim"], eps=1e-6)
        self.post_attention_layernorm = RMSNorm(cfg["emb_dim"], eps=1e-6)
        self.pre_feedforward_layernorm = RMSNorm(cfg["emb_dim"], eps=1e-6)
        self.post_feedforward_layernorm = RMSNorm(cfg["emb_dim"], eps=1e-6)

    def forward(
        self,
        x,
        float_mask_local,
        cos_global,
        sin_global,
        cos_local,
        sin_local,
    ):
        # Shortcut connection for attention block
        shortcut = x
        x = self.input_layernorm(x)

        if self.attn_type == "sliding_attention":
            attn_mask = float_mask_local   # precomputed buffer → SDPA efficient kernel
            cos = cos_local
            sin = sin_local
        else:
            attn_mask = None               # None → is_causal=True → FlashAttention kernel
            cos = cos_global
            sin = sin_global
        
        x_attn = self.att(x, attn_mask, cos, sin)
        x_attn = self.post_attention_layernorm(x_attn)
        x = shortcut + x_attn

        # Shortcut connection for feed forward block
        shortcut = x
        x_ffn = self.pre_feedforward_layernorm(x)
        x_ffn = self.ff(x_ffn)
        x_ffn = self.post_feedforward_layernorm(x_ffn)
        x = shortcut + x_ffn
        return x

class Gemma3Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg["layer_types"] is not None and len(cfg["layer_types"]) == cfg["n_layers"]
        
        # Main model parameters
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])

        self.blocks = nn.ModuleList([
            TransformerBlock(cfg, attn_type)for attn_type in cfg["layer_types"]
        ])

        self.final_norm = RMSNorm(cfg["emb_dim"], eps=1e-6)
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])
        # Weight tying: share the embedding matrix with the output projection.
        # Halves embedding parameter count and improves training stability.
        self.out_head.weight = self.tok_emb.weight
        self.cfg = cfg
        self.gradient_checkpointing = False

        self._init_weights()

        # Reusable utilities
        cos_local, sin_local = compute_rope_params(
            head_dim=cfg["head_dim"],
            theta_base=cfg["rope_local_base"],
            context_length=cfg["context_length"],
            dtype=torch.float32,
        )
        cos_global, sin_global = compute_rope_params(
            head_dim=cfg["head_dim"],
            theta_base=cfg["rope_base"],
            context_length=cfg["context_length"],
            dtype=torch.float32,
        )
        self.register_buffer("cos_local", cos_local, persistent=False)
        self.register_buffer("sin_local", sin_local, persistent=False)
        self.register_buffer("cos_global", cos_global, persistent=False)
        self.register_buffer("sin_global", sin_global, persistent=False)

        # Precompute the sliding-window float mask once — shape (seq, seq) in
        # cfg["dtype"] so SDPA receives it ready-to-use with no hot-path
        # allocation.  Global attention uses is_causal=True so needs no mask.
        float_mask_local = self._build_float_mask_local(
            cfg["context_length"], cfg["sliding_window"], cfg["dtype"]
        )
        self.register_buffer("float_mask_local", float_mask_local, persistent=False)
    
    def _init_weights(self):
        n_layers = self.cfg["n_layers"]
        residual_scale = 1.0 / (2 * n_layers) ** 0.5   # GPT-2 / LLaMA convention

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

        # Embedding (and tied out_head): init after the Linear loop so it isn't
        # overwritten by the loop above (out_head is a Linear and shares this weight).
        # std=0.02 keeps initial logits near ln(vocab_size) at startup.
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)

        # Scale projections that write *into* the residual stream by
        # 1/sqrt(2*L) so residual-stream variance stays O(1) at init
        # regardless of depth.
        for block in self.blocks:
            block.att.out_proj.weight.data.mul_(residual_scale)
            block.ff.fc3.weight.data.mul_(residual_scale)

    @staticmethod
    def _build_float_mask_local(seq_len: int, sliding_window: int, dtype) -> torch.Tensor:
        """Builds the additive float mask for sliding-window attention once at init."""
        ones     = torch.ones(seq_len, seq_len, dtype=torch.bool)
        causal   = torch.triu(ones, diagonal=1)
        far_past = torch.triu(ones, diagonal=sliding_window).T
        bool_mask = causal | far_past                          # True = block
        float_mask = torch.zeros(seq_len, seq_len, dtype=dtype)
        float_mask.masked_fill_(bool_mask, float("-inf"))
        return float_mask                                      # (seq, seq)

    def _create_masks(self, seq_len, device):
        ones = torch.ones((seq_len, seq_len), dtype=torch.bool, device=device)
    
        # mask_global (future is masked: j > i)
        #     j:  0 1 2 3 4 5 6 7
        #  i
        #     0:  0 1 1 1 1 1 1 1
        #     1:  0 0 1 1 1 1 1 1
        #     2:  0 0 0 1 1 1 1 1
        #     3:  0 0 0 0 1 1 1 1
        #     4:  0 0 0 0 0 1 1 1
        #     5:  0 0 0 0 0 0 1 1
        #     6:  0 0 0 0 0 0 0 1
        #     7:  0 0 0 0 0 0 0 0
        mask_global = torch.triu(ones, diagonal=1)
    
        # far_past (too far back is masked: i - j >= sliding_window)
        # where sliding_window = 4
        #     j:  0 1 2 3 4 5 6 7
        #  i
        #     0:  0 0 0 0 0 0 0 0
        #     1:  0 0 0 0 0 0 0 0
        #     2:  0 0 0 0 0 0 0 0
        #     3:  0 0 0 0 0 0 0 0
        #     4:  1 0 0 0 0 0 0 0
        #     5:  1 1 0 0 0 0 0 0
        #     6:  1 1 1 0 0 0 0 0
        #     7:  1 1 1 1 0 0 0 0
        far_past = torch.triu(ones, diagonal=self.cfg["sliding_window"]).T
    
        # Local (sliding_window) = future OR far-past
        # mask_local
        #     j:  0 1 2 3 4 5 6 7
        # i
        # 0:      0 1 1 1 1 1 1 1
        # 1:      0 0 1 1 1 1 1 1
        # 2:      0 0 0 1 1 1 1 1
        # 3:      0 0 0 0 1 1 1 1
        # 4:      1 0 0 0 0 1 1 1
        # 5:      1 1 0 0 0 0 1 1
        # 6:      1 1 1 0 0 0 0 1
        # 7:      1 1 1 1 0 0 0 0
        mask_local = mask_global | far_past
        return mask_global, mask_local

    def forward(self, input_ids, return_logits: bool = True):
        x = self.tok_emb(input_ids) * (self.cfg["emb_dim"] ** 0.5)

        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(
                    block,
                    x,
                    self.float_mask_local,
                    self.cos_global,
                    self.sin_global,
                    self.cos_local,
                    self.sin_local,
                    use_reentrant=False,
                )
            else:
                x = block(
                    x,
                    float_mask_local=self.float_mask_local,
                    cos_global=self.cos_global,
                    sin_global=self.sin_global,
                    cos_local=self.cos_local,
                    sin_local=self.sin_local,
                )

        x = self.final_norm(x).to(self.cfg["dtype"])
        if return_logits:
            return self.out_head(x)
        return x  # hidden states — caller handles the linear + loss

GEMMA3_CONFIG_310M = {
    "vocab_size": 32_064,
    "context_length": 32_768,
    "emb_dim": 1024,
    "n_heads": 8,
    "n_layers": 31,
    "hidden_dim": 2048,
    "head_dim": 128,
    "qk_norm": True,
    "n_kv_groups": 2,
    "rope_local_base": 10_000.0,
    "rope_base": 1_000_000.0,
    "sliding_window": 512,
    "layer_types": [
        "sliding_attention",  # 0
        "sliding_attention",  # 1
        "sliding_attention",  # 2
        "sliding_attention",  # 3
        "sliding_attention",  # 4
        "full_attention",     # 5
        "sliding_attention",  # 6
        "sliding_attention",  # 7
        "sliding_attention",  # 8
        "sliding_attention",  # 9
        "sliding_attention",  # 10
        "full_attention",     # 11
        "sliding_attention",  # 12
        "sliding_attention",  # 13
        "sliding_attention",  # 14
        "sliding_attention",  # 15
        "sliding_attention",  # 16
        "full_attention",     # 17
        "sliding_attention",  # 18
        "sliding_attention",  # 19
        "sliding_attention",  # 20
        "sliding_attention",  # 21
        "sliding_attention",  # 22
        "full_attention",     # 23
        "sliding_attention",  # 24
        "sliding_attention",  # 25
        "sliding_attention",  # 26
        "sliding_attention",  # 27
        "sliding_attention",  # 28
        "full_attention",     # 29
        "sliding_attention",  # 30
    ],
    "dtype": torch.bfloat16,
    "query_pre_attn_scalar": 128,
}

class SLM270Tokenizer:
    def __init__(self, tokenizer_dir: str):
        self._tok = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)
        self.eos_token_id = self._tok.eos_token_id
        self.pad_token_id = self._tok.pad_token_id

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text)

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids, skip_special_tokens=False)


def apply_chat_template(user_text, system_text=None):
    if (system_text is not None) and (system_text.strip() != ""):
        return f"<|system|>{system_text}<|end|><|user|>{user_text}<|end|><|assistant|>"
    return f"<|user|>{user_text}<|end|><|assistant|>"

if __name__ == "__main__":
    torch.manual_seed(42)
    model = Gemma3Model(GEMMA3_CONFIG_310M).to("cuda")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters (weight-tied): {total_params:,}")
