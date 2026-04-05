"""
chat.py  —  Interactive chat with SLM270

Usage:
    python chat.py                              # loads latest checkpoint
    python chat.py --checkpoint checkpoints/sft_ckpt_step00001000.pt
    python chat.py --temp 0.8 --top_p 0.9
    python chat.py --system "You are a helpful assistant."

Commands during chat:
    /reset   — clear conversation history
    /quit    — exit

Tool calls:
    After SFT, the model may emit <|tool_call|>JSON<|/tool_call|> blocks.
    chat.py intercepts these, executes them via math_tools.run_tool_calls,
    and injects <|tool_response|>result<|/tool_response|> back into context
    before continuing generation.  Multiple rounds per turn are supported.
"""

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from SLM270 import Gemma3Model, GEMMA3_CONFIG_270M, SLM270Tokenizer
from math_tools import run_tool_calls


# ── ANSI colours ──────────────────────────────────────────────────────────────
_CYAN   = "\033[36m"
_YELLOW = "\033[33m"
_GREEN  = "\033[32m"
_DIM    = "\033[2m"
_BOLD   = "\033[1m"
_RESET  = "\033[0m"

def _c(text, *codes):
    return "".join(codes) + text + _RESET


# ── Dataclasses needed to unpickle checkpoints ────────────────────────────────
@dataclass
class TrainConfig:
    total_tokens: int         = 50_000_000_000
    lr_flat_until_tokens: int = 14_000_000_000
    seq_len: int              = 1024
    batch_size: int           = 224
    grad_accum: int           = 1
    max_lr: float             = 1e-4
    min_lr: float             = 3e-5
    warmup_steps: int         = 500
    weight_decay: float       = 0.1
    grad_clip: float          = 1.0
    betas: tuple              = (0.9, 0.95)
    checkpoint_dir: str       = "checkpoints"
    ckpt_interval_tokens: int = 1_000_000_000
    ckpt_keep: int            = 10
    val_interval_tokens: int  = 1_000_000_000
    val_samples: int          = 1_000
    seed: int                 = 42
    wandb_project: str        = "SLM270"
    log_every: int            = 10

@dataclass
class SFTConfig:
    pretrain_checkpoint: str  = ""
    seq_len: int              = 1024
    batch_size: int           = 32
    grad_accum: int           = 2
    max_lr: float             = 3e-5
    min_lr: float             = 0.0
    warmup_ratio: float       = 0.05
    warmdown_ratio: float     = 0.30
    betas: tuple              = (0.9, 0.95)
    weight_decay: float       = 0.0
    grad_clip: float          = 1.0
    mmlu_epochs: int          = 3
    gsm8k_epochs: int         = 4
    identity_file: str        = "identity_conversations.jsonl"
    pack_buffer: int          = 200
    val_every_steps: int      = 200
    val_batch_size: int       = 16
    val_max_batches: int      = 80
    checkpoint_dir: str       = "checkpoints"
    ckpt_every_steps: int     = 500
    ckpt_keep: int            = 5
    seed: int                 = 42
    wandb_project: str        = "SLM270-SFT"
    log_every: int            = 10


# ── Checkpoint loading ────────────────────────────────────────────────────────

def latest_checkpoint(checkpoint_dir: str) -> str:
    """Return the most recent checkpoint (SFT preferred over pretrain)."""
    ckpts = [
        f for f in os.listdir(checkpoint_dir)
        if f.endswith(".pt") and (f.startswith("sft_ckpt_") or f.startswith("ckpt_"))
    ]
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}/")
    # SFT checkpoints first, then pretrain; within each group sort by step
    def sort_key(f):
        is_sft = f.startswith("sft_")
        m = re.search(r"step(\d+)", f)
        step = int(m.group(1)) if m else 0
        return (is_sft, step)
    return os.path.join(checkpoint_dir, sorted(ckpts, key=sort_key)[-1])


def load_model(checkpoint_path: str, device: torch.device):
    print(f"Loading {checkpoint_path} …")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    state_dict = ckpt["model_state_dict"]
    state_dict = {
        (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
        for k, v in state_dict.items()
    }

    model_cfg = {**GEMMA3_CONFIG_270M, "context_length": 1024}
    model = Gemma3Model(model_cfg).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    step   = ckpt.get("sft_step", ckpt.get("opt_step", "?"))
    tokens = ckpt.get("tokens_seen", 0)
    label  = "SFT" if "sft_step" in ckpt else "pretrain"
    print(f"  [{label}] step={step}  tokens_seen={tokens/1e9:.1f}B\n")
    return model


# ── Sampling ──────────────────────────────────────────────────────────────────

def _sample(logits: torch.Tensor, temperature: float, top_p: float) -> int:
    if temperature <= 0.0:
        return logits.argmax().item()
    logits = logits / temperature
    probs  = F.softmax(logits, dim=-1)
    if top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=0)
        cutoff = (cumulative - sorted_probs) > top_p
        sorted_probs[cutoff] = 0.0
        sorted_probs /= sorted_probs.sum()
        return sorted_idx[torch.multinomial(sorted_probs, 1)].item()
    return torch.multinomial(probs, 1).item()


def _next_token(model, ids: list[int], device: torch.device) -> torch.Tensor:
    """Run one forward pass and return the logits for the last position."""
    context_length = model.cfg["context_length"]
    window = ids[-context_length:]
    x      = torch.tensor([window], dtype=torch.long, device=device)
    seq_len = x.shape[1]
    orig = model.float_mask_local
    model.float_mask_local = orig[:seq_len, :seq_len]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(x)
    model.float_mask_local = orig
    return logits[0, -1, :]   # (vocab,)


# ── Generation with tool-call interception ────────────────────────────────────

@torch.no_grad()
def generate(
    model,
    input_ids: list[int],
    tokenizer,
    device: torch.device,
    max_new_tokens: int = 512,
    temperature: float  = 0.8,
    top_p: float        = 0.9,
    stop_ids: set[int]  = None,
) -> str:
    """
    Generate tokens one at a time, streaming to stdout.

    When <|tool_call|> … <|/tool_call|> is detected:
      1. Print the tool call in dim cyan.
      2. Execute it via math_tools.run_tool_calls.
      3. Print and inject <|tool_response|>result<|/tool_response|>.
      4. Continue generation (supports multiple rounds per turn).
    """
    raw_tok = tokenizer._tok

    # Special token IDs
    TC_OPEN  = raw_tok.encode("<|tool_call|>",    add_special_tokens=False)[0]  # 100
    TC_CLOSE = raw_tok.encode("<|/tool_call|>",   add_special_tokens=False)[0]  # 99
    TR_OPEN  = raw_tok.encode("<|tool_response|>", add_special_tokens=False)[0] # 98
    TR_CLOSE_IDS = raw_tok.encode("<|/tool_response|>", add_special_tokens=False)

    ids = list(input_ids)
    generated_ids: list[int] = []   # only model-generated tokens (not injected)

    in_tool_call    = False
    tc_open_pos     = None   # position in ids[] of the TC_OPEN token
    tc_token_buffer: list[int] = []

    steps = 0

    while steps < max_new_tokens:
        logits  = _next_token(model, ids, device)
        next_id = _sample(logits, temperature, top_p)
        steps  += 1

        # ── Stop on <|end|> / EOS (only outside a tool call) ──────────────
        if (not in_tool_call) and stop_ids and (next_id in stop_ids):
            break

        ids.append(next_id)
        generated_ids.append(next_id)

        # ── Tool-call open ─────────────────────────────────────────────────
        if next_id == TC_OPEN and not in_tool_call:
            in_tool_call    = True
            tc_open_pos     = len(ids) - 1
            tc_token_buffer = []
            print(_c("\n<|tool_call|>", _CYAN, _DIM), end="", flush=True)
            continue

        # ── Inside a tool call — buffer tokens ────────────────────────────
        if in_tool_call and next_id != TC_CLOSE:
            tc_token_buffer.append(next_id)
            tok_str = raw_tok.decode([next_id], skip_special_tokens=False)
            print(_c(tok_str, _CYAN, _DIM), end="", flush=True)
            continue

        # ── Tool-call close ────────────────────────────────────────────────
        if in_tool_call and next_id == TC_CLOSE:
            print(_c("<|/tool_call|>", _CYAN, _DIM), flush=True)
            in_tool_call = False

            # Decode the captured tool call JSON
            tc_json = raw_tok.decode(tc_token_buffer, skip_special_tokens=False).strip()

            # Execute
            try:
                results  = run_tool_calls(tc_json)
                tr_json  = json.dumps(results)
                result_display = tr_json
                ok = True
            except Exception as exc:
                tr_json = json.dumps({"error": str(exc)})
                result_display = tr_json
                ok = False

            # Display the injected tool response
            color = _YELLOW if ok else "\033[31m"   # yellow or red on error
            print(_c(f"<|tool_response|>{result_display}<|/tool_response|>",
                     color, _DIM), flush=True)

            # Inject into context (NOT counted as model-generated)
            tr_prefix_ids = raw_tok.encode(
                f"<|tool_response|>{tr_json}", add_special_tokens=False
            )
            tr_suffix_ids = TR_CLOSE_IDS
            ids.extend(tr_prefix_ids + tr_suffix_ids)
            # Remove these from generated_ids since we injected them
            del generated_ids[-1]   # remove TC_CLOSE we counted

            tc_token_buffer = []
            continue

        # ── Normal token ───────────────────────────────────────────────────
        tok_str = raw_tok.decode([next_id], skip_special_tokens=False)
        # Print visible tokens; skip lone special-token markers
        if not (tok_str.startswith("<|") and tok_str.endswith("|>")):
            print(tok_str, end="", flush=True)

    print()   # newline at end of turn
    return raw_tok.decode(generated_ids, skip_special_tokens=True)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",     default=None)
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--system",         default="",
                   help="Optional system prompt")
    p.add_argument("--temp",           type=float, default=0.8)
    p.add_argument("--top_p",          type=float, default=0.9)
    p.add_argument("--max_new_tokens", type=int,   default=512)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = args.checkpoint or latest_checkpoint(args.checkpoint_dir)
    model     = load_model(ckpt_path, device)
    tokenizer = SLM270Tokenizer(tokenizer_dir=".")
    raw_tok   = tokenizer._tok

    # Stop tokens
    end_ids  = raw_tok.encode("<|end|>", add_special_tokens=False)
    stop_ids = set(end_ids)
    if tokenizer.eos_token_id is not None:
        stop_ids.add(tokenizer.eos_token_id)

    print("SLM270 Chat  —  /reset to clear history, /quit to exit")
    if args.system:
        print(f"System: {args.system}")
    print(f"temp={args.temp}  top_p={args.top_p}  max_new_tokens={args.max_new_tokens}")
    print("-" * 60)

    history_ids: list[int] = []

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user_input:
            continue
        if user_input.lower() == "/quit":
            print("Bye.")
            break
        if user_input.lower() == "/reset":
            history_ids = []
            print("  [conversation reset]")
            continue

        # Build this turn's prompt tokens
        if not history_ids and args.system:
            prefix = f"<|system|>{args.system}<|end|>"
            turn_str = prefix + f"<|user|>{user_input}<|end|><|assistant|>"
        else:
            turn_str = f"<|user|>{user_input}<|end|><|assistant|>"

        # First turn gets BOS; subsequent turns do not
        turn_ids = raw_tok.encode(
            turn_str, add_special_tokens=(len(history_ids) == 0)
        )
        history_ids.extend(turn_ids)

        print("\nAssistant: ", end="", flush=True)
        reply = generate(
            model,
            history_ids,
            tokenizer,
            device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temp,
            top_p=args.top_p,
            stop_ids=stop_ids,
        )

        # Append assistant reply + <|end|> to history
        reply_ids = raw_tok.encode(reply, add_special_tokens=False)
        end_id    = raw_tok.encode("<|end|>", add_special_tokens=False)
        history_ids.extend(reply_ids + end_id)


if __name__ == "__main__":
    main()
