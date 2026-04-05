"""
chat.py  —  Interactive chat with SLM270

Usage:
    python chat.py                              # loads latest checkpoint
    python chat.py --checkpoint checkpoints/ckpt_step00113352_26.000B.pt
    python chat.py --temp 0.8 --top_p 0.9      # sampling params
    python chat.py --system "You are a helpful assistant."

Commands during chat:
    /reset   — clear conversation history
    /quit    — exit
"""

import argparse
import os
import re
import sys
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from SLM270 import Gemma3Model, GEMMA3_CONFIG_270M, SLM270Tokenizer, apply_chat_template


# ── Needed so torch.load can unpickle checkpoints saved from train.py ─────────
@dataclass
class TrainConfig:
    total_tokens: int          = 50_000_000_000
    lr_flat_until_tokens: int  = 14_000_000_000
    seq_len: int               = 1024
    batch_size: int            = 224
    grad_accum: int            = 1
    max_lr: float              = 1e-4
    min_lr: float              = 3e-5
    warmup_steps: int          = 500
    weight_decay: float        = 0.1
    grad_clip: float           = 1.0
    betas: tuple               = (0.9, 0.95)
    checkpoint_dir: str        = "checkpoints"
    ckpt_interval_tokens: int  = 1_000_000_000
    ckpt_keep: int             = 10
    val_interval_tokens: int   = 1_000_000_000
    val_samples: int           = 1_000
    seed: int                  = 42
    wandb_project: str         = "SLM270"
    log_every: int             = 10


# ── Helpers ───────────────────────────────────────────────────────────────────

def latest_checkpoint(checkpoint_dir: str) -> str:
    ckpts = sorted(
        [f for f in os.listdir(checkpoint_dir) if f.startswith("ckpt_") and f.endswith(".pt")],
        key=lambda f: int(re.search(r"ckpt_step(\d+)", f).group(1)),
    )
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}/")
    return os.path.join(checkpoint_dir, ckpts[-1])


def load_model(checkpoint_path: str, device: torch.device) -> tuple:
    print(f"Loading {checkpoint_path} …")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    state_dict = ckpt["model_state_dict"]
    unwanted = "_orig_mod."
    state_dict = {
        (k[len(unwanted):] if k.startswith(unwanted) else k): v
        for k, v in state_dict.items()
    }

    model_cfg = {**GEMMA3_CONFIG_270M, "context_length": 1024}
    model = Gemma3Model(model_cfg).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    step   = ckpt.get("opt_step", "?")
    tokens = ckpt.get("tokens_seen", 0)
    tokens_fmt = f"{tokens/1e9:.1f}B" if tokens else "?"
    print(f"  step={step:,}  tokens_seen={tokens_fmt}\n")

    return model, model_cfg


@torch.no_grad()
def generate(
    model,
    input_ids: list[int],
    tokenizer,
    device: torch.device,
    max_new_tokens: int = 512,
    temperature: float = 0.8,
    top_p: float = 0.9,
    stop_ids: set[int] = None,
) -> str:
    """Generate tokens one at a time, streaming to stdout."""
    context_length = model.cfg["context_length"]
    ids = list(input_ids)
    generated = []

    for _ in range(max_new_tokens):
        # Truncate to context window
        window = ids[-context_length:]
        x = torch.tensor([window], dtype=torch.long, device=device)

        # Slice the precomputed sliding-window mask to the actual sequence
        # length — it's built at context_length but SDPA requires it to
        # match the (seq, seq) attention weight shape exactly.
        seq_len = x.shape[1]
        original_mask = model.float_mask_local
        model.float_mask_local = original_mask[:seq_len, :seq_len]

        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)           # (1, seq, vocab)

        model.float_mask_local = original_mask

        next_logits = logits[0, -1, :]  # (vocab,)

        # Temperature + top-p sampling
        if temperature <= 0.0:
            next_id = next_logits.argmax().item()
        else:
            next_logits = next_logits / temperature
            probs = F.softmax(next_logits, dim=-1)

            if top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cumulative = torch.cumsum(sorted_probs, dim=0)
                # Remove tokens that push cumulative prob above top_p
                cutoff = (cumulative - sorted_probs) > top_p
                sorted_probs[cutoff] = 0.0
                sorted_probs /= sorted_probs.sum()
                next_id = sorted_idx[torch.multinomial(sorted_probs, 1)].item()
            else:
                next_id = torch.multinomial(probs, 1).item()

        if stop_ids and next_id in stop_ids:
            break

        ids.append(next_id)
        generated.append(next_id)

        # Decode and stream the new token
        token_str = tokenizer.decode([next_id])
        # Skip special tokens in output
        if not (token_str.startswith("<|") and token_str.endswith("|>")):
            print(token_str, end="", flush=True)

    print()  # newline after generation
    return tokenizer.decode(generated)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",      default=None,
                   help="Path to checkpoint (default: latest in checkpoints/)")
    p.add_argument("--checkpoint_dir",  default="checkpoints")
    p.add_argument("--system",          default="",
                   help="System prompt")
    p.add_argument("--temp",            type=float, default=0.8,
                   help="Sampling temperature (0 = greedy)")
    p.add_argument("--top_p",           type=float, default=0.9,
                   help="Top-p nucleus sampling")
    p.add_argument("--max_new_tokens",  type=int,   default=512)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = args.checkpoint or latest_checkpoint(args.checkpoint_dir)
    model, _ = load_model(ckpt_path, device)

    tokenizer = SLM270Tokenizer(tokenizer_dir=".")

    # Find the stop token id for <|end|>
    end_id  = tokenizer.encode("<|end|>")
    stop_ids = set(end_id) if end_id else set()
    if tokenizer.eos_token_id is not None:
        stop_ids.add(tokenizer.eos_token_id)

    print("SLM270 Chat  —  /reset to clear history, /quit to exit")
    if args.system:
        print(f"System: {args.system}")
    print(f"temp={args.temp}  top_p={args.top_p}  max_new_tokens={args.max_new_tokens}")
    print("-" * 60)

    # Conversation history as a flat token list
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

        # Build the prompt for this turn and append to history
        turn_prompt = apply_chat_template(user_input, system_text=args.system if not history_ids else None)
        turn_ids    = tokenizer.encode(turn_prompt)
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

        # Append assistant reply + end token to history
        reply_ids = tokenizer.encode(reply)
        history_ids.extend(reply_ids)
        end_token_ids = tokenizer.encode("<|end|>")
        history_ids.extend(end_token_ids)


if __name__ == "__main__":
    main()
