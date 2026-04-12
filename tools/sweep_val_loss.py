"""
ComputeValidationLoss.py

Iterates over every checkpoint in the checkpoints/ directory, runs the same
Wikipedia validation used during training, and prints a summary table.

Usage:
    python ComputeValidationLoss.py
    python ComputeValidationLoss.py --checkpoint_dir checkpoints --val_samples 1000
"""

import argparse
import math
import os
import re
import time
from dataclasses import dataclass

import torch


# Must be defined here so pickle can find it when loading checkpoints
# (train.py runs as __main__, so TrainConfig is stored as __main__.TrainConfig)
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
from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss
from tqdm import tqdm

from SLM270 import Gemma3Model, GEMMA3_CONFIG_270M, SLM270Tokenizer
from dataset import build_validation_batches, build_openwebtext_validation_batches


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--dataset",        default="wikipedia",
                   choices=["wikipedia", "openwebtext"],
                   help="Validation dataset: wikipedia (default) or openwebtext")
    p.add_argument("--val_samples",    type=int, default=1000,
                   help="Number of documents to validate on")
    p.add_argument("--batch_size",     type=int, default=224)
    p.add_argument("--seq_len",        type=int, default=1024)
    p.add_argument("--seed",           type=int, default=42)
    return p.parse_args()


# ── Validation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_validation(model, val_batches, device, fused_ce, out_head_weight):
    model.eval()
    total_loss   = 0.0
    total_tokens = 0

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    for batch in tqdm(val_batches, desc="  val", leave=False, dynamic_ncols=True):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels    = batch["labels"].to(device, non_blocking=True)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = model(input_ids, return_logits=False)          # (B, T, D)
            loss   = fused_ce(
                out_head_weight,
                hidden.reshape(-1, hidden.shape[-1]),               # (B*T, D)
                labels.reshape(-1),                                  # (B*T,)
            )

        n_tokens      = labels.numel()
        total_loss   += loss.item() * n_tokens
        total_tokens += n_tokens

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    tok_per_sec = total_tokens / elapsed

    avg_loss = total_loss / max(total_tokens, 1)
    return avg_loss, math.exp(min(avg_loss, 20)), tok_per_sec


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_checkpoint_into_model(model, path, device):
    """Load a checkpoint, stripping any _orig_mod. prefix from compiled models."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"]
    # torch.compile adds _orig_mod. prefix — strip it so we can load into a
    # plain (non-compiled) model
    unwanted = "_orig_mod."
    state_dict = {
        (k[len(unwanted):] if k.startswith(unwanted) else k): v
        for k, v in state_dict.items()
    }
    model.load_state_dict(state_dict)
    meta = {
        "opt_step":    ckpt.get("opt_step",    "?"),
        "tokens_seen": ckpt.get("tokens_seen", 0),
    }
    return meta


def parse_filename(fname):
    """Extract step and token count from filename for sorting."""
    m = re.search(r"ckpt_step(\d+)_(.+)\.pt", fname)
    if m:
        return int(m.group(1)), m.group(2)
    return 0, "?"


def format_tokens(n):
    if n >= 1_000_000_000_000:
        return f"{n / 1e12:.3f}T"
    if n >= 1_000_000_000:
        return f"{n / 1e9:.3f}B"
    return f"{n / 1e6:.2f}M"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device = torch.device("cuda")

    # Discover checkpoints
    ckpt_files = sorted(
        [f for f in os.listdir(args.checkpoint_dir)
         if f.startswith("ckpt_") and f.endswith(".pt")],
        key=lambda f: parse_filename(f)[0],
    )
    if not ckpt_files:
        print(f"No checkpoints found in {args.checkpoint_dir}/")
        return

    print(f"Found {len(ckpt_files)} checkpoints in {args.checkpoint_dir}/\n")

    # Build the model once (no torch.compile — not needed for eval)
    model_cfg = {**GEMMA3_CONFIG_270M, "context_length": args.seq_len}
    model = Gemma3Model(model_cfg).to(device)

    fused_ce       = LigerFusedLinearCrossEntropyLoss()
    out_head_weight = model.out_head.weight   # plain model, no _orig_mod layer

    # Build validation batches once
    tokenizer = SLM270Tokenizer(tokenizer_dir="tokenizer")
    print(f"Building validation set ({args.val_samples} {args.dataset} samples)…")
    if args.dataset == "openwebtext":
        val_batches = build_openwebtext_validation_batches(
            tokenizer,
            seq_len=args.seq_len,
            n_samples=args.val_samples,
            batch_size=args.batch_size,
            seed=args.seed,
        )
    else:
        val_batches = build_validation_batches(
            tokenizer,
            seq_len=args.seq_len,
            n_samples=args.val_samples,
            batch_size=args.batch_size,
            seed=args.seed,
        )
    total_val_tokens = sum(b["input_ids"].numel() for b in val_batches)
    print(f"  → {len(val_batches)} batches  ({total_val_tokens:,} tokens)\n")

    # Evaluate each checkpoint
    results = []
    for fname in ckpt_files:
        path = os.path.join(args.checkpoint_dir, fname)
        step, tok_label = parse_filename(fname)
        print(f"[{len(results)+1}/{len(ckpt_files)}]  {fname}")

        meta = load_checkpoint_into_model(model, path, device)
        val_loss, val_ppl, tok_per_sec = run_validation(model, val_batches, device, fused_ce, out_head_weight)

        results.append({
            "checkpoint": fname,
            "step":       meta["opt_step"],
            "tokens":     format_tokens(meta["tokens_seen"]) if meta["tokens_seen"] else tok_label,
            "val_loss":   val_loss,
            "val_ppl":    val_ppl,
            "tok_per_sec": tok_per_sec,
        })
        print(f"  loss={val_loss:.4f}  ppl={val_ppl:.2f}  ({tok_per_sec:,.0f} tok/s)\n")

    # ── Summary table ─────────────────────────────────────────────────────────
    col_ckpt  = max(len(r["checkpoint"]) for r in results)
    col_step  = max(len(str(r["step"]))  for r in results)
    col_tok   = max(len(r["tokens"])     for r in results)

    header = (
        f"{'Checkpoint':<{col_ckpt}}  "
        f"{'Step':>{col_step}}  "
        f"{'Tokens':>{col_tok}}  "
        f"{'Val Loss':>9}  "
        f"{'Perplexity':>10}  "
        f"{'tok/s':>10}"
    )
    sep = "-" * len(header)

    print("\n" + sep)
    print(header)
    print(sep)
    for r in results:
        print(
            f"{r['checkpoint']:<{col_ckpt}}  "
            f"{str(r['step']):>{col_step}}  "
            f"{r['tokens']:>{col_tok}}  "
            f"{r['val_loss']:>9.4f}  "
            f"{r['val_ppl']:>10.2f}  "
            f"{r['tok_per_sec']:>10,.0f}"
        )
    print(sep)

    best = min(results, key=lambda r: r["val_loss"])
    print(f"\nBest: {best['checkpoint']}  loss={best['val_loss']:.4f}  ppl={best['val_ppl']:.2f}")


if __name__ == "__main__":
    main()
