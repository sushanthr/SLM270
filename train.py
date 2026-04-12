"""
SLM270 training script.
  - BF16 autocast
  - Gradient accumulation (effective batch = BATCH_SIZE × GRAD_ACCUM)
  - Cosine LR schedule with linear warmup
  - Liger kernels: fused linear cross-entropy (lm_head)
  - WandB metrics + token-throughput tracking
  - Validation and checkpointing every 1B tokens (retain last 2 checkpoints)
"""

import os
import math
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import wandb
from tqdm import tqdm
from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss

from SLM270 import Gemma3Model, GEMMA3_CONFIG_270M, SLM270Tokenizer
from dataset import build_dataloader, build_climbmix_validation_batches, PrefetchLoader


# ── Training hyper-parameters ─────────────────────────────────────────────────

@dataclass
class TrainConfig:
    # Tokens / budget
    # nanochat uses target_param_data_ratio=12 × scaling_params (transformer_matrices + lm_head).
    # For nanochat's ~760M total model that works out to 12 × 435M ≈ 5.22B tokens.
    # Matching that here so runs are directly comparable.
    total_tokens: int          = 5_220_000_000     # 5.22 B  (matches nanochat's compute-optimal ratio=12)
    lr_flat_until_tokens: int  = 1_500_000_000     # hold max_lr for first ~29% of training, then cosine decay
    seq_len: int               = 1024

    # Batch
    batch_size: int     = 224
    grad_accum: int     = 1                 # effective batch = 64 × 1 × 1024 = 65 536 tok

    # Optimiser
    max_lr: float       = 1e-4
    min_lr: float       = 3e-5
    warmup_steps: int   = 500               # optimiser steps (not micro-steps)
    weight_decay: float = 0.1
    grad_clip: float    = 1.0
    betas: tuple        = (0.9, 0.95)

    # Checkpointing
    checkpoint_dir: str  = "checkpoints"
    ckpt_interval_tokens: int = 1_000_000_000   # checkpoint every 1B tokens
    ckpt_keep: int           = 10               # number of recent checkpoints to retain

    # Validation
    val_interval_tokens: int = 1_000_000_000    # run validation every 1B tokens
    val_samples: int         = 1_000            # ClimbMix documents to validate on
    val_batch_size: int      = 8                # small batch for val: 262K vocab × fp32 logits is ~8.6 GB at B=8

    # Misc
    seed: int           = 42
    wandb_project: str  = "SLM270"
    log_every: int      = 10                # log to WandB every N optimiser steps


CFG = TrainConfig()

# ── Resume settings (set both to None for a fresh run) ───────────────────────
RESUME_CHECKPOINT      = "checkpoints/ckpt_step00061036_14.000B.pt"
RESUME_WEIGHTS_ONLY    = True   # True = load model weights only, reset optimizer
WANDB_RESUME_RUN_ID    = "1rclqqot"

# Derived constants
TOKENS_PER_OPT_STEP = CFG.batch_size * CFG.seq_len * CFG.grad_accum   # 262 144
TOTAL_OPT_STEPS     = CFG.total_tokens // TOKENS_PER_OPT_STEP         # ≈ 190 735


# ── Helpers ───────────────────────────────────────────────────────────────────

def cosine_lr(opt_step: int) -> float:
    """Linear warmup → flat at max_lr until lr_flat_until_tokens → cosine decay to min_lr."""
    decay_start_step = CFG.lr_flat_until_tokens // TOKENS_PER_OPT_STEP
    if opt_step < CFG.warmup_steps:
        return CFG.max_lr * opt_step / max(1, CFG.warmup_steps)
    if opt_step < decay_start_step:
        return CFG.max_lr
    if opt_step >= TOTAL_OPT_STEPS:
        return CFG.min_lr
    progress = (opt_step - decay_start_step) / max(1, TOTAL_OPT_STEPS - decay_start_step)
    coeff    = 0.5 * (1.0 + math.cos(math.pi * progress))
    return CFG.min_lr + coeff * (CFG.max_lr - CFG.min_lr)


def format_tokens(n: int) -> str:
    """Human-readable token count: M / B / T."""
    if n >= 1_000_000_000_000:
        return f"{n / 1e12:.3f}T"
    if n >= 1_000_000_000:
        return f"{n / 1e9:.3f}B"
    return f"{n / 1e6:.2f}M"


def set_lr(optimizer, lr: float) -> None:
    for pg in optimizer.param_groups:
        pg["lr"] = lr


def save_checkpoint(model, optimizer, opt_step: int, tokens_seen: int) -> str:
    os.makedirs(CFG.checkpoint_dir, exist_ok=True)
    path = os.path.join(
        CFG.checkpoint_dir,
        f"ckpt_step{opt_step:08d}_{format_tokens(tokens_seen)}.pt",
    )
    torch.save(
        {
            "opt_step":          opt_step,
            "tokens_seen":       tokens_seen,
            "model_state_dict":  model.state_dict(),
            "optim_state_dict":  optimizer.state_dict(),
            "train_config":      CFG,
        },
        path,
    )
    return path


def rotate_checkpoints() -> None:
    """Delete old checkpoints, keeping only the CFG.ckpt_keep most recent ones."""
    ckpts = sorted(
        [
            os.path.join(CFG.checkpoint_dir, f)
            for f in os.listdir(CFG.checkpoint_dir)
            if f.startswith("ckpt_") and f.endswith(".pt")
        ],
        key=os.path.getmtime,
    )
    for old in ckpts[: -CFG.ckpt_keep]:
        os.remove(old)
        print(f"  ✗ Removed old checkpoint {old}")


# ── Validation ────────────────────────────────────────────────────────────────

def build_token_bytes(tokenizer, vocab_size: int, device) -> torch.Tensor:
    """
    Build a (vocab_size,) int32 tensor mapping each token id to its UTF-8 byte length.
    Special tokens (eos, pad, and anything in all_special_ids) get 0 so they are
    excluded from the bits-per-byte metric, matching nanochat's evaluate_bpb logic.
    """
    tok = tokenizer._tok
    special_ids = set(tok.all_special_ids) if tok.all_special_ids else set()
    sizes = []
    for token_id in range(vocab_size):
        if token_id in special_ids:
            sizes.append(0)
        else:
            decoded = tok.decode([token_id], skip_special_tokens=False)
            sizes.append(max(0, len(decoded.encode("utf-8"))))
    return torch.tensor(sizes, dtype=torch.int32, device=device)


@torch.no_grad()
def run_validation_bpb(model, val_batches, device, token_bytes) -> dict:
    """
    Evaluates the model on ClimbMix validation batches and returns val/bpb.

    bits-per-byte = sum(per_token_nats * (bytes > 0)) / (log(2) * sum(bytes))

    This is vocab-size-independent (unlike mean CE loss), so it stays
    comparable if the tokenizer or vocab size changes.  Special tokens
    contribute 0 bytes and are excluded from both the numerator and denominator.
    """
    model.eval()
    total_nats  = 0.0
    total_bytes = 0

    for batch in tqdm(val_batches, desc="  val", leave=False, dynamic_ncols=True):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels    = batch["labels"].to(device, non_blocking=True)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids, return_logits=True)             # (B, T, V)
        logits = logits.float()                                        # fp32 for stable CE

        loss_per_tok = F.cross_entropy(
            logits.view(-1, logits.shape[-1]),
            labels.view(-1),
            reduction="none",
        )                                                              # (B*T,)

        labels_flat    = labels.view(-1)
        bytes_per_tok  = token_bytes[labels_flat]                     # (B*T,)
        valid          = bytes_per_tok > 0
        total_nats    += (loss_per_tok * valid.float()).sum().item()
        total_bytes   += bytes_per_tok.sum().item()

    model.train()

    bpb = total_nats / (math.log(2) * total_bytes) if total_bytes > 0 else float("inf")
    return {"val/bpb": bpb}

# ── Main training loop ────────────────────────────────────────────────────────

def train() -> None:
    torch.manual_seed(CFG.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")

    # ── Model ──────────────────────────────────────────────────────────────────
    model_cfg = {**GEMMA3_CONFIG_270M, "context_length": CFG.seq_len}
    model = Gemma3Model(model_cfg).to(device)

    total_params  = sum(p.numel() for p in model.parameters())
    unique_params = total_params - model.tok_emb.weight.numel()   # weight-tied head

    model.gradient_checkpointing = True

    # torch.compile fuses RMSNorm, GeGLU, RoPE and attention kernels.
    model = torch.compile(model, mode="default", fullgraph=True)

    print(f"Parameters  : {total_params:,}  (unique: {unique_params:,})")
    print(f"Seq len     : {CFG.seq_len}")
    print(f"Batch (eff) : {CFG.batch_size} × {CFG.grad_accum} × {CFG.seq_len} = {TOKENS_PER_OPT_STEP:,} tok/step")
    print(f"Target      : {format_tokens(CFG.total_tokens)}  |  {TOTAL_OPT_STEPS:,} opt-steps")
    print(f"Checkpoints : every {format_tokens(CFG.ckpt_interval_tokens)}, keep last {CFG.ckpt_keep}")

    # Fused linear + cross-entropy: never materialises the (B×T × 262 144) logit
    # tensor — saves ~4.3 GB of VRAM per micro-batch.
    fused_ce = LigerFusedLinearCrossEntropyLoss()
    # out_head weight — unwrap the compiled model to reach the parameter directly
    out_head_weight = model._orig_mod.out_head.weight

    # ── Optimiser ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CFG.max_lr,
        betas=CFG.betas,
        weight_decay=CFG.weight_decay,
        fused=True,
    )

    # ── Tokeniser ─────────────────────────────────────────────────────────────
    tokenizer = SLM270Tokenizer(tokenizer_dir="tokenizer")

    # token_bytes: maps each token id → UTF-8 byte length (0 for special tokens)
    # Used in val/bpb computation. Built once here so validation is fast.
    print("Building token_bytes tensor…")
    token_bytes_tensor = build_token_bytes(tokenizer, GEMMA3_CONFIG_270M["vocab_size"], device)

    # ── Validation batches (materialised once from last ClimbMix shard) ─────────
    print(f"Building validation set ({CFG.val_samples} ClimbMix samples, last local shard)…")
    val_batches = build_climbmix_validation_batches(
        tokenizer,
        seq_len=CFG.seq_len,
        n_samples=CFG.val_samples,
        batch_size=CFG.val_batch_size,
        seed=CFG.seed,
    )
    print(f"  → {len(val_batches)} validation batches  "
          f"({sum(b['input_ids'].numel() for b in val_batches):,} tokens)")

    # ── WandB ─────────────────────────────────────────────────────────────────
    wandb_kwargs = dict(
        project=CFG.wandb_project,
        config={
            "total_tokens_B":   CFG.total_tokens / 1e9,
            "seq_len":          CFG.seq_len,
            "batch_size":       CFG.batch_size,
            "grad_accum":       CFG.grad_accum,
            "eff_batch_tokens": TOKENS_PER_OPT_STEP,
            "total_opt_steps":  TOTAL_OPT_STEPS,
            "max_lr":           CFG.max_lr,
            "min_lr":           CFG.min_lr,
            "warmup_steps":     CFG.warmup_steps,
            "weight_decay":     CFG.weight_decay,
            "params_total":     total_params,
            "params_unique":    unique_params,
            "val_interval_tokens_B": CFG.val_interval_tokens / 1e9,
            "val_samples":      CFG.val_samples,
        },
    )
    if WANDB_RESUME_RUN_ID:
        wandb_kwargs["id"]     = WANDB_RESUME_RUN_ID
        wandb_kwargs["resume"] = "must"
    wandb.init(**wandb_kwargs)
    # wandb.watch(model, log="gradients", log_freq=500)

    # ── State ──────────────────────────────────────────────────────────────────
    model.train()
    optimizer.zero_grad(set_to_none=True)

    tokens_seen = 0
    opt_step    = 0
    micro_step  = 0
    loss_accum  = 0.0

    # ── Resume from checkpoint ─────────────────────────────────────────────────
    if RESUME_CHECKPOINT:
        print(f"Loading checkpoint {RESUME_CHECKPOINT}…")
        ckpt = torch.load(RESUME_CHECKPOINT, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if not RESUME_WEIGHTS_ONLY:
            optimizer.load_state_dict(ckpt["optim_state_dict"])
        tokens_seen = ckpt["tokens_seen"]
        opt_step    = ckpt["opt_step"]
        del ckpt
        print(f"  → opt_step={opt_step:,}  tokens_seen={format_tokens(tokens_seen)}")

    # Next val/ckpt boundaries (rounded up to the next interval from where we are)
    next_val_tokens  = ((tokens_seen // CFG.val_interval_tokens)  + 1) * CFG.val_interval_tokens
    next_ckpt_tokens = ((tokens_seen // CFG.ckpt_interval_tokens) + 1) * CFG.ckpt_interval_tokens

    # Throughput window
    tput_t0         = time.perf_counter()
    tput_tokens     = 0

    # ── Progress bar ───────────────────────────────────────────────────────────
    pbar = tqdm(
        total=CFG.total_tokens,
        initial=tokens_seen,
        unit="tok",
        dynamic_ncols=True,
        bar_format=(
            "{desc} |{bar}| "
            "{percentage:5.2f}%  "
            "{elapsed}<{remaining}"
        ),
    )

    def update_pbar(tok_per_sec: float, loss: float) -> None:
        pbar.update(TOKENS_PER_OPT_STEP)
        seen_fmt = format_tokens(tokens_seen)
        # Colour-hint the scale bracket
        if tokens_seen >= 1e12:
            scale = "T"
        elif tokens_seen >= 1e9:
            scale = "B"
        else:
            scale = "M"
        pbar.set_description(
            f"[{seen_fmt} / {format_tokens(CFG.total_tokens)} {scale}]"
            f"  loss={loss:.4f}"
            f"  {tok_per_sec:>8,.0f} tok/s"
        )

    # ── DataLoader — fast-skip past already-seen docs ─────────────────────────
    skip_samples = 0
    if tokens_seen > 0:
        # Sample 200 docs to estimate avg tokens per JSONL line (takes ~1 s).
        # skip_samples = tokens_seen / avg_tokens_per_doc
        # Packing buffer error (≤ seq_len tokens) is negligible at this scale.
        import json as _json, glob as _glob, os as _os
        _shards = sorted(
            _glob.glob(_os.path.join(_os.path.dirname(_os.path.abspath("dataset.py")), "dataset", "part_*.jsonl")),
            key=lambda p: int(_os.path.basename(p).split("_")[1].split(".")[0]),
        )
        _sample_tokens, _n = 0, 0
        for _path in _shards:
            with open(_path) as _f:
                for _line in _f:
                    _obj = _json.loads(_line)
                    _ids = tokenizer.encode(_obj.get("text", ""))
                    if _ids:
                        _sample_tokens += len(_ids)
                        _n += 1
                    if _n >= 200:
                        break
            if _n >= 200:
                break
        avg_tokens_per_doc = _sample_tokens / _n
        skip_samples = int(tokens_seen / avg_tokens_per_doc)
        print(f"Avg tokens/doc: {avg_tokens_per_doc:.0f}  →  "
              f"skipping {skip_samples:,} JSONL docs (no tokenisation for skipped docs)")

    loader = PrefetchLoader(
        build_dataloader(
            tokenizer,
            seq_len=CFG.seq_len,
            batch_size=CFG.batch_size,
            seed=CFG.seed,
            skip_samples=skip_samples,
        ),
        buffer_size=4,
    )
    loader_iter = iter(loader)

    # ── Initial validation to confirm the checkpoint loaded correctly ──────────
    if RESUME_CHECKPOINT:
        print(f"\nInitial validation (checkpoint sanity-check)…")
        val_metrics = run_validation_bpb(model, val_batches, device, token_bytes_tensor)
        print(
            f"  [val @ resume / {format_tokens(tokens_seen)}]"
            f"  bpb={val_metrics['val/bpb']:.4f}\n"
        )
        wandb.log(
            {**val_metrics, "tokens/seen_B": tokens_seen / 1e9},
            step=opt_step,
        )

    # ── Training ───────────────────────────────────────────────────────────────
    for batch in loader_iter:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels    = batch["labels"].to(device, non_blocking=True)

        # ── Forward ───────────────────────────────────────────────────────────
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = model(input_ids, return_logits=False)          # (B, T, D)
            loss   = fused_ce(
                out_head_weight,
                hidden.reshape(-1, hidden.shape[-1]),               # (B*T, D)
                labels.reshape(-1),                                  # (B*T,)
            )
            loss   = loss / CFG.grad_accum                         # scale for accum

        loss.backward()

        batch_tokens  = input_ids.numel()
        tokens_seen  += batch_tokens
        tput_tokens  += batch_tokens
        loss_accum   += loss.item()
        micro_step   += 1

        # ── Optimiser step every grad_accum micro-batches ──────────────────────
        if micro_step % CFG.grad_accum == 0:
            opt_step += 1

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), CFG.grad_clip
            )
            lr = cosine_lr(opt_step)
            set_lr(optimizer, lr)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # ── Throughput ────────────────────────────────────────────────────
            now        = time.perf_counter()
            elapsed    = now - tput_t0
            tok_per_sec = tput_tokens / elapsed if elapsed > 0 else 0.0
            tput_t0    = now
            tput_tokens = 0

            displayed_loss = loss_accum   # already the mean CE loss
            loss_accum     = 0.0

            update_pbar(tok_per_sec, displayed_loss)

            # ── WandB logging ─────────────────────────────────────────────────
            if opt_step % CFG.log_every == 0:
                wandb.log(
                    {
                        "train/loss":         displayed_loss,
                        "train/perplexity":   math.exp(min(displayed_loss, 20)),
                        "train/lr":           lr,
                        "train/grad_norm":    grad_norm.item(),
                        "tokens/seen":        tokens_seen,
                        "tokens/seen_M":      tokens_seen / 1e6,
                        "tokens/seen_B":      tokens_seen / 1e9,
                        "tokens/seen_T":      tokens_seen / 1e12,
                        "perf/tok_per_sec":   tok_per_sec,
                        "perf/opt_step":      opt_step,
                    },
                    step=opt_step,
                )

            # ── Validation (every 1B tokens) ──────────────────────────────
            if tokens_seen >= next_val_tokens:
                val_metrics = run_validation_bpb(model, val_batches, device, token_bytes_tensor)
                print(
                    f"\n  [val @ {opt_step:,} steps / {format_tokens(tokens_seen)}]"
                    f"  bpb={val_metrics['val/bpb']:.4f}\n"
                )
                wandb.log(
                    {
                        **val_metrics,
                        "tokens/seen_B": tokens_seen / 1e9,
                    },
                    step=opt_step,
                )
                next_val_tokens += CFG.val_interval_tokens

            # ── Checkpoint (every 1B tokens, keep last 2) ─────────────────
            if tokens_seen >= next_ckpt_tokens:
                path = save_checkpoint(model, optimizer, opt_step, tokens_seen)
                print(f"\n  ✓ Checkpoint {path}  ({format_tokens(tokens_seen)})\n")
                rotate_checkpoints()
                wandb.log(
                    {
                        "checkpoint/step":         opt_step,
                        "checkpoint/tokens_seen_B": tokens_seen / 1e9,
                    },
                    step=opt_step,
                )
                next_ckpt_tokens += CFG.ckpt_interval_tokens

        # ── Budget check ──────────────────────────────────────────────────────
        if tokens_seen >= CFG.total_tokens:
            break

    pbar.close()
    wandb.finish()
    print(f"\nDone.  Tokens trained: {format_tokens(tokens_seen)}  |  Opt-steps: {opt_step:,}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train()
