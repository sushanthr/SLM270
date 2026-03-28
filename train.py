"""
SLM270 training script.
  - Transformer Engine BF16
  - Gradient accumulation (effective batch = BATCH_SIZE × GRAD_ACCUM)
  - Cosine LR schedule with linear warmup
  - WandB metrics + token-throughput tracking
  - 4 evenly-spaced checkpoints over 200B token budget
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

# Transformer Engine — BF16 training via torch.autocast
import transformer_engine.pytorch as te

from quartet2.linear import Quartet_II_linear

from SLM270 import Gemma3Model, GEMMA3_CONFIG_270M, SLM270Tokenizer
from dataset import build_dataloader, build_validation_batches, PrefetchLoader


# ── Training hyper-parameters ─────────────────────────────────────────────────

@dataclass
class TrainConfig:
    # Tokens / budget
    total_tokens: int   = 50_000_000_000    # 50 B  (10× Chinchilla for 270 M params)
    seq_len: int        = 1024

    # Batch
    batch_size: int     = 32
    grad_accum: int     = 2                 # effective batch = 32 × 2 × 1024 = 65 536 tok

    # Optimiser
    max_lr: float       = 3e-4
    min_lr: float       = 3e-5
    warmup_steps: int   = 2_000             # optimiser steps (not micro-steps)
    weight_decay: float = 0.1
    grad_clip: float    = 1.0
    betas: tuple        = (0.9, 0.95)

    # Checkpointing
    n_checkpoints: int  = 4
    checkpoint_dir: str = "checkpoints"

    # Validation
    val_every: int      = 100_000           # run validation every N optimiser steps
    val_samples: int    = 1_000             # Wikipedia documents to validate on

    # Quartet-II
    fp4_bwd_warmup: int = 100               # use BF16 backward for first N opt-steps

    # Misc
    seed: int           = 42
    wandb_project: str  = "SLM270"
    log_every: int      = 10                # log to WandB every N optimiser steps


CFG = TrainConfig()

# Derived constants
TOKENS_PER_OPT_STEP = CFG.batch_size * CFG.seq_len * CFG.grad_accum   # 65 536
TOTAL_OPT_STEPS     = CFG.total_tokens // TOKENS_PER_OPT_STEP         # ≈ 3 051 757

# Checkpoint at evenly-spaced milestones (25 / 50 / 75 / 100 %)
CHECKPOINT_AT = {
    round(TOTAL_OPT_STEPS * (i + 1) / CFG.n_checkpoints)
    for i in range(CFG.n_checkpoints)
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def cosine_lr(opt_step: int) -> float:
    """Linear warmup → cosine decay to min_lr."""
    if opt_step < CFG.warmup_steps:
        return CFG.max_lr * opt_step / max(1, CFG.warmup_steps)
    if opt_step >= TOTAL_OPT_STEPS:
        return CFG.min_lr
    progress = (opt_step - CFG.warmup_steps) / max(1, TOTAL_OPT_STEPS - CFG.warmup_steps)
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


def enable_fp4_backward(model: torch.nn.Module) -> int:
    """Flips disable_backward_quant=False on every Quartet_II_linear in the model."""
    n = 0
    for m in model.modules():
        if isinstance(m, Quartet_II_linear):
            m.disable_backward_quant = False
            n += 1
    return n


# ── Quartet-II linear swap ───────────────────────────────────────────────────

def replace_linear_with_quartet(model: torch.nn.Module) -> None:
    """
    Recursively replaces every nn.Linear with Quartet_II_linear in-place.
    Copies weights so training continues from the same initialisation.
    Must be called before torch.compile so the custom autograd ops are visible
    to the tracer.
    """
    for name, child in list(model.named_children()):
        if type(child) is torch.nn.Linear:
            q = Quartet_II_linear(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                four_over_six=True,
                disable_backward_quant=True,   # starts BF16; flipped to FP4 after warmup
                device=child.weight.device,
                dtype=child.weight.dtype,
            )
            q.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                q.bias.data.copy_(child.bias.data)
            setattr(model, name, q)
        else:
            replace_linear_with_quartet(child)


# ── Validation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_validation(model, val_batches, device) -> dict:
    """
    Evaluates the model on the pre-built Wikipedia validation batches.
    Returns a dict with val/loss and val/perplexity.
    """
    model.eval()
    total_loss   = 0.0
    total_tokens = 0

    for batch in tqdm(val_batches, desc="  val", leave=False, dynamic_ncols=True):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels    = batch["labels"].to(device, non_blocking=True)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids)
            loss   = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                reduction="sum",          # sum so we can weight by token count
            )

        n_tokens      = labels.numel()
        total_loss   += loss.item()
        total_tokens += n_tokens

    model.train()

    avg_loss = total_loss / max(total_tokens, 1)
    return {
        "val/loss":       avg_loss,
        "val/perplexity": math.exp(min(avg_loss, 20)),
    }


# ── Main training loop ────────────────────────────────────────────────────────

def train() -> None:
    torch.manual_seed(CFG.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch._dynamo.config.cache_size_limit = 256          # Quartet creates many guarded frames
    device = torch.device("cuda")

    # ── Model ──────────────────────────────────────────────────────────────────
    model_cfg = {**GEMMA3_CONFIG_270M, "context_length": CFG.seq_len}
    model = Gemma3Model(model_cfg).to(device)

    # Count params before the Quartet swap (tok_emb still accessible as plain attr)
    total_params  = sum(p.numel() for p in model.parameters())
    unique_params = total_params - model.tok_emb.weight.numel()   # weight-tied head

    # Swap all nn.Linear → Quartet_II_linear (NVFP4 fwd + FP4 bwd).
    # Must happen before torch.compile so the custom_op is visible to the tracer.
    replace_linear_with_quartet(model)
    n_quartet = sum(1 for m in model.modules() if isinstance(m, Quartet_II_linear))
    print(f"Quartet-II  : {n_quartet} linear layers swapped to NVFP4")

    model.gradient_checkpointing = False   # Liger fused CE handles the memory pressure

    # torch.compile fuses RMSNorm, SwiGLU, RoPE and attention kernels.
    # fullgraph=False allows graph breaks at Quartet's custom autograd boundaries.
    model = torch.compile(model, mode="default", fullgraph=False)

    print(f"Parameters  : {total_params:,}  (unique: {unique_params:,})")
    print(f"Seq len     : {CFG.seq_len}")
    print(f"Batch (eff) : {CFG.batch_size} × {CFG.grad_accum} × {CFG.seq_len} = {TOKENS_PER_OPT_STEP:,} tok/step")
    print(f"Target      : {format_tokens(CFG.total_tokens)}  |  {TOTAL_OPT_STEPS:,} opt-steps")
    print(f"Checkpoints : {sorted(CHECKPOINT_AT)}")

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

    # ── Tokeniser + DataLoader ─────────────────────────────────────────────────
    tokenizer = SLM270Tokenizer(tokenizer_dir=".")
    loader    = PrefetchLoader(
        build_dataloader(
            tokenizer,
            seq_len=CFG.seq_len,
            batch_size=CFG.batch_size,
            seed=CFG.seed,
        ),
        buffer_size=4,
    )

    # ── Validation batches (materialised once from Wikipedia stream) ───────────
    print(f"Building validation set ({CFG.val_samples} Wikipedia samples)…")
    val_batches = build_validation_batches(
        tokenizer,
        seq_len=CFG.seq_len,
        n_samples=CFG.val_samples,
        batch_size=CFG.batch_size,
        seed=CFG.seed,
    )
    print(f"  → {len(val_batches)} validation batches  "
          f"({sum(b['input_ids'].numel() for b in val_batches):,} tokens)")

    # ── WandB ─────────────────────────────────────────────────────────────────
    wandb.init(
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
            "val_every_steps":  CFG.val_every,
            "val_samples":      CFG.val_samples,
            "linear_backend":   "quartet2_nvfp4",
        },
    )
    wandb.watch(model, log="gradients", log_freq=500)

    # ── State ──────────────────────────────────────────────────────────────────
    model.train()
    optimizer.zero_grad(set_to_none=True)

    tokens_seen     = 0
    opt_step        = 0
    micro_step      = 0
    loss_accum      = 0.0

    # Throughput window
    tput_t0         = time.perf_counter()
    tput_tokens     = 0

    # ── Progress bar ───────────────────────────────────────────────────────────
    pbar = tqdm(
        total=CFG.total_tokens,
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

    # ── Training ───────────────────────────────────────────────────────────────
    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels    = batch["labels"].to(device, non_blocking=True)

        # ── Forward (BF16 via Transformer Engine autocast) ────────────────────
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

            # ── Enable FP4 backward after warmup ──────────────────────────────
            if opt_step == CFG.fp4_bwd_warmup:
                n = enable_fp4_backward(model)
                print(f"\n  FP4 backward enabled at step {opt_step} ({n} layers)\n")
                wandb.log({"event/fp4_bwd_enabled": opt_step}, step=opt_step)

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

            # ── Validation ───────────────────────────────────────────────────
            if opt_step % CFG.val_every == 0:
                val_metrics = run_validation(model, val_batches, device)
                print(
                    f"\n  [val @ {opt_step:,} steps / {format_tokens(tokens_seen)}]"
                    f"  loss={val_metrics['val/loss']:.4f}"
                    f"  ppl={val_metrics['val/perplexity']:.2f}\n"
                )
                wandb.log(
                    {
                        **val_metrics,
                        "tokens/seen_B": tokens_seen / 1e9,
                    },
                    step=opt_step,
                )

            # ── Checkpoint ───────────────────────────────────────────────────
            if opt_step in CHECKPOINT_AT:
                path = save_checkpoint(model, optimizer, opt_step, tokens_seen)
                print(f"\n  ✓ Checkpoint {path}  ({format_tokens(tokens_seen)})\n")
                wandb.log(
                    {
                        "checkpoint/step":         opt_step,
                        "checkpoint/tokens_seen_B": tokens_seen / 1e9,
                    },
                    step=opt_step,
                )

        # ── Budget check ──────────────────────────────────────────────────────
        if tokens_seen >= CFG.total_tokens:
            break

    pbar.close()
    wandb.finish()
    print(f"\nDone.  Tokens trained: {format_tokens(tokens_seen)}  |  Opt-steps: {opt_step:,}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train()
