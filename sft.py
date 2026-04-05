"""
SLM270 Supervised Fine-Tuning (SFT) script.
  - Starts from the 26B-token pretrained checkpoint
  - Data mixture: SmolTalk + GSM8KToolCall x4 + MMLU x3 + identity x2
    + OrcaMathRewritten (if parquet present)
  - Loss masked to assistant turns only; best-fit sequence packing
  - Tool-call turns trained on (mask=1); tool-response turns masked out (mask=0)
    so the model learns to call tools and write explanations, not to echo results
  - BF16 autocast + Liger fused linear CE (no logit materialisation)
  - LR schedule: linear warmup → flat → linear warmdown (progress-based)
  - WandB metrics; checkpoints saved with "sft_" prefix
"""

import os, sys, math, time, json
from dataclasses import dataclass

import torch
import wandb
from tqdm import tqdm
from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss

from SLM270 import Gemma3Model, GEMMA3_CONFIG_270M, SLM270Tokenizer

# nanochat task classes live one directory up
_NANOCHAT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "nanochat")
sys.path.insert(0, _NANOCHAT)
from tasks.common import TaskMixture
from tasks.mmlu import MMLU
from tasks.smoltalk import SmolTalk
from tasks.customjson import CustomJSON
from tasks.spellingbee import SimpleSpelling, SpellingBee

# SLM270-specific tool-call task classes
from gsm8k_toolcall import GSM8KToolCall
from orca_math import OrcaMathRewritten


# ── Configuration ─────────────────────────────────────────────────────────────

# Stub so pickle can deserialize pretrain checkpoints saved by train.py (__main__.TrainConfig)
@dataclass
class TrainConfig:
    total_tokens: int = 50_000_000_000
    lr_flat_until_tokens: int = 14_000_000_000
    seq_len: int = 1024
    batch_size: int = 224
    grad_accum: int = 1
    max_lr: float = 1e-4
    min_lr: float = 3e-5
    warmup_steps: int = 500
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    betas: tuple = (0.9, 0.95)
    checkpoint_dir: str = "checkpoints"
    ckpt_interval_tokens: int = 1_000_000_000
    ckpt_keep: int = 10
    val_interval_tokens: int = 1_000_000_000
    val_samples: int = 1_000
    seed: int = 42
    wandb_project: str = "SLM270"
    log_every: int = 10


@dataclass
class SFTConfig:
    # Starting checkpoint (26 B tokens)
    pretrain_checkpoint: str = "checkpoints/ckpt_step00113352_26.000B.pt"

    # Sequence / batch
    seq_len:    int   = 1024
    batch_size: int   = 32       # micro-batch sequences per step
    grad_accum: int   = 2        # effective ≈ 32 × 2 × 1024 = 65 536 tok/step

    # Learning rate schedule (progress goes 0 → 1 over the run)
    max_lr:          float = 3e-5
    min_lr:          float = 0.0
    warmup_ratio:    float = 0.05   # 5 % warmup
    warmdown_ratio:  float = 0.30   # 30 % warmdown

    # Optimiser
    betas:        tuple = (0.9, 0.95)
    weight_decay: float = 0.0
    grad_clip:    float = 1.0

    # Data mixture epochs
    mmlu_epochs:     int = 3
    gsm8k_epochs:    int = 4
    identity_file:   str = "identity_conversations.jsonl"

    # Packing buffer (number of pre-loaded conversations)
    pack_buffer: int = 200

    # Validation
    val_every_steps:  int = 200
    val_batch_size:   int = 16
    val_max_batches:  int = 80    # cap validation at ~80 batches

    # Checkpointing
    checkpoint_dir:  str = "checkpoints"
    ckpt_every_steps: int = 500
    ckpt_keep:       int = 5

    # Misc
    seed:           int = 42
    wandb_project:  str = "SLM270-SFT"
    log_every:      int = 10


CFG = SFTConfig()
TOKENS_PER_OPT_STEP = CFG.batch_size * CFG.seq_len * CFG.grad_accum   # ≈ 65 536


# ── Conversation rendering ────────────────────────────────────────────────────

def _user_content_to_str(content) -> str:
    """Flatten user/system content to a plain string (user turns are never structured)."""
    if isinstance(content, str):
        return content
    # Fallback: join any text parts (shouldn't normally happen for user turns)
    return "".join(p.get("text", "") for p in content if isinstance(p, dict))


def render_conversation(raw_tok, conversation: dict):
    """
    Tokenise a conversation and return (token_ids, loss_mask).

    loss_mask[i] = 1  → model-generated token (compute loss):
                        user/system prefix, tool_call blocks, plain answer text
    loss_mask[i] = 0  → environment-provided token (masked out):
                        user/system turns, tool_response blocks

    Chat format:
      [BOS] <|system|>…<|end|>                          (optional, mask=0)
      <|user|>…<|end|><|assistant|>                     (mask=0)
        <|tool_call|>…<|/tool_call|>                   (mask=1)
        <|tool_response|>…<|/tool_response|>            (mask=0)
        … (repeat for multi-round) …
        plain answer text                               (mask=1)
      <|end|>                                           (mask=1)

    Assistant content may be:
      - str                  → plain text response
      - list[dict]           → structured parts:
          {"type": "tool_call",     "text": "..."}  → <|tool_call|>…<|/tool_call|>  mask=1
          {"type": "tool_response", "text": "..."}  → <|tool_response|>…<|/tool_response|> mask=0
          {"type": "text",          "text": "..."}  → plain text  mask=1
    """
    messages = conversation["messages"]
    tok = raw_tok

    def enc(text: str, first: bool) -> list[int]:
        return tok.encode(text, add_special_tokens=first)

    all_ids: list[int] = []
    all_mask: list[int] = []
    first_piece = True

    # Optional system message
    start = 0
    if messages and messages[0]["role"] == "system":
        sys_text = _user_content_to_str(messages[0]["content"])
        ids = enc(f"<|system|>{sys_text}<|end|>", first_piece)
        all_ids.extend(ids)
        all_mask.extend([0] * len(ids))
        first_piece = False
        start = 1

    rest = messages[start:]
    for i in range(0, len(rest), 2):
        # ── User turn (mask=0) ────────────────────────────────────────────
        user_text = _user_content_to_str(rest[i]["content"])
        ids = enc(f"<|user|>{user_text}<|end|><|assistant|>", first_piece)
        all_ids.extend(ids)
        all_mask.extend([0] * len(ids))
        first_piece = False

        # ── Assistant turn ────────────────────────────────────────────────
        if i + 1 >= len(rest):
            continue

        content = rest[i + 1]["content"]

        if isinstance(content, str):
            # Plain text response — all tokens are trained on
            ids = enc(f"{content}<|end|>", False)
            all_ids.extend(ids)
            all_mask.extend([1] * len(ids))

        elif isinstance(content, list):
            # Structured content with tool_call / tool_response / text parts
            for part in content:
                ptype = part.get("type", "text")
                text  = part.get("text", "")

                if ptype == "tool_call":
                    # Model generates the tool call → train on it (mask=1)
                    ids = enc(f"<|tool_call|>{text}<|/tool_call|>", False)
                    all_ids.extend(ids)
                    all_mask.extend([1] * len(ids))

                elif ptype == "tool_response":
                    # Environment provides the result → mask out (mask=0)
                    ids = enc(f"<|tool_response|>{text}<|/tool_response|>", False)
                    all_ids.extend(ids)
                    all_mask.extend([0] * len(ids))

                elif ptype == "python":
                    # SpellingBee inline expression — render as <<expr (mask=1)
                    ids = enc(f"<<{text}", False)
                    all_ids.extend(ids)
                    all_mask.extend([1] * len(ids))

                elif ptype == "python_output":
                    # SpellingBee inline result — render as =result>> (mask=1)
                    ids = enc(f"={text}>>", False)
                    all_ids.extend(ids)
                    all_mask.extend([1] * len(ids))

                else:
                    # "text" → plain answer text, train on it
                    ids = enc(text, False)
                    all_ids.extend(ids)
                    all_mask.extend([1] * len(ids))

            # Closing <|end|> is part of the assistant turn → train on it
            ids = enc("<|end|>", False)
            all_ids.extend(ids)
            all_mask.extend([1] * len(ids))

    return all_ids, all_mask


# ── Sequence packing (best-fit) ───────────────────────────────────────────────

def sft_data_generator(raw_tok, dataset, batch_size: int, seq_len: int,
                        buffer_size: int = 200):
    """
    Yields (inputs, labels) pairs indefinitely until the dataset is exhausted.
    inputs : LongTensor[B, T]
    labels : LongTensor[B, T]  — non-assistant positions set to -100 (ignore)

    Uses best-fit packing: each row in the batch holds as many complete
    conversations as fit. When nothing fits the remaining space, the row is
    padded with EOS tokens (loss-masked).
    """
    dataset_size = len(dataset)
    row_capacity = seq_len + 1  # +1 so we can shift for targets
    pad_id       = raw_tok.eos_token_id   # EOS as neutral pad

    conv_buffer: list[tuple[list[int], list[int]]] = []
    cursor   = 0
    consumed = 0
    done     = False

    def refill():
        nonlocal cursor, done
        while len(conv_buffer) < buffer_size and cursor < dataset_size:
            conv = dataset[cursor]
            ids, mask = render_conversation(raw_tok, conv)
            # Skip conversations that are too long to ever fit
            if 1 <= len(ids) <= row_capacity:
                conv_buffer.append((ids, mask))
            cursor += 1
        if cursor >= dataset_size:
            done = True

    while True:
        rows      = []
        mask_rows = []

        for _ in range(batch_size):
            row      = []
            mask_row = []

            while len(row) < row_capacity:
                if len(conv_buffer) < buffer_size:
                    refill()
                if not conv_buffer:
                    # Dataset exhausted mid-row → pad remainder
                    remaining = row_capacity - len(row)
                    row.extend([pad_id] * remaining)
                    mask_row.extend([0] * remaining)
                    break

                remaining = row_capacity - len(row)

                # Best fit: largest conversation that still fits
                best_idx, best_len = -1, 0
                for i, (ids, _) in enumerate(conv_buffer):
                    if len(ids) <= remaining and len(ids) > best_len:
                        best_idx, best_len = i, len(ids)

                if best_idx >= 0:
                    ids, cmask = conv_buffer.pop(best_idx)
                    row.extend(ids)
                    mask_row.extend(cmask)
                    consumed += 1
                else:
                    # Nothing fits → pad the remainder
                    remaining = row_capacity - len(row)
                    row.extend([pad_id] * remaining)
                    mask_row.extend([0] * remaining)
                    break

            rows.append(row[:row_capacity])
            mask_rows.append(mask_row[:row_capacity])

        # Build tensors
        seq_tensor  = torch.tensor(rows,      dtype=torch.long)       # (B, T+1)
        mask_tensor = torch.tensor(mask_rows, dtype=torch.int8)       # (B, T+1)

        inputs  = seq_tensor[:, :-1]   # (B, T)
        labels  = seq_tensor[:, 1:].clone()   # (B, T)

        # Mask: shift mask by 1 to align with labels, then suppress non-assistant positions
        asst_mask = mask_tensor[:, 1:]           # (B, T)
        labels[asst_mask == 0] = -100

        yield inputs, labels, consumed

        if done and not conv_buffer:
            break


# ── LR schedule ──────────────────────────────────────────────────────────────

def get_lr(progress: float) -> float:
    """Linear warmup → flat at max_lr → linear warmdown to min_lr."""
    if progress < CFG.warmup_ratio:
        return CFG.max_lr * (progress / max(CFG.warmup_ratio, 1e-8))
    if progress <= 1.0 - CFG.warmdown_ratio:
        return CFG.max_lr
    decay = (progress - (1.0 - CFG.warmdown_ratio)) / CFG.warmdown_ratio
    return CFG.max_lr * (1.0 - decay) + CFG.min_lr * decay


def set_lr(optimizer, lr: float) -> None:
    for pg in optimizer.param_groups:
        pg["lr"] = lr


# ── Helpers ───────────────────────────────────────────────────────────────────

def format_tokens(n: int) -> str:
    if n >= 1e12: return f"{n/1e12:.3f}T"
    if n >= 1e9:  return f"{n/1e9:.3f}B"
    return f"{n/1e6:.2f}M"


def save_checkpoint(model, optimizer, step: int, tokens_seen: int) -> str:
    os.makedirs(CFG.checkpoint_dir, exist_ok=True)
    path = os.path.join(CFG.checkpoint_dir, f"sft_ckpt_step{step:08d}.pt")
    torch.save({
        "sft_step":         step,
        "tokens_seen":      tokens_seen,
        "model_state_dict": model.state_dict(),
        "optim_state_dict": optimizer.state_dict(),
        "sft_config":       CFG,
    }, path)
    return path


def rotate_checkpoints() -> None:
    ckpts = sorted(
        [os.path.join(CFG.checkpoint_dir, f)
         for f in os.listdir(CFG.checkpoint_dir)
         if f.startswith("sft_ckpt_") and f.endswith(".pt")],
        key=os.path.getmtime,
    )
    for old in ckpts[: -CFG.ckpt_keep]:
        os.remove(old)
        print(f"  ✗ Removed old SFT checkpoint {old}")


# ── Validation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_validation(model, val_gen_fn, device, fused_ce, out_head_weight) -> dict:
    """Evaluate loss on assistant tokens from the validation mixture."""
    model.eval()
    total_loss   = 0.0
    total_tokens = 0

    val_gen = val_gen_fn()
    for batch_idx, (inputs, labels, _) in enumerate(val_gen):
        if batch_idx >= CFG.val_max_batches:
            break
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        n_asst = (labels != -100).sum().item()
        if n_asst == 0:
            continue

        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = model(inputs, return_logits=False)
            loss   = fused_ce(
                out_head_weight,
                hidden.reshape(-1, hidden.shape[-1]),
                labels.reshape(-1),
            )

        total_loss   += loss.item() * n_asst
        total_tokens += n_asst

    model.train()
    avg_loss = total_loss / max(total_tokens, 1)
    return {
        "val/loss":       avg_loss,
        "val/perplexity": math.exp(min(avg_loss, 20)),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def train():
    torch.manual_seed(CFG.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")

    # ── Model ──────────────────────────────────────────────────────────────
    model_cfg = {**GEMMA3_CONFIG_270M, "context_length": CFG.seq_len}
    model = Gemma3Model(model_cfg).to(device)
    model.gradient_checkpointing = True
    model = torch.compile(model, mode="default", fullgraph=True)

    total_params  = sum(p.numel() for p in model.parameters())
    unique_params = total_params - model._orig_mod.tok_emb.weight.numel()
    print(f"Parameters  : {total_params:,}  (unique: {unique_params:,})")

    fused_ce       = LigerFusedLinearCrossEntropyLoss(ignore_index=-100)
    out_head_weight = model._orig_mod.out_head.weight

    # ── Optimiser ──────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CFG.max_lr,
        betas=CFG.betas,
        weight_decay=CFG.weight_decay,
        fused=True,
    )

    # ── Tokeniser ──────────────────────────────────────────────────────────
    tokenizer = SLM270Tokenizer(tokenizer_dir=".")
    raw_tok   = tokenizer._tok     # PreTrainedTokenizerFast

    # ── Load pretrained checkpoint ──────────────────────────────────────────
    print(f"Loading pretrained checkpoint: {CFG.pretrain_checkpoint}")
    ckpt = torch.load(CFG.pretrain_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    pretrain_tokens = ckpt.get("tokens_seen", 0)
    del ckpt
    print(f"  → pretrain tokens seen: {format_tokens(pretrain_tokens)}")

    # ── Dataset file inventory ───────────────────────────────────────────────
    _here = os.path.dirname(os.path.abspath(__file__))
    identity_path = os.path.join(_here, CFG.identity_file)
    orca_path     = os.path.join(_here, "orca_math_rewritten_llama.parquet")
    _cache_words  = os.path.expanduser("~/.cache/nanochat/words_alpha.txt")

    print("\n── Dataset inventory ────────────────────────────────────────────")
    _files = [
        ("SmolTalk (train)",         None,           True,  "HuggingFace auto-download"),
        ("MMLU auxiliary_train",     None,           True,  "HuggingFace auto-download"),
        ("GSM8K (tool-call)",        None,           True,  "HuggingFace auto-download"),
        ("SimpleSpelling word list", _cache_words,   True,  "auto-downloaded on first use"),
        ("SpellingBee word list",    _cache_words,   True,  "same file as SimpleSpelling"),
        ("Identity conversations",   identity_path,  True,  CFG.identity_file),
        ("Orca Math rewritten",      orca_path,      False, "orca_math_rewritten_llama.parquet"),
    ]
    _any_missing_required = False
    for label, path, required, note in _files:
        if path is None:
            status = "✓ remote"
        elif os.path.exists(path):
            size_mb = os.path.getsize(path) / 1e6
            status = f"✓ found  ({size_mb:.1f} MB)"
        elif required:
            status = "✗ MISSING  ← required"
            _any_missing_required = True
        else:
            status = "– not found (optional, skipping)"
        print(f"  {status:<40}  {label}  [{note}]")
    print("─" * 68)
    if _any_missing_required:
        raise FileNotFoundError(
            f"Required dataset file missing: {identity_path}\n"
            f"Place it in the SLM270/ directory and re-run."
        )
    print()

    # ── Datasets ────────────────────────────────────────────────────────────
    print("Loading training datasets…")

    # GSM8K with tool calls (replaces plain GSM8K)
    gsm8k_train = GSM8KToolCall(split="train")

    train_tasks = [
        SmolTalk(split="train"),
        CustomJSON(filepath=identity_path),
        CustomJSON(filepath=identity_path),          # 2 epochs of identity
        *[MMLU(subset="all", split="auxiliary_train")
          for _ in range(CFG.mmlu_epochs)],
        *[gsm8k_train for _ in range(CFG.gsm8k_epochs)],
        SimpleSpelling(size=200_000, split="train"),
        SpellingBee(size=80_000,    split="train"),
    ]

    # Orca Math (optional — only included if the parquet file exists)
    if os.path.exists(orca_path):
        print(f"  Found Orca Math parquet: {orca_path}")
        train_tasks.append(OrcaMathRewritten(parquet_path=orca_path))
    else:
        print(f"  Orca Math parquet not found ({orca_path}), skipping.")

    train_dataset = TaskMixture(train_tasks)
    print(f"  → {len(train_dataset):,} training conversations")

    val_tasks = [
        SmolTalk(split="test"),
        MMLU(subset="all", split="test", stop=5200),
        GSM8KToolCall(split="test"),
        SimpleSpelling(size=10_000, split="test"),
        SpellingBee(size=4_000,    split="test"),
    ]
    val_dataset = TaskMixture(val_tasks)
    print(f"  → {len(val_dataset):,} validation conversations")

    total_conversations = len(train_dataset)

    # ── WandB ───────────────────────────────────────────────────────────────
    wandb.init(
        project=CFG.wandb_project,
        config={
            "pretrain_tokens_B":  pretrain_tokens / 1e9,
            "seq_len":            CFG.seq_len,
            "batch_size":         CFG.batch_size,
            "grad_accum":         CFG.grad_accum,
            "eff_batch_tokens":   TOKENS_PER_OPT_STEP,
            "max_lr":             CFG.max_lr,
            "min_lr":             CFG.min_lr,
            "warmup_ratio":       CFG.warmup_ratio,
            "warmdown_ratio":     CFG.warmdown_ratio,
            "train_conversations": total_conversations,
            "val_conversations":   len(val_dataset),
            "mmlu_epochs":        CFG.mmlu_epochs,
            "gsm8k_epochs":       CFG.gsm8k_epochs,
            "params_total":       total_params,
            "params_unique":      unique_params,
        },
    )

    # ── Training state ──────────────────────────────────────────────────────
    model.train()
    optimizer.zero_grad(set_to_none=True)

    opt_step       = 0
    micro_step     = 0
    tokens_seen    = 0
    loss_accum     = 0.0
    progress       = 0.0

    next_val_step  = CFG.val_every_steps
    next_ckpt_step = CFG.ckpt_every_steps

    # ── Throughput ──────────────────────────────────────────────────────────
    tput_t0     = time.perf_counter()
    tput_tokens = 0

    # ── Progress bar ────────────────────────────────────────────────────────
    pbar = tqdm(
        total=total_conversations,
        unit="conv",
        dynamic_ncols=True,
        bar_format="{desc} |{bar}| {percentage:5.2f}%  {elapsed}<{remaining}",
    )

    def update_pbar(tok_per_sec: float, loss: float) -> None:
        pbar.update(0)
        pbar.set_description(
            f"[SFT {progress*100:.1f}%]"
            f"  loss={loss:.4f}"
            f"  lr={get_lr(progress):.2e}"
            f"  {tok_per_sec:>7,.0f} tok/s"
        )

    # ── Data generator ──────────────────────────────────────────────────────
    train_gen = sft_data_generator(
        raw_tok, train_dataset,
        batch_size=CFG.batch_size,
        seq_len=CFG.seq_len,
        buffer_size=CFG.pack_buffer,
    )

    def make_val_gen():
        return sft_data_generator(
            raw_tok, val_dataset,
            batch_size=CFG.val_batch_size,
            seq_len=CFG.seq_len,
            buffer_size=CFG.pack_buffer,
        )

    # ── Training loop ────────────────────────────────────────────────────────
    for inputs, labels, consumed in train_gen:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        n_asst = (labels != -100).sum().item()

        # ── Forward ─────────────────────────────────────────────────────────
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = model(inputs, return_logits=False)       # (B, T, D)
            loss   = fused_ce(
                out_head_weight,
                hidden.reshape(-1, hidden.shape[-1]),          # (B*T, D)
                labels.reshape(-1),                            # (B*T,)
            )
            loss   = loss / CFG.grad_accum

        loss.backward()

        batch_tokens  = n_asst    # only count tokens we actually train on
        tokens_seen  += batch_tokens
        tput_tokens  += inputs.numel()
        loss_accum   += loss.item()
        micro_step   += 1

        # Update progress (consumed = conversations processed so far)
        progress = min(consumed / max(total_conversations, 1), 1.0)

        # ── Optimiser step ───────────────────────────────────────────────────
        if micro_step % CFG.grad_accum == 0:
            opt_step += 1

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), CFG.grad_clip
            )
            lr = get_lr(progress)
            set_lr(optimizer, lr)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # Throughput
            now         = time.perf_counter()
            elapsed     = now - tput_t0
            tok_per_sec = tput_tokens / elapsed if elapsed > 0 else 0.0
            tput_t0     = now
            tput_tokens = 0

            displayed_loss = loss_accum
            loss_accum     = 0.0
            pbar.update(CFG.batch_size * CFG.grad_accum)

            update_pbar(tok_per_sec, displayed_loss)

            # ── WandB logging ────────────────────────────────────────────────
            if opt_step % CFG.log_every == 0:
                wandb.log(
                    {
                        "train/loss":        displayed_loss,
                        "train/perplexity":  math.exp(min(displayed_loss, 20)),
                        "train/lr":          lr,
                        "train/grad_norm":   grad_norm.item(),
                        "train/progress":    progress,
                        "tokens/seen":       tokens_seen,
                        "perf/tok_per_sec":  tok_per_sec,
                        "sft/opt_step":      opt_step,
                    },
                    step=opt_step,
                )

            # ── Validation ───────────────────────────────────────────────────
            if opt_step >= next_val_step:
                val_metrics = run_validation(
                    model, make_val_gen, device, fused_ce, out_head_weight
                )
                print(
                    f"\n  [val @ step {opt_step:,}  {progress*100:.1f}%]"
                    f"  loss={val_metrics['val/loss']:.4f}"
                    f"  ppl={val_metrics['val/perplexity']:.2f}\n"
                )
                wandb.log({**val_metrics, "sft/opt_step": opt_step}, step=opt_step)
                next_val_step += CFG.val_every_steps

            # ── Checkpoint ──────────────────────────────────────────────────
            if opt_step >= next_ckpt_step:
                path = save_checkpoint(model, optimizer, opt_step, tokens_seen)
                print(f"\n  ✓ SFT checkpoint {path}\n")
                rotate_checkpoints()
                wandb.log(
                    {"checkpoint/sft_step": opt_step},
                    step=opt_step,
                )
                next_ckpt_step += CFG.ckpt_every_steps

    pbar.close()

    # ── Final validation + checkpoint ────────────────────────────────────────
    print("\nFinal validation…")
    val_metrics = run_validation(
        model, make_val_gen, device, fused_ce, out_head_weight
    )
    print(
        f"  [val @ final / step {opt_step:,}]"
        f"  loss={val_metrics['val/loss']:.4f}"
        f"  ppl={val_metrics['val/perplexity']:.2f}"
    )
    wandb.log({**val_metrics, "sft/opt_step": opt_step}, step=opt_step)

    path = save_checkpoint(model, optimizer, opt_step, tokens_seen)
    print(f"  ✓ Final SFT checkpoint: {path}")

    wandb.finish()
    print(f"\nDone.  SFT opt-steps: {opt_step:,}  |  assistant tokens trained: {format_tokens(tokens_seen)}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train()
