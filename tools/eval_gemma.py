"""
Evaluate google/gemma-3-270m on the same Wikipedia validation set used in
training, so the result is directly comparable to the SLM270 val/loss numbers.

Each model is evaluated with its own tokenizer (same Wikipedia text, different
tokenisation), so the cross-entropy losses are on different token spaces.
This gives a fair model-vs-model comparison of next-token prediction quality.

Run on Mac (uses MPS if available, otherwise CPU):
    python eval_gemma.py

Requirements:
    pip install transformers accelerate
"""

import math
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_ID   = "google/gemma-3-270m"
SEQ_LEN    = 1024
BATCH_SIZE = 4      # small — Mac CPU/MPS has limited RAM
VAL_DOCS   = 500    # same seed/shuffle as training; 500 docs ≈ a few minutes on Mac
SEED       = 42     # must match train.py CFG.seed


# ── Device ────────────────────────────────────────────────────────────────────

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Device: MPS (Apple Silicon)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Device: CUDA")
else:
    device = torch.device("cpu")
    print("Device: CPU")


# ── Load model + tokenizer ────────────────────────────────────────────────────

print(f"\nLoading {MODEL_ID}…")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,   # MPS/CPU: float32 is safest
    device_map=None,             # we place it ourselves below
)
model = model.to(device)
model.eval()

total_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {total_params:,}")


# ── Build validation batches ───────────────────────────────────────────────────

print(f"\nBuilding validation set ({VAL_DOCS} Wikipedia docs, seq_len={SEQ_LEN})…")

ds = load_dataset(
    "wikimedia/wikipedia",
    "20231101.en",
    split="train",
    streaming=True,
    trust_remote_code=True,
)
ds = ds.shuffle(seed=SEED, buffer_size=10_000)

eos_id  = tokenizer.eos_token_id
needed  = SEQ_LEN + 1
buffer: list[int] = []
chunks: list[dict] = []

for sample in tqdm(ds.take(VAL_DOCS), total=VAL_DOCS, desc="  tokenising", unit="doc"):
    text = sample.get("text") or ""
    if not text.strip():
        continue
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        continue
    if eos_id is not None:
        ids.append(eos_id)
    buffer.extend(ids)

    while len(buffer) >= needed:
        chunk  = buffer[:needed]
        buffer = buffer[needed:]
        chunks.append({
            "input_ids": torch.tensor(chunk[:-1], dtype=torch.long),
            "labels":    torch.tensor(chunk[1:],  dtype=torch.long),
        })

# Collate into batches
batches = []
for i in range(0, len(chunks), BATCH_SIZE):
    group = chunks[i : i + BATCH_SIZE]
    if not group:
        continue
    batches.append({
        "input_ids": torch.stack([g["input_ids"] for g in group]),
        "labels":    torch.stack([g["labels"]    for g in group]),
    })

total_val_tokens = sum(b["input_ids"].numel() for b in batches)
print(f"  → {len(batches)} batches  ({total_val_tokens:,} tokens)")


# ── Evaluate ─────────────────────────────────────────────────────────────────

print(f"\nEvaluating {MODEL_ID}…")

total_loss   = 0.0
total_tokens = 0

with torch.no_grad():
    for batch in tqdm(batches, desc="  eval", unit="batch"):
        input_ids = batch["input_ids"].to(device)
        labels    = batch["labels"].to(device)

        outputs = model(input_ids=input_ids)
        logits  = outputs.logits                        # (B, T, vocab)

        # Flatten and compute token-level CE loss (sum, not mean)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),        # (B*T, vocab)
            labels.reshape(-1),                          # (B*T,)
            reduction="sum",
        )

        total_loss   += loss.item()
        total_tokens += labels.numel()

avg_loss   = total_loss / total_tokens
perplexity = math.exp(min(avg_loss, 20))

print(f"\n{'─'*45}")
print(f"Model      : {MODEL_ID}")
print(f"Val tokens : {total_tokens:,}  ({VAL_DOCS} docs × packed seq_len={SEQ_LEN})")
print(f"Val loss   : {avg_loss:.4f}")
print(f"Perplexity : {perplexity:.2f}")
print(f"{'─'*45}")
print()
print("Compare with your SLM270 val/loss (same Wikipedia, same seed, same seq_len):")
print("  11B tokens: 3.38  (best so far)")
print("  16-21B:     3.51–3.59  (current, flat-LR phase)")
