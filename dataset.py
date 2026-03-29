"""
Streaming dataset for SLM270 training.
Mixes HuggingFaceTB/smollm-corpus subsets:
  - cosmopedia-v2      (39.1M rows)
  - fineweb-edu-dedup  (190.0M rows)
Sequences are packed (no padding) into fixed-length chunks.

Validation: wikimedia/wikipedia 20231101.en — 1 000 random samples,
also opened in streaming mode and packed identically.
"""

import queue
import threading
import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset, interleave_datasets
from typing import Iterator, Dict, List

# ── Dataset registry ──────────────────────────────────────────────────────────
# Weights are proportional to row counts so sampling reflects corpus size.
DATASET_CONFIGS = [
    {
        "path": "OptimalScale/ClimbMix",
        "name": "default",
        "split": "train",
        "text_field": "text",
        "weight": 1.0,
    }
]


class PackedStreamingDataset(IterableDataset):
    """
    Streams text, tokenises on the fly, and packs tokens into fixed-length
    windows of size `seq_len`.  Each yielded sample contains:
        input_ids : LongTensor[seq_len]
        labels    : LongTensor[seq_len]   (input_ids shifted left by 1)
    """

    def __init__(self, tokenizer, seq_len: int = 1024, seed: int = 42):
        super().__init__()
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.seed = seed

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_mixed_stream(self):
        streams, weights = [], []
        for cfg in DATASET_CONFIGS:
            ds = load_dataset(
                cfg["path"],
                cfg["name"],
                split=cfg["split"],
                streaming=True,
                trust_remote_code=True,
            )
            # Keep only the text field to reduce memory
            ds = ds.select_columns([cfg["text_field"]])
            streams.append(ds)
            weights.append(cfg["weight"])

        total_weight = sum(weights)
        probabilities = [w / total_weight for w in weights]

        return interleave_datasets(
            streams,
            probabilities=probabilities,
            seed=self.seed,
            stopping_strategy="all_exhausted",  # repeat smaller sets until larger is done
        )

    # ── IterableDataset interface ─────────────────────────────────────────────

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        eos_id = self.tokenizer.eos_token_id
        needed = self.seq_len + 1          # need 1 extra token for the label shift
        buffer: list[int] = []

        for sample in self._build_mixed_stream():
            text: str = sample.get("text") or ""
            if not text.strip():
                continue

            ids: list[int] = self.tokenizer.encode(text)
            if not ids:
                continue

            # Append EOS as document boundary
            if eos_id is not None:
                ids.append(eos_id)

            buffer.extend(ids)

            # Drain full chunks
            while len(buffer) >= needed:
                chunk   = buffer[:needed]
                buffer  = buffer[needed:]
                yield {
                    "input_ids": torch.tensor(chunk[:-1], dtype=torch.long),
                    "labels":    torch.tensor(chunk[1:],  dtype=torch.long),
                }


# ── Validation dataset ────────────────────────────────────────────────────────

def build_validation_batches(
    tokenizer,
    seq_len: int = 1024,
    n_samples: int = 1000,
    batch_size: int = 8,
    seed: int = 42,
) -> List[Dict[str, torch.Tensor]]:
    """
    Streams wikimedia/wikipedia 20231101.en, takes the first `n_samples`
    documents, packs them into seq_len chunks, and returns a list of
    pre-built batches ready for eval.  Materialised once at startup so
    validation cost is constant and reproducible across runs.
    """
    ds = load_dataset(
        "wikimedia/wikipedia",
        "20231101.en",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )
    ds = ds.shuffle(seed=seed, buffer_size=10_000)

    eos_id = tokenizer.eos_token_id
    needed = seq_len + 1
    buffer: list[int] = []
    chunks: list[Dict[str, torch.Tensor]] = []

    for sample in ds.take(n_samples):
        text: str = sample.get("text") or ""
        if not text.strip():
            continue
        ids: list[int] = tokenizer.encode(text)
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
    batches: List[Dict[str, torch.Tensor]] = []
    for i in range(0, len(chunks), batch_size):
        group = chunks[i : i + batch_size]
        if not group:
            continue
        batches.append({
            "input_ids": torch.stack([g["input_ids"] for g in group]),
            "labels":    torch.stack([g["labels"]    for g in group]),
        })

    return batches


# ── Prefetch wrapper ──────────────────────────────────────────────────────────

class PrefetchLoader:
    """
    Wraps any DataLoader and uses a background thread to keep `buffer_size`
    batches ready, so the GPU never stalls waiting for CPU tokenization.
    """
    def __init__(self, loader: DataLoader, buffer_size: int = 4):
        self.loader = loader
        self.buffer_size = buffer_size

    def __iter__(self):
        q: queue.Queue = queue.Queue(maxsize=self.buffer_size)
        _DONE = object()

        def producer():
            for batch in self.loader:
                q.put(batch)
            q.put(_DONE)

        t = threading.Thread(target=producer, daemon=True)
        t.start()
        while True:
            item = q.get()
            if item is _DONE:
                break
            yield item

    def __len__(self):
        return len(self.loader)


# ── Training dataloader factory ───────────────────────────────────────────────

def build_dataloader(
    tokenizer,
    seq_len: int = 1024,
    batch_size: int = 8,
    seed: int = 42,
) -> DataLoader:
    """
    Returns a DataLoader backed by the packed streaming dataset.

    Note: num_workers=0 is required for HuggingFace streaming datasets unless
    you add explicit worker-rank sharding via worker_init_fn.
    """
    dataset = PackedStreamingDataset(tokenizer, seq_len=seq_len, seed=seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
