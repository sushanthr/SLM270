"""
Loads the rewritten Orca Math Word Problems dataset produced by
rewrite_dataset_llama.py.

Parquet schema:
  question        str   — the math word problem
  tool_call       str   — \n---\n-separated JSON arrays (one per round)
  tool_response   str   — \n---\n-separated JSON arrays (one per round)
  answer          str   — final explanation text after all tool calls
  correct_math    bool  — True when every tool call executed correctly

Only rows with correct_math=True are used.
Multi-round problems (tool_call contains multiple \n---\n sections) are
fully supported — each round becomes a separate tool_call/tool_response pair.
"""

import os
import sys
import json

import pandas as pd

_NANOCHAT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "nanochat")
sys.path.insert(0, _NANOCHAT)

from tasks.common import Task


class OrcaMathRewritten(Task):
    """
    Orca Math Word Problems rewritten with <|tool_call|>/<|tool_response|> turns.
    Requires the parquet output from rewrite_dataset_llama.py.
    """

    def __init__(self, parquet_path: str, **kwargs):
        super().__init__(**kwargs)
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(
                f"OrcaMathRewritten: parquet not found at {parquet_path!r}\n"
                f"Run rewrite_dataset_llama.py first to generate it."
            )

        df = pd.read_parquet(parquet_path)
        df = df[df["correct_math"] == True].reset_index(drop=True)
        self._data = df[["question", "tool_call", "tool_response", "answer"]].to_dict("records")
        print(f"OrcaMathRewritten: {len(self._data):,} examples loaded from {parquet_path}")

    @property
    def eval_type(self):
        return "generative"

    def num_examples(self) -> int:
        return len(self._data)

    def get_example(self, index: int) -> dict:
        row = self._data[index]

        # Split multi-round tool calls on the \n---\n separator
        tc_rounds = [s.strip() for s in row["tool_call"].split("\n---\n") if s.strip()]
        tr_rounds = [s.strip() for s in row["tool_response"].split("\n---\n") if s.strip()]

        parts: list[dict] = []
        for tc, tr in zip(tc_rounds, tr_rounds):
            parts.append({"type": "tool_call",     "text": tc})
            parts.append({"type": "tool_response", "text": tr})
        parts.append({"type": "text", "text": row["answer"].strip()})

        return {
            "messages": [
                {"role": "user",      "content": row["question"]},
                {"role": "assistant", "content": parts},
            ]
        }
