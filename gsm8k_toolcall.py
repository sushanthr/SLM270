"""
GSM8K task that converts <<expr=result>> annotations into proper
<|tool_call|> / <|tool_response|> turns using math_eval.

Each arithmetic step becomes a math_eval call batched into a single round:
  <|tool_call|>[{"name":"math_eval","arguments":{"expression":"2*12"}}, ...]<|/tool_call|>
  <|tool_response|>[24.0, 48.0, 16.0]<|/tool_response|>
  Cleaned explanation text...

Tool-response tokens are masked out in the loss (environment-provided).
Tool-call and answer tokens are trained on.

Examples that fail execution (invalid expressions) are skipped.
"""

import re
import json
import sys
import os

# nanochat base Task class
_NANOCHAT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "nanochat")
sys.path.insert(0, _NANOCHAT)

from datasets import load_dataset
from tasks.common import Task
from math_tools import run_tool_calls


# <<expr=result>>literal — the annotation and its trailing literal number
_GSM_ANNO_RE  = re.compile(r"<<([^>]+)=([^>]*)>>")       # capture expr, result
_GSM_CLEAN_RE = re.compile(r"<<[^>]+>>([\-\d,\.]+)?")    # for cleaning text
_FINAL_RE     = re.compile(r"#### (\-?[0-9\.,]+)")


def _convert(answer: str):
    """
    Return (tool_calls_json, tool_response_json, clean_text) or None on failure.
    All <<expr=result>> steps are batched into a single <|tool_call|> round.
    """
    matches = list(_GSM_ANNO_RE.finditer(answer))
    if not matches:
        return None

    tool_calls = [
        {"name": "math_eval", "arguments": {"expression": m.group(1).strip()}}
        for m in matches
    ]
    tc_str = json.dumps(tool_calls)

    try:
        results = run_tool_calls(tc_str)
        tr_str  = json.dumps(results)
    except Exception:
        return None

    # Remove <<expr=result>> annotations; keep the trailing literal digit(s)
    clean = _GSM_CLEAN_RE.sub(lambda m: m.group(1) or "", answer)
    return tc_str, tr_str, clean.strip()


class GSM8KToolCall(Task):
    """
    GSM8K (main split) with arithmetic steps as tool call turns.
    Only examples where every <<expr=result>> executes correctly are kept
    (~98 % of the dataset passes).
    """

    def __init__(self, split: str, **kwargs):
        super().__init__(**kwargs)
        assert split in ("train", "test"), "split must be train|test"
        ds = load_dataset("openai/gsm8k", "main", split=split).shuffle(seed=42)

        self._examples: list[dict] = []
        for i in range(len(ds)):
            row = ds[i]
            converted = _convert(row["answer"])
            if converted is None:
                continue
            tc_str, tr_str, clean = converted
            parts = [
                {"type": "tool_call",     "text": tc_str},
                {"type": "tool_response", "text": tr_str},
                {"type": "text",          "text": clean},
            ]
            self._examples.append({
                "question": row["question"],
                "parts":    parts,
                "answer":   row["answer"],   # kept for evaluate()
            })

        print(f"GSM8KToolCall ({split}): {len(self._examples)}/{len(ds)} examples converted")

    # ── Task interface ────────────────────────────────────────────────────────

    @property
    def eval_type(self):
        return "generative"

    def num_examples(self) -> int:
        return len(self._examples)

    def get_example(self, index: int) -> dict:
        ex = self._examples[index]
        return {
            "messages": [
                {"role": "user",      "content": ex["question"]},
                {"role": "assistant", "content": ex["parts"]},
            ]
        }

    def evaluate(self, conversation: dict, assistant_response: str) -> int:
        """Check if the #### answer in the response is correct."""
        # Ground truth
        ex = self._examples[conversation.get("_idx", 0)]
        ref = _FINAL_RE.search(ex["answer"])
        ref_num = ref.group(1).replace(",", "") if ref else None
        pred = _FINAL_RE.search(assistant_response)
        pred_num = pred.group(1).replace(",", "") if pred else None
        return int(ref_num == pred_num) if ref_num and pred_num else 0
