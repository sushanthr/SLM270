"""
Rewrites microsoft/orca-math-word-problems-200k using in-process vLLM.
Groups problems into batches of 10 (amortising the system prompt), then submits
ALL batch prompts at once via llm.chat() so the scheduler can pack them optimally.

Usage:
    python rewrite_dataset_llama.py --end 100

Options:
    --model      HuggingFace model ID (default: Qwen/Qwen3.5-9B)
    --batch-size Problems per prompt (default: 10)
    --start/--end  Process a slice of the dataset
    --max-model-len  Context window override (default: 16384)
"""

import argparse
import json
import signal
import sys
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

from math_tools import run_tool_calls, validate, MathEvalContext, time_limit

SYSTEM_PROMPT = Path("Prompt.md").read_text()
CHECKPOINT_FILE = "orca_rewrite_llama_checkpoint.parquet"
OUTPUT_FILE = "orca_math_rewritten_llama.parquet"
MAX_TOKENS_PER_ITEM = 1200  # output budget per problem (prompt=~2k tokens, ctx=16k, batch=10 → ~12k output)

# Guided-JSON schema returned by the model.
# tool_rounds: list of rounds; each round is a list of tool-call objects.
# answer: plain-text solution (no tool jargon).
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "tool_rounds": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "arguments": {"type": "object"},
                                },
                                "required": ["name", "arguments"],
                            },
                        },
                    },
                    "answer": {"type": "string"},
                },
                "required": ["tool_rounds", "answer"],
            },
        }
    },
    "required": ["results"],
}


def normalize_tool_calls(tool_calls: list[str]) -> tuple[list[str], list[str]]:
    """Normalize each round of tool calls and recompute tool responses.

    A single MathEvalContext is shared across ALL rounds so that variables
    assigned in round 1 (e.g. A=60-5) remain visible in round 2 (e.g. A).
    """
    norm_tcs: list[str] = []
    norm_trs: list[str] = []
    ctx = MathEvalContext()  # shared across rounds
    for tc_str in tool_calls:
        if not tc_str:
            norm_tcs.append(tc_str)
            norm_trs.append("")
            continue
        try:
            calls = json.loads(tc_str)
        except (json.JSONDecodeError, TypeError):
            norm_tcs.append(tc_str)
            norm_trs.append("")
            continue

        normalized = []
        for call in calls:
            if "arguments" in call:
                normalized.append(call)
            else:
                args = {k: v for k, v in call.items() if k != "name"}
                normalized.append({"name": call["name"], "arguments": args})

        norm_str = json.dumps(normalized)
        try:
            with time_limit(5):
                response_str = json.dumps(run_tool_calls(norm_str, ctx=ctx))
        except Exception:
            response_str = ""

        norm_tcs.append(norm_str)
        norm_trs.append(response_str)

    return norm_tcs, norm_trs


def _build_row(result: dict, idx: int, question: str) -> dict:
    """Convert one JSON result item into a dataset row dict."""
    tool_rounds = result.get("tool_rounds", [])
    answer = result.get("answer", "")

    tc_strs = [json.dumps(round_calls) for round_calls in tool_rounds]
    tcs, trs = normalize_tool_calls(tc_strs)

    correct = (
        all(validate(tc, tr) for tc, tr in zip(tcs, trs))
        if tcs
        else len(answer) > 10
    )
    return {
        "idx": idx,
        "question": question,
        "tool_call": "\n---\n".join(tcs),
        "tool_response": "\n---\n".join(trs),
        "answer": answer,
        "correct_math": correct,
    }


def build_messages(batch: list[tuple[int, str, str]]) -> list[dict]:
    """Build the chat messages for one batch of (idx, question, answer) tuples."""
    problem_lines = "\n\n".join(
        f"Problem {i + 1}:\nQuestion: {q}\nAnswer: {a}"
        for i, (_, q, a) in enumerate(batch)
    )
    user_content = (
        f"{problem_lines}\n\n"
        f"Process all {len(batch)} problems above and return a JSON object "
        f"with a 'results' array of {len(batch)} items."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def parse_output(text: str, batch: list[tuple[int, str, str]]) -> list[dict]:
    """Parse JSON text from one batch output into row dicts."""
    try:
        data = json.loads(text)
        results_list = data.get("results", [])
    except Exception as e:
        return [
            {
                "idx": idx, "question": question,
                "tool_call": "", "tool_response": "",
                "answer": f"ERROR: JSON parse failed: {e}", "correct_math": False,
            }
            for idx, question, _ in batch
        ]

    rows = []
    for i, item_result in enumerate(results_list[: len(batch)]):
        idx, question, _ = batch[i]
        rows.append(_build_row(item_result, idx, question))

    # Pad if model returned fewer results than requested
    for i in range(len(rows), len(batch)):
        idx, question, _ = batch[i]
        rows.append({
            "idx": idx, "question": question,
            "tool_call": "", "tool_response": "",
            "answer": "ERROR: missing from batch response", "correct_math": False,
        })
    return rows


def run_chunk(
    llm: LLM,
    sampling_params: SamplingParams,
    chunk: list[tuple[int, str, str]],
    batch_size: int,
) -> list[dict]:
    """Submit one chunk of problems to vLLM and return parsed row dicts."""
    batches = [chunk[i : i + batch_size] for i in range(0, len(chunk), batch_size)]
    conversations = [build_messages(b) for b in batches]

    outputs = llm.chat(
        conversations,
        sampling_params,
        use_tqdm=True,
        # Disable Qwen3 thinking mode — it emits <think>…</think> before the
        # JSON which breaks guided-decoding and parsing.
        chat_template_kwargs={"enable_thinking": False},
    )

    rows: list[dict] = []
    bar = tqdm(total=len(chunk), unit="sample", desc="Parsing", dynamic_ncols=True)
    for batch, output in zip(batches, outputs):
        batch_rows = parse_output(output.outputs[0].text, batch)
        rows.extend(batch_rows)
        bar.update(len(batch_rows))
    bar.close()
    return rows


def main(args):
    print("Loading dataset...")
    ds = load_dataset("microsoft/orca-math-word-problems-200k", split="train")
    end = args.end or len(ds)
    print(f"Processing indices {args.start}–{end}")

    # Resume from checkpoint
    done: set[int] = set()
    rows: list[dict] = []
    if Path(CHECKPOINT_FILE).exists():
        df_ckpt = pd.read_parquet(CHECKPOINT_FILE)
        done = set(df_ckpt["idx"].tolist())
        rows = df_ckpt.to_dict("records")
        print(f"Resuming: {len(done)} already done")

    todo = [
        (i, ds[i]["question"], ds[i]["answer"])
        for i in range(args.start, end)
        if i not in done
    ]
    print(f"Remaining: {len(todo)}")
    if not todo:
        print("Nothing to do.")
        return rows

    # --- vLLM setup (in-process) ---
    print(f"Loading model {args.model}...")
    llm = LLM(model=args.model, max_model_len=args.max_model_len)

    structured = StructuredOutputsParams(json=RESPONSE_SCHEMA)
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=MAX_TOKENS_PER_ITEM * args.batch_size,
        structured_outputs=structured,
    )

    # Split todo into chunks of --chunk-size; process and checkpoint each one.
    chunks = [
        todo[i : i + args.chunk_size] for i in range(0, len(todo), args.chunk_size)
    ]
    correct_count = sum(1 for r in rows if r.get("correct_math"))
    total_done = len(rows)

    for chunk_idx, chunk in enumerate(chunks):
        start_idx = chunk[0][0]
        end_idx = chunk[-1][0]
        print(f"\n--- Chunk {chunk_idx + 1}/{len(chunks)}: "
              f"dataset indices {start_idx}–{end_idx} ({len(chunk)} problems) ---")

        chunk_rows = run_chunk(llm, sampling_params, chunk, args.batch_size)
        rows.extend(chunk_rows)
        total_done += len(chunk_rows)
        correct_count += sum(1 for r in chunk_rows if r["correct_math"])

        pd.DataFrame(rows).to_parquet(CHECKPOINT_FILE, index=False)
        pct = 100 * correct_count / total_done if total_done else 0
        print(f"Checkpoint written — "
              f"total processed: {total_done}, "
              f"correct: {correct_count} ({pct:.1f}%)")

    print(f"\n--- Final Stats ---")
    print(f"Processed : {total_done}")
    print(f"Correct   : {correct_count} ({100 * correct_count / total_done:.1f}%)")
    return rows


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-8B",
                   help="HuggingFace model ID (default: Qwen/Qwen3-8B)")
    p.add_argument("--batch-size", type=int, default=10,
                   help="Problems per prompt (default: 10; prompt=~2k tokens, ctx=16k, leaves ~12k for output)")
    p.add_argument("--max-model-len", type=int, default=16384,
                   help="vLLM context window (default: 16384)")
    p.add_argument("--chunk-size", type=int, default=1000,
                   help="Problems per checkpoint chunk (default: 1000)")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    def shutdown(sig, frame):
        print("\nShutting down...")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    print(f"Model       : {args.model}")  # Qwen3-8B = text-only; Qwen3.5-9B = vision-language (needs flash_attn)
    print(f"Batch size  : {args.batch_size}")
    print(f"Chunk size  : {args.chunk_size}")
    print(f"Max ctx len : {args.max_model_len}")

    rows = main(args)

    final_df = (
        pd.DataFrame(rows)
        .sort_values("idx")
        .reset_index(drop=True)[
            ["question", "tool_call", "tool_response", "answer", "correct_math"]
        ]
    )
    final_df.to_parquet(OUTPUT_FILE, index=False)
    print(f"Saved to {OUTPUT_FILE}")
