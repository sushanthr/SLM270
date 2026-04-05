"""
Rewrites microsoft/orca-math-word-problems-200k using the Gemini API.
Adds tool_call, tool_response, answer, correct_math columns. Saves locally as parquet.

Usage:
    python rewrite_dataset_llama.py --end 16

    Set your API key via the GEMINI_API_KEY environment variable, or pass --api-key.

Options:
    --parallel      Number of concurrent requests (default: 8)
    --start / --end Process a slice of the dataset (default: full 200k)
    --api-key       Gemini API key (default: $GEMINI_API_KEY)
    --model         Gemini model name (default: gemini-3.1-flash-lite-preview)
"""

import argparse
import asyncio
import os
import json
import re
import signal
import sys
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from openai import AsyncOpenAI
from tqdm import tqdm

from math_tools import run_tool_calls, validate

SYSTEM_PROMPT = Path("Prompt.md").read_text()
CHECKPOINT_FILE = "orca_rewrite_llama_checkpoint.parquet"
OUTPUT_FILE = "orca_math_rewritten_llama.parquet"
MAX_TOKENS = 1024

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


def normalize_tool_calls(tool_calls: list[str]) -> tuple[list[str], list[str]]:
    """Normalize each round of tool calls into canonical {"name", "arguments"} format
    and re-compute tool_responses by actually executing the calls."""
    norm_tcs: list[str] = []
    norm_trs: list[str] = []
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
            results = run_tool_calls(norm_str)
            response_str = json.dumps(results)
        except Exception:
            response_str = ""

        norm_tcs.append(norm_str)
        norm_trs.append(response_str)

    return norm_tcs, norm_trs


def parse_response(text: str) -> tuple[list[str], list[str], str]:
    """Extract all tool_call/tool_response rounds and the final answer."""
    tc_matches = list(re.finditer(r"<\|tool_call\|>(.*?)<\|/tool_call\|>", text, re.DOTALL))
    tr_matches = list(re.finditer(r"<\|tool_response\|>(.*?)<\|/tool_response\|>", text, re.DOTALL))

    tool_calls = [m.group(1).strip() for m in tc_matches]
    tool_responses = [m.group(1).strip() for m in tr_matches]

    # Answer is everything after the last tool_response (or last tool_call if no response)
    if tr_matches:
        answer = text[tr_matches[-1].end():].strip()
    elif tc_matches:
        answer = text[tc_matches[-1].end():].strip()
    else:
        answer = text.strip()

    return tool_calls, tool_responses, answer



async def process_row(
    client: AsyncOpenAI, sem: asyncio.Semaphore,
    idx: int, question: str, answer: str, model: str
) -> dict:
    async with sem:
        for attempt in range(12):
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    max_tokens=MAX_TOKENS,
                    temperature=0.0,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"Question: {question}\nAnswer: {answer}"},
                    ],
                )
                text = resp.choices[0].message.content or ""
                tcs, trs, ans = parse_response(text)
                # Normalize flat tool calls and re-compute responses
                tcs, trs = normalize_tool_calls(tcs)
                # Join multiple rounds with separator for storage
                tc_joined = "\n---\n".join(tcs)
                tr_joined = "\n---\n".join(trs)
                correct = all(
                    validate(tc, tr) for tc, tr in zip(tcs, trs)
                ) if tcs else (len(ans) > 10)  # no tool calls but has a real answer = logic-only question
                # Small delay after success to pace requests
                await asyncio.sleep(2.5)
                return {
                    "idx": idx, "question": question,
                    "tool_call": tc_joined, "tool_response": tr_joined,
                    "answer": ans, "correct_math": correct,
                }
            except Exception as e:
                err_str = str(e)
                is_rate_limit = "429" in err_str or "rate" in err_str.lower()
                if attempt == 11:
                    return {
                        "idx": idx, "question": question,
                        "tool_call": "", "tool_response": "",
                        "answer": f"ERROR: {e}", "correct_math": False,
                    }
                # Longer backoff for rate limits
                delay = min(2 ** attempt * (8 if is_rate_limit else 1), 180)
                await asyncio.sleep(delay)


async def main(args):
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

    client = AsyncOpenAI(base_url=GEMINI_BASE_URL, api_key=args.api_key)
    sem = asyncio.Semaphore(args.parallel)
    tasks = [process_row(client, sem, i, q, a, args.model) for i, q, a in todo]

    correct_count = sum(1 for r in rows if r.get("correct_math"))
    total_done = len(rows)
    completed = 0

    bar = tqdm(total=len(tasks), unit="sample", dynamic_ncols=True)

    async def run():
        nonlocal correct_count, total_done, completed
        futs = [asyncio.ensure_future(t) for t in tasks]
        for fut in asyncio.as_completed(futs):
            result = await fut
            rows.append(result)
            completed += 1
            total_done += 1
            if result["correct_math"]:
                correct_count += 1

            pct = 100 * correct_count / total_done if total_done else 0
            bar.set_postfix(correct=f"{correct_count}/{total_done} ({pct:.1f}%)", refresh=False)
            bar.update(1)

            if completed % 10 == 0:
                pd.DataFrame(rows).to_parquet(CHECKPOINT_FILE, index=False)

    await run()
    bar.close()

    pd.DataFrame(rows).to_parquet(CHECKPOINT_FILE, index=False)

    print(f"\n--- Final Stats ---")
    print(f"Processed : {total_done}")
    print(f"Correct   : {correct_count} ({100*correct_count/total_done:.1f}%)")

    return rows


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--api-key", default=None,
                    help="Gemini API key (default: $GEMINI_API_KEY)")
    p.add_argument("--model", default="gemini-2.5-flash-lite",
                    help="Gemini model name (default: gemini-2.5-flash-lite)")
    p.add_argument("--parallel", type=int, default=8)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None)
    args = p.parse_args()

    # Resolve API key
    if not args.api_key:
        args.api_key = os.environ.get("GEMINI_API_KEY")
    if not args.api_key:
        print("ERROR: provide --api-key or set GEMINI_API_KEY environment variable")
        sys.exit(1)

    return args


if __name__ == "__main__":
    args = parse_args()

    def shutdown(sig, frame):
        print("\nShutting down...")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    print(f"Using model: {args.model}")
    rows = asyncio.run(main(args))

    final_df = (
        pd.DataFrame(rows)
        .sort_values("idx")
        .reset_index(drop=True)
        [["question", "tool_call", "tool_response", "answer", "correct_math"]]
    )
    final_df.to_parquet(OUTPUT_FILE, index=False)
    print(f"Saved to {OUTPUT_FILE}")
