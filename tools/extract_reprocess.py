"""
Extract questions from orca_math_rewritten_llama.parquet that need reprocessing.

Criteria (union — any one is sufficient):
  1. correct_math=False  (failed cases)
  2. Single tool call that produces only one scalar number
     e.g. [{"name": "math_eval", "arguments": {"expression": "M=15"}}]
  3. Last number in tool_response != last number in answer text
  4. Any tool call uses "operation": "perm"

Usage:
    python extract_reprocess.py [--parquet FILE] [--out FILE]
"""

import argparse
import json
import re
import pyarrow.parquet as pq
import pyarrow as pa


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_rounds(tc_str: str) -> list[list[dict]]:
    """Parse tool_call string (single round or ---separated rounds) into
    a list of rounds, where each round is a list of call dicts."""
    if not tc_str:
        return []
    rounds = []
    for part in tc_str.split("\n---\n"):
        part = part.strip()
        if not part:
            continue
        try:
            calls = json.loads(part)
            if isinstance(calls, dict):
                calls = [calls]
            rounds.append(calls)
        except Exception:
            pass
    return rounds


def parse_response_rounds(tr_str: str) -> list:
    """Parse tool_response string into a list of round results."""
    if not tr_str:
        return []
    results = []
    for part in tr_str.split("\n---\n"):
        part = part.strip()
        if not part:
            continue
        try:
            results.append(json.loads(part))
        except Exception:
            pass
    return results


def last_number(text: str) -> float | None:
    """Return the last numeric value found in *text*, or None."""
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return float(nums[-1]) if nums else None


def is_single_scalar_tool_call(tc_str: str) -> bool:
    """True if the entire tool_call is a single call that produces one scalar.

    Specifically: exactly one round, one call, and the call is math_eval with
    an expression that either assigns or evaluates to a single number (no list
    operations, no stats/sort/seq/numbers tools).
    """
    rounds = parse_rounds(tc_str)
    if len(rounds) != 1 or len(rounds[0]) != 1:
        return False
    call = rounds[0][0]
    if call.get("name") != "math_eval":
        return False
    expr = call.get("arguments", {}).get("expression", "")
    # Strip variable assignment prefix if present (e.g. "M=15" → "15")
    bare = re.sub(r'^[A-Za-z][A-Za-z0-9_]*\s*=\s*', '', expr).strip()
    # It's a single scalar if the bare expression is just a number
    return bool(re.fullmatch(r'-?\d+(?:\.\d+)?', bare))


def uses_perm(tc_str: str) -> bool:
    """True if any call in any round uses operation=perm."""
    for round_calls in parse_rounds(tc_str):
        for call in round_calls:
            if call.get("arguments", {}).get("operation") == "perm":
                return True
    return False


def last_number_mismatch(tr_str: str, answer: str) -> bool:
    """True if the last number in tool_response != last number in answer."""
    tr_last  = last_number(tr_str)
    ans_last = last_number(answer)
    if tr_last is None or ans_last is None:
        return False
    # Allow small floating-point tolerance
    return abs(tr_last - ans_last) > max(1e-6, 1e-6 * abs(ans_last))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", default="orca_math_rewritten_llama.parquet")
    p.add_argument("--out",     default="reprocess_questions.jsonl",
                   help="Output JSONL: one JSON object per line with idx + question + reasons")
    args = p.parse_args()

    print(f"Loading {args.parquet}…")
    table = pq.read_table(args.parquet)
    rows  = table.to_pylist()
    total = len(rows)
    print(f"  {total:,} rows total")

    counts = {
        "failed":        0,
        "single_scalar": 0,
        "last_num_mismatch": 0,
        "perm":          0,
    }

    to_reprocess = []

    for i, row in enumerate(rows):
        tc      = row.get("tool_call",     "") or ""
        tr      = row.get("tool_response", "") or ""
        answer  = row.get("answer",        "") or ""
        failed  = not row.get("correct_math", True)

        reasons = []
        if failed:
            reasons.append("failed")
            counts["failed"] += 1
        if is_single_scalar_tool_call(tc):
            reasons.append("single_scalar")
            counts["single_scalar"] += 1
        if last_number_mismatch(tr, answer):
            reasons.append("last_num_mismatch")
            counts["last_num_mismatch"] += 1
        if uses_perm(tc):
            reasons.append("perm")
            counts["perm"] += 1

        if reasons:
            to_reprocess.append({
                "idx":      row.get("idx", i),
                "question": row["question"],
                "reasons":  reasons,
            })

    print()
    print("── Breakdown ──────────────────────────────────────")
    print(f"  correct_math=False (failed)  : {counts['failed']:>7,}")
    print(f"  single scalar tool call      : {counts['single_scalar']:>7,}")
    print(f"  last-number mismatch         : {counts['last_num_mismatch']:>7,}")
    print(f"  uses perm operation          : {counts['perm']:>7,}")
    print(f"────────────────────────────────────────────────────")
    print(f"  Total to reprocess (union)   : {len(to_reprocess):>7,}  "
          f"({100*len(to_reprocess)/total:.1f}% of dataset)")
    print()

    print(f"Saving questions to {args.out}…")
    with open(args.out, "w") as f:
        for entry in to_reprocess:
            f.write(json.dumps(entry) + "\n")
    print(f"  Done — {len(to_reprocess):,} entries written.")


if __name__ == "__main__":
    main()
