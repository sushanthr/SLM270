"""
Simple interactive viewer for orca_math_rewritten_llama.parquet

Usage:
    python view_parquet.py [file.parquet]
    python view_parquet.py --wrong      # show only incorrect rows
"""

import sys
import os
import argparse
import textwrap
import shutil
import pandas as pd

W = shutil.get_terminal_size((80, 24)).columns  # auto-detect terminal width

def clear():
    os.system("cls" if os.name == "nt" else "clear")

def hr(char="─"):
    print(char * W)

def wrap(label, text, indent=2):
    prefix = " " * indent
    lines = textwrap.wrap(str(text), width=W - indent)
    print(f"\033[1m{label}\033[0m")
    for line in lines:
        print(prefix + line)

def print_raw(label, text):
    print(f"\033[1m{label}\033[0m")
    print(text if text else "(none)")

def show_row(row, idx, total):
    hr("═")
    correct = row.get("correct_math", "?")
    tag = "\033[32m✓ CORRECT\033[0m" if correct else "\033[31m✗ WRONG\033[0m"
    print(f"  Row {idx + 1} of {total}   {tag}")
    hr()
    wrap("Question", row.get("question", ""))
    print()
    wrap("Tool call", row.get("tool_call", "") or "(none)")
    print()
    wrap("Tool response", row.get("tool_response", "") or "(none)")
    print()
    print_raw("Answer", row.get("answer", ""))
    hr()

def summary(df):
    total = len(df)
    correct = df["correct_math"].sum() if "correct_math" in df.columns else "?"
    wrong = total - correct
    print(f"\n  Rows: {total:,}   ✓ {correct:,}  ({100*correct/total:.1f}%)   ✗ {wrong:,}  ({100*wrong/total:.1f}%)\n")
    print("  Columns:", ", ".join(df.columns.tolist()))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", nargs="?", default="orca_math_rewritten_llama.parquet")
    parser.add_argument("--wrong", action="store_true", help="Show only incorrect rows")
    parser.add_argument("--correct", action="store_true", help="Show only correct rows")
    args = parser.parse_args()

    print(f"\nLoading {args.file}…")
    df = pd.read_parquet(args.file).reset_index(drop=True)
    summary(df)

    if args.wrong:
        df = df[df["correct_math"] == False].reset_index(drop=True)
        print(f"  Filtering to {len(df):,} wrong rows.\n")
    elif args.correct:
        df = df[df["correct_math"] == True].reset_index(drop=True)
        print(f"  Filtering to {len(df):,} correct rows.\n")

    total = len(df)
    i = 0

    while True:
        clear()
        show_row(df.iloc[i], i, total)
        print("  Commands:  [Enter] next   [p] prev   [N] jump to row N   [q] quit\n")
        try:
            cmd = input("  > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if cmd in ("q", "quit"):
            break
        elif cmd in ("", "n"):
            i = min(i + 1, total - 1)
        elif cmd == "p":
            i = max(i - 1, 0)
        elif cmd.isdigit():
            n = int(cmd) - 1
            if 0 <= n < total:
                i = n
            else:
                print(f"  Row must be between 1 and {total}")
        else:
            print("  Unknown command.")

if __name__ == "__main__":
    main()
