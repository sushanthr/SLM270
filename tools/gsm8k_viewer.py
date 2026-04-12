#!/usr/bin/env python3
"""
GSM8K / Orca-Math Tool-Call Conversation Viewer
------------------------------------------------
Shows the exact conversation sequence SLM270 will be trained on,
with colour-coded regions and tool-response validation.

Colour key:
  ● grey          user prompt (mask=0, not trained on)
  ● cyan bold     <|tool_call|> … <|/tool_call|>  (mask=1, model generates)
  ● yellow dim    <|tool_response|> … <|/tool_response|>  (mask=0, env-provided)
  ● green         answer / explanation text  (mask=1, model generates)
  ● blue          special delimiter tokens
  ● red/green     ✗/✓ badge showing whether chat.py would produce the same response

Navigation
----------
  ← / h        previous sample
  → / l        next sample
  ↑ / k        scroll up
  ↓ / j        scroll down
  f            jump to next failing sample
  F            jump to previous failing sample
  g            jump to sample number
  q            quit

Usage
-----
  python gsm8k_viewer.py [train|test]                      # GSM8K interactive viewer
  python gsm8k_viewer.py [train|test] --stats              # print pass/fail stats then exit
  python gsm8k_viewer.py --parquet FILE.parquet            # view any orca-math-style parquet
  python gsm8k_viewer.py --parquet FILE.parquet --all      # include rows where correct_math=False
  python gsm8k_viewer.py --parquet FILE.parquet --stats    # stats for parquet file
"""

import argparse
import sys
import os
import json
import curses
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_NANOCHAT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "nanochat")
sys.path.insert(0, _NANOCHAT)

from transformers import PreTrainedTokenizerFast
from gsm8k_toolcall import GSM8KToolCall
from math_tools import run_tool_calls, _approx_equal, time_limit
from extract_reprocess import is_single_scalar_tool_call, last_number_mismatch, uses_perm


# ── Validation ────────────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    ok:       bool
    computed: str   # json.dumps of what run_tool_calls produced
    stored:   str   # json string from the dataset
    error:    str   # non-empty on exception


def validate_sample(conv: dict) -> ValidationResult:
    """
    Re-execute the tool_call in *conv* and check whether the result matches
    the stored tool_response — i.e. what chat.py would inject at runtime.
    Returns ok=True for samples that have no tool calls.
    """
    content = conv["messages"][1]["content"]
    if not isinstance(content, list):
        return ValidationResult(ok=True, computed="", stored="", error="")

    tc_text = next((p["text"] for p in content if p.get("type") == "tool_call"),  None)
    tr_text = next((p["text"] for p in content if p.get("type") == "tool_response"), None)

    if tc_text is None:
        return ValidationResult(ok=True, computed="", stored="", error="")

    stored = tr_text or ""
    try:
        with time_limit(5):
            computed_obj = run_tool_calls(tc_text)
        computed     = json.dumps(computed_obj)
        stored_obj   = json.loads(stored) if stored else None
        ok = _approx_equal(computed_obj, stored_obj)
        return ValidationResult(ok=ok, computed=computed, stored=stored, error="")
    except Exception as exc:
        return ValidationResult(ok=False, computed="", stored=stored, error=str(exc))


def compute_all_validations(dataset, verbose: bool = True) -> list[ValidationResult]:
    results = []
    n = len(dataset)
    for i in range(n):
        if verbose and i % 500 == 0:
            print(f"  {i}/{n}…", end="\r", flush=True)
        results.append(validate_sample(dataset[i]))
    if verbose:
        print(" " * 30, end="\r")
    return results


# ── Stats mode (--stats) ──────────────────────────────────────────────────────

def run_stats(dataset, split: str) -> None:
    n = len(dataset)
    print(f"Validating {n} {split} samples…")
    results = compute_all_validations(dataset, verbose=True)

    passes  = [r for r in results if r.ok]
    fails   = [(i, results[i]) for i in range(n) if not results[i].ok]

    pct = 100 * len(passes) / n if n else 0
    print(f"\nPass : {len(passes):>5} / {n}  ({pct:.1f}%)")
    print(f"Fail : {len(fails):>5} / {n}  ({100-pct:.1f}%)\n")

    if not fails:
        print("All samples pass — chat.py will produce matching tool responses.")
        return

    print(f"{'─'*72}")
    print(f"Failing samples ({len(fails)})")
    print(f"{'─'*72}")
    for rank, (i, r) in enumerate(fails):
        conv     = dataset[i]
        question = conv["messages"][0]["content"]
        q_short  = question[:80].replace("\n", " ")
        content  = conv["messages"][1]["content"]
        tc_text  = next((p["text"] for p in content if p.get("type") == "tool_call"), "")

        print(f"\n[{i+1:>5}] {q_short}…")
        print(f"  tool_call  : {tc_text[:120]}")
        if r.error:
            print(f"  error      : {r.error}")
        else:
            print(f"  stored     : {r.stored}")
            print(f"  computed   : {r.computed}")
            print(f"  match      : ✗")

        if rank >= 49:
            remaining = len(fails) - 50
            if remaining > 0:
                print(f"\n… and {remaining} more failures (use the viewer to browse all).")
            break


# ── Colour palette (curses colour-pair indices) ────────────────────────────

C_USER      = 1    # dim white   – user / system turns (masked)
C_TC_DELIM  = 2    # bold cyan   – <|tool_call|> / <|/tool_call|>
C_TC_BODY   = 3    # cyan        – tool_call JSON body
C_TR_DELIM  = 4    # yellow      – <|tool_response|> / <|/tool_response|>
C_TR_BODY   = 5    # dim yellow  – tool_response JSON body
C_ANSWER    = 6    # bold green  – answer / explanation text
C_SPECIAL   = 7    # bold blue   – <|user|> <|end|> <|assistant|> etc.
C_HEADER    = 8    # bold white on blue – header bar
C_STATUS    = 9    # black on white – status bar
C_PASS      = 10   # bold green  – ✓ badge
C_FAIL      = 11   # bold red    – ✗ badge + diff text

ATTR = {}


def setup_colors():
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(C_USER,     curses.COLOR_WHITE,   -1)
    curses.init_pair(C_TC_DELIM, curses.COLOR_CYAN,    -1)
    curses.init_pair(C_TC_BODY,  curses.COLOR_CYAN,    -1)
    curses.init_pair(C_TR_DELIM, curses.COLOR_YELLOW,  -1)
    curses.init_pair(C_TR_BODY,  curses.COLOR_YELLOW,  -1)
    curses.init_pair(C_ANSWER,   curses.COLOR_GREEN,   -1)
    curses.init_pair(C_SPECIAL,  curses.COLOR_BLUE,    -1)
    curses.init_pair(C_HEADER,   curses.COLOR_WHITE,   curses.COLOR_BLUE)
    curses.init_pair(C_STATUS,   curses.COLOR_BLACK,   curses.COLOR_WHITE)
    curses.init_pair(C_PASS,     curses.COLOR_GREEN,   -1)
    curses.init_pair(C_FAIL,     curses.COLOR_RED,     -1)

    ATTR[C_USER]     = curses.color_pair(C_USER)    | curses.A_DIM
    ATTR[C_TC_DELIM] = curses.color_pair(C_TC_DELIM)| curses.A_BOLD
    ATTR[C_TC_BODY]  = curses.color_pair(C_TC_BODY)
    ATTR[C_TR_DELIM] = curses.color_pair(C_TR_DELIM)| curses.A_DIM
    ATTR[C_TR_BODY]  = curses.color_pair(C_TR_BODY) | curses.A_DIM
    ATTR[C_ANSWER]   = curses.color_pair(C_ANSWER)  | curses.A_BOLD
    ATTR[C_SPECIAL]  = curses.color_pair(C_SPECIAL) | curses.A_BOLD
    ATTR[C_HEADER]   = curses.color_pair(C_HEADER)  | curses.A_BOLD
    ATTR[C_STATUS]   = curses.color_pair(C_STATUS)
    ATTR[C_PASS]     = curses.color_pair(C_PASS)    | curses.A_BOLD
    ATTR[C_FAIL]     = curses.color_pair(C_FAIL)    | curses.A_BOLD


# ── Span building ─────────────────────────────────────────────────────────────

Span = tuple  # (text: str, color: int, mask: int)


def pretty_json(s: str) -> str:
    try:
        return json.dumps(json.loads(s), indent=2)
    except Exception:
        return s


def build_spans(conversation: dict, vr: ValidationResult | None = None) -> list[Span]:
    """
    Convert a conversation dict into a list of (text, colour_pair_id, mask) spans.
    If *vr* is a failing ValidationResult, a diff block is appended at the end.
    """
    spans: list[Span] = []
    messages = conversation["messages"]

    def add(text, color, mask):
        spans.append((text, color, mask))

    add("<bos>", C_SPECIAL, 0)

    start = 0
    if messages and messages[0]["role"] == "system":
        add("<|system|>", C_SPECIAL, 0)
        add(messages[0]["content"], C_USER, 0)
        add("<|end|>", C_SPECIAL, 0)
        start = 1

    rest = messages[start:]
    for i in range(0, len(rest), 2):
        add("<|user|>",      C_SPECIAL, 0)
        add(rest[i]["content"], C_USER, 0)
        add("<|end|>",       C_SPECIAL, 0)
        add("<|assistant|>", C_SPECIAL, 0)

        if i + 1 >= len(rest):
            continue

        content = rest[i + 1]["content"]

        if isinstance(content, str):
            add(content,   C_ANSWER,  1)
            add("<|end|>", C_SPECIAL, 1)

        elif isinstance(content, list):
            for part in content:
                ptype = part.get("type", "text")
                text  = part.get("text", "")
                if ptype == "tool_call":
                    add("<|tool_call|>",  C_TC_DELIM, 1)
                    add("\n" + pretty_json(text) + "\n", C_TC_BODY, 1)
                    add("<|/tool_call|>", C_TC_DELIM, 1)
                elif ptype == "tool_response":
                    add("<|tool_response|>",  C_TR_DELIM, 0)
                    add("\n" + pretty_json(text) + "\n", C_TR_BODY, 0)
                    add("<|/tool_response|>", C_TR_DELIM, 0)
                else:
                    add("\n" + text, C_ANSWER, 1)
            add("\n<|end|>", C_SPECIAL, 1)

    # ── Validation diff block ─────────────────────────────────────────────
    if vr is not None and not vr.ok:
        add("\n\n── chat.py validation ✗ ──────────────────\n", C_FAIL, 0)
        if vr.error:
            add(f"  ERROR   : {vr.error}\n", C_FAIL, 0)
        else:
            add(f"  stored  : {vr.stored}\n",   C_TR_BODY, 0)
            add(f"  computed: {vr.computed}\n",  C_FAIL,    0)

    return spans


# ── Token count summary ───────────────────────────────────────────────────────

def token_summary(raw_tok, ids: list, mask: list) -> str:
    total   = len(ids)
    trained = sum(mask)
    masked  = total - trained
    return f"tok:{total}  ✓train:{trained}  ✗mask:{masked}"


# ── Span → wrapped lines ──────────────────────────────────────────────────────

Line = list  # list of (text_chunk, colour, mask)


def wrap_spans(spans: list[Span], width: int) -> list[Line]:
    lines: list[Line] = [[]]
    col = 0
    for text, color, mask in spans:
        segments = text.split("\n")
        for seg_idx, seg in enumerate(segments):
            if seg_idx > 0:
                lines.append([])
                col = 0
            while seg:
                remaining = width - col
                if remaining <= 0:
                    lines.append([])
                    col = 0
                    remaining = width
                chunk = seg[:remaining]
                lines[-1].append((chunk, color, mask))
                col  += len(chunk)
                seg   = seg[remaining:]
    return lines


# ── Drawing ───────────────────────────────────────────────────────────────────

def draw_header(win, idx: int, total: int, tok_sum: str,
                split: str, vr: ValidationResult, n_fail: int):
    h, w = win.getmaxyx()
    badge = " ✓ " if vr.ok else " ✗ "
    badge_color = ATTR[C_PASS] if vr.ok else ATTR[C_FAIL]

    fail_info = f"  fails:{n_fail}/{total}" if n_fail else ""
    title = f" [{split}] {idx+1}/{total}{fail_info} "
    right = f" {tok_sum} "
    pad   = w - len(title) - len(badge) - len(right)

    try:
        win.addstr(0, 0, title[:w], ATTR[C_HEADER])
        col = len(title)
        if col < w:
            win.addstr(0, col, badge[:w - col], badge_color)
            col += len(badge)
        if col < w:
            right_str = (" " * max(pad, 0) + right)[:w - col]
            win.addstr(0, col, right_str, ATTR[C_HEADER])
    except curses.error:
        pass


def draw_legend(win, row: int, width: int):
    items = [
        ("user(masked)", C_USER),
        ("<tool_call>(trained)", C_TC_DELIM),
        ("tool_response(masked)", C_TR_DELIM),
        ("answer(trained)", C_ANSWER),
        ("✓pass", C_PASS),
        ("✗fail", C_FAIL),
    ]
    col = 0
    for label, color in items:
        entry = f"█{label}  "
        if col + len(entry) > width:
            break
        try:
            win.addstr(row, col, "█", ATTR[color])
            win.addstr(row, col + 1, label + "  ")
        except curses.error:
            pass
        col += len(entry)


def draw_status(win, scroll: int, max_scroll: int, n_fail: int):
    h, w = win.getmaxyx()
    fail_hint = "  f/F=next/prev-fail" if n_fail else ""
    keys = f"  ←/h prev  →/l next  ↑↓/jk scroll  g goto{fail_hint}  q quit"
    pct  = f"  line {scroll}/{max_scroll}" if max_scroll > 0 else ""
    line = (keys + pct).ljust(w)[:w]
    try:
        win.addstr(h - 1, 0, line, ATTR[C_STATUS])
    except curses.error:
        pass


def render_sample(win, lines: list[Line], scroll: int, tok_sum: str,
                  idx: int, total: int, split: str,
                  vr: ValidationResult, n_fail: int):
    h, w = win.getmaxyx()
    win.erase()
    draw_header(win, idx, total, tok_sum, split, vr, n_fail)
    draw_legend(win, 1, w)

    content_rows = h - 3
    for row_offset, line_chunks in enumerate(lines[scroll: scroll + content_rows]):
        y = 2 + row_offset
        x = 0
        for chunk, color, _mask in line_chunks:
            if x >= w:
                break
            chunk = chunk[:w - x]
            try:
                win.addstr(y, x, chunk, ATTR[color])
            except curses.error:
                pass
            x += len(chunk)

    draw_status(win, scroll, max(0, len(lines) - content_rows), n_fail)
    win.refresh()


def prompt_jump(win, total: int) -> int | None:
    h, w = win.getmaxyx()
    prompt = f" Jump to sample (1-{total}): "
    try:
        win.addstr(h - 1, 0, prompt.ljust(w), ATTR[C_STATUS])
        win.refresh()
        curses.echo()
        buf = win.getstr(h - 1, len(prompt), 6).decode("utf-8").strip()
        curses.noecho()
        val = int(buf)
        if 1 <= val <= total:
            return val - 1
    except Exception:
        pass
    return None


# ── Viewer ────────────────────────────────────────────────────────────────────

def viewer(stdscr, dataset, raw_tok, split: str, validations: list[ValidationResult]):
    curses.curs_set(0)
    stdscr.keypad(True)
    setup_colors()

    total      = len(dataset)
    fail_idxs  = [i for i, r in enumerate(validations) if not r.ok]
    n_fail     = len(fail_idxs)
    idx        = 0
    scroll     = 0
    cache: dict[int, tuple[list[Line], str]] = {}

    def get_cached(i):
        if i not in cache:
            conv    = dataset[i]
            vr      = validations[i]
            spans   = build_spans(conv, vr)
            from sft import render_conversation
            ids, mask = render_conversation(raw_tok, conv)
            tok_sum = token_summary(raw_tok, ids, mask)
            _, w    = stdscr.getmaxyx()
            lines   = wrap_spans(spans, max(w - 1, 40))
            cache[i] = (lines, tok_sum)
        return cache[i]

    while True:
        h, w = stdscr.getmaxyx()

        # Invalidate span cache on resize (width changed)
        if cache:
            sample_lines = next(iter(cache.values()))[0]
            if sample_lines:
                row_width = sum(len(c) for c, _, _ in sample_lines[0])
                if row_width > w or (w - row_width) > 5:
                    cache.clear()

        lines, tok_sum = get_cached(idx)
        content_rows   = h - 3
        max_scroll      = max(0, len(lines) - content_rows)
        scroll          = min(scroll, max_scroll)

        render_sample(stdscr, lines, scroll, tok_sum,
                      idx, total, split, validations[idx], n_fail)

        key = stdscr.getch()

        if key in (ord('q'), ord('Q'), 27):
            break
        elif key in (curses.KEY_RIGHT, ord('l'), ord('n')):
            idx = (idx + 1) % total; scroll = 0
        elif key in (curses.KEY_LEFT, ord('h'), ord('p')):
            idx = (idx - 1) % total; scroll = 0
        elif key in (curses.KEY_DOWN, ord('j')):
            scroll = min(scroll + 1, max_scroll)
        elif key in (curses.KEY_UP, ord('k')):
            scroll = max(scroll - 1, 0)
        elif key in (curses.KEY_NPAGE, ord(' ')):
            scroll = min(scroll + content_rows, max_scroll)
        elif key == curses.KEY_PPAGE:
            scroll = max(scroll - content_rows, 0)
        elif key == ord('f') and fail_idxs:
            # next failure after current idx
            nxt = next((i for i in fail_idxs if i > idx), fail_idxs[0])
            idx = nxt; scroll = 0
        elif key == ord('F') and fail_idxs:
            # previous failure before current idx
            prv = next((i for i in reversed(fail_idxs) if i < idx), fail_idxs[-1])
            idx = prv; scroll = 0
        elif key in (curses.KEY_HOME, ord('g')):
            new = prompt_jump(stdscr, total)
            if new is not None:
                idx = new; scroll = 0
        elif key == curses.KEY_RESIZE:
            cache.clear()


# ── Parquet dataset wrapper ───────────────────────────────────────────────────

class ParquetDataset:
    """
    Wraps an orca-math-style parquet file (columns: question, tool_call,
    tool_response, answer, correct_math) into the same dict-list interface
    that GSM8KToolCall uses so the viewer works unchanged.
    """

    def __init__(self, path: str, include_all: bool = False,
                 single_scalar: bool = False, last_num_mismatch: bool = False,
                 perm: bool = False):
        import pyarrow.parquet as pq
        table = pq.read_table(path)
        rows = table.to_pylist()
        filter_mode = single_scalar or last_num_mismatch or perm
        self._examples = []
        for row in rows:
            tc  = row.get("tool_call",     "") or ""
            tr  = row.get("tool_response", "") or ""
            ans = row.get("answer",        "") or ""

            if filter_mode:
                # Keep only rows matching any requested filter
                match = (
                    (single_scalar    and is_single_scalar_tool_call(tc)) or
                    (last_num_mismatch and last_number_mismatch(tr, ans))  or
                    (perm             and uses_perm(tc))
                )
                if not match:
                    continue
            elif not include_all and not row.get("correct_math", True):
                continue

            if tc:
                parts = [
                    {"type": "tool_call",     "text": tc},
                    {"type": "tool_response", "text": tr},
                    {"type": "text",          "text": ans},
                ]
            else:
                parts = ans  # plain string if no tool call
            self._examples.append({
                "messages": [
                    {"role": "user",      "content": row["question"]},
                    {"role": "assistant", "content": parts},
                ]
            })

        skipped = len(rows) - len(self._examples)
        print(f"ParquetDataset: {len(self._examples)}/{len(rows)} rows loaded"
              + (f" ({skipped} skipped)" if skipped else ""))

    def __len__(self):
        return len(self._examples)

    def __getitem__(self, i):
        return self._examples[i]


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="GSM8K / Orca-Math tool-call viewer / validator")
    p.add_argument("split",    nargs="?", default="train", choices=["train", "test"],
                   help="GSM8K split to load (ignored when --parquet is given)")
    p.add_argument("--parquet", metavar="FILE",
                   help="Load an orca-math-style parquet file instead of GSM8K")
    p.add_argument("--all",              action="store_true",
                   help="Include parquet rows where correct_math=False (default: skip them)")
    p.add_argument("--single-scalar",    action="store_true",
                   help="Show only rows whose entire tool_call is one scalar math_eval")
    p.add_argument("--last-num-mismatch", action="store_true",
                   help="Show only rows where last number in tool_response != last number in answer")
    p.add_argument("--perm",             action="store_true",
                   help="Show only rows that use the perm operation")
    p.add_argument("--stats",            action="store_true",
                   help="Print pass/fail statistics and failure cases, then exit")
    args = p.parse_args()

    if args.parquet:
        split_label = os.path.basename(args.parquet)
        print(f"Loading {args.parquet}…", flush=True)
        dataset = ParquetDataset(
            args.parquet,
            include_all=args.all,
            single_scalar=args.single_scalar,
            last_num_mismatch=args.last_num_mismatch,
            perm=args.perm,
        )
    else:
        split_label = args.split
        print(f"Loading GSM8KToolCall ({args.split})…", flush=True)
        dataset = GSM8KToolCall(split=args.split)

    if not dataset:
        print("No examples loaded."); sys.exit(1)

    if args.stats:
        run_stats(dataset, split_label)
        return

    raw_tok = PreTrainedTokenizerFast.from_pretrained(
        os.path.dirname(os.path.abspath(__file__))
    )

    print(f"Pre-computing tool-response validation for {len(dataset)} samples…", flush=True)
    validations = compute_all_validations(dataset, verbose=True)
    n_fail = sum(1 for r in validations if not r.ok)
    print(f"  ✓ {len(validations)-n_fail}  ✗ {n_fail}  — launching viewer…")

    curses.wrapper(viewer, dataset, raw_tok, split_label, validations)


if __name__ == "__main__":
    main()
