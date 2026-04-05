"""
Implementations of all math tools from Prompt.md.
Used to validate model-generated tool_call / tool_response pairs.
"""

import json
import math
import re
from itertools import combinations as _combinations, permutations as _permutations
from math import comb, factorial, floor, ceil, gcd, isclose, perm, sqrt
from statistics import mean, median

# ---------------------------------------------------------------------------
# Shared context  (single-letter variables, Ans, and list variables)
# ---------------------------------------------------------------------------

class MathEvalContext:
    """Unified variable store used by ALL tools.

    Scalars (float) and lists (list[float]) live side-by-side.
    * ``Ans`` always holds the last scalar result.
    * Any single-letter (or short name) variable can hold a scalar or a list.
    """

    def __init__(self):
        self.vars: dict = {"Ans": 0}

    # -- helpers to resolve a "values" argument --------------------------
    def resolve_values(self, values):
        """If *values* is a string variable name, look it up; otherwise pass through."""
        if isinstance(values, str):
            name = values.lstrip("$")          # accept both "P" and "$P"
            if name in self.vars:
                v = self.vars[name]
                if isinstance(v, list):
                    return v
                return [v]                     # wrap scalar in list
            raise ValueError(f"Unknown variable: {values!r}")
        return values                          # already a list

    # -- math_eval -------------------------------------------------------
    def eval(self, expression: str) -> "float | list[float]":
        expr = expression.strip()
        # Variable assignment: NAME=expr
        m = re.match(r'^([A-Za-z][A-Za-z0-9_]*)\s*=\s*(.+)$', expr)
        if m:
            var_name, expr = m.group(1), m.group(2).strip()
        else:
            var_name = None

        # List literal assignment:  N=[3,5,2] or N=[Y,N,J,T]
        m_list = re.match(r'^\[(.+)\]$', expr)
        if m_list:
            inner = m_list.group(1)
            elements = [e.strip() for e in inner.split(',')]
            try:
                result = [float(self._eval_raw(e)) for e in elements]
                self.vars["Ans"] = result
                if var_name:
                    self.vars[var_name] = result
                return result
            except Exception:
                pass  # fall through to normal eval

        # Resolve list indexing BEFORE element-wise mapping.
        # E.g. "P[2]" -> the float at index 2 of list P, "Ans[0]" etc.
        expr = self._resolve_indexing(expr)

        # If the entire expression is just a variable name that holds a list,
        # return a copy of that list (e.g. "N=P" copies list P into N).
        if expr in self.vars and isinstance(self.vars[expr], list):
            result = list(self.vars[expr])
            self.vars["Ans"] = result
            if var_name:
                self.vars[var_name] = result
            return result

        # Check if the expression references any *list* variable.
        # If so, map the expression element-wise over that list (or zip multiple).
        list_vars = self._find_all_list_vars(expr)
        if list_vars:
            # Get all referenced lists; they must be the same length.
            lists = {name: self.vars[name] for name in list_vars}
            length = len(next(iter(lists.values())))
            results = []
            for i in range(length):
                one_expr = expr
                for name in list_vars:
                    one_expr = re.sub(rf'\b{re.escape(name)}\b', str(lists[name][i]), one_expr)
                results.append(float(self._eval_raw(one_expr)))
            self.vars["Ans"] = results
            if var_name:
                self.vars[var_name] = results
            return results

        result = float(self._eval_raw(expr))
        self.vars["Ans"] = result
        if var_name:
            self.vars[var_name] = result
        return result

    def _resolve_indexing(self, expr: str) -> str:
        """Replace VAR[i] with the actual value from the list variable.
        If VAR is a scalar, VAR[0] returns the scalar itself."""
        def _repl(m):
            name = m.group(1)
            idx = int(m.group(2))
            if name in self.vars:
                val = self.vars[name]
                if isinstance(val, list):
                    if 0 <= idx < len(val):
                        return str(val[idx])
                else:
                    # Scalar: VAR[0] → the scalar value
                    return str(val)
            return m.group(0)  # leave unchanged if unresolvable
        return re.sub(r'\b([A-Za-z][A-Za-z0-9_]*)\[(\d+)\]', _repl, expr)

    def _find_list_var(self, expr: str) -> str | None:
        """Return the name of the first variable in *expr* that is a list, or None."""
        for name, val in self.vars.items():
            if isinstance(val, list) and re.search(rf'\b{re.escape(name)}\b', expr):
                return name
        return None

    def _find_all_list_vars(self, expr: str) -> list[str]:
        """Return names of all variables in *expr* that are lists."""
        found = []
        for name, val in self.vars.items():
            if isinstance(val, list) and re.search(rf'\b{re.escape(name)}\b', expr):
                found.append(name)
        return found

    def _eval_raw(self, expr: str) -> float:
        expr = expr.replace("^", "**")
        # Support modulo operator as %
        # Build namespace: scalars only (lists are resolved before calling this)
        ns = {
            "__builtins__": {},
            "abs": abs, "round": round, "floor": floor, "ceil": ceil,
            "sqrt": sqrt, "pow": pow, "mod": lambda a, b: a % b,
            "ipart": lambda x: float(int(x)),
            "fpart": lambda x: x - int(x),
        }
        for k, v in self.vars.items():
            if not isinstance(v, list):
                ns[k] = v
        return eval(expr, ns)  # noqa: S307 — local validation only


def math_eval(expression: str, ctx: MathEvalContext) -> "float | list[float]":
    return ctx.eval(expression)


# ---------------------------------------------------------------------------
# math_stats
# ---------------------------------------------------------------------------

def _mode(values: list[float]) -> float:
    counts: dict = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    return max(counts, key=lambda k: counts[k])


def math_stats(operation: str, values, ctx: MathEvalContext) -> float:
    values = ctx.resolve_values(values)
    ops = {
        "min":    min,
        "max":    max,
        "sum":    sum,
        "avg":    mean,
        "median": median,
        "mode":   _mode,
        "count":  len,
    }
    if operation not in ops:
        raise ValueError(f"Unknown math_stats operation: {operation!r}")
    result = float(ops[operation](values))
    ctx.vars["Ans"] = result
    return result


# ---------------------------------------------------------------------------
# math_sort
# ---------------------------------------------------------------------------

def math_sort(operation: str, values, ctx: MathEvalContext) -> list | int:
    values = ctx.resolve_values(values)
    if operation == "sort":
        result = sorted(values)
    elif operation == "sortd":
        result = sorted(values, reverse=True)
    elif operation == "sortidx":
        result = sorted(range(len(values)), key=lambda i: values[i])
    elif operation == "minidx":
        result = int(min(range(len(values)), key=lambda i: values[i]))
    elif operation == "maxidx":
        result = int(max(range(len(values)), key=lambda i: values[i]))
    else:
        raise ValueError(f"Unknown math_sort operation: {operation!r}")
    ctx.vars["Ans"] = result
    return result


# ---------------------------------------------------------------------------
# math_numbers  (number theory + combinatorics)
# ---------------------------------------------------------------------------

def _lcm(a: int, b: int) -> int:
    return abs(a * b) // gcd(a, b)


def math_numbers(operation: str, a, b, ctx: MathEvalContext) -> float:
    # Resolve variable references: if a or b is a string like "T", look it up
    if isinstance(a, str):
        a = ctx.vars.get(a, a)
    if isinstance(b, str) and b is not None:
        b = ctx.vars.get(b, b)
    a = int(a) if a is not None else None
    b = int(b) if b is not None else None
    ops = {
        "gcd":       lambda: float(gcd(a, b)),
        "lcm":       lambda: float(_lcm(a, b)),
        "remainder": lambda: float(a % b),
        "factorial": lambda: float(factorial(a)),
        "ncr":       lambda: float(comb(a, b)),
        "npr":       lambda: float(perm(a, b)),
    }
    if operation not in ops:
        raise ValueError(f"Unknown math_numbers operation: {operation!r}")
    result = ops[operation]()
    ctx.vars["Ans"] = result
    return result


# ---------------------------------------------------------------------------
# math_seq
# ---------------------------------------------------------------------------

def math_seq(operation: str, ctx: MathEvalContext,
             expression: str | None = None,
             start: int | None = None, end: int | None = None,
             values=None,
             r: int | None = None,
             store: str | None = None,
             min_val: float | None = None,
             max_val: float | None = None) -> list[float]:
    # Resolve values from context if it's a variable name
    if values is not None and not isinstance(values, list):
        values = ctx.resolve_values(values)

    if operation == "seq":
        results = []
        for x in range(int(start), int(end) + 1):
            expr = expression.replace("^", "**")
            ns = {"__builtins__": {}, "x": x, "sqrt": sqrt, "abs": abs,
                  "floor": floor, "ceil": ceil}
            results.append(float(eval(expr, ns)))  # noqa: S307
    elif operation == "cumsum":
        total = 0.0
        results = []
        for v in values:
            total += v
            results.append(total)
    elif operation == "perm":
        if r is None:
            r = len(values)
        results = []
        for p in _permutations(values, r):
            num = 0
            for digit in p:
                num = num * 10 + int(digit)
            results.append(float(num))
    elif operation == "comb":
        if r is None:
            r = len(values)
        results = []
        for c in _combinations(values, r):
            num = 0
            for digit in c:
                num = num * 10 + int(digit)
            results.append(float(num))
    elif operation == "filter":
        results = list(values)  # copy
    else:
        raise ValueError(f"Unknown math_seq operation: {operation!r}")

    # Apply optional min/max filtering (available for ALL operations)
    if min_val is not None:
        results = [v for v in results if v >= min_val]
    if max_val is not None:
        results = [v for v in results if v <= max_val]

    # Store result under the given name, and always update Ans
    if store:
        ctx.vars[store] = results
    ctx.vars["Ans"] = results
    return results


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

TOOL_MAP = {
    "math_eval":    lambda args, ctx: math_eval(args["expression"], ctx),
    "math_stats":   lambda args, ctx: math_stats(args["operation"], args["values"], ctx),
    "math_sort":    lambda args, ctx: math_sort(args["operation"], args["values"], ctx),
    "math_numbers": lambda args, ctx: math_numbers(args["operation"], args.get("a"), args.get("b"), ctx),
    "math_seq":     lambda args, ctx: math_seq(
        args["operation"], ctx,
        expression=args.get("expression"),
        start=args.get("start"),
        end=args.get("end"),
        values=args.get("values"),
        r=args.get("r"),
        store=args.get("store"),
        min_val=args.get("min"),
        max_val=args.get("max"),
    ),
}


def run_tool_calls(tool_call_json: str) -> list:
    """Execute a JSON array of tool calls in order, return list of results."""
    calls = json.loads(tool_call_json)
    ctx = MathEvalContext()
    results = []
    for call in calls:
        args = call["arguments"]
        fn = TOOL_MAP[call["name"]]
        result = fn(args, ctx)
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _approx_equal(a, b, rel_tol=1e-6) -> bool:
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_approx_equal(x, y, rel_tol) for x, y in zip(a, b))
    try:
        return isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=1e-9)
    except (TypeError, ValueError):
        return str(a) == str(b)


def validate(tool_call_str: str, tool_response_str: str) -> bool:
    """Return True if executing tool_calls produces the expected tool_responses."""
    if not tool_call_str or not tool_response_str:
        return False
    try:
        actual = run_tool_calls(tool_call_str)
        expected = json.loads(tool_response_str)
        return _approx_equal(actual, expected)
    except Exception:
        return False
