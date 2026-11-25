"""Hermetic Agent — orchestration entry point.

Runs the full closed loop against a deliberately tricky algorithmic problem and
prints a markdown execution report: every iteration, the failure class that
drove each heal, token spend, peak memory, and the final verdict.

Usage:
    python main.py            # run the built-in challenge
    python main.py --preflight  # offline: prove the AST gate works, no API/Docker
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from typing import Any, Dict, List

from hermetic.config import settings
from hermetic.engine import HermeticEngine
from hermetic.sandbox import DockerSandbox
from hermetic.schemas import ExecutionStatus, TelemetryRecord

# --------------------------------------------------------------------------- #
# The challenge: interval merging. Small surface, big edge-case blast radius —
# touching vs. overlapping boundaries, nesting, unsorted input, empties. Classic
# off-by-one territory that punishes sloppy comparisons, so the healer earns it.
# --------------------------------------------------------------------------- #
PROBLEM = textwrap.dedent(
    """
    Implement `merge_intervals(intervals)` in a module importable as `solution`.

    Given a list of `[start, end]` integer intervals (unsorted, possibly empty,
    possibly nested or duplicated), return the minimal list of merged,
    non-overlapping intervals, sorted ascending by start.

    Rules:
      - Intervals that merely TOUCH must merge: [1, 2] and [2, 3] -> [1, 3].
      - A fully nested interval is absorbed: [1, 10] and [3, 4] -> [1, 10].
      - Input may be empty; return [].
      - Do not mutate the input list.
      - Each returned interval is a two-element list [start, end].
    """
).strip()

ENTRYPOINT_HINT = "The tests do: `from solution import merge_intervals`."

TESTS = textwrap.dedent(
    '''
    from solution import merge_intervals


    def test_empty():
        assert merge_intervals([]) == []

    def test_single():
        assert merge_intervals([[1, 4]]) == [[1, 4]]

    def test_disjoint_sorted():
        assert merge_intervals([[1, 2], [5, 6]]) == [[1, 2], [5, 6]]

    def test_overlap():
        assert merge_intervals([[1, 3], [2, 6]]) == [[1, 6]]

    def test_touching_merges():
        # boundary case: touching intervals must merge
        assert merge_intervals([[1, 2], [2, 3]]) == [[1, 3]]

    def test_nested_absorbed():
        assert merge_intervals([[1, 10], [3, 4]]) == [[1, 10]]

    def test_unsorted_input():
        assert merge_intervals([[5, 6], [1, 3], [2, 4]]) == [[1, 4], [5, 6]]

    def test_does_not_mutate_input():
        data = [[3, 4], [1, 2]]
        snapshot = [list(x) for x in data]
        merge_intervals(data)
        assert data == snapshot

    def test_many_merges():
        intervals = [[i, i + 2] for i in range(0, 200, 1)]
        merged = merge_intervals(intervals)
        assert merged == [[0, 201]]
    '''
).strip()


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def render_report(state: Dict[str, Any]) -> str:
    history: List[Dict[str, Any]] = state.get("telemetry_history", []) or []
    tokens = state.get("token_usage_total", {}) or {}
    lines: List[str] = []
    lines.append("# Hermetic Agent — Execution Report\n")

    verdict = "✅ SUCCESS" if state.get("final_success") else "❌ FAILED"
    lines.append(f"**Verdict:** {verdict}")
    lines.append(f"**Reason:** {state.get('termination_reason', 'n/a')}")
    lines.append(f"**Heal iterations used:** {state.get('iteration_count', 0)}")
    lines.append(
        f"**Tokens:** {tokens.get('total_tokens', 0)} "
        f"(prompt {tokens.get('prompt_tokens', 0)}, "
        f"output {tokens.get('output_tokens', 0)})"
    )
    lines.append(f"**Peak sandbox memory:** {state.get('peak_memory_bytes', 0) / 1024:.1f} KiB\n")

    lines.append("## Iteration trace\n")
    for rec_dict in history:
        rec = TelemetryRecord.model_validate(rec_dict)
        marker = {
            ExecutionStatus.SUCCESS: "✅",
            ExecutionStatus.SEMANTIC_FAIL: "🟠",
            ExecutionStatus.RUNTIME_FAIL: "🔴",
            ExecutionStatus.AST_FAIL: "⛔",
        }.get(rec.status, "•")
        lines.append(f"### {marker} Iteration {rec.iteration} — {rec.status.value}")
        if rec.tests_total:
            lines.append(f"- tests: {rec.tests_passed}/{rec.tests_total} passed")
        if rec.static_violations:
            lines.append("- AST violations: " + "; ".join(rec.static_violations))
        for sf in rec.semantic_failures:
            detail = f"  - `{sf.test_name}`: {sf.message}"
            if sf.expected is not None:
                detail += f" (expected={sf.expected}, actual={sf.actual})"
            lines.append(detail)
        if rec.timed_out:
            lines.append("- ⏱️ watchdog killed a hung execution")
        lines.append("")

    final_code = state.get("current_code")
    if final_code:
        lines.append("## Final solution\n")
        lines.append("```python")
        lines.append(final_code.strip())
        lines.append("```")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Offline preflight — proves the security gate without an API key or Docker.
# --------------------------------------------------------------------------- #
def run_preflight() -> int:
    from hermetic import security

    samples = {
        "clean solution": "def merge_intervals(x):\n    return sorted(x)\n",
        "network escape": "import socket\ndef merge_intervals(x):\n    return x\n",
        "eval injection": "def merge_intervals(x):\n    return eval('x')\n",
        "dunder escape": "def f():\n    return ().__class__.__bases__\n",
        "syntax error": "def merge_intervals(x)\n    return x\n",
    }
    print("# AST pre-flight sanitizer\n")
    all_expected = True
    for label, code in samples.items():
        result = security.sanitize(code)
        expected_ok = label == "clean solution"
        if result.ok != expected_ok:
            all_expected = False
        status = "PASS ✅" if result.ok else "BLOCK ⛔"
        print(f"- {label}: {status}")
        for v in result.violations:
            print(f"    - {v}")
    return 0 if all_expected else 1


def cli() -> int:
    parser = argparse.ArgumentParser(description="Hermetic Agent")
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Offline demo of the AST gate (no API key / Docker needed).",
    )
    args = parser.parse_args()

    if args.preflight:
        return run_preflight()

    if not settings.gemini_api_key:
        print(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and add your "
            "key, or run `python main.py --preflight` for the offline demo.",
            file=sys.stderr,
        )
        return 2

    engine = HermeticEngine(backend=DockerSandbox(settings))
    try:
        final = engine.solve(
            problem=PROBLEM, tests=TESTS, entrypoint_hint=ENTRYPOINT_HINT
        )
    except Exception as exc:  # noqa: BLE001 — surface infra errors cleanly
        print(f"Engine error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(render_report(final))
    return 0 if final.get("final_success") else 1


if __name__ == "__main__":
    raise SystemExit(cli())
