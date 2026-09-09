"""Turn raw sandbox output into a structured, model-readable diagnosis.

This is the "dual-signal" layer. Naive agents only see a stack trace; the
Hermetic Agent separates two failure classes so the healer can reason about the
right thing:

  * **Runtime telemetry** — the process crashed, hung, or raised a non-assertion
    exception (``IndexError``, ``KeyError``, ``TimeoutError`` from the watchdog).
  * **Semantic telemetry** — the code ran clean but an *assertion* diverged; we
    surface which test, and expected-vs-actual where we can recover it.

Everything is derived from the JSON envelope the in-container harness prints,
plus the pytest-json-report payload inside it.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from hermetic.schemas import (
    ExecutionStatus,
    SemanticFailure,
    TelemetryRecord,
)

_START = "<<<HERMETIC_TELEMETRY_START>>>"
_END = "<<<HERMETIC_TELEMETRY_END>>>"

# Recover "assert <actual> == <expected>" shapes from pytest's rewritten asserts.
_ASSERT_EQ = re.compile(r"assert\s+(.+?)\s*==\s*(.+)")


def extract_envelope(stdout: str) -> Optional[Dict[str, Any]]:
    """Pull the JSON telemetry envelope out of the container's stdout stream."""
    start = stdout.find(_START)
    end = stdout.find(_END)
    if start == -1 or end == -1 or end < start:
        return None
    blob = stdout[start + len(_START) : end].strip()
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        return None


def _is_assertion(message: str) -> bool:
    return "AssertionError" in message or message.strip().startswith("assert")


def _split_expected_actual(message: str):
    """Best-effort (actual, expected) recovery from an assertion message."""
    match = _ASSERT_EQ.search(message.replace("\n", " "))
    if not match:
        return None, None
    actual = match.group(1).strip()
    expected = match.group(2).strip()
    # Trim pytest's trailing "+ where ..." explanations.
    expected = re.split(r"\s+\+ where|\s+where ", expected)[0].strip()
    return actual, expected


def _collect_failures(report: Dict[str, Any]) -> List[Dict[str, str]]:
    """Return [{test_name, message}] for every non-passing test."""
    failures: List[Dict[str, str]] = []
    for test in report.get("tests", []):
        outcome = test.get("outcome")
        if outcome in ("passed", "skipped"):
            continue
        name = test.get("nodeid", "<unknown>").split("::")[-1]
        message = ""
        # The failing phase carries the crash; check call, then setup/collect.
        for phase in ("call", "setup", "teardown"):
            info = test.get(phase) or {}
            crash = info.get("crash") or {}
            if crash.get("message"):
                message = crash["message"]
                break
            if info.get("longrepr"):
                message = str(info["longrepr"])
                break
        failures.append({"test_name": name, "message": message.strip()})
    return failures


def build_record(
    *,
    iteration: int,
    stdout: str,
    stderr: str,
    exit_code: Optional[int],
    timed_out: bool,
) -> TelemetryRecord:
    """Assemble the canonical :class:`TelemetryRecord` for one sandbox run."""
    # 1. Watchdog kill dominates everything — the process never finished.
    if timed_out:
        return TelemetryRecord(
            iteration=iteration,
            status=ExecutionStatus.RUNTIME_FAIL,
            stdout=stdout,
            stderr=(stderr or "")
            + "\nResourceExhaustionError: infinite loop or deadlock detected "
            "(watchdog wall-clock ceiling exceeded).",
            exit_code=exit_code,
            timed_out=True,
        )

    envelope = extract_envelope(stdout)
    peak = int(envelope.get("peak_memory_bytes", 0)) if envelope else 0
    report = envelope.get("report") if envelope else None

    # 2. No parseable report => the harness itself crashed. Pure runtime failure.
    if not report:
        return TelemetryRecord(
            iteration=iteration,
            status=ExecutionStatus.RUNTIME_FAIL,
            stdout=stdout,
            stderr=stderr or "No telemetry envelope produced; execution crashed "
            "before tests could run (import-time or syntax error at runtime).",
            exit_code=exit_code,
            peak_memory_bytes=peak,
        )

    summary = report.get("summary", {})
    total = int(summary.get("total", summary.get("collected", 0)) or 0)
    passed = int(summary.get("passed", 0) or 0)
    failures = _collect_failures(report)

    # 3. Classify failures: any non-assertion exception => runtime, else semantic.
    semantic: List[SemanticFailure] = []
    has_runtime_exception = False
    for fail in failures:
        msg = fail["message"]
        if _is_assertion(msg):
            actual, expected = _split_expected_actual(msg)
            semantic.append(
                SemanticFailure(
                    test_name=fail["test_name"],
                    message=_first_line(msg),
                    expected=expected,
                    actual=actual,
                )
            )
        else:
            has_runtime_exception = True
            semantic.append(
                SemanticFailure(
                    test_name=fail["test_name"], message=_first_line(msg)
                )
            )

    if has_runtime_exception:
        status = ExecutionStatus.RUNTIME_FAIL
    elif failures:
        status = ExecutionStatus.SEMANTIC_FAIL
    elif total > 0 and passed == total:
        status = ExecutionStatus.SUCCESS
    else:
        # Nothing collected / nothing ran — treat as runtime failure so we heal.
        status = ExecutionStatus.RUNTIME_FAIL

    return TelemetryRecord(
        iteration=iteration,
        status=status,
        stdout=stdout,
        stderr=stderr,
        exit_code=exit_code,
        peak_memory_bytes=peak,
        tests_total=total,
        tests_passed=passed,
        semantic_failures=semantic,
    )


def _first_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return text.strip()
