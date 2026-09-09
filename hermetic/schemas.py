"""Structured contracts for the whole engine.

Two families live here:

1. **LLM I/O schemas** (``CodeSynthesisSchema``, ``CodeRefactorSchema``) — the
   *only* shape the model is allowed to speak in. We never parse free text.
2. **Telemetry / state models** — the deterministic payloads the state machine
   carries between nodes.

The LangGraph ``AgentState`` itself is a ``TypedDict`` (LangGraph merges partial
dict updates from each node), while the richer payloads are Pydantic models so
they validate and serialize cleanly into the state.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, TypedDict

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# LLM structured-output schemas (passed to Gemini as response_schema)
# --------------------------------------------------------------------------- #
class CodeSynthesisSchema(BaseModel):
    """First-pass solution. The model must fill every field."""

    big_o_analysis: str = Field(
        description="Strict analysis of target time and space complexity, e.g. "
        "'Time O(n log n), Space O(n)', with a one-line justification."
    )
    edge_cases: List[str] = Field(
        default_factory=list,
        description="Concrete edge cases the solution explicitly handles "
        "(empty input, single element, negatives, overflow, duplicates, ...).",
    )
    target_code: str = Field(
        description="Clean, executable Python source ONLY. No markdown fences, "
        "no backticks, no prose. Must define exactly the functions/classes the "
        "test suite imports."
    )


class CodeRefactorSchema(BaseModel):
    """Repair pass. Driven by structured failure telemetry, not vibes."""

    failure_diagnosis: str = Field(
        description="Root-cause isolation. State whether the fault is STATIC "
        "(forbidden import / syntax), RUNTIME (exception / timeout / OOM), or "
        "SEMANTIC (a specific assertion diverged), and why."
    )
    remedial_strategy: str = Field(
        description="The concrete change plan targeting the failing assertion or "
        "exception. If a stagnation warning is present, this MUST describe a "
        "different algorithmic approach than the previous attempt."
    )
    corrected_code: str = Field(
        description="The fully rewritten Python source ONLY. No markdown fences, "
        "no backticks, no prose."
    )


# --------------------------------------------------------------------------- #
# Deterministic telemetry payloads
# --------------------------------------------------------------------------- #
class ExecutionStatus(str, Enum):
    PENDING = "PENDING"
    AST_FAIL = "AST_FAIL"          # blocked pre-flight, never reached the sandbox
    RUNTIME_FAIL = "RUNTIME_FAIL"  # raised / crashed / timed out / OOM
    SEMANTIC_FAIL = "SEMANTIC_FAIL"  # ran clean but failed assertions
    SUCCESS = "SUCCESS"


class SemanticFailure(BaseModel):
    """One failed test case, extracted from the pytest JSON report."""

    test_name: str
    message: str = Field(description="Assertion message / short crash summary.")
    expected: Optional[str] = None
    actual: Optional[str] = None


class TelemetryRecord(BaseModel):
    """A single loop iteration's full diagnostic footprint.

    Appended to ``telemetry_history`` and fed verbatim into the healer so the
    model reasons over hard evidence rather than a summary.
    """

    iteration: int
    status: ExecutionStatus
    stdout: str = ""
    stderr: str = ""
    exit_code: Optional[int] = None
    timed_out: bool = False
    peak_memory_bytes: int = 0
    tests_total: int = 0
    tests_passed: int = 0
    semantic_failures: List[SemanticFailure] = Field(default_factory=list)
    # Populated only when the AST sanitizer rejects the code pre-flight.
    static_violations: List[str] = Field(default_factory=list)

    def is_success(self) -> bool:
        return self.status == ExecutionStatus.SUCCESS

    def as_feedback(self) -> str:
        """Render this record as compact, model-readable feedback."""
        lines = [f"# Iteration {self.iteration} result: {self.status.value}"]
        if self.static_violations:
            lines.append("## Static (AST) violations — code never ran:")
            lines.extend(f"  - {v}" for v in self.static_violations)
        if self.timed_out:
            lines.append(
                "## Watchdog: execution exceeded the hard wall-clock ceiling "
                "(infinite loop / deadlock). Reduce complexity."
            )
        if self.tests_total:
            lines.append(
                f"## Tests: {self.tests_passed}/{self.tests_total} passed."
            )
        for sf in self.semantic_failures:
            detail = f"  - {sf.test_name}: {sf.message}"
            if sf.expected is not None or sf.actual is not None:
                detail += f" (expected={sf.expected!r}, actual={sf.actual!r})"
            lines.append(detail)
        if self.peak_memory_bytes:
            lines.append(
                f"## Peak traced memory: {self.peak_memory_bytes / 1024:.1f} KiB"
            )
        if self.stderr.strip():
            lines.append("## stderr (tail):")
            lines.append(_tail(self.stderr, 2000))
        return "\n".join(lines)


def _tail(text: str, limit: int) -> str:
    text = text.rstrip()
    return text if len(text) <= limit else "...(truncated)...\n" + text[-limit:]


# --------------------------------------------------------------------------- #
# LangGraph state
# --------------------------------------------------------------------------- #
class AgentState(TypedDict, total=False):
    """Single source of truth threaded through every node.

    ``total=False`` lets nodes return partial updates that LangGraph merges.
    """

    # Inputs (set once at kickoff)
    problem_specification: str
    static_unit_tests: str
    entrypoint_hint: str  # human note on what the tests import

    # Evolving code state
    current_code: Optional[str]
    code_history: List[str]
    ast_hash_history: List[str]

    # Loop bookkeeping
    iteration_count: int
    execution_status: str  # ExecutionStatus value
    stagnation_warning: bool

    # Diagnostics
    telemetry_history: List[Dict[str, Any]]  # serialized TelemetryRecord
    peak_memory_bytes: int
    token_usage_total: Dict[str, int]

    # Terminal
    final_success: bool
    termination_reason: str
