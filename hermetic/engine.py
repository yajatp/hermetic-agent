"""The deterministic control plane: a LangGraph state machine.

The LLM is powerful but non-deterministic; this graph is the opposite, and it is
in charge. The model only ever *fills fields* — the graph decides what runs,
when to heal, and when to stop.

Topology::

    START -> synthesize -> validate_ast --(fail)--> refactor ─┐
                               │(pass)                          │
                               ▼                                │
                          execute_sandbox <────────────────────┘
                               │
                          analyze  ──(success | budget spent)──> END
                               │
                               └────────(recoverable failure)──> refactor

``validate_ast`` runs on every freshly generated candidate — from both
``synthesize`` and ``refactor`` — so untrusted code is vetted before it can ever
reach a container.
"""

from __future__ import annotations

from typing import Optional

from hermetic import security
from hermetic.config import Settings, settings
from hermetic.llm import LLMClient
from hermetic.sandbox import ExecutionBackend
from hermetic.schemas import (
    AgentState,
    ExecutionStatus,
    TelemetryRecord,
)
from hermetic.telemetry import build_record


class HermeticEngine:
    """Wires the nodes, backend, and LLM into a runnable LangGraph app."""

    def __init__(
        self,
        backend: ExecutionBackend,
        llm: Optional[LLMClient] = None,
        config: Settings = settings,
    ) -> None:
        self.backend = backend
        self.llm = llm or LLMClient(config)
        self.cfg = config
        self.app = self._build_graph()

    # ------------------------------------------------------------------ #
    # Graph construction
    # ------------------------------------------------------------------ #
    def _build_graph(self):
        from langgraph.graph import END, START, StateGraph

        graph = StateGraph(AgentState)
        graph.add_node("synthesize", self.synthesize_node)
        graph.add_node("validate_ast", self.validate_ast_node)
        graph.add_node("execute_sandbox", self.execute_node)
        graph.add_node("analyze", self.analyze_node)
        graph.add_node("refactor", self.refactor_node)

        graph.add_edge(START, "synthesize")
        graph.add_edge("synthesize", "validate_ast")
        graph.add_edge("refactor", "validate_ast")

        # After AST validation: clean code runs; rejected code heals (if budget).
        graph.add_conditional_edges(
            "validate_ast",
            self._route_after_ast,
            {"execute": "execute_sandbox", "refactor": "refactor", "end": END},
        )
        graph.add_edge("execute_sandbox", "analyze")
        graph.add_conditional_edges(
            "analyze",
            self._route_after_analyze,
            {"refactor": "refactor", "end": END},
        )
        return graph.compile()

    # ------------------------------------------------------------------ #
    # Nodes
    # ------------------------------------------------------------------ #
    def synthesize_node(self, state: AgentState) -> dict:
        result, usage = self.llm.synthesize(
            problem=state["problem_specification"],
            tests=state["static_unit_tests"],
            entrypoint_hint=state.get("entrypoint_hint", ""),
        )
        return {
            "current_code": result.target_code,
            "iteration_count": 0,
            "execution_status": ExecutionStatus.PENDING.value,
            "token_usage_total": _accum(state.get("token_usage_total"), usage),
        }

    def validate_ast_node(self, state: AgentState) -> dict:
        """Pre-flight AST gate. Records a static-fail telemetry entry on reject."""
        code = state["current_code"] or ""
        result = security.sanitize(code)
        if result.ok:
            return {"execution_status": ExecutionStatus.PENDING.value}

        record = TelemetryRecord(
            iteration=state.get("iteration_count", 0),
            status=ExecutionStatus.AST_FAIL,
            static_violations=result.violations,
        )
        return {
            "execution_status": ExecutionStatus.AST_FAIL.value,
            "telemetry_history": _append(
                state.get("telemetry_history"), record.model_dump()
            ),
        }

    def execute_node(self, state: AgentState) -> dict:
        """Run the vetted candidate in the sandbox and parse its telemetry."""
        raw = self.backend.run(
            solution_code=state["current_code"] or "",
            test_code=state["static_unit_tests"],
        )
        record = build_record(
            iteration=state.get("iteration_count", 0),
            stdout=raw.stdout,
            stderr=raw.stderr,
            exit_code=raw.exit_code,
            timed_out=raw.timed_out,
        )
        return {
            "execution_status": record.status.value,
            "peak_memory_bytes": record.peak_memory_bytes,
            "telemetry_history": _append(
                state.get("telemetry_history"), record.model_dump()
            ),
        }

    def analyze_node(self, state: AgentState) -> dict:
        """Fold the latest result into terminal flags. Routing reads these."""
        status = ExecutionStatus(state["execution_status"])
        if status == ExecutionStatus.SUCCESS:
            return {"final_success": True, "termination_reason": "all tests passed"}
        return {"final_success": False}

    def refactor_node(self, state: AgentState) -> dict:
        """Heal: record the failed attempt, check stagnation, ask for a rewrite."""
        failed_code = state["current_code"] or ""
        history = state.get("code_history") or []

        stagnating = security.is_stagnating(
            failed_code, history, self.cfg.stagnation_threshold
        )

        latest = _latest_record(state)
        feedback = latest.as_feedback() if latest else "Unknown failure."

        result, usage = self.llm.refactor(
            problem=state["problem_specification"],
            tests=state["static_unit_tests"],
            failed_code=failed_code,
            feedback=feedback,
            stagnating=stagnating,
        )
        return {
            "current_code": result.corrected_code,
            "iteration_count": state.get("iteration_count", 0) + 1,
            "stagnation_warning": stagnating,
            "code_history": history + [failed_code],
            "ast_hash_history": (state.get("ast_hash_history") or [])
            + [security.structural_hash(failed_code)],
            "token_usage_total": _accum(state.get("token_usage_total"), usage),
        }

    # ------------------------------------------------------------------ #
    # Routers (pure functions of state — the deterministic core)
    # ------------------------------------------------------------------ #
    def _route_after_ast(self, state: AgentState) -> str:
        if state["execution_status"] != ExecutionStatus.AST_FAIL.value:
            return "execute"
        # Rejected pre-flight: heal if there's budget, else give up.
        if state.get("iteration_count", 0) < self.cfg.max_iterations:
            return "refactor"
        return "end"

    def _route_after_analyze(self, state: AgentState) -> str:
        if state.get("final_success"):
            return "end"
        if state.get("iteration_count", 0) >= self.cfg.max_iterations:
            return "end"
        return "refactor"

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #
    def solve(
        self, *, problem: str, tests: str, entrypoint_hint: str = ""
    ) -> AgentState:
        self.backend.ensure_ready()
        initial: AgentState = {
            "problem_specification": problem,
            "static_unit_tests": tests,
            "entrypoint_hint": entrypoint_hint,
            "current_code": None,
            "code_history": [],
            "ast_hash_history": [],
            "iteration_count": 0,
            "execution_status": ExecutionStatus.PENDING.value,
            "stagnation_warning": False,
            "telemetry_history": [],
            "peak_memory_bytes": 0,
            "token_usage_total": {},
            "final_success": False,
            "termination_reason": "",
        }
        # recursion_limit guards against any pathological loop in the graph itself.
        final = self.app.invoke(
            initial, config={"recursion_limit": 4 * self.cfg.max_iterations + 10}
        )
        if not final.get("final_success") and not final.get("termination_reason"):
            final["termination_reason"] = (
                f"exhausted {self.cfg.max_iterations} healing iterations"
            )
        return final


# --------------------------------------------------------------------------- #
# Small state helpers (kept pure so nodes stay readable)
# --------------------------------------------------------------------------- #
def _append(seq: Optional[list], item) -> list:
    return (seq or []) + [item]


def _accum(total: Optional[dict], usage: dict) -> dict:
    total = dict(total or {})
    for key, value in usage.items():
        total[key] = total.get(key, 0) + value
    return total


def _latest_record(state: AgentState) -> Optional[TelemetryRecord]:
    history = state.get("telemetry_history") or []
    if not history:
        return None
    return TelemetryRecord.model_validate(history[-1])
