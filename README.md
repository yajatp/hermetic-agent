# Hermetic Agent

A zero-trust, closed-loop autonomous coding engine. It **synthesizes** Python
solutions, **statically vets** them, **executes** them inside a hardened,
air-gapped sandbox, ingests **dual-signal telemetry** (runtime *and* semantic),
and **self-heals** — all inside a deterministic LangGraph state machine. No human
in the debugging loop.

Built with **LangGraph · Gemini (Google Gen AI SDK) · Pydantic v2 · Docker**.

---

## Why it's more than "self-healing"

The generative model is powerful but non-deterministic. Hermetic wraps it in a
deterministic control plane: the LLM only ever *fills structured fields*; the
graph decides what runs, when to heal, and when to stop. The hard parts are the
systems around the model:

| Concern | How Hermetic handles it |
|---|---|
| Untrusted code reaching the host | **AST pre-flight** blocks dangerous imports / `eval` / dunder escapes *before* Docker starts |
| Air-gapping | `--network none` + `network_disabled`, all Linux caps dropped, non-root user, `no-new-privileges` |
| Runaway resource use | `mem_limit`, `cpu_shares`, `pids_limit` (fork-bomb guard), read-only rootfs, size-capped tmpfs |
| Infinite loops / deadlocks | **Thread-level watchdog** SIGKILLs a hung container at a hard wall-clock ceiling |
| Wrong-but-running code | **Semantic telemetry** from the pytest JSON report — which test, expected vs. actual |
| Environment drift | **Version-pinned, reproducible image**; base digest resolved & recorded at build |
| Repair loops that spin | **Stagnation circuit breaker**: structural AST hash + normalized Levenshtein force a strategy pivot |
| Vendor lock-in on isolation | `ExecutionBackend` ABC — swap Docker for Firecracker/gVisor without touching the engine |

## Architecture

```
START -> synthesize -> validate_ast --(fail)--> refactor ─┐
                           │(pass)                          │
                           ▼                                │
                      execute_sandbox <────────────────────┘
                           │
                      analyze ──(success | budget spent)──> END
                           │
                           └────────(recoverable failure)──> refactor
```

| File | Role |
|---|---|
| `hermetic/schemas.py` | Pydantic LLM schemas + telemetry models + LangGraph `AgentState` |
| `hermetic/security.py` | AST sanitizer, forbidden-import walker, Levenshtein / loop defense |
| `hermetic/sandbox.py` | Hardened Docker backend, cross-arch build, threaded watchdog (`ExecutionBackend` ABC) |
| `hermetic/telemetry.py` | Dual-signal parsing: runtime crashes vs. semantic assertion failures |
| `hermetic/llm.py` | Gemini structured-output client + synthesizer/healer system prompts |
| `hermetic/engine.py` | LangGraph nodes, edges, deterministic routers |
| `hermetic/Dockerfile.sandbox` + `hermetic/sandbox_runner.py` | Reproducible image + in-container harness |
| `main.py` | Orchestration runner, challenge problem, markdown report |

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # needs Python 3.11+
cp .env.example .env                      # add your GEMINI_API_KEY

# Offline — prove the AST security gate works (no API key / Docker needed):
python main.py --preflight

# Full loop (needs a Docker daemon + GEMINI_API_KEY):
python main.py
```

The first full run builds the pinned sandbox image once (the only step that
needs network); every execution after that is fully air-gapped.

## Requirements

- Python **3.11+**
- A running **Docker** daemon (Linux containers; builds natively on Apple
  Silicon and x86_64)
- A **Gemini API key** (`GEMINI_API_KEY`)

## Extending

- **Different isolation**: implement `ExecutionBackend` (Firecracker/gVisor) and
  pass it to `HermeticEngine`.
- **New problems**: call `engine.solve(problem=..., tests=..., entrypoint_hint=...)`
  with your own pytest suite.
- **Tighter/looser policy**: tune limits in `.env` (`HERMETIC_*`).
