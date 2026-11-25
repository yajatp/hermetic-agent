# Hermetic Agent

**A self-repairing Python coding engine with a deterministic controller and a network-disabled Docker runner.**

Given a function specification and a pytest suite, Hermetic Agent asks Gemini for a candidate, screens it with an AST policy, and runs accepted code in a restricted container. It feeds test failures and runtime errors back into the next attempt. LangGraph controls retries and stopping conditions.

![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white) ![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-24292F) ![Docker](https://img.shields.io/badge/Runner-Docker-2496ED?logo=docker&logoColor=white) [![Offline checks](https://github.com/yajatp/hermetic-agent/actions/workflows/preflight.yml/badge.svg)](https://github.com/yajatp/hermetic-agent/actions/workflows/preflight.yml)

## Try it

Use Python 3.11 or 3.12 for the pinned dependency set.

```bash
git clone https://github.com/yajatp/hermetic-agent.git
cd hermetic-agent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
hermetic --preflight
```

The offline check takes no API key or Docker daemon. It verifies that a valid candidate passes and rejected constructs are blocked. An unexpected outcome exits with a nonzero status.

For the built-in interval-merging challenge, start a Docker daemon, copy `.env.example` to `.env`, set `GEMINI_API_KEY`, and run `hermetic` from the clone directory. Set `HERMETIC_MODEL` to choose another Gemini model that supports structured output. The host calls Gemini, and Docker needs network access to build the sandbox image on the first run. The containers that execute candidates run with networking disabled.

## What it does

| Area | What you can inspect |
|---|---|
| Generation and repair | Pydantic-validated candidate code and structured repair feedback |
| Execution | AST preflight, a non-root container, a read-only root filesystem, and resource limits |
| Verification | Runtime errors and failed assertions handled as distinct failure classes |
| Stopping | Retry limits and detection of structurally similar attempts |
| Reporting | A Markdown trace of iterations, outcomes, model usage, and peak memory |

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

## Scope

The included benchmark is a single Python problem with a known pytest suite. Treat the AST screen as an early filter and the Docker settings as execution constraints. This repository demonstrates the repair loop and its controls.

## License

[MIT](LICENSE).
