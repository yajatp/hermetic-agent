"""Gemini integration — structured outputs only, no free-text parsing.

Every model call is pinned to a Pydantic ``response_schema`` so the engine
receives validated objects, never a blob of prose it has to regex. The two
system prompts encode the agent's cognitive stance:

  * the **synthesizer** is an error-averse engineer aiming for optimal
    complexity and exhaustive edge-case coverage;
  * the **healer** is a cold automated debugger that reads telemetry, isolates a
    root cause, and rewrites — no apologies, no meta-commentary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Type, TypeVar

from pydantic import BaseModel

from hermetic.config import Settings, settings
from hermetic.schemas import CodeRefactorSchema, CodeSynthesisSchema

T = TypeVar("T", bound=BaseModel)

SYNTHESIZER_SYSTEM = """\
You are an elite, error-averse Python engineer inside a hermetic, air-gapped \
execution engine. You will be given a problem specification and the exact test \
suite that will judge your code.

Rules:
- Target optimal time and space complexity; state it honestly.
- Handle extreme inputs: empty, single-element, negatives, duplicates, very \
large sizes, and type boundaries.
- Define EXACTLY the functions/classes the test suite imports from `solution`. \
Read the tests carefully to match names and signatures.
- Your runtime is sealed: no network, no filesystem, no os/sys/subprocess. Use \
only pure-Python standard-library data structures and math.
- Output ONLY through the provided schema. `target_code` must be raw executable \
Python with no markdown fences and no commentary."""

HEALER_SYSTEM = """\
You are an automated debugger inside a hermetic execution engine. You are shown \
code that failed, together with STRUCTURED telemetry describing exactly how it \
failed (static AST violations, a runtime exception/timeout, or a specific \
semantic assertion that diverged).

Rules:
- Do NOT apologize. Do NOT narrate your feelings or say "I see the issue". \
Diagnose and fix.
- Classify the fault: STATIC vs RUNTIME vs SEMANTIC, and fix that exact class.
- If a STAGNATION WARNING is present, your previous approach is a dead end: \
discard it and adopt a genuinely different algorithm (e.g. recursion -> DP, \
brute force -> two-pointer/hashing). Do not merely tweak lines.
- Respect the sealed runtime: no network, no filesystem, no os/sys/subprocess.
- Output ONLY through the provided schema. `corrected_code` must be raw \
executable Python with no markdown fences and no commentary."""


@dataclass
class LLMResult:
    """A parsed structured response plus its token accounting."""

    value: BaseModel
    usage: Dict[str, int]


class LLMClient:
    """Thin, typed wrapper over the Google Gen AI SDK."""

    def __init__(self, config: Settings = settings) -> None:
        self.cfg = config
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=self.cfg.require_api_key())
        return self._client

    def _generate(
        self, *, system: str, prompt: str, schema: Type[T], temperature: float
    ) -> Tuple[T, Dict[str, int]]:
        from google.genai import types

        response = self.client.models.generate_content(
            model=self.cfg.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system,
                response_mime_type="application/json",
                response_schema=schema,
                temperature=temperature,
            ),
        )
        parsed = getattr(response, "parsed", None)
        if not isinstance(parsed, schema):
            # Defensive fallback: validate the raw JSON text ourselves.
            parsed = schema.model_validate_json(response.text)

        usage = {"prompt_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        meta = getattr(response, "usage_metadata", None)
        if meta is not None:
            usage = {
                "prompt_tokens": int(getattr(meta, "prompt_token_count", 0) or 0),
                "output_tokens": int(
                    getattr(meta, "candidates_token_count", 0) or 0
                ),
                "total_tokens": int(getattr(meta, "total_token_count", 0) or 0),
            }
        return parsed, usage

    def synthesize(
        self, *, problem: str, tests: str, entrypoint_hint: str
    ) -> Tuple[CodeSynthesisSchema, Dict[str, int]]:
        prompt = (
            f"# Problem specification\n{problem}\n\n"
            f"# Entry point\n{entrypoint_hint}\n\n"
            f"# Test suite (this is exactly what will run against your code)\n"
            f"```python\n{tests}\n```\n\n"
            "Synthesize a correct, efficient solution."
        )
        return self._generate(
            system=SYNTHESIZER_SYSTEM,
            prompt=prompt,
            schema=CodeSynthesisSchema,
            temperature=0.2,
        )

    def refactor(
        self,
        *,
        problem: str,
        tests: str,
        failed_code: str,
        feedback: str,
        stagnating: bool,
    ) -> Tuple[CodeRefactorSchema, Dict[str, int]]:
        warning = (
            "\n\n!!! STAGNATION WARNING: your recent attempts are structurally "
            "near-identical and keep failing. You MUST change algorithmic "
            "strategy, not tweak lines.\n"
            if stagnating
            else ""
        )
        prompt = (
            f"# Problem specification\n{problem}\n\n"
            f"# Test suite\n```python\n{tests}\n```\n\n"
            f"# Code that FAILED\n```python\n{failed_code}\n```\n\n"
            f"# Structured failure telemetry\n{feedback}{warning}\n\n"
            "Diagnose the root cause and rewrite the solution to pass."
        )
        # Slightly warmer on repair so it can explore alternatives.
        return self._generate(
            system=HEALER_SYSTEM,
            prompt=prompt,
            schema=CodeRefactorSchema,
            temperature=0.4 if not stagnating else 0.7,
        )
