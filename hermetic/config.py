"""Central configuration.

Everything the engine and sandbox need is resolved once, here, from the
environment (with sane defaults). Keeping this in one immutable object means
no module reaches into ``os.environ`` on its own — a single source of truth
for the whole run.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # dotenv is optional at runtime; env may be set another way.
    pass


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    return float(raw) if raw not in (None, "") else default


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    return int(raw) if raw not in (None, "") else default


@dataclass(frozen=True)
class Settings:
    """Immutable, fully-resolved configuration for a single engine run."""

    # --- LLM ---
    gemini_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY")
    )
    model: str = field(
        default_factory=lambda: os.getenv("HERMETIC_MODEL", "gemini-2.0-flash")
    )

    # --- Loop control ---
    max_iterations: int = field(
        default_factory=lambda: _env_int("HERMETIC_MAX_ITERATIONS", 5)
    )
    stagnation_threshold: float = field(
        default_factory=lambda: _env_float("HERMETIC_STAGNATION_THRESHOLD", 0.02)
    )

    # --- Sandbox hardening ---
    wallclock_seconds: float = field(
        default_factory=lambda: _env_float("HERMETIC_WALLCLOCK_SECONDS", 5.0)
    )
    mem_limit: str = field(
        default_factory=lambda: os.getenv("HERMETIC_MEM_LIMIT", "128m")
    )
    cpu_shares: int = field(
        default_factory=lambda: _env_int("HERMETIC_CPU_SHARES", 512)
    )
    pids_limit: int = field(
        default_factory=lambda: _env_int("HERMETIC_PIDS_LIMIT", 50)
    )
    image_tag: str = field(
        default_factory=lambda: os.getenv("HERMETIC_IMAGE_TAG", "hermetic-sandbox:1")
    )
    base_image: str = field(
        default_factory=lambda: os.getenv("HERMETIC_BASE_IMAGE", "python:3.11-slim")
    )

    def require_api_key(self) -> str:
        if not self.gemini_api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Copy .env.example to .env and fill it in."
            )
        return self.gemini_api_key


# Module-level singleton; import `settings` everywhere.
settings = Settings()
