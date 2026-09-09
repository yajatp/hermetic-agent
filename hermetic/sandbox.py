"""The hardened compute sandbox.

Treat every byte of generated code as hostile. This module gives it the
smallest, most locked-down box we can build and still get useful telemetry back:

  * **Air-gapped** — ``network_disabled`` + ``network_mode="none"``.
  * **Resource-capped** — memory, CPU shares, and a PID ceiling (fork-bomb guard).
  * **Read-only root** — the only writable surface is a size-capped tmpfs at
    ``/workspace``; nothing the code writes survives or touches the host.
  * **Deprivileged** — non-root user, all capabilities dropped, no new privileges.
  * **Reproducible** — a version-pinned image whose base digest is resolved and
    recorded at build time, so the sandbox is never a moving target.
  * **Watchdogged** — a thread-level wall-clock ceiling that SIGKILLs a hung
    container even if Docker's own limits don't trip.

``ExecutionBackend`` is an ABC so the Docker implementation can later be swapped
for a microVM backend (Firecracker / gVisor) without the engine noticing.
"""

from __future__ import annotations

import io
import platform
import tarfile
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from hermetic.config import Settings, settings

_RUNNER_CMD = ["python", "/opt/hermetic/runner.py"]
_DOCKERFILE = Path(__file__).with_name("Dockerfile.sandbox")


@dataclass
class RawExecution:
    """Untyped output of one sandbox run, handed to the telemetry parser."""

    stdout: str
    stderr: str
    exit_code: Optional[int]
    timed_out: bool
    wall_seconds: float


class ExecutionBackend(ABC):
    """Swap-in seam for alternative isolation technologies."""

    @abstractmethod
    def ensure_ready(self) -> None:
        """Build/pull whatever the backend needs before the first run."""

    @abstractmethod
    def run(self, *, solution_code: str, test_code: str) -> RawExecution:
        """Execute one (solution, tests) pair under full isolation."""

    @property
    @abstractmethod
    def image_reference(self) -> str:
        """A stable, reproducible identifier for what actually runs."""


def _host_platform() -> str:
    """Map the host CPU to a Docker platform string.

    Lets the same image run natively on Apple Silicon (arm64) and x86_64 cloud
    hosts without qemu emulation — emulation is slow enough to trip a 5s
    watchdog and produce false timeouts.
    """
    machine = platform.machine().lower()
    if machine in ("arm64", "aarch64"):
        return "linux/arm64"
    if machine in ("x86_64", "amd64"):
        return "linux/amd64"
    return f"linux/{machine}"


def _tar_bytes(files: dict) -> bytes:
    """Pack ``{name: text}`` into an in-memory tar for ``put_archive``."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for name, text in files.items():
            data = text.encode("utf-8")
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


class DockerSandbox(ExecutionBackend):
    """Docker-backed implementation of the hardened sandbox."""

    def __init__(self, config: Settings = settings) -> None:
        self.cfg = config
        self._client = None
        self._resolved_digest: Optional[str] = None

    # -- lifecycle ---------------------------------------------------------- #
    @property
    def client(self):
        if self._client is None:
            import docker  # imported lazily so the module loads without Docker

            self._client = docker.from_env()
        return self._client

    @property
    def image_reference(self) -> str:
        if self._resolved_digest:
            return f"{self.cfg.image_tag}@{self._resolved_digest}"
        return self.cfg.image_tag

    def ensure_ready(self) -> None:
        """Build the pinned sandbox image if it isn't already present."""
        import docker

        try:
            self.client.images.get(self.cfg.image_tag)
        except docker.errors.ImageNotFound:
            self.client.images.build(
                path=str(_DOCKERFILE.parent),
                dockerfile=_DOCKERFILE.name,
                tag=self.cfg.image_tag,
                platform=_host_platform(),
                rm=True,
                pull=True,
            )
        image = self.client.images.get(self.cfg.image_tag)
        # Record the concrete base digest so the run is auditable / reproducible.
        digests = image.attrs.get("RepoDigests") or []
        self._resolved_digest = image.id  # sha256:... of the built image
        if digests:
            self._resolved_digest = digests[0].split("@")[-1]

    # -- execution ---------------------------------------------------------- #
    def run(self, *, solution_code: str, test_code: str) -> RawExecution:
        container = self._create_container()
        started = time.monotonic()
        timed_out = False
        exit_code: Optional[int] = None
        try:
            # Inject files into the writable tmpfs, then start.
            container.put_archive(
                "/workspace",
                _tar_bytes(
                    {"solution.py": solution_code, "test_suite.py": test_code}
                ),
            )
            container.start()

            # Thread-level watchdog: block on wait(), but only up to the ceiling.
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(container.wait)
                try:
                    result = future.result(timeout=self.cfg.wallclock_seconds)
                    exit_code = int(result.get("StatusCode", -1))
                except FuturesTimeout:
                    timed_out = True
                    self._kill(container)

            stdout = self._logs(container, stdout=True, stderr=False)
            stderr = self._logs(container, stdout=False, stderr=True)
            return RawExecution(
                stdout=stdout,
                stderr=stderr,
                exit_code=exit_code,
                timed_out=timed_out,
                wall_seconds=time.monotonic() - started,
            )
        finally:
            self._remove(container)

    # -- helpers ------------------------------------------------------------ #
    def _create_container(self):
        """Create (but don't start) a maximally constrained container."""
        return self.client.containers.create(
            image=self.cfg.image_tag,
            command=_RUNNER_CMD,
            platform=_host_platform(),
            # --- isolation ---
            network_disabled=True,
            network_mode="none",
            # --- resource caps ---
            mem_limit=self.cfg.mem_limit,
            memswap_limit=self.cfg.mem_limit,  # deny swap escape hatch
            cpu_shares=self.cfg.cpu_shares,
            pids_limit=self.cfg.pids_limit,
            # --- filesystem ---
            read_only=True,
            tmpfs={"/workspace": "rw,size=16m,mode=1777,noexec,nosuid,nodev"},
            working_dir="/workspace",
            # --- privileges ---
            user="sandbox",
            cap_drop=["ALL"],
            security_opt=["no-new-privileges"],
            # --- io ---
            stdin_open=False,
            tty=False,
            detach=True,
            environment={"PYTHONDONTWRITEBYTECODE": "1", "HOME": "/workspace"},
        )

    @staticmethod
    def _kill(container) -> None:
        try:
            container.kill(signal="SIGKILL")
        except Exception:
            pass

    @staticmethod
    def _remove(container) -> None:
        try:
            container.remove(force=True, v=True)
        except Exception:
            pass

    @staticmethod
    def _logs(container, *, stdout: bool, stderr: bool) -> str:
        try:
            raw = container.logs(stdout=stdout, stderr=stderr)
            return raw.decode("utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001
            return f"<log capture failed: {exc}>"
