"""In-container harness. Runs UNTRUSTED code. Never import project modules here.

This file is baked into the sandbox image (immutable) and executed as the
container's entrypoint command. It:

  1. puts /workspace on sys.path so the test suite can ``import solution``,
  2. runs the test suite under pytest with the JSON reporter,
  3. measures peak traced memory with tracemalloc, and
  4. prints a single JSON envelope between sentinel markers on stdout.

We ship results out over stdout (not a bind mount) so the container's root
filesystem can stay read-only and its workspace an ephemeral tmpfs.
"""

import json
import sys
import tracemalloc

WORKSPACE = "/workspace"
REPORT_PATH = "/workspace/report.json"
START = "<<<HERMETIC_TELEMETRY_START>>>"
END = "<<<HERMETIC_TELEMETRY_END>>>"


def main() -> None:
    if WORKSPACE not in sys.path:
        sys.path.insert(0, WORKSPACE)

    tracemalloc.start()
    return_code = -1
    try:
        import pytest

        return_code = int(
            pytest.main(
                [
                    "-q",
                    "-p", "no:cacheprovider",
                    "--json-report",
                    f"--json-report-file={REPORT_PATH}",
                    f"{WORKSPACE}/test_suite.py",
                ]
            )
        )
    except SystemExit as exc:  # pytest occasionally raises instead of returning
        return_code = int(exc.code) if exc.code is not None else -1
    except BaseException as exc:  # noqa: BLE001 — must never crash unframed
        sys.stderr.write(f"harness error: {type(exc).__name__}: {exc}\n")

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    try:
        with open(REPORT_PATH) as fh:
            report = json.load(fh)
    except Exception:
        report = None

    envelope = {
        "pytest_returncode": return_code,
        "peak_memory_bytes": int(peak),
        "report": report,
    }
    sys.stdout.write("\n" + START + "\n")
    sys.stdout.write(json.dumps(envelope))
    sys.stdout.write("\n" + END + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
