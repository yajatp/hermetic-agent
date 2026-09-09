"""Pre-flight defense and anti-stagnation circuit breakers.

Two responsibilities, both pure-stdlib and both cheap enough to run every
iteration *before* we pay for a container:

1. **AST sanitizer** — compile the model's code to an Abstract Syntax Tree and
   walk it. If it won't parse, or it reaches for the host (dangerous imports,
   ``eval``/``exec``, dunder escape hatches), we reject it here and route
   straight back to the healer. Docker never spins up. This is the cheapest and
   most reliable line of defense: a container that can't reach the network still
   shouldn't be handed code that tries to.

2. **Loop defense** — an AST-normalized structural hash plus a normalized
   Levenshtein distance. Together they detect the agent oscillating between
   near-identical broken attempts, so the engine can force a strategy pivot
   instead of burning all five iterations on cosmetic edits.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import List, Set

# Modules that give code a path to the host, the network, other processes, or
# arbitrary deserialization. Algorithmic solutions never need these.
FORBIDDEN_MODULES: Set[str] = {
    "os", "sys", "subprocess", "socket", "shutil", "ctypes", "multiprocessing",
    "threading", "asyncio", "importlib", "pickle", "marshal", "shelve",
    "requests", "urllib", "urllib2", "http", "ftplib", "smtplib", "telnetlib",
    "pty", "signal", "resource", "mmap", "fcntl", "pathlib", "tempfile",
    "webbrowser", "builtins", "gc", "inspect", "code", "codeop",
}

# Bare names that are dangerous to *call*, regardless of import.
FORBIDDEN_CALLS: Set[str] = {
    "eval", "exec", "compile", "__import__", "open", "input", "globals",
    "locals", "vars", "getattr", "setattr", "delattr", "memoryview",
}

# Attribute suffixes used for sandbox escapes (e.g. ().__class__.__bases__...).
FORBIDDEN_ATTRS: Set[str] = {
    "__subclasses__", "__globals__", "__bases__", "__mro__", "__builtins__",
    "__code__", "__closure__", "__dict__", "__getattribute__", "__reduce__",
}


@dataclass
class SanitizationResult:
    ok: bool
    violations: List[str] = field(default_factory=list)


class _Sanitizer(ast.NodeVisitor):
    def __init__(self) -> None:
        self.violations: List[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.split(".")[0]
            if root in FORBIDDEN_MODULES:
                self._flag(node, f"forbidden import: {alias.name!r}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        root = (node.module or "").split(".")[0]
        if root in FORBIDDEN_MODULES:
            self._flag(node, f"forbidden import-from: {node.module!r}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Name) and func.id in FORBIDDEN_CALLS:
            self._flag(node, f"forbidden call: {func.id}(...)")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in FORBIDDEN_ATTRS:
            self._flag(node, f"forbidden attribute access: .{node.attr}")
        self.generic_visit(node)

    def _flag(self, node: ast.AST, msg: str) -> None:
        line = getattr(node, "lineno", "?")
        self.violations.append(f"line {line}: {msg}")


def sanitize(code: str) -> SanitizationResult:
    """Parse and statically vet untrusted solution code.

    Returns ``ok=False`` with human-readable violations for either a syntax
    error or any forbidden construct.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return SanitizationResult(
            ok=False,
            violations=[f"syntax error at line {exc.lineno}: {exc.msg}"],
        )
    sanitizer = _Sanitizer()
    sanitizer.visit(tree)
    return SanitizationResult(
        ok=not sanitizer.violations, violations=sanitizer.violations
    )


# --------------------------------------------------------------------------- #
# Loop / stagnation defense
# --------------------------------------------------------------------------- #
def structural_hash(code: str) -> str:
    """A hash of the code's *structure*, insensitive to formatting/comments.

    Uses ``ast.dump`` (without attributes) so that two attempts differing only
    in whitespace, comments, or variable-free reformatting collapse to the same
    fingerprint. Falls back to a raw hash if the code won't parse.
    """
    import hashlib

    try:
        tree = ast.parse(code)
        normalized = ast.dump(tree, annotate_fields=False, include_attributes=False)
    except SyntaxError:
        normalized = code.strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def levenshtein(a: str, b: str) -> int:
    """Classic edit distance, O(len(a) * len(b)) with an O(min) row buffer."""
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            insert = previous[j] + 1
            delete = current[j - 1] + 1
            substitute = previous[j - 1] + (ca != cb)
            current.append(min(insert, delete, substitute))
        previous = current
    return previous[-1]


def normalized_distance(a: str, b: str) -> float:
    """Levenshtein distance scaled to [0, 1] by the longer string length."""
    longest = max(len(a), len(b))
    if longest == 0:
        return 0.0
    return levenshtein(a, b) / longest


def is_stagnating(
    candidate: str, history: List[str], threshold: float
) -> bool:
    """Decide whether ``candidate`` is a near-duplicate of the last attempt.

    Two independent triggers:
      * structural-hash collision with *any* prior attempt (exact re-emission), or
      * a normalized edit distance below ``threshold`` vs. the immediately
        preceding attempt (cosmetic tweaking of a broken approach).
    """
    if not history:
        return False
    if structural_hash(candidate) in {structural_hash(h) for h in history}:
        return True
    return normalized_distance(candidate, history[-1]) < threshold
