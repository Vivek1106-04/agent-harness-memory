"""Path confinement for the tool runtime. See SECURITY.md section 5.

Every filesystem path that originates in model output goes through `resolve_within`. It is
shared by `LocalToolRuntime` and `MockToolRuntime` so that both refuse exactly the same
paths - a mock that is more permissive than the real runtime hides the bug it exists to
catch.
"""

from __future__ import annotations

from pathlib import Path


class SandboxViolationError(Exception):
    """A path escaped the sandbox root. Callers turn this into an error Observation."""


def resolve_within(root: Path, user_path: str) -> Path:
    """Resolve `user_path` under `root`, refusing anything that escapes it.

    Catches `../` traversal, absolute paths and symlink escapes, because both sides are
    fully resolved before the descendant check.
    """
    resolved_root = root.resolve()
    candidate = (resolved_root / user_path).resolve()
    if candidate != resolved_root and resolved_root not in candidate.parents:
        raise SandboxViolationError(f"path {user_path!r} resolves outside the sandbox root")
    return candidate
