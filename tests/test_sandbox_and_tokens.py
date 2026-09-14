from __future__ import annotations

from pathlib import Path

import pytest

from harness.sandbox import SandboxViolationError, resolve_within
from harness.tokens import CHARS_PER_TOKEN, approx_token_count

ESCAPES = ["../outside.txt", "../../etc/passwd", "/etc/passwd", "a/../../outside.txt"]


def test_path_inside_the_root_resolves(tmp_path: Path) -> None:
    resolved = resolve_within(tmp_path, "notes/day1.txt")
    assert resolved == (tmp_path / "notes" / "day1.txt").resolve()


def test_the_root_itself_is_allowed(tmp_path: Path) -> None:
    assert resolve_within(tmp_path, ".") == tmp_path.resolve()


@pytest.mark.parametrize("path", ESCAPES)
def test_escaping_paths_are_refused(tmp_path: Path, path: str) -> None:
    with pytest.raises(SandboxViolationError):
        resolve_within(tmp_path, path)


def test_symlink_escape_is_refused(tmp_path: Path) -> None:
    """Both sides are fully resolved, so a symlink out of the sandbox is caught."""
    # Arrange
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "sandbox"
    root.mkdir()
    (root / "link").symlink_to(outside)

    # Act / Assert
    with pytest.raises(SandboxViolationError):
        resolve_within(root, "link/secret.txt")


def test_token_count_is_a_ceiling_division() -> None:
    assert approx_token_count("") == 0
    assert approx_token_count("a") == 1
    assert approx_token_count("a" * CHARS_PER_TOKEN) == 1
    assert approx_token_count("a" * (CHARS_PER_TOKEN + 1)) == 2
