from __future__ import annotations

from pathlib import Path

from harness.config import load_config
from harness.provenance import UNKNOWN_COMMIT, build_provenance, git_commit
from harness.types import PAYLOAD_VERSION
from tests.test_config import ENV, VALID

COMMIT = "1b4ee04148faf693bdfe7c50b62a17162f0d2e76"


def _git_dir(tmp_path: Path, head: str, ref_value: str | None = None) -> Path:
    git = tmp_path / ".git"
    (git / "refs" / "heads").mkdir(parents=True)
    (git / "HEAD").write_text(head)
    if ref_value is not None:
        (git / "refs" / "heads" / "main").write_text(ref_value)
    return tmp_path


def test_commit_is_read_from_the_branch_ref(tmp_path: Path) -> None:
    root = _git_dir(tmp_path, "ref: refs/heads/main\n", COMMIT + "\n")
    assert git_commit(root) == COMMIT


def test_detached_head_is_read_directly(tmp_path: Path) -> None:
    root = _git_dir(tmp_path, COMMIT + "\n")
    assert git_commit(root) == COMMIT


def test_packed_ref_is_used_when_the_loose_ref_is_absent(tmp_path: Path) -> None:
    root = _git_dir(tmp_path, "ref: refs/heads/main\n")
    (root / ".git" / "packed-refs").write_text(f"# pack-refs\n{COMMIT} refs/heads/main\n")
    assert git_commit(root) == COMMIT


def test_outside_a_checkout_the_commit_is_unknown(tmp_path: Path) -> None:
    assert git_commit(tmp_path) == UNKNOWN_COMMIT


def test_a_ref_with_no_loose_and_no_packed_file_is_unknown(tmp_path: Path) -> None:
    root = _git_dir(tmp_path, "ref: refs/heads/main\n")
    assert git_commit(root) == UNKNOWN_COMMIT


def test_a_ref_absent_from_packed_refs_is_unknown(tmp_path: Path) -> None:
    root = _git_dir(tmp_path, "ref: refs/heads/main\n")
    (root / ".git" / "packed-refs").write_text(f"{COMMIT} refs/heads/other\n")
    assert git_commit(root) == UNKNOWN_COMMIT


def test_provenance_carries_every_pinned_identity(tmp_path: Path) -> None:
    # Arrange
    path = tmp_path / "config.yaml"
    path.write_text(VALID)
    config = load_config(path, env=ENV)

    # Act
    provenance = build_provenance("run-1", config, repo_root=tmp_path)

    # Assert
    assert provenance.payload_version == PAYLOAD_VERSION
    assert provenance.model_id == "qwen3-8b"
    assert provenance.model_revision == "def456"
    assert provenance.embedding_revision == "abc123"
    assert provenance.seeds == {"run": 1}
    assert len(provenance.config_hash) == 64
