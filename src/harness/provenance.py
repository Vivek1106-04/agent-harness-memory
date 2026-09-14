"""Assembles the provenance header stamped into every run.

Read from the repository files rather than by shelling out to git: no subprocess, no shell,
and it works in a container where git is absent.
"""

from __future__ import annotations

from pathlib import Path

from harness.config import HarnessConfig, config_hash
from harness.types import PAYLOAD_VERSION, RunProvenance

UNKNOWN_COMMIT = "unknown"


def git_commit(repo_root: str | Path = ".") -> str:
    """Current commit id, or `UNKNOWN_COMMIT` outside a checkout."""
    head = Path(repo_root, ".git", "HEAD")
    if not head.is_file():
        return UNKNOWN_COMMIT
    content = head.read_text().strip()
    if not content.startswith("ref:"):
        return content
    ref = content.removeprefix("ref:").strip()
    ref_file = Path(repo_root, ".git", ref)
    if ref_file.is_file():
        return ref_file.read_text().strip()
    return _from_packed_refs(Path(repo_root, ".git", "packed-refs"), ref)


def build_provenance(
    run_id: str,
    config: HarnessConfig,
    repo_root: str | Path = ".",
) -> RunProvenance:
    return RunProvenance(
        run_id=run_id,
        config_hash=config_hash(config),
        git_commit=git_commit(repo_root),
        payload_version=PAYLOAD_VERSION,
        model_id=config.llm.model,
        model_revision=config.llm.revision,
        embedding_model=config.embedding.model,
        embedding_revision=config.embedding.revision,
        seeds={"run": config.run.seed},
    )


def _from_packed_refs(packed: Path, ref: str) -> str:
    if not packed.is_file():
        return UNKNOWN_COMMIT
    for line in packed.read_text().splitlines():
        if line.endswith(f" {ref}"):
            return line.split(" ", 1)[0]
    return UNKNOWN_COMMIT
