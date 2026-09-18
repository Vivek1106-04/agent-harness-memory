from __future__ import annotations

from pathlib import Path

import pytest

from harness.config import ConfigError, config_hash, load_config

VALID = """
run: {seed: 1, max_steps: 35, run_id_prefix: exp1}
planner: {type: single_step, max_steps: 35}
memory: {type: vector, k: 5}
context: {type: budgeted, token_budget: 8000}
tools: {type: mock, sandbox_root: ./.sandbox}
evaluator: {type: always_fail}
embedding: {model: Qwen/Qwen3-Embedding-0.6B, revision: abc123}
llm:
  provider: local
  base_url: ${HARNESS_LLM_BASE_URL}
  model: qwen3-8b
  revision: def456
  max_output_tokens: 300
  temperature: 0.0
limits: {max_tokens_per_run: 500000, max_spend_usd: 25.0}
"""

ENV = {"HARNESS_LLM_BASE_URL": "http://localhost:8000/v1"}


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text)
    return path


def test_valid_config_loads_and_expands_environment_references(tmp_path: Path) -> None:
    # Arrange
    path = _write(tmp_path, VALID)

    # Act
    config = load_config(path, env=ENV)

    # Assert
    assert config.llm.base_url == "http://localhost:8000/v1"
    assert config.memory.type == "vector"


def test_component_specific_keys_are_preserved(tmp_path: Path) -> None:
    # Arrange
    path = _write(tmp_path, VALID)

    # Act
    config = load_config(path, env=ENV)

    # Assert
    assert config.memory.model_extra == {"k": 5}


def test_unknown_top_level_key_is_an_error(tmp_path: Path) -> None:
    # Arrange
    path = _write(tmp_path, VALID + "\nscheduler: {type: fifo}\n")

    # Act / Assert
    with pytest.raises(ConfigError, match="scheduler"):
        load_config(path, env=ENV)


def test_unknown_key_inside_a_fixed_block_is_an_error(tmp_path: Path) -> None:
    # Arrange
    path = _write(
        tmp_path,
        VALID.replace("limits: {max_tokens_per_run", "limits: {typo_key: 1, max_tokens_per_run"),
    )

    # Act / Assert
    with pytest.raises(ConfigError, match="typo_key"):
        load_config(path, env=ENV)


def test_missing_environment_variable_fails_loudly(tmp_path: Path) -> None:
    # Arrange
    path = _write(tmp_path, VALID)

    # Act / Assert
    with pytest.raises(ConfigError, match="HARNESS_LLM_BASE_URL"):
        load_config(path, env={})


def test_a_credential_in_config_is_refused(tmp_path: Path) -> None:
    """Keys come from environment variables only. SECURITY.md section 1."""
    # Arrange
    leaked = VALID.replace("revision: def456", "revision: sk-abcdefghijklmnopqrstuvwxyz0123")
    path = _write(tmp_path, leaked)

    # Act / Assert
    with pytest.raises(ConfigError, match="credential"):
        load_config(path, env=ENV)


def test_non_positive_max_steps_is_refused(tmp_path: Path) -> None:
    # Arrange
    path = _write(
        tmp_path, VALID.replace("max_steps: 35, run_id_prefix", "max_steps: 0, run_id_prefix")
    )

    # Act / Assert
    with pytest.raises(ConfigError):
        load_config(path, env=ENV)


def test_unknown_llm_provider_is_refused(tmp_path: Path) -> None:
    # Arrange
    path = _write(tmp_path, VALID.replace("provider: local", "provider: telepathy"))

    # Act / Assert
    with pytest.raises(ConfigError):
        load_config(path, env=ENV)


def test_top_level_must_be_a_mapping(tmp_path: Path) -> None:
    # Arrange
    path = _write(tmp_path, "- just\n- a list\n")

    # Act / Assert
    with pytest.raises(ConfigError, match="mapping"):
        load_config(path, env=ENV)


def test_environment_references_inside_a_list_are_expanded(tmp_path: Path) -> None:
    # Arrange
    path = _write(
        tmp_path,
        VALID.replace(
            "memory: {type: vector, k: 5}",
            "memory: {type: vector, paths: ['${HARNESS_LLM_BASE_URL}']}",
        ),
    )

    # Act
    config = load_config(path, env=ENV)

    # Assert
    assert config.memory.model_extra == {"paths": ["http://localhost:8000/v1"]}


def test_a_credential_hidden_in_a_list_is_refused(tmp_path: Path) -> None:
    # Arrange
    path = _write(
        tmp_path,
        VALID.replace(
            "memory: {type: vector, k: 5}",
            "memory: {type: vector, keys: ['sk-abcdefghijklmnopqrstuvwxyz0123']}",
        ),
    )

    # Act / Assert
    with pytest.raises(ConfigError, match="credential"):
        load_config(path, env=ENV)


def test_hash_is_stable_across_loads(tmp_path: Path) -> None:
    # Arrange
    path = _write(tmp_path, VALID)

    # Act
    first = config_hash(load_config(path, env=ENV))
    second = config_hash(load_config(path, env=ENV))

    # Assert
    assert first == second


def test_hash_ignores_the_endpoint_but_not_the_model(tmp_path: Path) -> None:
    """The same experiment on two machines must carry the same identity."""
    # Arrange
    path = _write(tmp_path, VALID)
    baseline = config_hash(load_config(path, env=ENV))

    # Act
    other_machine = config_hash(
        load_config(path, env={"HARNESS_LLM_BASE_URL": "http://10.0.0.4:8000/v1"})
    )
    other_model = config_hash(
        load_config(_write(tmp_path, VALID.replace("model: qwen3-8b", "model: qwen3-4b")), env=ENV)
    )

    # Assert
    assert other_machine == baseline
    assert other_model != baseline


def test_hash_changes_when_a_component_changes(tmp_path: Path) -> None:
    # Arrange
    path = _write(tmp_path, VALID)
    baseline = config_hash(load_config(path, env=ENV))

    # Act
    swapped = config_hash(
        load_config(_write(tmp_path, VALID.replace("type: vector", "type: graph")), env=ENV)
    )

    # Assert
    assert swapped != baseline


SHIPPED_CONFIGS = sorted(Path(__file__).resolve().parents[1].joinpath("configs").glob("*.yaml"))


@pytest.mark.parametrize("path", SHIPPED_CONFIGS, ids=lambda path: path.name)
def test_every_shipped_config_validates(path: Path) -> None:
    """A config in the repo is a declared experiment cell. A typo in one is a cell that
    dies hours into a sweep rather than at load."""
    assert config_hash(load_config(path, env=ENV))
