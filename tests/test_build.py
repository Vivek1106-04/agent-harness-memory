"""The builder. Config in, five layers out, with no implementation named in harness source."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from harness.build import BuildError, build_component, build_llm, build_loop
from harness.components.memory_episodic import EpisodicMemory
from harness.components.memory_vector import VectorMemory
from harness.components.planner_react import ReActPlanner
from harness.components.tools_local import LocalToolRuntime
from harness.config import ComponentConfig, HarnessConfig, load_config
from harness.gcp_auth import GoogleAdcAuth
from harness.interfaces import MemoryProvider
from harness.llm_http import LLMTransportError
from harness.registry import register
from harness.trace import MemoryTraceSink, TraceBus
from harness.types import MemoryItem, MemoryQuery
from tests.fakes import FakeLLMClient

CONFIG_TEMPLATE: dict[str, Any] = {
    "run": {"seed": 1, "max_steps": 4, "run_id_prefix": "test", "retrieval_k": 3},
    "suite": {
        "count": 2,
        "chain_length": 2,
        "recall_distance": 12,
        "distractor_density": 0.5,
        "horizon": 20,
        "seed": 1000,
    },
    "planner": {"type": "react", "max_steps": 4},
    "memory": {"type": "noop"},
    "context": {"type": "budgeted", "token_budget": 4000},
    "tools": {"type": "task_docs", "max_output_chars": 500},
    "evaluator": {"type": "checker"},
    "embedding": {"model": "stub/model", "revision": "abc123"},
    "llm": {
        "provider": "local",
        "base_url": "https://endpoint.invalid/v1",
        "model": "test-model",
        "revision": "r1",
        "max_output_tokens": 64,
        "temperature": 0.0,
    },
    "limits": {"max_tokens_per_run": 100000, "max_spend_usd": 1.0},
}


@register("memory", name="_needs_nothing")
class _BareMemory(MemoryProvider):
    """A provider that names no service. The builder must offer it none."""

    def write(self, item: MemoryItem) -> None: ...

    def retrieve(self, query: MemoryQuery) -> list[MemoryItem]:
        return []

    def reset(self) -> None: ...


@register("evaluator", name="_not_an_evaluator")
class _WrongLayer:
    """Registered under the evaluator layer without implementing it."""


def write_config(tmp_path: Path, **sections: dict[str, Any]) -> Path:
    """Overlay `sections` onto the template. A None value removes that key."""
    data = {key: dict(value) for key, value in CONFIG_TEMPLATE.items()}
    for key, overrides in sections.items():
        merged = {**data[key], **overrides}
        data[key] = {name: value for name, value in merged.items() if value is not None}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def config(tmp_path: Path, **sections: dict[str, Any]) -> HarnessConfig:
    return load_config(write_config(tmp_path, **sections))


def test_a_component_receives_only_the_services_it_names(tmp_path: Path) -> None:
    """The wiring matches on parameter name, never on the implementation."""
    # Arrange
    services = {"llm": FakeLLMClient(["x"]), "sandbox_root": tmp_path}

    # Act
    component = build_component("memory", ComponentConfig(type="_needs_nothing"), services)

    # Assert
    assert isinstance(component, _BareMemory)


def test_a_component_receives_the_service_it_does_name(tmp_path: Path) -> None:
    # Arrange
    services: dict[str, Any] = {"sandbox_root": tmp_path}

    # Act
    component = build_component("tools", ComponentConfig(type="local"), services)

    # Assert
    assert isinstance(component, LocalToolRuntime)


def test_config_values_win_over_services() -> None:
    """`sandbox_root: ./somewhere` in a YAML file has to mean something."""
    # Arrange
    block = ComponentConfig(type="local", sandbox_root="from-config")
    services: dict[str, Any] = {"sandbox_root": Path("from-service")}

    # Act
    component = build_component("tools", block, services)

    # Assert
    assert isinstance(component, LocalToolRuntime)
    assert component._sandbox_root == Path("from-config")


def test_an_unexpected_component_argument_is_a_build_error() -> None:
    with pytest.raises(BuildError, match="memory/noop"):
        build_component("memory", ComponentConfig(type="noop", nonsense=1), {})


def test_a_class_registered_under_the_wrong_layer_is_refused(tmp_path: Path) -> None:
    """Better here than forty steps into a run that cost money."""
    # Arrange
    subject = config(tmp_path, evaluator={"type": "_not_an_evaluator"})

    # Act / Assert
    with pytest.raises(BuildError, match="Evaluator"):
        build_loop(subject, TraceBus("r", MemoryTraceSink()), FakeLLMClient(["x"]))


def test_build_loop_assembles_the_components_the_config_names(tmp_path: Path) -> None:
    # Arrange
    subject = config(tmp_path, memory={"type": "episodic"})

    # Act
    built = build_loop(subject, TraceBus("r", MemoryTraceSink()), FakeLLMClient(["x"]))

    # Assert
    assert isinstance(built.loop._memory, EpisodicMemory)
    assert isinstance(built.loop._planner, ReActPlanner)


def test_vector_memory_is_handed_an_encoder_built_from_the_embedding_block(
    tmp_path: Path,
) -> None:
    """One embedding model per run, and it is the one in the config hash."""
    # Arrange
    subject = config(tmp_path, memory={"type": "vector"})

    # Act
    built = build_loop(subject, TraceBus("r", MemoryTraceSink()), FakeLLMClient(["x"]))

    # Assert
    memory = built.loop._memory
    assert isinstance(memory, VectorMemory)
    assert memory._encoder._model_name == "stub/model"  # type: ignore[attr-defined]
    assert memory._encoder._revision == "abc123"  # type: ignore[attr-defined]


def test_the_condition_label_is_the_memory_type(tmp_path: Path) -> None:
    built = build_loop(
        config(tmp_path, memory={"type": "oracle"}),
        TraceBus("r", MemoryTraceSink()),
        FakeLLMClient(["x"]),
    )

    assert built.condition == "oracle"


def test_the_token_budget_is_read_from_the_assembled_context_manager(tmp_path: Path) -> None:
    """`FullContextManager` declares an unbounded budget that no YAML file states."""
    # Arrange
    subject = config(tmp_path, context={"type": "full", "token_budget": None})

    # Act
    built = build_loop(subject, TraceBus("r", MemoryTraceSink()), FakeLLMClient(["x"]))

    # Assert
    assert built.token_budget > 4000


def test_the_retrieval_budget_comes_from_the_run_block(tmp_path: Path) -> None:
    built = build_loop(config(tmp_path), TraceBus("r", MemoryTraceSink()), FakeLLMClient(["x"]))

    assert built.loop._retrieval_k == 3


def test_the_loop_counts_tokens_the_way_the_client_does(tmp_path: Path) -> None:
    """The budget and the write-path cost must not be measured on two different rulers."""
    # Arrange
    llm = FakeLLMClient(["x"])

    # Act
    built = build_loop(config(tmp_path), TraceBus("r", MemoryTraceSink()), llm)

    # Assert
    assert built.loop._count_tokens == llm.count_tokens


def test_build_llm_refuses_when_the_named_key_is_unset(tmp_path: Path) -> None:
    # Arrange
    subject = config(tmp_path, llm={"api_key_env": "DEFINITELY_UNSET_HARNESS_KEY"})

    # Act / Assert
    with pytest.raises(LLMTransportError, match="DEFINITELY_UNSET_HARNESS_KEY"):
        build_llm(subject, env={})


def test_build_llm_with_google_login_authenticates_every_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    sentinel = GoogleAdcAuth(credentials=object())
    monkeypatch.setattr(GoogleAdcAuth, "from_environment", classmethod(lambda cls: sentinel))
    subject = config(tmp_path, llm={"auth": "google_adc"})

    # Act
    client = build_llm(subject, env={})

    # Assert
    assert client._auth is sentinel


def test_build_llm_needs_no_key_for_a_local_endpoint(tmp_path: Path) -> None:
    client = build_llm(config(tmp_path), env={})

    assert client.count_tokens("abcd") == 1


SHIPPED_CONFIGS = sorted(Path(__file__).resolve().parents[1].joinpath("configs").glob("*.yaml"))
SHIPPED_ENV = {"HARNESS_LLM_BASE_URL": "http://localhost:8000/v1"}


@pytest.mark.parametrize("path", SHIPPED_CONFIGS, ids=lambda path: path.name)
def test_every_shipped_config_assembles(path: Path, tmp_path: Path) -> None:
    """Validating is not enough: a config can be well-formed and still name a component
    argument no implementation accepts. That failure belongs here, not four hours into a
    sweep."""
    # Arrange
    subject = load_config(path, env=SHIPPED_ENV)

    # Act
    built = build_loop(
        subject, TraceBus("r", MemoryTraceSink()), FakeLLMClient(["{}"]), sandbox_root=tmp_path
    )

    # Assert
    assert built.condition == subject.memory.type
