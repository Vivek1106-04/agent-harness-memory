"""Config in, runnable loop out. The one place the harness turns names into objects.

Every component is resolved through the registry, so nothing here names an implementation
class and a swap is a config edit. The only interesting problem is that components need
things a YAML file cannot hold - a live HTTP client, an encoder, the sandbox path - and
solving it with `if config.memory == "vector": ...` would make the substitutability claim
false (AGENTS.md rule 4).

So: a fixed set of **services** is offered to every component, and a component takes what
it names in its constructor. `ReActPlanner` names `llm`, `VectorMemory` names `encoder`,
`LocalToolRuntime` names `sandbox_root`, and a provider that names none receives none. The
match is on the parameter name, never on the implementation, so a third-party component
installed through an entry point is wired by the same rule with no change here.

Config wins over services. A value written in the component's own block is passed through
untouched, which is what keeps `sandbox_root: ./.sandbox` in a YAML file meaningful.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harness.config import ComponentConfig, HarnessConfig
from harness.embedding import SentenceTransformerEncoder
from harness.interfaces import (
    ContextManager,
    Evaluator,
    LLMClient,
    MemoryProvider,
    Planner,
    ToolRuntime,
)
from harness.llm_http import HttpLLMClient
from harness.loop import AgentLoop
from harness.registry import resolve
from harness.trace import TraceBus

DEFAULT_SANDBOX_ROOT = Path(".sandbox")


class BuildError(Exception):
    """A config that validated but cannot be assembled. Raised before any model call."""


@dataclass(frozen=True)
class BuiltRun:
    """The loop, plus the two labels a results row needs and only the components know.

    `token_budget` is read from the assembled context manager rather than from the config
    block, because `FullContextManager` declares an unbounded budget that no YAML file
    states. `condition` is the memory type: in this study the memory layer is the treatment.
    """

    loop: AgentLoop
    token_budget: int
    condition: str


def build_llm(config: HarnessConfig, env: dict[str, str] | None = None) -> HttpLLMClient:
    """One client for both providers: they differ in a base URL and a key, not in code."""
    return HttpLLMClient.from_env(
        base_url=config.llm.base_url,
        model=config.llm.model,
        temperature=config.llm.temperature,
        api_key_env=config.llm.api_key_env,
        timeout_s=config.llm.timeout_s,
        max_attempts=config.llm.max_attempts,
        env=env,
    )


def build_component(layer: str, component: ComponentConfig, services: dict[str, Any]) -> Any:
    """Resolve `component.type` in `layer` and construct it from its config plus services."""
    cls = resolve(layer, component.type)
    kwargs: dict[str, Any] = component.model_dump(exclude={"type"})
    accepted = inspect.signature(cls).parameters
    for name, value in services.items():
        if name in accepted and name not in kwargs:
            kwargs[name] = value
    try:
        return cls(**kwargs)
    except TypeError as error:
        raise BuildError(f"{layer}/{component.type}: {error}") from error


def build_loop(
    config: HarnessConfig,
    bus: TraceBus,
    llm: LLMClient,
    sandbox_root: Path = DEFAULT_SANDBOX_ROOT,
) -> BuiltRun:
    """Assemble the five layers named by `config` into a runnable loop.

    `llm` is passed in rather than built here so that a caller can share one client across
    every cell of a sweep, and so that a test can assemble the whole loop without a network.
    """
    counter = _counter(llm)
    services: dict[str, Any] = {
        "llm": llm,
        "encoder": SentenceTransformerEncoder(config.embedding.model, config.embedding.revision),
        "sandbox_root": sandbox_root,
        "count_tokens": counter,
    }
    context = _typed(build_component("context", config.context, services), ContextManager)
    loop = AgentLoop(
        planner=_typed(build_component("planner", config.planner, services), Planner),
        context=context,
        memory=_typed(build_component("memory", config.memory, services), MemoryProvider),
        tools=_typed(build_component("tools", config.tools, services), ToolRuntime),
        evaluator=_typed(build_component("evaluator", config.evaluator, services), Evaluator),
        bus=bus,
        max_steps=config.run.max_steps,
        retrieval_k=config.run.retrieval_k,
        count_tokens=counter,
    )
    return BuiltRun(loop=loop, token_budget=context.token_budget, condition=config.memory.type)


def _counter(llm: LLMClient) -> Callable[[str], int]:
    """The budget and the loop count tokens the same way, whichever client is in use."""
    return llm.count_tokens


def _typed(component: Any, expected: type[Any]) -> Any:
    """A registered class that does not implement its layer is a config error, not a crash
    forty steps later.

    `expected` is one of the five layer ABCs, so it is typed loosely: a `type[T]` parameter
    would make mypy refuse an abstract class at every call site, and the check being made
    here is precisely that a concrete class satisfies an abstract one.
    """
    if not isinstance(component, expected):
        raise BuildError(
            f"{type(component).__name__} is registered under a layer it does not implement; "
            f"expected {expected.__name__}"
        )
    return component
