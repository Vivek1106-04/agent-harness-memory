"""Registration without forking.

Components declare themselves; the registry discovers them. A third-party package installs
alongside the harness and its provider becomes selectable by name in config. Nothing in the
harness names an implementation class.

If a swap requires touching an import, a factory function, or a match statement inside
harness source, the component is not really substitutable and the premise is rhetoric.

SECURITY: entry-point discovery imports third-party code with your privileges. The
conformance suite is a specification, not a sandbox - passing it proves a component honours
the interface, never that it is safe. See SECURITY.md section 7.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from importlib.metadata import entry_points
from typing import Any, TypeVar

LAYERS = ("memory", "planner", "context", "tools", "evaluator")

_REGISTRY: dict[str, dict[str, type[Any]]] = {layer: {} for layer in LAYERS}
_DISCOVERED = False

T = TypeVar("T")


class RegistryError(Exception):
    pass


def register(layer: str, *, name: str) -> Callable[[type[T]], type[T]]:
    """Class decorator registering an implementation under `layer` as `name`."""
    if layer not in _REGISTRY:
        raise RegistryError(f"unknown layer {layer!r}; expected one of {LAYERS}")

    def decorator(cls: type[T]) -> type[T]:
        existing = _REGISTRY[layer].get(name)
        if existing is not None and existing is not cls:
            owner = f"{existing.__module__}.{existing.__qualname__}"
            raise RegistryError(f"{layer}/{name} already registered by {owner}")
        _REGISTRY[layer][name] = cls
        return cls

    return decorator


def discover() -> None:
    """Load third-party components declared through entry points.

    A package exposes one like this in its own pyproject.toml:

        [project.entry-points."harness.memory"]
        my_memory = "my_package.memory:MyMemory"
    """
    global _DISCOVERED
    if _DISCOVERED:
        return
    importlib.import_module("harness.components")  # registers the built-ins
    for layer in LAYERS:
        for ep in entry_points(group=f"harness.{layer}"):
            cls = ep.load()  # executes third-party code - see SECURITY.md section 7
            register(layer, name=ep.name)(cls)
    _DISCOVERED = True


def resolve(layer: str, name: str) -> type[Any]:
    """Look up a registered implementation.

    An unknown name is an error, never a silent fallback to a default: a typo'd component
    in a config must fail loudly rather than quietly run the wrong experiment.
    """
    discover()
    if layer not in _REGISTRY:
        raise RegistryError(f"unknown layer {layer!r}; expected one of {LAYERS}")
    try:
        return _REGISTRY[layer][name]
    except KeyError:
        available = sorted(_REGISTRY[layer]) or ["<none registered>"]
        raise RegistryError(
            f"no {layer} named {name!r}. Available: {', '.join(available)}"
        ) from None


def available(layer: str) -> list[str]:
    discover()
    return sorted(_REGISTRY.get(layer, {}))
