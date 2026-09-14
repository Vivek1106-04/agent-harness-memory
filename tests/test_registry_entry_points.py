"""Entry-point discovery: how a third-party component registers without forking.

The mechanism is exercised with a fake entry point rather than by installing a package, so
the test runs offline and leaves no state in the environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

import harness.registry as registry


@dataclass
class _FakeEntryPoint:
    name: str
    target: type[Any]

    def load(self) -> type[Any]:
        return self.target


class _ExternalMemory:
    capabilities = frozenset({"semantic"})


@pytest.fixture
def rediscoverable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registry, "_DISCOVERED", False)


def test_a_third_party_component_becomes_selectable_by_name(
    rediscoverable: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    def fake_entry_points(group: str) -> list[_FakeEntryPoint]:
        if group == "harness.memory":
            return [_FakeEntryPoint(name="external_memory", target=_ExternalMemory)]
        return []

    monkeypatch.setattr(registry, "entry_points", fake_entry_points)

    # Act
    resolved = registry.resolve("memory", "external_memory")

    # Assert
    assert resolved is _ExternalMemory


def test_discovery_runs_once(rediscoverable: None, monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    calls: list[str] = []

    def counting_entry_points(group: str) -> list[_FakeEntryPoint]:
        calls.append(group)
        return []

    monkeypatch.setattr(registry, "entry_points", counting_entry_points)

    # Act
    registry.discover()
    registry.discover()

    # Assert
    assert calls == [f"harness.{layer}" for layer in registry.LAYERS]
