from __future__ import annotations

import pytest

from harness.registry import RegistryError, available, register, resolve


def test_built_in_components_resolve_by_name() -> None:
    assert resolve("planner", "single_step").__name__ == "SingleStepPlanner"
    assert resolve("context", "full").__name__ == "FullContextManager"
    assert resolve("tools", "mock").__name__ == "MockToolRuntime"
    assert resolve("evaluator", "always_fail").__name__ == "AlwaysFailEvaluator"


def test_available_lists_registered_names() -> None:
    assert "single_step" in available("planner")


def test_unknown_name_fails_loudly_with_the_alternatives() -> None:
    with pytest.raises(RegistryError, match="single_step"):
        resolve("planner", "raect")


def test_unknown_layer_is_an_error() -> None:
    with pytest.raises(RegistryError):
        resolve("scheduler", "anything")

    with pytest.raises(RegistryError):
        register("scheduler", name="anything")


def test_duplicate_registration_under_one_name_is_refused() -> None:
    # Arrange
    @register("memory", name="registry_test_dummy")
    class _First:
        pass

    # Act / Assert
    with pytest.raises(RegistryError, match="already registered"):

        @register("memory", name="registry_test_dummy")
        class _Second:
            pass


def test_re_registering_the_same_class_is_idempotent() -> None:
    @register("memory", name="registry_test_idempotent")
    class _Only:
        pass

    register("memory", name="registry_test_idempotent")(_Only)
