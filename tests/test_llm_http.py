"""The model endpoint, tested without one. Every request is served by a MockTransport."""

from __future__ import annotations

import json

import httpx
import pytest

from harness.llm_http import (
    HttpLLMClient,
    LLMTransportError,
    to_messages,
)
from harness.prompts import SYSTEM_INSTRUCTIONS
from harness.types import ContextWindow

BASE_URL = "https://endpoint.invalid/v1"
MODEL = "test-model"
NO_BACKOFF = 0.0


def _window(segments: list[tuple[str, str]] | None = None) -> ContextWindow:
    return ContextWindow(
        segments=segments
        if segments is not None
        else [("system", SYSTEM_INSTRUCTIONS), ("goal", "find the code")],
        token_count=12,
        dropped=[],
    )


def _completion(text: str) -> dict[str, object]:
    return {"choices": [{"message": {"role": "assistant", "content": text}}]}


def _client(
    handler: object,
    api_key: str | None = None,
    max_attempts: int = 1,
) -> HttpLLMClient:
    transport = httpx.MockTransport(handler)  # type: ignore[arg-type]
    return HttpLLMClient(
        base_url=BASE_URL,
        model=MODEL,
        api_key=api_key,
        max_attempts=max_attempts,
        backoff_s=NO_BACKOFF,
        client=httpx.Client(transport=transport),
    )


def test_returns_the_first_choice_text() -> None:
    # Arrange
    client = _client(lambda request: httpx.Response(200, json=_completion("Action: {}")))

    # Act
    text = client.complete(_window(), max_output_tokens=50)

    # Assert
    assert text == "Action: {}"


def test_posts_to_the_chat_completions_path() -> None:
    # Arrange
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_completion("ok"))

    # Act
    _client(handler).complete(_window(), max_output_tokens=50)

    # Assert
    assert str(seen[0].url) == f"{BASE_URL}/chat/completions"


def test_a_trailing_slash_on_the_base_url_does_not_double_up() -> None:
    # Arrange
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_completion("ok"))

    client = HttpLLMClient(
        base_url=f"{BASE_URL}/",
        model=MODEL,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    # Act
    client.complete(_window(), max_output_tokens=50)

    # Assert
    assert str(seen[0].url) == f"{BASE_URL}/chat/completions"


def test_the_request_carries_the_model_and_the_output_cap() -> None:
    # Arrange
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_completion("ok"))

    # Act
    _client(handler).complete(_window(), max_output_tokens=123)

    # Assert
    body = json.loads(seen[0].content)
    assert body["model"] == MODEL
    assert body["max_tokens"] == 123
    assert body["temperature"] == 0.0


def test_a_key_becomes_a_bearer_header() -> None:
    # Arrange
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_completion("ok"))

    # Act
    _client(handler, api_key="sk-not-a-real-key").complete(_window(), max_output_tokens=10)

    # Assert
    assert seen[0].headers["authorization"] == "Bearer sk-not-a-real-key"


def test_no_key_sends_no_authorization_header() -> None:
    """A local vLLM server needs none, and sending an empty one is a confusing 401."""
    # Arrange
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_completion("ok"))

    # Act
    _client(handler).complete(_window(), max_output_tokens=10)

    # Assert
    assert "authorization" not in seen[0].headers


def test_from_env_reads_the_key_by_name() -> None:
    # Arrange
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_completion("ok"))

    client = HttpLLMClient.from_env(
        base_url=BASE_URL,
        model=MODEL,
        api_key_env="TEST_HARNESS_KEY",
        env={"TEST_HARNESS_KEY": "sk-from-env"},
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    # Act
    client.complete(_window(), max_output_tokens=10)

    # Assert
    assert seen[0].headers["authorization"] == "Bearer sk-from-env"


def test_from_env_refuses_to_start_when_the_named_key_is_unset() -> None:
    """A sweep that discovers this on its first call has already booked a slot."""
    with pytest.raises(LLMTransportError, match="TEST_HARNESS_KEY"):
        HttpLLMClient.from_env(
            base_url=BASE_URL, model=MODEL, api_key_env="TEST_HARNESS_KEY", env={}
        )


def test_from_env_without_a_named_variable_needs_no_key() -> None:
    client = HttpLLMClient.from_env(base_url=BASE_URL, model=MODEL, env={})

    assert isinstance(client, HttpLLMClient)


def test_the_key_never_appears_in_a_transport_error() -> None:
    # Arrange
    client = _client(lambda request: httpx.Response(400), api_key="sk-not-a-real-key")

    # Act / Assert
    with pytest.raises(LLMTransportError) as error:
        client.complete(_window(), max_output_tokens=10)
    assert "sk-not-a-real-key" not in str(error.value)


@pytest.mark.parametrize("status", [429, 500, 503])
def test_a_transient_status_is_retried(status: int) -> None:
    # Arrange
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(status)
        if len(attempts) < 3:
            return httpx.Response(status)
        return httpx.Response(200, json=_completion("recovered"))

    # Act
    text = _client(handler, max_attempts=3).complete(_window(), max_output_tokens=10)

    # Assert
    assert text == "recovered"
    assert len(attempts) == 3


def test_a_client_error_is_not_retried() -> None:
    """A 400 is our bug. Retrying it spends the budget four times to learn nothing."""
    # Arrange
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(400)
        return httpx.Response(400)

    # Act / Assert
    with pytest.raises(LLMTransportError):
        _client(handler, max_attempts=4).complete(_window(), max_output_tokens=10)
    assert len(attempts) == 1


def test_a_transport_failure_is_retried_then_reraised() -> None:
    # Arrange
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        raise httpx.ConnectError("no route to host")

    # Act / Assert
    with pytest.raises(httpx.ConnectError):
        _client(handler, max_attempts=3).complete(_window(), max_output_tokens=10)
    assert len(attempts) == 3


def test_persistent_rate_limiting_surfaces_as_a_transport_error() -> None:
    # Arrange
    client = _client(lambda request: httpx.Response(429), max_attempts=2)

    # Act / Assert
    with pytest.raises(LLMTransportError, match="429"):
        client.complete(_window(), max_output_tokens=10)


def test_a_non_object_body_is_a_transport_error() -> None:
    client = _client(lambda request: httpx.Response(200, json=["not", "an", "object"]))

    with pytest.raises(LLMTransportError, match="not a JSON object"):
        client.complete(_window(), max_output_tokens=10)


def test_max_attempts_below_one_is_refused() -> None:
    with pytest.raises(ValueError, match="max_attempts"):
        HttpLLMClient(base_url=BASE_URL, model=MODEL, max_attempts=0)


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"choices": []},
        {"choices": "not a list"},
        {"choices": ["not an object"]},
        {"choices": [{"message": "not an object"}]},
        {"choices": [{"message": {"content": None}}]},
    ],
)
def test_a_missing_completion_is_an_empty_string_not_an_exception(body: dict[str, object]) -> None:
    """The planner already treats an unusable reply as an expected condition."""
    # Arrange
    client = _client(lambda request: httpx.Response(200, json=body))

    # Act
    text = client.complete(_window(), max_output_tokens=10)

    # Assert
    assert text == ""


def test_token_counting_is_the_arithmetic_approximation() -> None:
    """Identical on every machine and for every provider, which is what the budget needs."""
    client = _client(lambda request: httpx.Response(200, json=_completion("ok")))

    assert client.count_tokens("abcd" * 10) == 10


def test_closing_the_client_closes_the_transport() -> None:
    # Arrange
    inner = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})))
    client = HttpLLMClient(base_url=BASE_URL, model=MODEL, client=inner)

    # Act
    with client:
        pass

    # Assert
    assert inner.is_closed


def test_system_segments_become_the_system_message() -> None:
    # Arrange
    window = _window(
        [("system", "instructions"), ("goal", "the goal"), ("observation", "step 1 output")]
    )

    # Act
    messages = to_messages(window)

    # Assert
    assert messages[0] == {"role": "system", "content": "instructions"}
    assert messages[1]["content"] == "the goal\n\nstep 1 output"


def test_a_window_with_no_system_segment_still_carries_the_instructions() -> None:
    """The structural defence against retrieved content displacing the instructions."""
    messages = to_messages(_window([("goal", "the goal")]))

    assert messages[0]["content"] == SYSTEM_INSTRUCTIONS


def test_several_system_segments_are_joined_in_order() -> None:
    window = _window([("system", "first"), ("system", "second"), ("goal", "g")])

    assert to_messages(window)[0]["content"] == "first\n\nsecond"
