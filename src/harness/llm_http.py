"""The model endpoint. One client, both paths.

`HttpLLMClient` speaks the OpenAI chat-completions shape, which is what a local vLLM
server and every hosted endpoint we would consider both expose. The two paths in
`LLMConfig.provider` therefore differ only in a base URL and whether a key is sent, not in
code - a second adapter would be two things to keep in step for no measured difference.

Three properties this file exists to hold:

* **The key never enters the config, the trace or an exception.** It is read from the
  environment by the name `llm.api_key_env` gives, and it is checked at construction so a
  missing key fails before a sweep starts rather than four hours in. SECURITY.md section 1.
* **Transient failure is retried, and only transient failure.** A timeout, a dropped
  connection, a 429 or a 5xx are the API having a bad minute. A 400 is our bug and retrying
  it just spends the budget four times.
* **A refusal is not an exception.** An empty or missing completion is returned as an empty
  string, because the planner already treats malformed model output as an expected
  condition. Only transport failure raises, and the runner records that cell as failed and
  leaves it for a rerun.
"""

from __future__ import annotations

import os
from types import TracebackType

import httpx
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from harness.interfaces import LLMClient
from harness.prompts import SYSTEM_INSTRUCTIONS
from harness.segments import ROLE_SYSTEM
from harness.tokens import approx_token_count
from harness.types import ContextWindow

DEFAULT_TIMEOUT_S = 120.0
DEFAULT_MAX_ATTEMPTS = 4
DEFAULT_BACKOFF_S = 1.0
MAX_BACKOFF_S = 30.0

#: Status codes worth a second attempt: rate limiting and server-side faults.
RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

ROLE_ASSISTANT_INPUT = "user"
COMPLETIONS_PATH = "/chat/completions"


class LLMTransportError(Exception):
    """The endpoint could not be reached, or refused the request. Never carries the key."""


class _RetryableStatusError(LLMTransportError):
    """An HTTP status that is worth another attempt. Internal to the retry policy."""


class HttpLLMClient(LLMClient):
    def __init__(
        self,
        base_url: str,
        model: str,
        temperature: float = 0.0,
        api_key: str | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        backoff_s: float = DEFAULT_BACKOFF_S,
        client: httpx.Client | None = None,
    ) -> None:
        if max_attempts < 1:
            raise ValueError(f"max_attempts must be at least 1, got {max_attempts}")
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._temperature = temperature
        self._max_attempts = max_attempts
        self._backoff_s = backoff_s
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = client or httpx.Client(timeout=timeout_s)
        self._headers = headers

    @classmethod
    def from_env(
        cls,
        base_url: str,
        model: str,
        temperature: float = 0.0,
        api_key_env: str | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        env: dict[str, str] | None = None,
        client: httpx.Client | None = None,
    ) -> HttpLLMClient:
        """Build a client, reading the key from the environment variable `api_key_env`.

        A local server needs no key, so `api_key_env` may be None. When it is named and
        unset the constructor refuses: a sweep that discovers this on its first call has
        already paid for a scheduler slot.
        """
        source = os.environ if env is None else env
        api_key = None
        if api_key_env is not None:
            api_key = source.get(api_key_env)
            if not api_key:
                raise LLMTransportError(
                    f"environment variable {api_key_env} is not set. Keys come from the "
                    "environment only - see SECURITY.md section 1."
                )
        return cls(
            base_url=base_url,
            model=model,
            temperature=temperature,
            api_key=api_key,
            timeout_s=timeout_s,
            max_attempts=max_attempts,
            client=client,
        )

    def complete(self, window: ContextWindow, max_output_tokens: int) -> str:
        payload = {
            "model": self._model,
            "messages": to_messages(window),
            "max_tokens": max_output_tokens,
            "temperature": self._temperature,
        }
        response = self._post_with_retry(payload)
        return _first_choice_text(response)

    def count_tokens(self, text: str) -> int:
        """The arithmetic approximation, deliberately, not the provider's tokenizer.

        The budget has to bind identically on every machine and for every provider, or the
        8k cell means one thing on Gemini and another on a local model and Experiment 2
        compares two different budgets. See DECISIONS.md.
        """
        return approx_token_count(text)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> HttpLLMClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def _post_with_retry(self, payload: dict[str, object]) -> dict[str, object]:
        retrying = Retrying(
            stop=stop_after_attempt(self._max_attempts),
            wait=wait_exponential(multiplier=self._backoff_s, max=MAX_BACKOFF_S),
            retry=retry_if_exception_type((_RetryableStatusError, httpx.TransportError)),
            reraise=True,
        )
        for attempt in retrying:
            with attempt:
                return self._post(payload)
        raise LLMTransportError("retry policy exhausted without a result")  # pragma: no cover

    def _post(self, payload: dict[str, object]) -> dict[str, object]:
        response = self._client.post(
            f"{self._base_url}{COMPLETIONS_PATH}", json=payload, headers=self._headers
        )
        if response.status_code in RETRYABLE_STATUS:
            raise _RetryableStatusError(f"{self._model}: endpoint returned {response.status_code}")
        if response.status_code >= 400:
            raise LLMTransportError(f"{self._model}: endpoint returned {response.status_code}")
        body = response.json()
        if not isinstance(body, dict):
            raise LLMTransportError(f"{self._model}: response body was not a JSON object")
        return body


def to_messages(window: ContextWindow) -> list[dict[str, str]]:
    """Flatten the window into the two-message shape every endpoint accepts.

    System segments become the system message; goal, memory and observation segments become
    one user message in assembly order. Collapsing to two messages rather than mapping each
    role onto a chat turn keeps the prompt identical across providers, which is what makes a
    cross-model confirmation run a comparison rather than a coincidence.
    """
    system = [text for role, text in window.segments if role == ROLE_SYSTEM]
    body = [text for role, text in window.segments if role != ROLE_SYSTEM]
    return [
        {"role": "system", "content": "\n\n".join(system) or SYSTEM_INSTRUCTIONS},
        {"role": ROLE_ASSISTANT_INPUT, "content": "\n\n".join(body)},
    ]


def _first_choice_text(body: dict[str, object]) -> str:
    """An absent or empty completion is an empty string, which the planner already handles."""
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    return content if isinstance(content, str) else ""
