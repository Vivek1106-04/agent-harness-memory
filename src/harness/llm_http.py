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
* **Spend is metered from the endpoint's own usage report** and priced from config. The
  client refuses a call once `max_spend_usd` is reached, and raises `SpendLimitError`,
  which stops the sweep rather than failing every remaining cell one by one.
* **A refusal is not an exception.** An empty or missing completion is returned as an empty
  string, because the planner already treats malformed model output as an expected
  condition. Only transport failure raises, and the runner records that cell as failed and
  leaves it for a rerun.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from types import TracebackType

import httpx
from tenacity import (
    RetryCallState,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from harness.interfaces import LLMClient
from harness.prompts import SYSTEM_INSTRUCTIONS
from harness.runner import SweepAbortedError
from harness.segments import ROLE_SYSTEM
from harness.tokens import approx_token_count
from harness.types import ContextWindow

DEFAULT_TIMEOUT_S = 120.0
DEFAULT_MAX_ATTEMPTS = 4
DEFAULT_BACKOFF_S = 1.0
MAX_BACKOFF_S = 30.0
#: Longest server-requested wait honoured. A per-minute quota asks for under a minute; a
#: longer request means a daily quota, and the consecutive-failure stop handles that.
MAX_RETRY_AFTER_S = 90.0
TOKENS_PER_MILLION = 1_000_000
#: Gemini occasionally answers `finish_reason: stop` with no content and zero output tokens
#: (about 1 call in 11 in the first smoke run). Asking again is a transport retry, not a
#: second opinion: an empty reply carries no model output to prefer or discard.
EMPTY_COMPLETION_RETRIES = 2

#: Gemini puts the wait in the error body ("retryDelay": "36s"); others use the header.
_RETRY_DELAY = re.compile(r'"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"')

HTTP_TOO_MANY_REQUESTS = 429
#: A 429 whose body names a per-day quota is not transient: Gemini's free tier reports
#: `GenerateRequestsPerDayPerProjectPerModel-FreeTier`.
DAILY_QUOTA_MARKER = "PerDay"

#: Status codes worth a second attempt: rate limiting and server-side faults.
RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

ROLE_ASSISTANT_INPUT = "user"
COMPLETIONS_PATH = "/chat/completions"


class LLMTransportError(Exception):
    """The endpoint could not be reached, or refused the request. Never carries the key."""


class _RetryableStatusError(LLMTransportError):
    """An HTTP status that is worth another attempt. Internal to the retry policy."""

    def __init__(self, message: str, retry_after_s: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s


class SpendLimitError(SweepAbortedError):
    """The configured spend limit is reached. No further call is made."""


class DailyQuotaError(SweepAbortedError):
    """The provider's per-day quota is spent. Retrying today cannot succeed; resume tomorrow
    or on a paid tier. Gemini names the quota (`...PerDay...`) in the 429 body."""


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
        reasoning_effort: str | None = None,
        pricing: Pricing | None = None,
        auth: httpx.Auth | None = None,
    ) -> None:
        if max_attempts < 1:
            raise ValueError(f"max_attempts must be at least 1, got {max_attempts}")
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._temperature = temperature
        self._max_attempts = max_attempts
        self._backoff_s = backoff_s
        self._reasoning_effort = reasoning_effort
        self._pricing = pricing or Pricing()
        self._spent_usd = self._pricing.already_spent_usd
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = client or httpx.Client(timeout=timeout_s)
        self._headers = headers
        self._auth = auth

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
        reasoning_effort: str | None = None,
        pricing: Pricing | None = None,
        auth: httpx.Auth | None = None,
    ) -> HttpLLMClient:
        """Build a client, reading the key from the environment variable `api_key_env`.

        A local server needs no key, so `api_key_env` may be None. When it is named and
        unset the constructor refuses: a sweep that discovers this on its first call has
        already paid for a scheduler slot. `auth` is for a credential that changes during a
        sweep, such as a Google login, and is applied to every request.
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
            reasoning_effort=reasoning_effort,
            pricing=pricing,
            auth=auth,
        )

    @property
    def spent_usd(self) -> float:
        return self._spent_usd

    def complete(self, window: ContextWindow, max_output_tokens: int) -> str:
        limit = self._pricing.max_spend_usd
        if limit is not None and self._spent_usd >= limit:
            raise SpendLimitError(
                f"spend limit reached: ${self._spent_usd:.2f} of ${limit:.2f} "
                "(limits.max_spend_usd)"
            )
        messages = to_messages(window)
        payload: dict[str, object] = {
            "model": self._model,
            "messages": messages,
            "max_tokens": max_output_tokens,
            "temperature": self._temperature,
        }
        if self._reasoning_effort is not None:
            payload["reasoning_effort"] = self._reasoning_effort
        text = ""
        for _ in range(1 + EMPTY_COMPLETION_RETRIES):
            response = self._post_with_retry(payload)
            text = _first_choice_text(response)
            self._spent_usd += self._pricing.cost(response, messages, text)
            if text.strip():
                break
        return text

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
            wait=_server_or_exponential(self._backoff_s),
            retry=retry_if_exception_type((_RetryableStatusError, httpx.TransportError)),
            reraise=True,
        )
        for attempt in retrying:
            with attempt:
                return self._post(payload)
        raise LLMTransportError("retry policy exhausted without a result")  # pragma: no cover

    def _post(self, payload: dict[str, object]) -> dict[str, object]:
        response = self._client.post(
            f"{self._base_url}{COMPLETIONS_PATH}",
            json=payload,
            headers=self._headers,
            auth=self._auth or httpx.USE_CLIENT_DEFAULT,
        )
        if response.status_code == HTTP_TOO_MANY_REQUESTS and DAILY_QUOTA_MARKER in response.text:
            raise DailyQuotaError(
                f"{self._model}: the provider's daily request quota is spent - rerun the same "
                "command after it resets (completed runs are skipped) or enable billing"
            )
        if response.status_code in RETRYABLE_STATUS:
            raise _RetryableStatusError(
                f"{self._model}: endpoint returned {response.status_code}",
                retry_after_s=_retry_after(response),
            )
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


class Pricing:
    """What a call costs, and when to stop paying.

    Priced from the endpoint's `usage` block when it sends one, which is what the provider
    bills. Without one - a local server, or a provider that omits it - the arithmetic token
    count stands in, so the limit still binds rather than silently reading zero.
    """

    def __init__(
        self,
        input_usd_per_m: float = 0.0,
        output_usd_per_m: float = 0.0,
        max_spend_usd: float | None = None,
        already_spent_usd: float = 0.0,
    ) -> None:
        self.input_usd_per_m = input_usd_per_m
        self.output_usd_per_m = output_usd_per_m
        self.max_spend_usd = max_spend_usd
        self.already_spent_usd = already_spent_usd

    def cost(self, body: dict[str, object], messages: list[dict[str, str]], text: str) -> float:
        prompt, completion = _usage(body)
        if prompt is None:
            prompt = sum(approx_token_count(message["content"]) for message in messages)
        if completion is None:
            completion = approx_token_count(text)
        return (
            prompt * self.input_usd_per_m + completion * self.output_usd_per_m
        ) / TOKENS_PER_MILLION


def _usage(body: dict[str, object]) -> tuple[int | None, int | None]:
    """(prompt, billed output) tokens. Output is total minus prompt when total is given,
    because thinking tokens are billed as output and some endpoints report them only there."""
    usage = body.get("usage")
    if not isinstance(usage, dict):
        return None, None
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    total = usage.get("total_tokens")
    if not isinstance(prompt, int):
        return None, None
    if isinstance(total, int) and total >= prompt:
        return prompt, total - prompt
    return prompt, completion if isinstance(completion, int) else None


def _retry_after(response: httpx.Response) -> float | None:
    header = response.headers.get("retry-after")
    if header is not None:
        try:
            return float(header)
        except ValueError:
            return None
    match = _RETRY_DELAY.search(response.text)
    return float(match.group(1)) if match else None


def _server_or_exponential(backoff_s: float) -> Callable[[RetryCallState], float]:
    """Wait as long as the server asked, capped; otherwise back off exponentially."""
    exponential = wait_exponential(multiplier=backoff_s, max=MAX_BACKOFF_S)

    def wait(state: RetryCallState) -> float:
        error = state.outcome.exception() if state.outcome is not None else None
        requested = getattr(error, "retry_after_s", None)
        if isinstance(requested, float):
            return min(requested, MAX_RETRY_AFTER_S)
        return float(exponential(state))

    return wait
