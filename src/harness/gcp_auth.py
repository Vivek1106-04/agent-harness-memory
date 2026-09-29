"""Google Cloud login for a Vertex AI endpoint.

Vertex AI serves the same OpenAI chat-completions shape as every other endpoint, but takes
an OAuth access token rather than an API key, and that token expires after an hour. A sweep
runs for several, so the token is checked before every request and refreshed when stale.

The credential comes from Application Default Credentials - `gcloud auth
application-default login` on a laptop - never from config. SECURITY.md section 1.
"""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

import google.auth
import google.auth.exceptions
import google.auth.transport.requests
import httpx

from harness.llm_http import LLMTransportError

CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
_LOGIN_HINT = "run `gcloud auth application-default login` and retry"


class GoogleAdcAuth(httpx.Auth):
    def __init__(self, credentials: Any) -> None:
        self._credentials = credentials
        self._request = google.auth.transport.requests.Request()

    @classmethod
    def from_environment(cls) -> GoogleAdcAuth:
        """Load the machine's Google login and prove it yields a token before a sweep starts."""
        try:
            credentials, _project = google.auth.default(scopes=[CLOUD_PLATFORM_SCOPE])
            auth = cls(credentials)
            auth._refresh_if_stale()
        except google.auth.exceptions.GoogleAuthError as error:
            raise LLMTransportError(f"no usable Google login: {_LOGIN_HINT}") from error
        return auth

    def auth_flow(self, request: httpx.Request) -> Generator[httpx.Request, httpx.Response, None]:
        self._refresh_if_stale()
        request.headers["Authorization"] = f"Bearer {self._credentials.token}"
        yield request

    def _refresh_if_stale(self) -> None:
        if not self._credentials.valid:
            self._credentials.refresh(self._request)
