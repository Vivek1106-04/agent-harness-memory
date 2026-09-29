"""Google Cloud auth, tested without Google. Credentials are a fake that counts refreshes."""

from __future__ import annotations

import google.auth
import google.auth.exceptions
import httpx
import pytest

from harness.gcp_auth import GoogleAdcAuth
from harness.llm_http import LLMTransportError


class _FakeCredentials:
    def __init__(self, is_valid: bool) -> None:
        self.valid = is_valid
        self.token: str | None = "tok-initial" if is_valid else None
        self.refreshes = 0

    def refresh(self, request: object) -> None:
        self.refreshes += 1
        self.valid = True
        self.token = f"tok-{self.refreshes}"


def _authorised(auth: GoogleAdcAuth) -> httpx.Request:
    request = httpx.Request("POST", "https://endpoint.invalid/v1/chat/completions")
    return next(auth.auth_flow(request))


def test_sends_the_current_token_as_a_bearer_header() -> None:
    # Arrange
    auth = GoogleAdcAuth(_FakeCredentials(is_valid=True))

    # Act
    request = _authorised(auth)

    # Assert
    assert request.headers["authorization"] == "Bearer tok-initial"


def test_refreshes_an_expired_token_before_the_request() -> None:
    """An access token lasts an hour and a sweep runs for several."""
    # Arrange
    credentials = _FakeCredentials(is_valid=False)
    auth = GoogleAdcAuth(credentials)

    # Act
    request = _authorised(auth)

    # Assert
    assert credentials.refreshes == 1
    assert request.headers["authorization"] == "Bearer tok-1"


def test_a_valid_token_is_not_refreshed() -> None:
    # Arrange
    credentials = _FakeCredentials(is_valid=True)
    auth = GoogleAdcAuth(credentials)

    # Act
    _authorised(auth)
    _authorised(auth)

    # Assert
    assert credentials.refreshes == 0


def test_from_environment_refuses_to_start_without_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same rule as a missing key: fail before the sweep, not on its first call."""

    # Arrange
    def no_credentials(scopes: list[str]) -> object:
        raise google.auth.exceptions.DefaultCredentialsError("none found")

    monkeypatch.setattr(google.auth, "default", no_credentials)

    # Act / Assert
    with pytest.raises(LLMTransportError, match="application-default login"):
        GoogleAdcAuth.from_environment()


def test_from_environment_refuses_to_start_when_the_token_cannot_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    class _Revoked(_FakeCredentials):
        def refresh(self, request: object) -> None:
            raise google.auth.exceptions.RefreshError("reauthentication needed")

    monkeypatch.setattr(google.auth, "default", lambda scopes: (_Revoked(False), None))

    # Act / Assert
    with pytest.raises(LLMTransportError, match="application-default login"):
        GoogleAdcAuth.from_environment()


def test_from_environment_checks_the_token_up_front(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    credentials = _FakeCredentials(is_valid=False)
    monkeypatch.setattr(google.auth, "default", lambda scopes: (credentials, "btp-vivek"))

    # Act
    auth = GoogleAdcAuth.from_environment()

    # Assert
    assert credentials.refreshes == 1
    assert _authorised(auth).headers["authorization"] == "Bearer tok-1"
