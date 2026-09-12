from __future__ import annotations

import pytest

from harness.redaction import (
    PLACEHOLDER,
    contains_secret,
    looks_like_secret,
    redact,
    redact_payload,
)

SECRETS = [
    "sk-abcdefghijklmnopqrstuvwxyz0123",
    "AIzaSyA1234567890abcdefghijklmnopqrstu",
    "gsk_abcdefghijklmnopqrstuvwxyz012345",
    "hf_abcdefghijklmnopqrstuvwxyz012345",
    "ghp_abcdefghijklmnopqrstuvwxyz012345",
    "Zm9vYmFyQmF6MTIzNDU2Nzg5MFF3ZXJ0eVVpb3BBc2Rm",
]

KEEPERS = [
    "the safe combination is 4271",
    "9f2b1c4e8a7d6b5c3e2f1a0b9c8d7e6f5a4b3c2d1e0f9a8b7c6d5e4f3a2b1c0d",
    "harness.memory.vector",
]


@pytest.mark.parametrize("secret", SECRETS)
def test_secrets_are_replaced(secret: str) -> None:
    assert redact(f"key={secret} rest") == f"key={PLACEHOLDER} rest"
    assert contains_secret(secret)


@pytest.mark.parametrize("text", KEEPERS)
def test_ordinary_text_and_hashes_survive(text: str) -> None:
    """A config hash and a git commit are hex and are recorded on purpose."""
    assert redact(text) == text
    assert not contains_secret(text)


def test_a_short_token_is_never_treated_as_a_secret() -> None:
    assert not looks_like_secret("Abc123")


def test_payload_is_redacted_recursively_without_mutation() -> None:
    # Arrange
    payload = {
        "prompt": "use sk-abcdefghijklmnopqrstuvwxyz0123 please",
        "nested": {"items": ["AIzaSyA1234567890abcdefghijklmnopqrstu", 7]},
        "tuple": ("sk-abcdefghijklmnopqrstuvwxyz0123",),
        "count": 3,
    }
    original = str(payload)

    # Act
    cleaned = redact_payload(payload)

    # Assert
    assert PLACEHOLDER in cleaned["prompt"]
    assert cleaned["nested"]["items"] == [PLACEHOLDER, 7]
    assert cleaned["tuple"] == (PLACEHOLDER,)
    assert cleaned["count"] == 3
    assert str(payload) == original
