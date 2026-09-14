"""Credential redaction, shared by the trace bus and the config loader.

Traces capture full prompts and responses. A key pasted into a prompt during debugging
would otherwise be written to disk and pushed, so redaction happens at emit time, not at
read time. See SECURITY.md section 1.

The same detector rejects credentials found in config files, where they are never allowed:
keys come from environment variables only.
"""

from __future__ import annotations

import re
from typing import Any

PLACEHOLDER = "[REDACTED]"

#: Provider key shapes we can name. Prefix-anchored, so they are precise.
KEY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"AIza[0-9A-Za-z_\-]{20,}"),
    re.compile(r"gsk_[A-Za-z0-9]{20,}"),
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
)

#: Catch-all for unrecognised providers: a long high-entropy run.
GENERIC_MIN_LENGTH = 40
_CANDIDATE = re.compile(rf"[A-Za-z0-9+/_\-]{{{GENERIC_MIN_LENGTH},}}={{0,2}}")


def looks_like_secret(token: str) -> bool:
    """Mixed-case alphanumeric run of significant length.

    Pure-hex runs are excluded: config hashes and git commit ids are hex, they are recorded
    in every trace header on purpose, and redacting them would destroy provenance.
    """
    if len(token) < GENERIC_MIN_LENGTH:
        return False
    if all(c in "0123456789abcdefABCDEF" for c in token):
        return False
    has_lower = any(c.islower() for c in token)
    has_upper = any(c.isupper() for c in token)
    has_digit = any(c.isdigit() for c in token)
    return has_lower and has_upper and has_digit


def redact(text: str) -> str:
    for pattern in KEY_PATTERNS:
        text = pattern.sub(PLACEHOLDER, text)
    return _CANDIDATE.sub(
        lambda m: PLACEHOLDER if looks_like_secret(m.group()) else m.group(), text
    )


def contains_secret(text: str) -> bool:
    if any(pattern.search(text) for pattern in KEY_PATTERNS):
        return True
    return any(looks_like_secret(match.group()) for match in _CANDIDATE.finditer(text))


def redact_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Redact every string in a trace payload, recursively. Returns a new payload."""
    return {key: _redact_value(value) for key, value in payload.items()}


def _redact_value(value: Any) -> Any:
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        return {key: _redact_value(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_redact_value(inner) for inner in value]
    if isinstance(value, tuple):
        return tuple(_redact_value(inner) for inner in value)
    return value
