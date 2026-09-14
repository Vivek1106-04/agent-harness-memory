"""Token accounting used by the context manager before a model tokenizer exists.

The budget must bind deterministically on every machine, so the default counter is an
arithmetic approximation rather than a tokenizer: no model download, no revision drift, no
network in unit tests. Week 4 swaps `LLMClient.count_tokens` in for the real runs; the
budget guarantee is tested against whichever counter the manager was given.
"""

from __future__ import annotations

CHARS_PER_TOKEN = 4


def approx_token_count(text: str) -> int:
    """Ceiling division of characters by `CHARS_PER_TOKEN`. Empty text costs nothing."""
    if not text:
        return 0
    return -(-len(text) // CHARS_PER_TOKEN)
