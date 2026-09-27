"""How an observation is cut into memory items. One strategy, frozen across conditions.

SPEC.md section 8 lists the chunking strategy among the frozen controls: every memory
provider receives the same items, so a difference between conditions is a difference in
representation, not in what was stored.

Why chunk at all. A 900-token document stored whole comes back whole: five retrieved
items fill most of an 8k window, and the budget - which protects memory ahead of history -
leaves the agent a handful of recent steps. In the first real run the memory conditions
lost track of which documents they had read and looped, which penalised memory for being
present. Sentences keep a retrieved fact about twenty tokens long.

Sentences, not fixed windows. A fact is one sentence by construction, so a sentence split
never cuts one in half; a fixed-size window would, and half a fact reads as a whole fact
to the model.
"""

from __future__ import annotations

import re

#: A sentence ends at . ! or ? followed by whitespace; a line break also ends a chunk, so a
#: directory listing becomes one item per name.
_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")


def chunk_text(text: str) -> list[str]:
    """Non-empty sentences of `text`, in order. Text with no boundary is one chunk."""
    return [chunk.strip() for chunk in _BOUNDARY.split(text) if chunk.strip()]
