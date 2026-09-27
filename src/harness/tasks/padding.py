"""Document padding - burying one fact in a document long enough to be worth evicting.

Why this exists. With one-sentence documents, twenty of them fit in an 8k window many times
over: NoOp solves every task, the bracket is zero, and the forces-recall screen strikes the
whole suite (DECISIONS.md, 2026-09-27). Padding each document to `document_tokens` makes
the window hold only a handful of documents, so a fact read `recall_distance` steps ago has
genuinely left it and the budget axis of Experiment 2 means what it says.

The filler is inert on purpose: no digits, no identifier-shaped names, and never the
attribute the goal asks about. A filler sentence that looked like a fact would turn padding
into distractors and change the task, not just its length.

Pure, like the generator: the same spec yields the same documents on every machine.
"""

from __future__ import annotations

import hashlib
import random

from harness.tokens import approx_token_count

#: Characters of the padding digest used to seed its RNG. Separate from the generator's
#: stream so that padding a suite never changes its codes, entities or distractors.
SEED_DIGEST_CHARS = 16

FILLER: tuple[str, ...] = (
    "The quarterly maintenance log was reviewed and filed without further comment.",
    "Staff rotated through the north corridor during the afternoon inspection.",
    "A shipment of office supplies arrived late and was stored in the east annex.",
    "The committee postponed its discussion of the parking arrangements again.",
    "Weather conditions remained mild for most of the reporting period.",
    "Several visitors asked about the opening hours of the reading room.",
    "The heating system was serviced and no faults were recorded by the engineer.",
    "Minutes from the previous meeting were approved with minor corrections.",
    "A reminder was circulated about keeping the fire exits clear at all times.",
    "The cafeteria menu was revised to include more seasonal options.",
    "Cleaning schedules were adjusted to cover the weekend shifts as well.",
    "The notice board in the lobby was reorganised by the front desk team.",
    "Lighting in the stairwell was upgraded as part of the routine works.",
    "An informal survey found that most people preferred the new seating plan.",
    "The library catalogue migration continued slowly but without data loss.",
    "Delivery vans were asked to use the rear entrance during peak hours.",
)


def pad(content: str, document_tokens: int, key: str) -> str:
    """Return `content` surrounded by filler until the document reaches `document_tokens`.

    `key` identifies the document (task, seed and document name), so each document gets
    its own filler order and its own fact position. Zero or a target already met returns
    `content` unchanged, which keeps every unpadded suite byte-identical to before.
    """
    if approx_token_count(content) >= document_tokens:
        return content
    rng = _rng(key)
    filler: list[str] = []
    while approx_token_count(" ".join([content, *filler])) < document_tokens:
        filler.append(FILLER[rng.randrange(len(FILLER))])
    position = rng.randint(0, len(filler))
    return " ".join([*filler[:position], content, *filler[position:]])


def _rng(key: str) -> random.Random:
    digest = hashlib.sha256(f"padding:{key}".encode()).hexdigest()
    return random.Random(int(digest[:SEED_DIGEST_CHARS], 16))  # noqa: S311 - not cryptographic
