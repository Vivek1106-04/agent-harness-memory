"""Segment vocabulary shared by every context manager.

`ContextWindow.segments` carries `(role, text)` pairs and `dropped` carries identifiers.
Both are read by the trace bus and by the Experiment 2 analysis, so the vocabulary is fixed
here rather than invented per implementation.

`dropped` entries are identifiers and reasons only - never evicted payloads. SECURITY.md section 4.
"""

from __future__ import annotations

ROLE_SYSTEM = "system"
ROLE_GOAL = "goal"
ROLE_MEMORY = "memory"
ROLE_OBSERVATION = "observation"
#: One compact line per step taken. See `prompts.render_action_log`.
ROLE_ACTIONS = "actions"
ACTIONS_ID = "actions:log"

#: Roles the budget may never evict. See SECURITY.md section 3 - these are also the structural
#: defence against injected content displacing the instructions.
PROTECTED_ROLES = frozenset({ROLE_SYSTEM, ROLE_GOAL})


def observation_id(step: int) -> str:
    return f"obs:{step}"


def memory_id(provenance: str) -> str:
    return f"mem:{provenance}"


def dropped_entry(identifier: str, reason: str) -> str:
    return f"{identifier}|{reason}"
