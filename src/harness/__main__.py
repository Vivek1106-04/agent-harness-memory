"""Entry point for `python -m harness`.

Exercised by `tests/test_entry_point.py`, which runs it in a subprocess - the only way an
entry point can be tested at all. Coverage does not follow a subprocess, so these two
lines are excluded rather than left to look untested.
"""

from harness.cli import main  # pragma: no cover

raise SystemExit(main())  # pragma: no cover
