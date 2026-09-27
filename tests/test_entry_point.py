"""`python -m harness` has to work, and only a subprocess can prove it does.

One test, no model, no config: the entry point is either importable and wired to the
parser or it is not, and finding that out during a gate run is the expensive way.
"""

from __future__ import annotations

import subprocess
import sys


def test_the_module_entry_point_is_wired_to_the_parser() -> None:
    # Act
    result = subprocess.run(
        [sys.executable, "-m", "harness", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    # Assert
    assert result.returncode == 0
    assert "run" in result.stdout


def test_an_unknown_subcommand_exits_non_zero() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "harness", "nonsense"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
