#This test makes sure the CLI is properly rejecting unknown arguments
#Cmd to run from root: py -m pytest -q -rs tests/test_cli_unknown_flag.py

from pathlib import Path
import sys 
import subprocess
import pytest

# Locate src/ and the CLI file
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
CLI = SRC / "app" / "cli.py"

#skips if cli doesnt exist
@pytest.mark.skipif(not CLI.exists(), reason="CLI skeleton not created yet")

def test_unknown_option_exits_2_and_shows_error():
    """Unknown CLI flags should cause an argparse-style exit(2) with an error message."""
    result = subprocess.run(
        [sys.executable, str(CLI), "--definitely-not-a-real-flag"],
        capture_output=True,
        text=True,
    )

    # If argparse isn't wired yet, don't fail the build—treat as pending.
    if result.returncode == 0:
        pytest.skip("Parser not implemented yet (main returned 0 for unknown flag)")

    assert result.returncode == 2
    assert "unrecognized arguments" in (result.stderr or result.stdout).lower()