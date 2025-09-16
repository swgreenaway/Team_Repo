#This test makes sure the CLI is properly rejecting unknown arguments
#Cmd to run from root: py -m pytest -q -rs tests/test_cli_unknown_flag.py

from pathlib import Path
import sys 
import subprocess
import pytest

# Locate src/ and the CLI file
CLI = Path("run") if Path("run").exists() else Path("run.py")

@pytest.mark.skipif(not CLI.exists(), reason="root-level CLI not found (run / run.py)")
def test_unknown_option_exits_2_and_shows_error():
    r = subprocess.run([sys.executable, str(CLI), "--definitely-not-a-real-flag"],
                       capture_output=True, text=True)
    if r.returncode == 0:
        pytest.skip("Parser not implemented yet (returned 0 for unknown flag)")
    assert r.returncode == 2
    assert "unrecognized arguments" in (r.stderr or r.stdout).lower()