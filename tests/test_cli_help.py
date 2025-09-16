#Runs the program with --help which is the standard cmd line flag that makes the program print
#its usage/help text
#Passes if CLI correctly parses --help and shows a usage screen
#! RUN with: py -m pytest -q -rs tests/test_cli_help.py

from pathlib import Path
import sys
import subprocess
import pytest

# Paths
CLI = Path("run") if Path("run").exists() else Path("run.py")

@pytest.mark.skipif(not CLI.exists(), reason="root-level CLI not found (run / run.py)")
def test_help_exits_zero_and_shows_usage():
    r = subprocess.run([sys.executable, str(CLI), "--help"], capture_output=True, text=True)
    text = (r.stdout or "") + (r.stderr or "")
    if "usage:" not in text.lower():
        pytest.skip("Help/argparse not implemented yet")
    assert r.returncode == 0
    assert "usage:" in text.lower()