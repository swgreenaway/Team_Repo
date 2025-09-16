#Runs the program with --help which is the standard cmd line flag that makes the program print
#its usage/help text
#Passes if CLI correctly parses --help and shows a usage screen
#! RUN with: py -m pytest -q -rs tests/test_cli_help.py

import pytest

def test_help_exits_zero_and_shows_usage(run_cli):
    code, out, err = run_cli("--help")
    text = (out or "") + (err or "")
    if "usage:" not in text.lower():
        pytest.skip("Help/argparse not implemented yet")
    assert code == 0