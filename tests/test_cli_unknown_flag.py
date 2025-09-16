#This test makes sure the CLI is properly rejecting unknown arguments
#Cmd to run from root: py -m pytest -q -rs tests/test_cli_unknown_flag.py

import pytest

def test_unknown_option_exits_2_and_shows_error(run_cli):
    code, out, err = run_cli("--definitely-not-a-real-flag")
    text = (err or out or "").lower()
    if "required: command" in text:
        # requires a subcommand; try one if available
        h_code, h_out, h_err = run_cli("--help")
        if "install" not in (h_out + h_err).lower():
            pytest.skip("Parser requires subcommand; none available to test unknown flags")
        code, out, err = run_cli("install", "--definitely-not-a-real-flag")
        text = (err or out or "").lower()
    assert code == 2
    assert "unrecognized arguments" in text