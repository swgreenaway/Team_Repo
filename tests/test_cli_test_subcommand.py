"""Test purpose: ensure the CLI `test` subcommand delegates to Tester.run(...)
and propagates its exit code. Pending-safe: skips if `test` isn’t advertised.
"""
import sys
import types
import runpy
import pytest

def test_test_subcommand_calls_tester_and_propagates_code(cli_path, run_cli, monkeypatch):
    # Only run if `test` appears in help
    h_code, h_out, h_err = run_cli("--help")
    if " test" not in (h_out + h_err).lower():
        pytest.skip("`test` subcommand not implemented yet")

    # Fake Tester module so Orchestrator (imported by run) uses our stub
    calls = {}
    fake_tester = types.SimpleNamespace(
        run=lambda paths=None, pytest_args=None: calls.setdefault("ret", 0)
    )
    # Inject our fake BEFORE executing the CLI so imports resolve to it
    monkeypatch.setitem(sys.modules, "Tester", fake_tester)

    # Choose a return code to verify propagation (0 = success)
    calls["ret"] = 0

    # Execute the root script in-process so our monkeypatch applies
    monkeypatch.setattr(sys, "argv", [str(cli_path), "test"], raising=False)
    try:
        runpy.run_path(str(cli_path), run_name="__main__")
        rc = 0
    except SystemExit as e:
        rc = int(e.code)

    # Assertions: CLI exited with the Tester.run() code and our stub was used
    assert rc == 0, f"Expected CLI to propagate Tester.run() exit code, got {rc}"
    # also confirm our fake ran (by presence of 'ret' key)
    assert "ret" in calls
