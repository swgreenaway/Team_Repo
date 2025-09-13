"""
UML reference (Install branch):
Parse CLI args → INSTALL? → Run dependency install (pip) → 
   on success: print success & exit 0
   on failure: print error & exit 1

This test verifies the success path:
- calling the CLI with 'install' triggers a pip install invocation
- exit code is 0 when the underlying pip call succeeds

It auto-skips if the CLI module or 'install' subcommand isn't present yet.
If present, fakes pip via monkeypatch to avoid a real install

*Command to run test (run from repo root): py -m pytest -q -rs tests/test_install_cli.py
"""

from pathlib import Path
import sys
import types
import pytest

#Make 'import app' work without installing the package
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

CLI_PATH = SRC / "app" / "cli.py"

@pytest.mark.skipif(not CLI_PATH.exists(), reason="CLI skeleton not created yet")

#monkeypatch and capsys are built in pytest fixtures 
#monkeypatch patches functions during a test and automatically restores them afterwards
#capsys grabs what the code prints out to sys.stdout or sys.stderr, currently not used
def test_install_invokes_pip_and_exits_zero(monkeypatch):
    """
    Contract (from UML):
      install -> runs pip install -> exit 0 on success

    This test:
      - SKIPS if the parser or 'install' isn't implemented yet.
      - When present, fakes pip via monkeypatch to avoid real installs.
    """
    try:
        import app.cli as cli
    except Exception as e:
        pytest.skip(f"CLI not importable yet: {e}")

    # If there is a parser, require that 'install' shows up in help.
    if not hasattr(cli, "build_parser"):
        pytest.skip("build_parser() not implemented yet")
    try:
        help_text = cli.build_parser().format_help()
    except Exception as e:
        pytest.skip(f"parser not functional yet: {e}")
    if "install" not in help_text:
        pytest.skip("'install' subcommand not implemented yet")

    # Fake out pip so no real installation happens.
    calls = []
    def fake_run(cmd, *a, **kw):
        calls.append(cmd)
        return type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()
    monkeypatch.setattr(subprocess, "run", fake_run, raising=True)

    # Support either main(argv=None) or main() reading sys.argv
    assert hasattr(cli, "main"), "CLI should expose main()"
    try:
        sig = inspect.signature(cli.main)
        if len(sig.parameters) == 0:
            # No-arg main(): patch argv
            monkeypatch.setattr(sys, "argv", ["app", "install"], raising=False)
            rc = cli.main()
        else:
            rc = cli.main(["install"])
    except SystemExit as e:
        rc = e.code

    # Success path per UML
    assert rc == 0
    assert calls, "Expected a pip command to be invoked"
    joined = " ".join(map(str, calls[0])) if isinstance(calls[0], (list, tuple)) else str(calls[0])
    assert "pip" in joined and "install" in joined