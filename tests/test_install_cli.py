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
import sys, runpy, subprocess, pytest

CLI = Path("run") if Path("run").exists() else Path("run.py")

@pytest.mark.skipif(not CLI.exists(), reason="root-level CLI not found (run / run.py)")
def test_install_subcommand_invokes_pip_and_exits_zero(monkeypatch):
    # Only activate when 'install' is advertised
    h = subprocess.run([sys.executable, str(CLI), "--help"], capture_output=True, text=True)
    if "install" not in ((h.stdout or "") + (h.stderr or "")):
        pytest.skip("'install' subcommand not implemented yet")

    calls = []
    def fake_run(cmd, *a, **kw):
        calls.append(cmd)
        return type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()
    monkeypatch.setattr(subprocess, "run", fake_run, raising=True)

    # Run the script in-process so the monkeypatch applies
    monkeypatch.setattr(sys, "argv", [str(CLI), "install"], raising=False)
    try:
        runpy.run_path(str(CLI), run_name="__main__")
        rc = 0
    except SystemExit as e:
        rc = int(e.code)

    assert rc == 0, "Expected exit 0 on successful install"
    assert calls, "Expected a pip command to be invoked"
    joined = " ".join(map(str, calls[0])) if isinstance(calls[0], (list, tuple)) else str(calls[0])
    assert "pip" in joined and "install" in joined