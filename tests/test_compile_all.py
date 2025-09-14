#Cmd to run from root: py -m pytest -q -rs tests/test_compile_all.py

import compileall
from pathlib import Path

# Checks that all files in src compile
def test_all_python_files_compile():
    src = Path("src")
    assert src.exists(), "Expected a src/ folder"
    ok = compileall.compile_dir(str(src), quiet=1)
    assert ok, "Some files failed to compile."