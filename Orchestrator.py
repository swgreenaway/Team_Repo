import time
import sys
import Url_Parser
import Installer
import Tester
from pathlib import Path

def install_dependencies() -> int:
    return Installer.run()
    

def run_tests(pytest_args=None) -> int:
    if pytest_args is None:
        try:
            i = sys.argv.index("test")
            pytest_args = sys.argv[i+1:]   # everything after 'test'
        except ValueError:
            pytest_args = []
    return Tester.run(pytest_args=pytest_args)


def process_urls(file_path: Path) -> int:
    Url_Parser.process_urls(file_path)

    return 0


