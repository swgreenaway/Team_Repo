import time
import Url_Parser
import Installer
import Tester
from pathlib import Path

def install_dependencies() -> int:
    return Installer.run()
    

def run_tests() -> int:
    return Tester.run()


def process_urls(file_path: Path) -> int:
    Url_Parser.process_urls(file_path)

    return 0


