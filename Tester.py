# run  (or run.py)
import argparse, subprocess, sys

def build_parser():
    p = argparse.ArgumentParser(prog="run", description="Team CLI")
    subs = p.add_subparsers(dest="command", required=True)

    # existing subcommands … e.g., install = subs.add_parser("install")

    # --- test subcommand ---
    test_p = subs.add_parser("test", help="Run pytest suite")
    test_p.add_argument("paths", nargs="*", default=["tests"], help="Files/dirs to test (default: tests)")
    test_p.add_argument("-q", "--quiet", action="store_true", help="Quiet pytest output")
    test_p.add_argument("-v", "--verbose", action="store_true", help="Verbose pytest output (-v)")
    return p

def cmd_test(args) -> int:
    cmd = [sys.executable, "-m", "pytest"]
    if args.quiet:
        cmd.append("-q")
    if args.verbose:
        cmd.append("-v")
    cmd += args.paths
    print("Running:", " ".join(cmd))
    return subprocess.run(cmd).returncode

def run(paths=None, pytest_args=None) -> int:
    """Run pytest and return its exit code."""
    if paths is None:
        paths = ["tests"]
    if pytest_args is None:
        pytest_args = ["-q", "-rs"]  # quiet + show skip reasons

    cmd = [sys.executable, "-m", "pytest", *pytest_args, *paths]
    print("Running:", " ".join(cmd))
    return subprocess.run(cmd).returncode

def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "test":
        return cmd_test(args)

    # elif args.command == "install": ...
    return 0

if __name__ == "__main__":
    raise SystemExit(main())