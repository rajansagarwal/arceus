#!/usr/bin/env python3
"""
Plain-Python test runner for this repository.

- Discovers tests in tests/test_*.py
- Uses unittest discovery and TextTestRunner
- Exits with non-zero status code on failures/errors
- Supports running from repository root: `PYTHONPATH=. python tests/run_tests.py`

This avoids pytest and third-party deps.
"""
import os
import sys
import unittest
from types import SimpleNamespace


def ensure_repo_root_on_path():
    # Ensure repo root (parent of tests/) is on sys.path
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    return repo_root


def discover_tests(start_dir: str = None):
    if start_dir is None:
        start_dir = os.path.dirname(__file__)
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=start_dir, pattern="test_*.py")
    return suite


def main(argv=None):
    ensure_repo_root_on_path()
    suite = discover_tests()
    verbosity = 2 if os.environ.get("VERBOSE", "0") == "1" else 1
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    summary = SimpleNamespace(
        run=result.testsRun,
        failures=len(result.failures),
        errors=len(result.errors),
        skipped=len(getattr(result, 'skipped', [])),
    )

    print("\n==== Test Summary ====")
    print(f"Ran: {summary.run}")
    print(f"Failures: {summary.failures}")
    print(f"Errors: {summary.errors}")
    print(f"Skipped: {summary.skipped}")

    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
