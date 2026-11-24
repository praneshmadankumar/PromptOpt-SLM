"""Minimal test runner for environments without pytest.

Executes every top-level test_* function in tests/test_core.py.
Exit code 0 means all tests passed.
"""
import importlib
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    mod = importlib.import_module("tests.test_core")
    names = sorted(n for n in dir(mod) if n.startswith("test_"))
    passed, failed = 0, []
    for name in names:
        try:
            getattr(mod, name)()
            passed += 1
        except Exception:  # report every failure, keep going
            failed.append((name, traceback.format_exc(limit=2)))
    for name, tb in failed:
        print(f"FAIL {name}\n{tb}")
    print(f"{passed} passed, {len(failed)} failed, {len(names)} total")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
