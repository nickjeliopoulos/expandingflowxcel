"""Run the test suite with nothing installed but torch.

The suite is written for pytest and real pytest is the supported path -- but
pytest is not always present in a research env, and installing it is not always
worth a round trip. This runner injects a ~60-line pytest shim into
``sys.modules`` and then discovers and runs the same test functions, unmodified.

    python scripts/run_tests.py                # everything
    python scripts/run_tests.py test_expand    # one module (substring match)
    python scripts/run_tests.py -v             # full tracebacks

Exit code is the failure count.

Scope of the shim: ``mark.parametrize`` (including stacked, via cartesian
product), ``mark.<anything>`` as a no-op, ``skip``, ``raises``, ``fixture``.
That is everything tests/ currently uses -- verified by grep, not by hope. If a
test starts needing more (``approx``, real fixtures, ``xfail``), install pytest
rather than growing this; the shim is a convenience, not a second test
framework.
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import pathlib
import sys
import traceback
import types

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


class Skipped(Exception):
    pass


def _shim() -> types.ModuleType:
    m = types.ModuleType("pytest")

    class _Mark:
        def parametrize(self, argnames, argvalues, **_):
            names = ([a.strip() for a in argnames.split(",")]
                     if isinstance(argnames, str) else list(argnames))

            def deco(fn):
                groups = list(getattr(fn, "_params", []))
                groups.append((names, list(argvalues)))
                fn._params = groups
                return fn
            return deco

        def __getattr__(self, _name):          # @pytest.mark.gpu, .slow, ...
            def deco(fn=None, **__):
                return fn if fn is not None else (lambda f: f)
            return deco

    class _Raises:
        def __init__(self, exc, **_):
            self.exc = exc

        def __enter__(self):
            return self

        def __exit__(self, t, v, tb):
            if t is None:
                raise AssertionError(f"expected {self.exc.__name__}, none raised")
            return issubclass(t, self.exc)

    def skip(reason="", **_):
        raise Skipped(reason)

    m.mark = _Mark()
    m.skip = skip
    m.raises = _Raises
    m.fixture = lambda *a, **k: (lambda f: f)
    m.Skipped = Skipped
    return m


def _cases(fn):
    """Expand stacked parametrize decorators into a list of (kwargs, label)."""
    groups = getattr(fn, "_params", None)
    if not groups:
        return [({}, "")]
    per_group = []
    for names, values in groups:
        opts = []
        for v in values:
            opts.append({names[0]: v} if len(names) == 1 else dict(zip(names, v)))
        per_group.append(opts)
    out = []
    for combo in itertools.product(*per_group):
        kw = {}
        for d in combo:
            kw.update(d)
        out.append((kw, "[" + ",".join(f"{k}={v}" for k, v in kw.items()) + "]"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("filter", nargs="?", default="")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()

    sys.modules["pytest"] = _shim()

    import torch
    print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}"
          f"{'  ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}")
    print("(pytest shim -- real pytest is the supported path; see docstring)\n")

    files = sorted((ROOT / "tests").glob("test_*.py"))
    if a.filter:
        files = [f for f in files if a.filter in f.name]

    n_pass = n_fail = n_skip = 0
    failures = []

    for path in files:
        spec = importlib.util.spec_from_file_location(path.stem, path)
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except Exception as e:
            print(f"{path.name}: IMPORT FAILED -- {type(e).__name__}: {e}")
            if a.verbose:
                traceback.print_exc()
            n_fail += 1
            failures.append((path.name, "<import>", traceback.format_exc()))
            continue

        tests = [(n, o) for n, o in vars(mod).items()
                 if n.startswith("test_") and callable(o)]
        if not tests:
            continue
        print(f"{path.name}")
        for name, fn in sorted(tests):
            for kw, label in _cases(fn):
                tag = f"  {name}{label}"
                try:
                    fn(**kw)
                    n_pass += 1
                    print(f"    ok   {name}{label}")
                except Skipped as e:
                    n_skip += 1
                    print(f"    skip {name}{label}  ({e})")
                except Exception as e:
                    n_fail += 1
                    failures.append((path.name, f"{name}{label}",
                                     traceback.format_exc()))
                    print(f"    FAIL {name}{label}  {type(e).__name__}: "
                          f"{str(e).splitlines()[0][:100] if str(e) else ''}")
        print()

    if failures and a.verbose:
        print("=" * 70)
        for f, t, tb in failures:
            print(f"\n--- {f}::{t} ---\n{tb}")

    print(f"{n_pass} passed, {n_fail} failed, {n_skip} skipped")
    if failures and not a.verbose:
        print("re-run with -v for tracebacks")
    return n_fail


if __name__ == "__main__":
    sys.exit(main())
