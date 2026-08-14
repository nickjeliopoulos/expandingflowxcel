"""Backend registry -- the single seam between reference math and fast kernels.

Every microbenchmark target is declared once as an *op name* with one or more
*backends*.  Exactly one backend per op must be tagged ``reference``: slow,
loop-heavy, obviously-correct, float64.  It is the ground truth that
``tests/test_backend_parity.py`` checks every other backend against, and it is
never used on the benchmark path.

Adding a Triton kernel is then a two-line change with zero refactoring::

    @register("interpolant", "triton")
    def _interp_triton(labels, t_local, ...):
        ...

and it automatically acquires (a) a correctness test against the reference and
(b) a row in every benchmark sweep that covers ``interpolant``.

This is deliberately ~60 lines with no config system.  Resist growing it.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable

_REGISTRY: Dict[str, Dict[str, "Impl"]] = {}


@dataclass
class Impl:
    fn: Callable
    op: str
    backend: str
    reference: bool = False
    #: predicate returning None if runnable, else a string reason to skip
    available: Callable[[], str | None] = field(default=lambda: None)
    #: free-form notes surfaced in benchmark reports
    note: str = ""


def register(op: str, backend: str, *, reference: bool = False,
             available: Callable[[], str | None] | None = None, note: str = ""):
    def deco(fn):
        impls = _REGISTRY.setdefault(op, {})
        if backend in impls:
            raise KeyError(f"{op}/{backend} already registered")
        if reference and any(i.reference for i in impls.values()):
            raise KeyError(f"{op} already has a reference backend")
        impls[backend] = Impl(fn, op, backend, reference,
                              available or (lambda: None), note)
        return fn
    return deco


def get(op: str, backend: str = "torch") -> Callable:
    try:
        return _REGISTRY[op][backend].fn
    except KeyError as e:
        have = sorted(_REGISTRY.get(op, {}))
        raise KeyError(f"no backend {backend!r} for op {op!r}; have {have}") from e


def reference_of(op: str) -> Impl:
    for impl in _REGISTRY[op].values():
        if impl.reference:
            return impl
    raise KeyError(f"op {op!r} has no reference backend -- add one before benchmarking")


def backends(op: str) -> Iterable[Impl]:
    return list(_REGISTRY[op].values())


def ops() -> Iterable[str]:
    return sorted(_REGISTRY)


def requires_triton() -> str | None:
    try:
        import triton  # noqa: F401
    except Exception as e:  # pragma: no cover
        return f"triton unavailable: {e}"
    return None


@functools.lru_cache(maxsize=1)
def load_all() -> None:
    """Import every op module so decorators fire. Called by bench/ and tests/."""
    from eflow.ops import (  # noqa: F401
        adaln, edge_ops, expand, expand_child, expand_graph, gap_counts,
        flow_map, insertion_sample, interpolant, local_time, poisson_nll,
        semigroup_ce,
    )
