"""The registry's load-bearing invariant: backends of one op are interchangeable.

An op whose backends take different arguments cannot be checked against a common
reference and cannot be swapped in a sweep, which defeats the point of the
registry. This broke once already (com_free/segment took a CSR triple while its
siblings took dense (x, mask)) and was caught by scripts/smoke.py rather than by
a test -- so here is the test.

Signature comparison ignores optional keyword-only tuning parameters (sigma,
eps, chunk, ...), which backends are free to differ on; it pins the REQUIRED
positional arguments, which they are not.
"""
from __future__ import annotations

import inspect

import pytest

from eflow.ops import registry

registry.load_all()


def required_params(fn):
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):        # torch.compile wrappers, C funcs
        return None
    return [n for n, p in sig.parameters.items()
            if p.default is inspect.Parameter.empty
            and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]


@pytest.mark.parametrize("op", sorted(registry.ops()))
def test_backends_share_a_call_signature(op):
    impls = registry.backends(op)
    sigs = {}
    for impl in impls:
        params = required_params(impl.fn)
        if params is None:
            continue
        sigs[impl.backend] = params
    if len(sigs) < 2:
        pytest.skip(f"{op} has fewer than two introspectable backends")
    ref = next(iter(sigs.values()))
    for backend, params in sigs.items():
        assert params == ref, (
            f"{op}/{backend} takes {params} but a sibling takes {ref}; "
            "backends of one op must be interchangeable"
        )


@pytest.mark.parametrize("op", sorted(registry.ops()))
def test_op_has_at_most_one_reference(op):
    n = sum(1 for i in registry.backends(op) if i.reference)
    assert n <= 1, f"{op} has {n} reference backends"


def test_every_op_is_importable_and_nonempty():
    ops = list(registry.ops())
    assert ops, "load_all() registered nothing -- an ops module failed to import"
    for op in ops:
        assert registry.backends(op), f"{op} registered with no backends"
