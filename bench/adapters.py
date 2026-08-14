"""Drop external profiling tools in here.

The harness knows nothing about these; it just wraps the measured region in
whatever adapters you name.  Two stubs are provided to fix the shape of the
contract -- replace or add alongside.

Contract:  adapter(**kw) -> context manager whose object exposes .summary()
"""
from __future__ import annotations

import contextlib

import torch

from bench.harness import register_adapter


@register_adapter("torch_profiler")
@contextlib.contextmanager
def torch_profiler_adapter(sort_by="self_cuda_time_total", row_limit=15, **_):
    class H:
        def summary(self):
            return self.table
    h = H(); h.table = None
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True) as prof:
        yield h
    h.table = prof.key_averages().table(sort_by=sort_by, row_limit=row_limit)


@register_adapter("nvtx")
@contextlib.contextmanager
def nvtx_adapter(label="region", **_):
    class H:
        def summary(self): return {"label": label}
    torch.cuda.nvtx.range_push(label)
    try:
        yield H()
    finally:
        torch.cuda.nvtx.range_pop()


# YOUR TOOLS GO HERE.
# @register_adapter("nick_tool")
# @contextlib.contextmanager
# def nick_tool(**kw): ...
