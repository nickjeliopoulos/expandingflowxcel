"""Thin measurement core. Deliberately small; extend via adapters.py, not here.

Design rules
------------
1. No config framework, no plugin discovery, no CLI magic. A benchmark is a
   function that returns a Result.
2. Timing is CUDA-event based with explicit warmup and L2 flush. If you need
   kernel-level attribution, that is an *adapter*, not a change to this file.
3. Every Result carries the shapes and dtypes it was measured at, because a
   number without a shape is not a measurement.
4. Memory is reported as peak *allocated* (what the method actually needs) and
   peak *reserved* (what the allocator held), because the gap between them is
   itself a finding for ops that thrash on [B,L,V] temporaries.
"""
from __future__ import annotations

import contextlib
import dataclasses
import gc
import json
import platform
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import torch

_L2_FLUSH_BYTES = 256 * 1024 * 1024


@dataclass
class Result:
    op: str
    backend: str
    shape: Dict[str, int]
    dtype: str
    ms_mean: float
    ms_median: float
    ms_p10: float
    ms_p90: float
    peak_alloc_mib: float
    peak_reserved_mib: float
    n_iters: int
    achieved_gbps: Optional[float] = None
    achieved_tflops: Optional[float] = None
    note: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def as_row(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        d.update({f"shape_{k}": v for k, v in d.pop("shape").items()})
        d.pop("extra")
        return d


class _L2Flusher:
    def __init__(self, device):
        self.buf = torch.empty(_L2_FLUSH_BYTES // 4, dtype=torch.int32, device=device)

    def __call__(self):
        self.buf.zero_()


def benchmark(fn: Callable[[], Any], *, op: str, backend: str,
              shape: Dict[str, int], dtype: torch.dtype,
              warmup: int = 10, iters: int = 50, flush_l2: bool = True,
              bytes_moved: Optional[int] = None, flops: Optional[int] = None,
              note: str = "", device: str = "cuda") -> Result:
    dev = torch.device(device)
    flusher = _L2Flusher(dev) if (flush_l2 and dev.type == "cuda") else (lambda: None)

    for _ in range(warmup):
        fn()
    if dev.type == "cuda":
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    times: List[float] = []
    if dev.type == "cuda":
        starts = [torch.cuda.Event(True) for _ in range(iters)]
        ends = [torch.cuda.Event(True) for _ in range(iters)]
        for i in range(iters):
            flusher()
            starts[i].record(); fn(); ends[i].record()
        torch.cuda.synchronize()
        times = [s.elapsed_time(e) for s, e in zip(starts, ends)]
        peak_a = torch.cuda.max_memory_allocated() / 2 ** 20
        peak_r = torch.cuda.max_memory_reserved() / 2 ** 20
    else:
        for _ in range(iters):
            t0 = time.perf_counter(); fn(); times.append((time.perf_counter() - t0) * 1e3)
        peak_a = peak_r = float("nan")

    times.sort()
    med = statistics.median(times)
    res = Result(op=op, backend=backend, shape=dict(shape), dtype=str(dtype),
                 ms_mean=statistics.fmean(times), ms_median=med,
                 ms_p10=times[int(0.1 * len(times))], ms_p90=times[int(0.9 * len(times)) - 1],
                 peak_alloc_mib=peak_a, peak_reserved_mib=peak_r,
                 n_iters=iters, note=note)
    if bytes_moved:
        res.achieved_gbps = bytes_moved / (med * 1e-3) / 1e9
    if flops:
        res.achieved_tflops = flops / (med * 1e-3) / 1e12
    return res


# ---------------------------------------------------------------- adapters --

_ADAPTERS: Dict[str, Callable] = {}


def register_adapter(name: str):
    """Seam for external profiling tools.

    An adapter is a context manager factory: ``adapter(**kw) -> contextmanager``
    that wraps the measured region and, on exit, attaches whatever it collected
    to ``result.extra[name]``.  Drop your own tooling in adapters.py and it
    composes with every benchmark in bench/ without touching harness.py.
    """
    def deco(fn):
        _ADAPTERS[name] = fn
        return fn
    return deco


@contextlib.contextmanager
def with_adapters(names: List[str], result_extra: Dict[str, Any], **kw):
    stack = contextlib.ExitStack()
    with stack:
        handles = {n: stack.enter_context(_ADAPTERS[n](**kw)) for n in names if n in _ADAPTERS}
        yield handles
    for n, h in handles.items():
        result_extra[n] = getattr(h, "summary", lambda: None)()


def env_fingerprint() -> Dict[str, Any]:
    d = {"torch": torch.__version__, "python": platform.python_version(),
         "platform": platform.platform()}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        d.update({"gpu": p.name, "sm": f"{p.major}.{p.minor}",
                  "mem_gib": round(p.total_memory / 2 ** 30, 1),
                  "cuda": torch.version.cuda,
                  "sms": p.multi_processor_count})
    try:
        import triton
        d["triton"] = triton.__version__
    except Exception:
        d["triton"] = None
    return d


def write_results(results: List[Result], path: str) -> None:
    with open(path, "w") as f:
        json.dump({"env": env_fingerprint(),
                   "results": [r.as_row() for r in results]}, f, indent=2)
