"""Does it run at all? Exercises every registered backend at tiny shapes.

This is the first thing to run on a new machine, before any benchmark. It
answers one question -- "does this code execute here" -- and deliberately does
not measure anything. Shapes come from the `smoke` config (B=2, L=8, V=32), so
it allocates megabytes, not gigabytes, and finishes in seconds on CPU.

    python scripts/smoke.py            # auto: cuda if available, else cpu
    python scripts/smoke.py --cpu      # force cpu
    python scripts/smoke.py -v         # show full tracebacks

Exit code is the number of failures, so it composes with CI.

Backends that report themselves unavailable (no Triton, no working
torch.compile) are SKIPped, not failed -- that is the expected state on a
Windows box without Triton.
"""
from __future__ import annotations

import argparse
import sys
import traceback

sys.path.insert(0, "src")
sys.path.insert(0, ".")

import torch  # noqa: E402

from bench.roofline import CONFIGS  # noqa: E402
from eflow.ops import registry  # noqa: E402

registry.load_all()

OK, FAIL, SKIP = "ok", "FAIL", "skip"


def build(op, c, device, dtype):
    """Tiny inputs per op. Mirrors bench/bench_ops.py::make_inputs but small."""
    B, L, V, D = c.B, c.L, c.V, c.D
    f = dict(device=device, dtype=dtype)
    simplex = lambda: torch.softmax(torch.randn(B, L, V, **f), -1)
    if op == "local_time":
        return dict(t_ins=torch.rand(B, L, **f), t=torch.tensor([0.5], device=device))
    if op == "interpolant":
        return dict(labels=torch.randint(0, V, (B, L), device=device),
                    t_local=torch.rand(B, L, **f),
                    active=torch.rand(B, L, device=device) > 0.4, V=V)
    if op == "gap_counts":
        return dict(active=torch.rand(B, L, device=device) > 0.4)
    if op == "poisson_nll":
        return dict(a=torch.randint(0, 4, (B, L), device=device).to(dtype),
                    b=torch.rand(B, L, **f) + 0.1)
    if op == "insertion_sample":
        return dict(I_hat=torch.rand(B, L + 1, device=device, dtype=torch.float32) * 2,
                    budget=torch.randint(1, L, (B,), device=device))
    if op == "flow_map":
        return dict(logits=torch.randn(B, L, V, **f), x=torch.randn(B, L, V, **f),
                    s=torch.tensor(0.2, device=device), t=torch.tensor(0.8, device=device))
    if op == "semigroup_ce":
        return dict(student_logits=torch.randn(B, L, V, **f, requires_grad=True),
                    psi_su=simplex(), psi_ut=simplex(),
                    s=torch.tensor(0.2, device=device), u=torch.tensor(0.5, device=device),
                    t=torch.tensor(0.8, device=device),
                    mask=(torch.rand(B, L, device=device) > 0.3).to(dtype))
    if op == "adaln":
        return dict(x=torch.randn(B, L, D, **f), shift=torch.randn(B, L, D, **f),
                    scale=torch.randn(B, L, D, **f))
    if op == "edge_time":
        return dict(t_local=torch.rand(B, L, **f),
                    active=torch.rand(B, L, device=device) > 0.3)
    if op == "symmetrize":
        return dict(E=torch.randn(B, L, L, 3, **f))
    if op == "child_expand":
        return dict(x0_heavy=torch.randn(B, L, 3, **f),
                    parent=torch.randint(0, L, (B, L), device=device))
    if op == "com_free":
        return dict(x=torch.randn(B, L, 3, **f),
                    mask=torch.rand(B, L, device=device) > 0.3)
    if op == "expand_sampling":
        active = torch.rand(B, L, device=device) > 0.5
        counts = torch.zeros(B, L + 1, dtype=torch.long, device=device)
        counts[:, 0] = 1
        return dict(x=torch.randn(B, L, V, **f), t_local=torch.rand(B, L, **f) * active,
                    active=active, counts=counts)
    if op == "expand_training":
        return dict(state_x=torch.randn(B, L, V, **f), t_ins=torch.rand(B, L, **f),
                    s=torch.tensor(0.3, device=device), t=torch.tensor(0.7, device=device))
    if op == "flow_map_cont":
        return dict(x_expanded=torch.randn(B, L, 3, **f), v=torch.randn(B, L, 3, **f),
                    s=torch.tensor(0.2, device=device), t=torch.tensor(0.8, device=device))
    if op == "expand_graph":
        active = torch.rand(B, L, device=device) > 0.5
        counts = torch.zeros(B, L + 1, dtype=torch.long, device=device)
        counts[:, 0] = 1
        return dict(x=torch.randn(B, L, 4, **f), E=torch.randn(B, L, L, 5, **f),
                    active=active, counts=counts)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()

    device = "cpu" if (a.cpu or not torch.cuda.is_available()) else "cuda"
    dtype = torch.float32
    c = CONFIGS["smoke"]

    print(f"torch {torch.__version__}  device={device}  "
          f"cuda={torch.cuda.is_available()}")
    if device == "cuda":
        print(f"gpu   {torch.cuda.get_device_name(0)}")
    print(f"shapes B={c.B} L={c.L} V={c.V} D={c.D}\n")

    n_ok = n_fail = n_skip = 0
    rows = []
    for op in registry.ops():
        kw = build(op, c, device, dtype)
        for impl in registry.backends(op):
            name = f"{op}/{impl.backend}"
            if kw is None:
                rows.append((SKIP, name, "no smoke fixture")); n_skip += 1
                continue
            why = impl.available()
            if why:
                rows.append((SKIP, name, why)); n_skip += 1
                continue
            try:
                out = impl.fn(**{k: v for k, v in kw.items()})
                t = out[0] if isinstance(out, tuple) else out
                if torch.is_tensor(t) and t.dtype.is_floating_point:
                    if not torch.isfinite(t).all():
                        raise ValueError("output contains nan/inf")
                rows.append((OK, name, "")); n_ok += 1
            except Exception as e:
                msg = traceback.format_exc() if a.verbose else f"{type(e).__name__}: {e}"
                rows.append((FAIL, name, msg)); n_fail += 1

    for status, name, msg in rows:
        mark = {OK: "  ok  ", FAIL: " FAIL ", SKIP: " skip "}[status]
        print(f"[{mark}] {name:<34}{msg if status != OK else ''}")

    print(f"\n{n_ok} ok, {n_fail} failed, {n_skip} skipped")

    # Hygiene: an op with no reference backend cannot be checked by
    # test_backend_parity, so a kernel for it would land unverified.
    orphans = [op for op in registry.ops()
               if not any(i.reference for i in registry.backends(op))]
    if orphans:
        print(f"\nNOTE: {len(orphans)} op(s) have no reference backend and are "
              f"therefore unverifiable:\n  " + ", ".join(orphans) +
              "\n  Add one before writing a kernel for any of these.")
    return n_fail


if __name__ == "__main__":
    sys.exit(main())
