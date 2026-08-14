"""Op-level sweeps. Every registered backend x every shape in the sweep.

    python -m bench.bench_ops --config lm1b --ops interpolant,semigroup_ce
    python -m bench.bench_ops --sweep V --op semigroup_ce
"""
from __future__ import annotations

import argparse

import torch

from bench.harness import benchmark, write_results
from bench.roofline import CONFIGS, op_costs
from eflow.ops import registry

registry.load_all()


def make_inputs(op, c, device, dtype):
    B, L, V, D = c.B, c.L, c.V, c.D
    g = torch.Generator(device=device).manual_seed(0)
    if op == "interpolant":
        return dict(labels=torch.randint(0, V, (B, L), device=device),
                    t_local=torch.rand(B, L, device=device, dtype=dtype),
                    active=torch.rand(B, L, device=device) > 0.4, V=V)
    if op == "semigroup_ce":
        mk = lambda: torch.softmax(torch.randn(B, L, V, device=device, dtype=dtype), -1)
        return dict(student_logits=torch.randn(B, L, V, device=device, dtype=dtype,
                                               requires_grad=True),
                    psi_su=mk(), psi_ut=mk(),
                    s=torch.tensor(0.2, device=device), u=torch.tensor(0.5, device=device),
                    t=torch.tensor(0.8, device=device),
                    mask=(torch.rand(B, L, device=device) > 0.3).to(dtype))
    if op == "flow_map":
        return dict(logits=torch.randn(B, L, V, device=device, dtype=dtype),
                    x=torch.randn(B, L, V, device=device, dtype=dtype),
                    s=torch.tensor(0.2, device=device), t=torch.tensor(0.8, device=device))
    if op == "adaln":
        return dict(x=torch.randn(B, L, D, device=device, dtype=dtype),
                    shift=torch.randn(B, L, D, device=device, dtype=dtype),
                    scale=torch.randn(B, L, D, device=device, dtype=dtype))
    if op == "gap_counts":
        return dict(active=torch.rand(B, L, device=device) > 0.4)
    if op == "insertion_sample":
        return dict(I_hat=torch.rand(B, L + 1, device=device, dtype=torch.float32) * 2,
                    budget=torch.randint(1, L, (B,), device=device))
    raise KeyError(op)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="lm1b", choices=list(CONFIGS))
    ap.add_argument("--ops", default="interpolant,flow_map,semigroup_ce,adaln,"
                                     "gap_counts,insertion_sample")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--backward", action="store_true",
                    help="time fwd+bwd; the loss ops are ~half backward")
    ap.add_argument("--out", default="results_ops.json")
    a = ap.parse_args()

    c = CONFIGS[a.config]
    dtype = getattr(torch, a.dtype)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    costs = op_costs(c)
    results = []

    for op in a.ops.split(","):
        kw = make_inputs(op, c, device, dtype)
        for impl in registry.backends(op):
            if impl.reference:
                continue          # reference is O(B*L) python; never benchmark it
            why = impl.available()
            if why:
                print(f"skip {op}/{impl.backend}: {why}")
                continue

            def run(fn=impl.fn, kw=kw):
                out = fn(**kw)
                if a.backward:
                    t = out if torch.is_tensor(out) else out[0]
                    (t.sum() if t.ndim else t).backward(retain_graph=True)

            try:
                r = benchmark(run, op=op, backend=impl.backend,
                              shape={"B": c.B, "L": c.L, "V": c.V, "D": c.D},
                              dtype=dtype, device=device,
                              bytes_moved=costs.get(op, {}).get("bytes_ideal"),
                              flops=costs.get(op, {}).get("flops"),
                              note=impl.note)
            except Exception as e:                      # OOM is itself a result
                print(f"FAIL {op}/{impl.backend}: {type(e).__name__}: {e}")
                continue
            results.append(r)
            print(f"{op:<18}{impl.backend:<10}{r.ms_median:8.3f} ms  "
                  f"{r.peak_alloc_mib:8.1f} MiB  "
                  f"{(r.achieved_gbps or 0):7.1f} GB/s")

    write_results(results, a.out)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
