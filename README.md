# expandingflow

Reference operators and microbenchmarks for **Expanding Flow Maps** (EFM) —
Tang & Chatterjee, [arXiv:2607.21585](https://arxiv.org/abs/2607.21585).

EFM factors the map between two timesteps into a learned **expand operator**
(which grows the state with conditional noise) and a **transport map** (which
denoises the expanded state). This repository implements those operators and
measures what they cost.

**It is not a training framework.** There is no dataloader, no training loop, no
checkpointing, and no evaluation metric — by design. The goal is per-operator
characterization to drive kernel design, not end-to-end reproduction. See
[`docs/PLAN.md`](docs/PLAN.md) for the operator inventory, the analytic cost
model, and the build order.

## Install

```bash
uv sync                          # core + dev group
uv sync --extra triton           # once you start writing kernels
uv sync --extra report           # pandas/matplotlib, for benchmark tables
```

## Quick start

```bash
# 1. Does it run here at all? Tiny shapes, seconds, no GPU required.
python scripts/smoke.py

# 2. Analytic FLOP/byte model. CPU-only -- no GPU, no torch import.
python -m bench.roofline --config lm1b

# 3. Correctness. Every backend is checked against a float64 reference.
pytest                            # preferred
python scripts/run_tests.py       # same suite, no pytest needed (torch only)

# Op-level sweep across every registered backend.
python -m bench.bench_ops --config lm1b --ops interpolant,semigroup_ce,adaln
```

## Layout

| path | what |
|---|---|
| `src/eflow/ops/` | every microbenchmark target, one file per operator |
| `src/eflow/ops/registry.py` | the op × backend seam |
| `src/eflow/reference/` | literal transcription of Alg. 1–4; the correctness oracle |
| `src/eflow/schedules/` | α(t), hazard, inverse CDF, τ(t) time warp |
| `src/eflow/models/` | TinyDDiT + insertion head; stubs for borrowed backbones |
| `scripts/smoke.py` | "does this execute here" check; run this first on a new box |
| `scripts/run_tests.py` | runs tests/ without pytest installed |
| `bench/roofline.py` | analytic cost model, runs on CPU |
| `bench/harness.py` | timing + memory core, deliberately thin |
| `bench/adapters.py` | plug external profiling tools in here |
| `tests/test_identities.py` | Prop. 4.1 / 5.1 residuals against an exact oracle |
| `docs/PLAN.md` | design document and build order |

Each op module's docstring carries the paper equation numbers, the traffic
accounting, and the kernel contract. **Read those before optimizing anything** —
several ops look expensive and are not, and one looks cheap and is not.

## Three conventions worth knowing before you edit

**1. Fixed-size buffers, never ragged tensors.** The paper's `x_s` has length
`d(s)`; we always carry `[B, L, V]` plus an `active` mask. Dynamic shapes defeat
CUDA graphs and `torch.compile`, and would make every measurement a recompile.
Alg. 2 L12 already "expands once to `d(t)`", so the training path loses nothing.
The padding waste this creates is an experimental variable (ablation A1), not a
bug to quietly fix.

**2. Every op has a `reference` backend, and it is never benchmarked.** Slow,
loop-based, float64, transcribed line by line from the algorithm boxes. It is
the oracle for `tests/test_backend_parity.py`. When a fast path and the
reference disagree, the fast path is wrong — tolerances do not get relaxed to
make a kernel pass.

**3. The registry is the only extension point.** Adding a backend is two lines,
and it inherits a correctness test and a row in every benchmark sweep:

```python
from eflow.ops.registry import register, requires_triton

@register("interpolant", "triton", available=requires_triton)
def interpolant_triton(labels, t_local, active, V, sigma=1.0, **kw):
    ...
```

External profiling tools attach the same way, via `bench/adapters.py`: an
adapter is a context-manager factory whose object exposes `.summary()`, and its
output lands in `result.extra[name]`. `harness.py` never needs to change.

## Status

Phase 0 complete. `scripts/smoke.py` and `bench/roofline.py` verified on an
RTX 3090 Ti (torch 2.5.1+cu121, no Triton): 31 backends execute, 0 fail.

**No timing has been measured yet.** Every number in `docs/PLAN.md` is analytic,
not observed. Treat them as predictions to falsify, not results.

Known environment gaps on that box, both expected and both handled by
`available()` guards rather than failures:

* **no Triton** — every `triton` backend skips.
* **`torch.compile` is dead**, and not only on GPU: inductor fails at the C++
  stage (no reachable MSVC `cl`), so the CPU path is out too. This matters more
  than it looks — `compile` is the baseline a hand-written kernel has to beat,
  so it must work on whatever machine the real benchmarks run on, or every
  reported speedup is measured against eager and is therefore flattering.

Four of those predictions, in profit order:

1. **The truncated binomial draw of Alg. 3 L4 is not sequential.** Left-to-right
   truncation collapses to `min(cumsum(ℓ), B)` differencing — a prefix sum, a
   clamp, and a diff instead of `L+1` dependent kernel launches. Proof sketch in
   `src/eflow/ops/insertion_sample.py`; verified exhaustively in
   `tests/test_insertion.py`. Already implemented.
2. **The consistency loss holds ~5.6 GiB it does not need** at the LM1B config —
   six live `[B,L,V]` tensors. Since `dL/dlogits = w·(softmax(logits) − ψ̄)`, a
   streaming kernel stores nothing for backward.
3. **Per-position adaLN (App. C.3) costs +8% FLOPs and +1.7 GiB activations**
   versus the standard broadcast form — the largest unadvertised cost in the
   method, and the clearest architecture/systems co-design lever.
4. **Padding waste is worst exactly where the method spends its time** (~95% at
   `t=0.1` under the E.3 cosine schedule). Active positions are contiguous after
   compaction, so varlen attention applies directly — a layout decision, not a
   kernel.

## When the authors release their code

`ExpandingFlowMaps` is expected at
[github.com/sophtang/ExpandingFlowMaps](https://github.com/sophtang/ExpandingFlowMaps).
`src/eflow/reference/ref_seq.py` is a literal transcription of Alg. 1–4
specifically so a line-by-line diff localizes any divergence to a numbered
equation. `docs/PLAN.md` §7 lists the five places divergence is most likely.
