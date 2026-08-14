# `expanding-flow-maps-bench` — Repository Plan

Reference implementation and microbenchmark harness for **Expanding Flow Maps**
(Tang & Chatterjee, arXiv:2607.21585).

**Scope discipline.** This repo implements the *operators*, not the *system*.
There is no dataloader, no training loop, no checkpointing, no evaluation
metric, no distributed anything. Every file exists either to (a) define an
operator correctly, (b) prove it is correct, or (c) measure it. If a file does
none of those three things it does not belong here.

The deliverable is a characterization: for each novel operator, what it costs,
why, and what the ceiling is. Kernels come after that, informed by it.

---

## 1. What is actually novel here

Stripping away the borrowed backbones (GeoDiff dual-encoder, DeFoG graph
transformer, DDiT), EFM contributes exactly three mechanisms:

1. **Per-position local time.** Each coordinate carries its own clock
   `t_i = (t - t_i^ins)/(1 - t_i^ins)` (Eq. 12). This is what makes the state
   ragged in time as well as in length, and it is what forces conditioning to
   become per-position.
2. **The expand operator.** A learned, stochastic, count-predicting insertion of
   new coordinates into an existing state (Eq. 27–28, Alg. 3).
3. **The two-time semigroup objective on the expanded state.** Cross-entropy
   against a convex mixture of two stop-grad denoiser outputs, evaluated once at
   the terminal dimension `d(t)` (Eq. 32–33, Alg. 2).

Everything expensive traces back to one of those three. The benchmark plan is
organized around them.

---

## 2. Operator inventory

Shapes at the LM1B config from App. E.3 (`B=128, L=128, V=30522, D=768,
cond=128, 12 blocks`, bf16). One `[B,L,V]` tensor is **0.93 GiB**.

| ID | Operator | Paper ref | Shape | Character | Why it matters |
|----|----------|-----------|-------|-----------|----------------|
| OP-1 | Local time + activity mask | Eq. 12/83 | `[B,L]` | launch-bound | dependency of every fused kernel |
| OP-2 | Expanding interpolant | Eq. 84/93, Alg.1 L8 | `[B,L]`→`[B,L,V]` | **memory-bound, 7× traffic overhang** | cheapest large win |
| OP-3 | Expand operator (seq) | Eq. 27, Alg. 3 | `[B,L,V]` permute | memory-bound, irregular | training path should be free; sampling path is not |
| OP-3g | Expand operator (graph) | Eq. 97–98 | `[B,L,L,Ve]` | blocked scatter | only the A×A block moves |
| OP-4 | Gap counts + Poisson NLL | Eq. 25–26, 96 | `[B,L+1]` | launch-bound | 3 launches on a tiny tensor |
| OP-5 | Truncated binomial sampling | Eq. 28, Alg. 3 L4 | `[B,L+1]` | **naively O(L) dependent launches** | see §4.1 — collapses to a cumsum |
| OP-6 | Flow map affine combine | Eq. 30/100 | `[B,L,V]` | memory-bound, 2.3× | 5 passes → 1 |
| OP-7 | Semigroup CE + adaptive weight | Eq. 32–33, 87 | `[B,L,V]` ×6 | **5.6 GiB live residency** | flagship kernel |
| OP-8 | Per-position adaLN | App. C.3 | `[B,L,6D]` | +8.1% FLOPs, +1.7 GiB | biggest unadvertised cost |
| OP-9 | Edge time / symmetrize | Eq. 92/94, D.6 | `[B,L,L,Ve]` | strided transpose-add | quadratic; scaling question |
| OP-10 | Child expand + COM projection | Sec. 3(iii), E.1 | `[B,A,3]` | ragged segmented reduction | every step, ragged batch |
| OP-11 | Time reparameterization `τ(t)` | App. C.2 | scalar | negligible | correctness only |

Analytic model in `bench/roofline.py`; the numbers above are its output, not
estimates typed by hand.

---

## 3. Repository structure

```
expandingflow/
├── README.md
├── pyproject.toml                  # hatchling, src layout, uv dependency-groups
├── requirements.txt                # mirror of [project.dependencies]
├── docs/PLAN.md                    # this file
├── src/eflow/
│   ├── state.py                    # SeqState / GraphState / PointCloudState
│   ├── schedules/
│   │   ├── insertion.py            # alpha(t), hazard, inverse CDF, rho_{s,t}
│   │   ├── timewarp.py             # tau(t) reparameterization (App. C.2)
│   │   └── interp.py               # beta(t) mixing coefficient (Eq. 85)
│   ├── ops/                        # ← every microbenchmark target
│   │   ├── registry.py             # THE SEAM: op × backend, auto-tested
│   │   ├── local_time.py           interpolant.py        expand.py
│   │   ├── expand_graph.py         expand_child.py       gap_counts.py
│   │   ├── poisson_nll.py          insertion_sample.py   flow_map.py
│   │   ├── semigroup_ce.py         adaln.py              edge_ops.py
│   ├── losses/{diagonal,consistency,insertion,weighting}.py
│   ├── models/{ddit_block,insertion_head,time_embed,stubs}.py
│   ├── sampling/{seq,graph}.py     # Alg. 4
│   └── reference/{ref_seq,ref_graph}.py   # literal Alg. 1–4 transcription
├── bench/
│   ├── harness.py                  # thin core: CUDA events, peak memory
│   ├── adapters.py                 # ← your profiling tools plug in here
│   ├── roofline.py                 # analytic FLOP/byte model, CPU-only
│   ├── bench_ops.py  bench_step.py  bench_sampling.py  bench_ablations.py
│   └── report.py
├── tests/
│   ├── test_identities.py          # Prop. 4.1 / 5.1 residuals on an exact oracle
│   ├── test_insertion.py           # scan == loop, exhaustively
│   ├── test_backend_parity.py      # every backend vs. its reference, automatic
│   ├── test_schedules.py  test_expand.py  test_graph_equivariance.py
└── scripts/{fit_timewarp.py,run_all_benchmarks.sh}
```

Benchmark shapes live in `bench/roofline.py::CONFIGS` as the single source of
truth, not in a YAML config tree. There are three of them (`lm1b`, `qm9_graph`,
`geom_drugs`) and they are read by both the analytic model and the sweeps;
splitting them across two formats would guarantee they drift.

### Three structural decisions worth arguing about

**(a) Fixed-size buffers, never ragged tensors.** The paper's `x_s` has length
`d(s)`; we always carry `[B, L, V]` plus an `active` mask. Dynamic shapes defeat
CUDA graphs and `torch.compile`, and would make every measurement a recompile.
Alg. 2 L12 already "expands once to `d(t)`", so the training path loses nothing.
The padding waste this creates is not a bug to hide — it is ablation A1.

**(b) A `reference` backend for every op, and it is never benchmarked.** Slow,
loop-based, float64, transcribed literally from the paper's algorithm boxes. It
is the oracle for `test_backend_parity.py`. The moment a fast path and the
reference disagree, the fast path is wrong — no negotiating tolerances upward.

**(c) The registry is the only extension point.** Adding a Triton kernel is
`@register("interpolant", "triton")` and nothing else: it inherits a correctness
test and a row in every benchmark sweep. No config system, no plugin discovery.
`bench/adapters.py` is the matching seam for your external profiling tools —
they wrap the measured region without `harness.py` knowing they exist.

---

## 4. Findings already available from the analytic model

These come out of reading the paper and running `bench/roofline.py`; they shape
the kernel priority order before a single GPU hour is spent.

### 4.1 The truncated binomial draw is not sequential — *verified*

Alg. 3 L4 says draw `l_i ~ Binomial(...)` and "truncate the joint draw
left-to-right so that `Σ l_i ≤ L - n_s`". The literal reading is a loop over
`L+1 = 129` gaps carrying a running budget — 129 **dependent** kernel launches
per sampling step, on the critical path of the 1-step generation that is the
method's headline result.

It is not sequential. Left-to-right truncation is `S_i = min(S_{i-1} + l_i, B)`,
`l'_i = S_i - S_{i-1}`. The family `f_c(x) = min(x + c, B)` with `c ≥ 0` is
closed under composition:

```
f_c2(f_c1(x)) = min(min(x+c1, B) + c2, B) = min(x + c1 + c2, B + c2, B)
              = min(x + (c1+c2), B)                        since c2 ≥ 0
```

so composing gaps just adds counts, and therefore `S_i = min(cumsum(l)_i, B)`.
The whole truncation is **a prefix sum, a clamp, and a diff** — three vectorized
ops, no loop, no scan of composed maps.

Verified exact against the loop over 20,000 random `(counts, budget)` cases; the
check lives in `tests/test_insertion.py::test_scan_equals_loop_exhaustively`.
Already implemented as the `torch` backend. This one is free.

### 4.2 The loss holds 5.6 GiB it does not need

A naive autograd implementation of Eq. 32–33 + 87 holds six live `[B,L,V]`
tensors simultaneously: `psi_su`, `psi_ut`, `psibar`, student logits,
`log_softmax`, and the gradient w.r.t. logits. That is **5.59 GiB for the loss
alone**, comparable to the rest of the model.

None of it is required. The backward through soft-target CE with a stop-grad
target is `dL/dlogits = w · (softmax(logits) - psibar)`, which depends only on
`logits` and `psibar`. So a streaming kernel can compute logsumexp online, form
the mixture in registers as it reads `psi_su`/`psi_ut`, accumulate the Eq. 87
`‖Δ‖²` reduction in the *same* pass, and store nothing for backward:

| variant | loss residency | note |
|---|---|---|
| naive autograd | 5.59 GiB | 6 × `[B,L,V]` |
| chunked over L | ~0.4 GiB | no kernel needed — do this first |
| fused streaming CE | 0.93 GiB | logits only |
| + fused vocab projection | **0.023 GiB** | Cut-Cross-Entropy style; ~240× |

The chunked variant is already written (`semigroup_ce_chunked`) precisely so the
Triton kernel has an honest baseline to beat rather than being credited for a
win that was just better scheduling.

### 4.3 Per-position adaLN is the biggest unadvertised cost

App. C.3 makes source-time conditioning **per position**, so modulation is
`[B,L,6D]` instead of `[B,6D]`. At the E.3 config:

```
per-block core (attn qkvo + mlp + scores)   238.4 GFLOP
modulation, broadcast (standard DiT)          0.15 GFLOP   (0.06%)
modulation, per-position (EFM)               19.3  GFLOP   (8.1%)

modulation activations, broadcast            13.5 MiB total
modulation activations, per-position          1.69 GiB total   (~128×)
```

~8% extra FLOPs is tolerable; 1.7 GiB of extra live activations is not, and the
conversion of a free broadcast into a materialized elementwise op on the largest
hidden tensor in each block is a real architectural cost the paper does not
price. Four mitigations are worth racing (ablation A2), and one is genuinely
interesting: under a fixed insertion grid the number of *distinct* local times
at global time `t` is bounded by the number of insertion events so far, which is
`≪ L` early in the trajectory — so the modulation GEMM could be computed once
per distinct time and gathered.

**Counterpoint worth keeping honest:** the per-position backward is *easier*,
not harder, than the broadcast one — `dscale`/`dshift` are `[B,L,D]` with no
cross-position reduction. Some of the forward cost is bought back.

### 4.4 Padding waste is severe exactly where the method spends its time

Under the E.3 cosine schedule with `t_ins_end = 0.5`:

| global `t` | `α(t)` | active | dense waste | attention waste |
|---|---|---|---|---|
| 0.10 | 0.049 | 6/128 | 95.1% | 99.8% |
| 0.20 | 0.191 | 24/128 | 80.9% | 96.4% |
| 0.30 | 0.412 | 53/128 | 58.8% | 83.0% |
| 0.40 | 0.691 | 88/128 | 30.9% | 52.3% |
| ≥0.50 | 1.000 | 128/128 | 0% | 0% |

Half the time axis is in the insertion phase, and early in it the model is doing
almost no useful work per FLOP. Because active positions are *contiguous* after
compaction (Eq. 24), varlen FlashAttention applies directly with no exotic
masking. This is likely the single largest end-to-end win available and it needs
no custom kernel — it needs a layout decision.

### 4.5 A training step is 3–4× a denoiser step, and two of the passes are free to batch

Alg. 2 needs four network evaluations: `psihat^(t)_{s,u}`, `psihat_{u,t}(Φ)`,
`psihat_{t,t}` (diagonal target), and the student `psihat_{s,t}`. Three are
stop-grad. The two target forwards have **identical shapes** and differ only in
`(s,t)` conditioning, so concatenating them into a single `2B` forward should
improve occupancy and halve launch count at zero cost. Test this before writing
any kernel — it is the cheapest thing on the list after §4.1.

Separately, E.3 merges `L_DEFM` and `L_insert` by **gradient surgery**, which
needs per-loss gradients and therefore either two backward passes or a fused
dual accumulation. Time it explicitly; it must not hide inside the loss number.

---

## 5. Build order

Each phase ends in something measurable. Do not start a phase before the
previous one's tests pass.

**Phase 0 — skeleton (½ day).** `state.py`, `registry.py`, `schedules/`,
`harness.py`, `roofline.py`. Exit: `python -m bench.roofline --config lm1b` runs
and `pytest tests/test_schedules.py` passes. *(Done — written and syntax-checked,
but not yet executed against a real torch install. First job on a GPU box is to
run `scripts/run_all_benchmarks.sh` and fix whatever falls over.)*

**Phase 1 — sequence path, reference-first (2 days).** Literal transcription of
Alg. 1–4 into `reference/ref_seq.py`, then `torch` backends for OP-1..OP-7.
Exit: `test_identities.py` and `test_backend_parity.py` green; `bench_ops.py`
produces a table for every op.

**Phase 2 — model in context (1–2 days).** `TinyDDiT` + `InsertionHead`, then
`bench_step.py` with per-phase nvtx ranges. Exit: a stacked bar of Alg. 2 phase →
ms at 1/2/4 steps, with and without batched target forwards.

**Phase 3 — free wins, no kernels (1 day).** Ablations A1 (varlen), A4 (chunked
loss), A5 (expand paths), A6 (CUDA graphs), plus the batched-target-forward test.
Exit: a documented speedup with zero Triton written. *This phase is the control
group for everything after it.*

**Phase 4 — graph + continuous paths (2 days).** OP-3g, OP-9, OP-10. Exit:
`test_graph_equivariance.py` green; edge-storage ablation A7 swept over
`L ∈ {9, 32, 64, 128, 181}`.

**Phase 5 — kernels, in profit order.** Only now, and only against Phase-3
baselines:
1. `semigroup_ce` fused streaming CE → then fused-linear-CE (§4.2)
2. `interpolant` procedural-noise write (§4.2 of the op table, 7× overhang)
3. `adaln` fused LN + per-position affine (§4.3)
4. `flow_map` fused softmax + affine
5. `expand_sampling` fused permute + procedural fill
6. graph `symmetrize` tiled transpose-add

Every kernel lands with a parity test against the float64 reference and a
before/after row in the benchmark table. No exceptions.

---

## 6. Deliberate non-goals

- **No training script, no dataloader, no eval metrics.** If a number requires
  a trained checkpoint it is out of scope for this repo.
- **No reimplementation of GeoDiff or DeFoG.** Shape-faithful stubs in
  `models/stubs.py`, clearly labelled. They are not EFM contributions and
  benchmarking them as such would be misleading.
- **No premature Triton.** Phase 3 exists to establish how much of the
  apparent kernel opportunity is really a scheduling or layout problem. Some of
  it will be.
- **No accuracy claims.** This repo measures cost. Quality comparisons need the
  authors' code or a full training run, and we have neither yet.

---

## 7. When the authors release their code

The repo is arranged so that the reference backends become the diff target.
`reference/ref_seq.py` is a literal transcription of Alg. 1–4, so a
line-by-line comparison against the released implementation localizes any
divergence to a specific equation. The things most likely to differ, and worth
checking first:

1. Whether `psi^(t)_{s,u}` is evaluated in the *expand-once* form (Eq. 82) or the
   native form (Eq. 81) — the paper says Alg. 2 uses expand-once, and they are
   equal only under assumptions A2–A3.
2. The exact clamping in `phi(a,b)` at `a = 0`, which is where the insertion loss
   is numerically fragile.
3. Whether the diagonal target `D_t` is a teacher denoiser or the student's own
   diagonal (Eq. 32 permits both; E.2 and E.3 make different choices).
4. Whether gradient surgery is applied per-parameter or per-tensor.
5. The `t_denoise-min` value for one-step sampling (App. C.5) — never stated
   numerically.
