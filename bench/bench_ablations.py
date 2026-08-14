"""The experiments that turn microbenchmarks into a co-design argument.

Each ablation isolates one design choice the paper makes implicitly and prices
it. These are the plots that go in a paper.

A1  padded vs. varlen
    The fixed-buffer state pads to L, but only alpha(t)*L positions are active.
    Under the LM1B cosine schedule with t_ins_end = 0.5:
        t=0.10  active   6/128   dense waste 95%   attention waste 99.8%
        t=0.20  active  24/128   dense waste 81%   attention waste 96.4%
        t=0.30  active  53/128   dense waste 59%   attention waste 83.0%
        t=0.40  active  88/128   dense waste 31%   attention waste 52.3%
        t>=0.50 active 128/128   -- pure denoising phase, no waste
    Since active positions are contiguous after compaction (Eq. 24), varlen
    FlashAttention applies directly. Measure the recoverable fraction as a
    function of t and of t_ins_end.

A2  per-position vs. broadcast adaLN
    +8.1% FLOPs and +1.7 GiB activations at the LM1B config (bench/roofline.py).
    Race: full per-position / low-rank modulation / shared-across-blocks /
    gather-from-distinct-times (few distinct t_i early in the trajectory).

A3  binomial vs. Poisson insertion (Eq. 28)
    Cost and variance vs. step count. The paper argues Poisson for many-step and
    binomial for few-step; verify the crossover empirically.

A4  loss residency: naive / chunked / fused / fused-linear-CE
    6 x [B,L,V] -> 1x -> [B,L,D]. Report peak allocated and ms.

A5  expand: training (mask-only) vs. sampling (data-moving)
    Confirm the training path really is free and quantify the sampling path.

A6  CUDA graphs on/off for the 1-step sampling call
    The insertion path is launch-latency bound; 1-step generation is the
    headline use case. Measure how much of it is pure launch overhead.

A7  edge storage: full symmetric vs. upper-triangular only (graph path)
    Halves traffic and makes the D.6 invariant structural rather than enforced.
    Sweep L in {9, 32, 64, 128, 181} to find where it starts to matter.
"""
from __future__ import annotations

ABLATIONS = ["padded_vs_varlen", "adaln_variants", "binomial_vs_poisson",
             "loss_residency", "expand_paths", "cuda_graphs", "edge_storage"]

if __name__ == "__main__":
    raise SystemExit("TODO(step-2): implement ablations in the order listed in PLAN.md")
