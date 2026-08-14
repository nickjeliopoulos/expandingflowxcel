"""Attribute a full Alg. 2 (EFM training) step. This is the number that matters.

Alg. 2 needs FOUR network evaluations per step:
    psihat^{(t)}_{s,u}(x^eps_s)          target,  no grad
    psihat_{u,t}(Phi^{(t)}_{s,u})        target,  no grad
    psihat_{t,t}(x_t)                    diagonal target (or teacher), no grad
    psihat_{s,t}(x^eps_s)                student,  fwd + bwd
so a step is ~3 forwards + 1 forward/backward, i.e. roughly 3-4x a plain
denoiser step. This script measures the actual split rather than assuming it.

Two free systems wins to test here before writing any kernel:

  1. BATCH THE TARGET FORWARDS. The two stop-grad target passes have identical
     shapes and differ only in (s,t) conditioning. Concatenating them into one
     2B forward should improve occupancy and halve launch count. Costs nothing.

  2. GRADIENT SURGERY (E.3) merges L_DEFM and L_insert by projecting conflicting
     gradients, which needs per-loss gradients -- either two backwards or a
     fused dual accumulation. Time it separately; it must not hide inside the
     loss timing.

    python -m bench.bench_step --config lm1b --breakdown
"""
from __future__ import annotations

import argparse

import torch

from bench.harness import benchmark, write_results
from bench.roofline import CONFIGS

PHASES = ["interpolant", "expand_training", "target_fwd_su", "target_fwd_ut",
          "target_fwd_diag", "student_fwd", "loss", "backward", "insertion_head",
          "grad_surgery", "optimizer"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="lm1b", choices=list(CONFIGS))
    ap.add_argument("--batched-targets", action="store_true",
                    help="fuse the two stop-grad target forwards into one 2B pass")
    ap.add_argument("--breakdown", action="store_true")
    ap.add_argument("--out", default="results_step.json")
    a = ap.parse_args()
    raise SystemExit(
        "TODO(step-1): wire models/ddit_block.py + losses/ into a single "
        "Alg. 2 step, then time each phase in PHASES with nvtx ranges.\n"
        "Deliverable: a stacked bar of phase -> ms, at 1/2/4-step configs, "
        "with and without --batched-targets."
    )


if __name__ == "__main__":
    main()
