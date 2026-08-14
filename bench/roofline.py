"""Analytic FLOP / byte model for every EFM op. Runs on CPU, needs no GPU.

Purpose: know what a kernel *should* cost before writing it, so a benchmark
number can be read as "3% of peak bandwidth" rather than "12 ms".  Every op in
ops/ should have an entry here, and bench_ops.py divides measured throughput by
these to report achieved fraction of roofline.

Run:  python -m bench.roofline --config lm1b
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict

GiB = 2 ** 30


@dataclass
class Config:
    name: str
    B: int; L: int; V: int; D: int; cond: int; blocks: int
    bytes_per_elem: int = 2       # bf16


CONFIGS = {
    # Tiny shapes for smoke-testing that the code RUNS. Not representative of
    # anything -- never quote a number measured at this config.
    "smoke":      Config("smoke", B=2, L=8, V=32, D=32, cond=16, blocks=2),
    # App. E.3 -- LM1B, bert-base-uncased vocab, DDiT backbone
    "lm1b":       Config("lm1b", B=128, L=128, V=30522, D=768, cond=128, blocks=12),
    # App. E.2 -- QM9 graphs, DeFoG graph transformer
    "qm9_graph":  Config("qm9_graph", B=256, L=9, V=4, D=256, cond=64, blocks=9),
    # App. E.1 -- GEOM-Drugs, GeoDiff dual encoder (atoms in place of L)
    "geom_drugs": Config("geom_drugs", B=128, L=181, V=3, D=128, cond=64, blocks=10),
}


def op_costs(c: Config) -> Dict[str, Dict[str, float]]:
    b = c.bytes_per_elem
    BLV = c.B * c.L * c.V
    BLD = c.B * c.L * c.D
    out: Dict[str, Dict[str, float]] = {}

    def add(op, bytes_ideal, bytes_naive, flops=0.0, note=""):
        out[op] = {"bytes_ideal": bytes_ideal, "bytes_naive": bytes_naive,
                   "traffic_overhang": bytes_naive / max(bytes_ideal, 1),
                   "flops": flops, "note": note}

    add("local_time", 2 * c.B * c.L * 4, 6 * c.B * c.L * 4, 3 * c.B * c.L,
        "launch-latency bound; fuse into interpolant")

    add("interpolant", BLV * b, (4 + 3) * BLV * b, 2 * BLV,
        "one_hot + randn + 2 lerp temporaries vs. 1 procedural write")

    add("expand_sampling", 2 * BLV * b, 4 * BLV * b, 0,
        "only ACTIVE rows need to move; naive touches full buffer")

    add("flow_map", 3 * BLV * b, 7 * BLV * b, 4 * BLV,
        "softmax(2 passes) + 2 mul + add vs. 1 streaming pass")

    add("semigroup_ce", 4 * BLV * b, 11 * BLV * b, 6 * BLV,
        "fwd+bwd; naive holds 6 x [B,L,V] live")

    add("adaln", 2 * BLD * b + c.B * c.L * 6 * c.D * b,
        4 * BLD * b + 2 * c.B * c.L * 6 * c.D * b,
        2 * c.B * c.L * c.cond * 6 * c.D,
        "per-position modulation: [B,L,6D] not [B,6D]")

    add("gap_counts", 2 * c.B * c.L * 8, 6 * c.B * c.L * 8, c.B * c.L,
        "3 launches on a tiny tensor -> pure launch overhead")

    add("insertion_sample", 3 * c.B * c.L * 4, 3 * c.B * c.L * 4, 5 * c.B * c.L,
        "loop form is O(L) DEPENDENT LAUNCHES; scan form is 3 launches")

    add("edge_symmetrize", 2 * c.B * c.L * c.L * 5 * b, 4 * c.B * c.L * c.L * 5 * b, 0,
        "strided transpose-add, applied after every jump")
    return out


def backbone_costs(c: Config) -> Dict[str, float]:
    attn = 4 * c.B * c.L * c.D * c.D * 2
    mlp = 2 * c.B * c.L * c.D * (4 * c.D) * 2
    scores = 2 * c.B * c.L * c.L * c.D * 2
    core = attn + mlp + scores
    mod_b = c.B * c.cond * 6 * c.D * 2
    mod_p = c.B * c.L * c.cond * 6 * c.D * 2
    head = c.B * c.L * c.D * c.V * 2
    return {
        "core_per_block_gflop": core / 1e9,
        "adaln_broadcast_per_block_gflop": mod_b / 1e9,
        "adaln_perpos_per_block_gflop": mod_p / 1e9,
        "adaln_perpos_pct_of_core": 100 * mod_p / core,
        "backbone_tflop": c.blocks * core / 1e12,
        "adaln_perpos_overhead_tflop": c.blocks * mod_p / 1e12,
        "adaln_perpos_act_gib": c.blocks * c.B * c.L * 6 * c.D * c.bytes_per_elem / GiB,
        "adaln_broadcast_act_gib": c.blocks * c.B * 6 * c.D * c.bytes_per_elem / GiB,
        "vocab_proj_tflop": head / 1e12,
        "blv_tensor_gib": c.B * c.L * c.V * c.bytes_per_elem / GiB,
        # Alg. 2 needs psi_su, psi_ut, psibar, logits, log_softmax, grad
        "alg2_naive_loss_residency_gib": 6 * c.B * c.L * c.V * c.bytes_per_elem / GiB,
        "alg2_fused_loss_residency_gib": 1 * c.B * c.L * c.V * c.bytes_per_elem / GiB,
        "alg2_fused_linear_ce_residency_gib": c.B * c.L * c.D * c.bytes_per_elem / GiB,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="lm1b", choices=list(CONFIGS))
    a = ap.parse_args()
    c = CONFIGS[a.config]
    print(f"=== {c.name}  B={c.B} L={c.L} V={c.V} D={c.D} blocks={c.blocks} ===\n")
    print(f"{'op':<20}{'ideal MiB':>11}{'naive MiB':>11}{'overhang':>10}  note")
    for op, d in op_costs(c).items():
        print(f"{op:<20}{d['bytes_ideal']/2**20:>11.1f}{d['bytes_naive']/2**20:>11.1f}"
              f"{d['traffic_overhang']:>9.1f}x  {d['note']}")
    print("\n--- backbone / residency ---")
    for k, v in backbone_costs(c).items():
        print(f"  {k:<40}{v:>12.3f}")


if __name__ == "__main__":
    main()
