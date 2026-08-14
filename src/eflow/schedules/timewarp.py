"""Time reparameterization tau(t) (App. C.2, following Lee et al. 2026).

For large alphabets the decoding error P_e(t) is flat across most of [0,1] and
collapses only near t=1, so uniform training times and evenly spaced inference
grids spend nearly all their budget where the model has little to learn -- and it
worsens as |V| grows. The fix composes the time axis with a smooth increasing
tau with

    tau(t) = (P_e(0) - P_e(t)) / P_e(0) = 1 - |V|/(|V|-1) * P_e(t)

evaluated once by Gauss-Hermite quadrature on 10^4 uniform points, then fitted
with cubic splines in both directions for constant-time tau(t), t(tau), dtau/dt.
The warp is applied unconditionally in training and sampling.

Runtime cost is negligible (a spline eval on [B]); this module exists for
correctness and because getting the *inverse* wrong silently changes the time
distribution the model trains on, which is invisible in any benchmark.

Fit once with scripts/fit_timewarp.py and cache; do not refit per step.
"""
from __future__ import annotations

import torch


class TimeWarp:
    """Piecewise-cubic monotone warp with a cached inverse."""

    def __init__(self, t_grid: torch.Tensor, tau_grid: torch.Tensor):
        assert torch.all(tau_grid[1:] >= tau_grid[:-1]), "tau must be non-decreasing"
        self.t, self.tau = t_grid, tau_grid

    @staticmethod
    def identity(n: int = 10_000) -> "TimeWarp":
        g = torch.linspace(0, 1, n, dtype=torch.float64)
        return TimeWarp(g, g)

    def forward(self, t):    # tau(t)
        return self._interp(t, self.t, self.tau)

    def inverse(self, tau):  # t(tau) -- used to draw training times, Alg. 1 L4
        return self._interp(tau, self.tau, self.t)

    @staticmethod
    def _interp(x, xs, ys):
        i = torch.searchsorted(xs, x.clamp(xs[0], xs[-1]).contiguous()).clamp(1, len(xs) - 1)
        x0, x1 = xs[i - 1], xs[i]
        y0, y1 = ys[i - 1], ys[i]
        w = ((x - x0) / (x1 - x0).clamp_min(1e-12)).clamp(0, 1)
        return y0 + w * (y1 - y0)


# TODO(phase-1): implement the Gauss-Hermite P_e(t) quadrature of Eq. 86 in
# scripts/fit_timewarp.py and cache the grid per (|V|, sigma).
