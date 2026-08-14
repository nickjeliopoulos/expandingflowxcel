"""Insertion schedules alpha(t) and the derived hazard / inverse CDF.

alpha(t) = Pr[t_ins_i <= t]  (Eq. 23), with alpha(0)=0, alpha(1)=1.

Three quantities are needed and are easy to get wrong independently, so each
schedule supplies all three and ``tests/test_schedules.py`` checks them against
finite differences and against empirical samples:

  alpha(t)          cumulative insertion fraction
  hazard(t)         lambda_t = alpha'(t) / (1 - alpha(t))   -- Eq. 25
  inverse_cdf(u)    t_ins = alpha^{-1}(u),  u ~ U(0,1)      -- Alg. 1 L6

The insertion cutoff of App. C.1 is applied uniformly by reparameterising
alpha_{t_end}(t) = alpha(min(t / t_end, 1)), so no schedule implements it
itself.  ``t_ins_end`` values from the paper: 0.5 (LM1B, E.3), 0.6 (conformer,
E.1), 1.0 (QM9 graph, E.2).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

_EPS = 1e-7


@dataclass
class Schedule:
    t_ins_end: float = 1.0

    # --- subclasses implement these on the *unclipped* [0,1] axis ---
    def _alpha(self, t: torch.Tensor) -> torch.Tensor: ...
    def _alpha_prime(self, t: torch.Tensor) -> torch.Tensor: ...
    def _inv(self, u: torch.Tensor) -> torch.Tensor: ...

    # --- public API applies the cutoff of App. C.1 ---
    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        return self._alpha((t / self.t_ins_end).clamp(max=1.0))

    def alpha_prime(self, t: torch.Tensor) -> torch.Tensor:
        inside = t < self.t_ins_end
        return torch.where(inside,
                           self._alpha_prime((t / self.t_ins_end).clamp(max=1.0)) / self.t_ins_end,
                           torch.zeros_like(t))

    def hazard(self, t: torch.Tensor) -> torch.Tensor:
        """lambda_t = alpha'(t) / (1 - alpha(t)); the diagonal weight in Eq. 26."""
        return self.alpha_prime(t) / (1.0 - self.alpha(t)).clamp_min(_EPS)

    def sample_t_ins(self, shape, device=None, generator=None) -> torch.Tensor:
        u = torch.rand(shape, device=device, generator=generator)
        return self._inv(u) * self.t_ins_end

    def rho(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """rho_{s,t} = (alpha_t - alpha_s)/(1 - alpha_s): fraction of the
        remaining tokens inserted over (s, t].  Eq. 25 / Alg. 2 L18."""
        a_s, a_t = self.alpha(s), self.alpha(t)
        return ((a_t - a_s) / (1.0 - a_s).clamp_min(_EPS)).clamp(0.0, 1.0)


class CosineSchedule(Schedule):
    """alpha(t) = 1 - cos(pi/2 * t).  Used for LM1B (E.3) and conformers (E.1).
    Median insertion time is (2/3) * t_ins_end, as stated in E.3."""

    def _alpha(self, t):        return 1.0 - torch.cos(math.pi / 2 * t)
    def _alpha_prime(self, t):  return (math.pi / 2) * torch.sin(math.pi / 2 * t)
    def _inv(self, u):          return (2 / math.pi) * torch.arccos((1.0 - u).clamp(-1, 1))


@dataclass
class PolynomialSchedule(Schedule):
    """The rho(t) = (r t^{r-1})/(1 - t^r) form of E.2 integrates to
    alpha(t) = t^r.  r = 0.5 for QM9 graphs, giving median t_ins = 0.25."""
    r: float = 0.5

    def _alpha(self, t):        return t.clamp_min(0).pow(self.r)
    def _alpha_prime(self, t):  return self.r * t.clamp_min(_EPS).pow(self.r - 1)
    def _inv(self, u):          return u.pow(1.0 / self.r)


class LinearSchedule(Schedule):
    def _alpha(self, t):        return t
    def _alpha_prime(self, t):  return torch.ones_like(t)
    def _inv(self, u):          return u


SCHEDULES = {"cosine": CosineSchedule, "polynomial": PolynomialSchedule,
             "linear": LinearSchedule}
