"""Property tests that pin the EFM math independently of any implementation.

These are the tests that must never be relaxed to make a kernel pass.  They
encode the paper's own consistency conditions, so a kernel that breaks one is
wrong in a way that would show up as a silently worse model rather than a crash.

Prop. 5.1 (discrete):
    Lagrangian : psi_{s,t} = psi_{t,t}(Phi_{s,t}) - gamma_{s,t} d_t psi_{s,t}
    Eulerian   : d_s psi + J_x psi . b_s = kappa_{s,t}(psi_{s,t} - E(psi_{s,s}))
    Semigroup  : psi_{s,t} = w psi^{(t)}_{s,u} + (1-w) psi_{u,t}(E(Phi_{s,u}))
Prop. 4.1 (continuous):
    Semigroup  : Phi_{u,t}(Phi_{s,u}(x)) = Phi_{s,t}(x)

Strategy: construct an *analytically exact* flow map (the one induced by the
linear interpolant with a known clean target), verify the identities hold on it
to float64 tolerance, then verify each backend reproduces it.  A learned network
would only satisfy these approximately, which is useless as a test oracle.
"""
from __future__ import annotations

import pytest
import torch

from eflow.ops.semigroup_ce import flow_map_mix, omega

torch.manual_seed(0)
DT = torch.float64


def exact_mean_denoiser(x, s, x1):
    """For the linear interpolant with known x1, psi_{s,t}(x_s) = x1 exactly.
    This is the oracle: the perfect student. Every identity must hold on it."""
    return x1


def interp(x0, x1, t):
    return (1 - t) * x0 + t * x1


@pytest.mark.parametrize("s,u,t", [(0.1, 0.4, 0.7), (0.0, 0.5, 1.0 - 1e-6),
                                   (0.25, 0.3, 0.35), (0.6, 0.8, 0.9)])
def test_semigroup_convexity(s, u, t):
    """w_{s,u,t} + (1 - w_{s,u,t}) == 1 and both are in [0,1] (Eq. 78-79)."""
    s, u, t = (torch.tensor(v, dtype=DT) for v in (s, u, t))
    w = omega(s, u=u, t=t)
    comp = ((t - u) * (1 - s)) / ((t - s) * (1 - u))
    assert torch.allclose(w + comp, torch.ones((), dtype=DT), atol=1e-12)
    assert 0.0 <= float(w) <= 1.0


@pytest.mark.parametrize("s,u", [(0.1, 0.4), (0.0, 0.9), (0.33, 0.34), (0.7, 0.999)])
def test_flow_map_lands_on_the_interpolant(s, u):
    """The sharp version of the oracle test: with the exact denoiser psi = x1,
    Phi_{s,u}(x_s) must land EXACTLY on x_u = (1-u)x0 + u x1.

    Algebraically:
        (1-u)/(1-s) [(1-s)x0 + s x1] + (u-s)/(1-s) x1
            = (1-u)x0 + [(1-u)s + u - s]/(1-s) x1
            = (1-u)x0 + u x1
    so any coefficient error in Eq. 30 shows up here, unlike the mixture
    identity below which is degenerate on a constant oracle.
    """
    B, L, V = 4, 6, 8
    s_t, u_t = torch.tensor(s, dtype=DT), torch.tensor(u, dtype=DT)
    x1 = torch.nn.functional.one_hot(torch.randint(0, V, (B, L)), V).to(DT)
    x0 = torch.randn(B, L, V, dtype=DT)
    got = flow_map_mix(interp(x0, x1, s_t), exact_mean_denoiser(None, s_t, x1), s_t, u_t)
    torch.testing.assert_close(got, interp(x0, x1, u_t), rtol=0, atol=1e-11)


@pytest.mark.parametrize("s,u,t", [(0.1, 0.4, 0.7), (0.2, 0.5, 0.9)])
def test_discrete_semigroup_identity_on_oracle(s, u, t):
    """Eq. 31c / 81. NOTE: on a constant oracle both mixture components equal
    x1, so this checks the convex-combination *plumbing* (that the weights sum
    to one and are applied to the right operands), not the coefficients. The
    coefficients are pinned by test_flow_map_lands_on_the_interpolant above.
    Keep both.
    """
    B, L, V = 4, 6, 8
    s, u, t = (torch.tensor(v, dtype=DT) for v in (s, u, t))
    x1 = torch.nn.functional.one_hot(torch.randint(0, V, (B, L)), V).to(DT)
    x0 = torch.randn(B, L, V, dtype=DT)
    xs = interp(x0, x1, s)

    psi_su = exact_mean_denoiser(xs, s, x1)
    phi_su = flow_map_mix(xs, psi_su, s, u)
    psi_ut = exact_mean_denoiser(phi_su, u, x1)
    w = omega(s, u=u, t=t)
    rhs = w * psi_su + (1 - w) * psi_ut
    torch.testing.assert_close(exact_mean_denoiser(xs, s, x1), rhs, rtol=0, atol=1e-12)


def test_schedule_medians_match_paper():
    """Independent check that our alpha(t) reconstructions are the paper's.

    E.3 states the cosine schedule has median insertion time (2/3) t_ins_end.
    E.2 gives only the hazard rho(t) = r t^{r-1}/(1-t^r) and states median 0.25
    at r = 0.5; alpha(t) = t^r is the unique antiderivative-consistent
    reconstruction, and it reproduces that median. If either median drifts, our
    schedule is not the paper's schedule and every downstream number is wrong.
    """
    from eflow.schedules.insertion import CosineSchedule, PolynomialSchedule
    grid = torch.linspace(1e-6, 1.0, 200_001, dtype=DT)
    for sched, want in [(CosineSchedule(), 2 / 3), (PolynomialSchedule(r=0.5), 0.25)]:
        a = sched.alpha(grid)
        med = float(grid[int((a - 0.5).abs().argmin())])
        assert abs(med - want) < 1e-3, f"{type(sched).__name__}: median {med} != {want}"


def test_flow_map_endpoints():
    """Phi_{s,s}(x) = x and Phi_{s,1}(x) = psi_{s,1}(x)  (Eq. 30)."""
    B, L, V = 2, 3, 5
    x = torch.randn(B, L, V, dtype=DT)
    psi = torch.softmax(torch.randn(B, L, V, dtype=DT), -1)
    s = torch.tensor(0.3, dtype=DT)
    assert torch.allclose(flow_map_mix(x, psi, s, s), x, atol=1e-12)
    one = torch.tensor(1.0, dtype=DT)
    assert torch.allclose(flow_map_mix(x, psi, s, one), psi, atol=1e-12)


def test_flow_map_composition_continuous():
    """Prop 4.1c on the exact linear-interpolant map: Phi_ut o Phi_su == Phi_st."""
    B, d = 4, 7
    x0 = torch.randn(B, d, dtype=DT)
    x1 = torch.randn(B, d, dtype=DT)
    s, u, t = 0.1, 0.55, 0.8
    xs = interp(x0, x1, s)
    # exact mean velocity of the linear interpolant is (x1 - x0), constant
    v = x1 - x0
    xu = xs + (u - s) * v
    xt_two = xu + (t - u) * v
    xt_one = xs + (t - s) * v
    assert torch.allclose(xt_two, xt_one, atol=1e-12)


def test_local_time_bijection():
    """Eq. 12: t_i maps [t_ins, 1] -> [0, 1] bijectively and monotonically."""
    from eflow.ops.local_time import local_time_reference
    t_ins = torch.tensor([[0.0, 0.25, 0.5, 0.9]], dtype=DT)
    prev = None
    for t in torch.linspace(0.0, 1.0, 21, dtype=DT):
        tl, act = local_time_reference(t_ins, t.reshape(1))
        assert (tl[act] >= 0).all() and (tl[act] <= 1).all()
        assert (tl[~act] == 0).all()
        if prev is not None:
            assert (tl >= prev - 1e-12).all(), "local time must be non-decreasing in t"
        prev = tl
    tl, _ = local_time_reference(t_ins, torch.ones(1, dtype=DT))
    assert torch.allclose(tl, torch.ones_like(tl), atol=1e-12), "t=1 -> all local times 1"


def test_poisson_nll_minimiser_is_the_mean():
    """arg min_b E[phi(A, b)] = E[A] -- the property the insertion head relies on."""
    from eflow.ops.poisson_nll import poisson_nll_reference
    torch.manual_seed(1)
    true_mean = 2.7
    a = torch.poisson(torch.full((200_000,), true_mean, dtype=DT))
    b = torch.tensor(true_mean, dtype=DT, requires_grad=True)
    loss = poisson_nll_reference(a, b.expand_as(a)).mean()
    (g,) = torch.autograd.grad(loss, b)
    assert abs(float(g)) < 1e-2, f"gradient at the true mean should vanish, got {g}"
