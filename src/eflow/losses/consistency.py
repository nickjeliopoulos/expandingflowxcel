"""Consistency objectives.

Discrete (Eq. 32-33, Alg. 2 L15/L21) -- implemented in ops/semigroup_ce.py
because it is a fused-kernel target, not a composition of ops. This module
holds the *continuous* variants of Eq. 22, which are separate objectives with
genuinely different cost profiles and are worth benchmarking against each other:

  L_LSD (22a)  Lagrangian self-distillation
               needs d/dt of Phi -> one JVP per step
  L_ESD (22b)  Eulerian self-distillation
               needs d/ds of Phi AND the Jacobian-vector product
               grad_x Phi . b_s -- note the Jacobian is RECTANGULAR here
               (d(t) x d(s), Prop. 4.1) because the expand operator changes
               dimension. That is the one place EFM's identities differ
               structurally from fixed-dimension flow maps, and it is the
               interesting autodiff question in the continuous path.
  L_PSD (22c)  Semigroup / progressive self-distillation
               needs the extra Phi_{s,u} evaluation but NO derivatives

Cost ordering to establish empirically: PSD needs an extra forward; LSD and ESD
need forward-mode AD through the network. On modern PyTorch that is
torch.func.jvp, whose cost relative to an extra forward is the thing to measure.
The paper uses the semigroup form for the discrete path; whether that choice is
also the cheapest one in the continuous path is an open question this repo can
answer.
"""
