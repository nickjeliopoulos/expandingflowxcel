"""Adaptive loss weighting (Eq. 87) and gradient surgery (App. E.3).

    w_{s,t}(x) = sg[ (||psi_{s,t}(x) - psibar_{s,t}(x)||^2 + c)^{-r} ],  c=1e-6, r=0.5

Downweights large student/target mismatches. Note it is a *per-position* weight
requiring a reduction over V -- naively a second full pass over [B,L,V], which is
why ops/semigroup_ce.py folds it into the CE pass instead.

Gradient surgery merges L_DEFM and L_insert. It needs per-loss gradients, so it
costs either two backward passes or a fused dual accumulation. Implement both
and time them; this must be measured separately from the loss itself or it will
be silently attributed to the wrong thing.
"""
