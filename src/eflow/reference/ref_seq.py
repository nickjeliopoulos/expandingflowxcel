"""Literal, slow, obviously-correct transcription of Alg. 1-4 for sequences.

Rules for this file, which exists to be the oracle:
  * python loops are FINE and preferred where they match the algorithm box
  * float64 throughout
  * no vectorization tricks, no fused anything, no in-place ops
  * every line should map to a numbered line of an algorithm box, and say which
  * never import from ops/ -- if it shares code with the fast path it cannot
    catch a bug in the fast path

When the authors release their implementation, THIS is the file to diff against.
See PLAN.md section 7 for the five places divergence is most likely.
"""
