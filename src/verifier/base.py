"""
src/verifier/base.py
--------------------
Abstract base class for all verifier implementations.

The verifier is the core quality-gate component described in §3 of the paper.
It computes v_s(q, ỹ_s) = σ(g_φ(q, ỹ_s)) ∈ [0, 1], i.e., a scalar
confidence score that the ThresholdGate then compares against τ_s.

The CascadeRouter only ever calls this interface — concrete verifiers
(e.g., a fine-tuned cross-encoder, a rule-based heuristic) are hot-swapped
by passing a different subclass instance at construction time.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.types import Answer, Query


class BaseVerifier(ABC):
    """
    Abstract verifier: maps a (query, answer) pair to a confidence score.

    Subclasses
    ----------
    (to be added as concrete implementations are built)

    Contract
    --------
    - score() must return a value in [0.0, 1.0].
    - score() must NOT modify either the Query or the Answer object.
    - Implementations are responsible for any model loading / warm-up in
      their __init__; score() should be stateless with respect to its args.
    """

    @abstractmethod
    def score(self, query: Query, answer: Answer) -> float:
        """
        Compute a confidence score for the (query, answer) pair.

        This corresponds to v_s(q, ỹ_s) in the paper.  The ThresholdGate
        compares the returned value against the per-stage threshold τ_s
        to decide whether to accept the answer or escalate to the next tier.

        Parameters
        ----------
        query:
            The original input request.  Do not mutate it.
        answer:
            The candidate answer produced by the current stage.
            Do not mutate it.

        Returns
        -------
        float
            Confidence score in [0.0, 1.0].  Higher means more confident
            that *answer* correctly resolves *query*.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
