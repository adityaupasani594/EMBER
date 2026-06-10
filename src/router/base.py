"""
src/router/base.py
------------------
Abstract base class for all routing-policy implementations.

The routing policy π is the top-level orchestrator described in §2 of the
paper.  It receives a raw query and returns a final Answer by deciding
which cascade tier(s) to invoke and when to halt.

Concrete subclasses will encode different decision strategies (e.g., greedy
threshold-based cascade, bandit, learned policy), but the CascadeRunner only
ever calls this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.types import Answer, Query


class RoutingPolicy(ABC):
    """
    Abstract routing policy: maps a query to a final answer.

    Subclasses
    ----------
    (to be added as concrete implementations are built)

    Contract
    --------
    - route() must return exactly one Answer.
    - route() must NOT modify the Query object.
    - All resource accounting (energy, latency) is the responsibility of
      the concrete implementation or its injected EnergyMeter.
    """

    @abstractmethod
    def route(self, query: Query) -> Answer:
        """
        Execute the routing policy and return a final answer for *query*.

        This corresponds to the policy π that solves the optimisation
        problem in §2 of the paper — maximising utility subject to the
        energy budget B_E and latency budget B_L.

        Parameters
        ----------
        query:
            The input request to be answered.  Do not mutate it.

        Returns
        -------
        Answer
            The final, accepted answer for *query*, including the stage
            at which routing halted and any recorded resource usage.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
