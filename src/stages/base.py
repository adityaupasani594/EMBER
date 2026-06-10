"""
src/stages/base.py
------------------
Abstract base class that every stage in the cascade must implement.
The CascadeRouter only ever talks to this interface — it never imports
a concrete stage directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.types import Answer, Query, StageID


class BaseStage(ABC):
    """
    A single tier in the verifier-guided cascade.

    Subclasses
    ----------
    Stage0Reflex          (src/stages/stage0.py)
    Stage1Knowledge       (src/stages/stage1.py)
    Stage2Reasoning       (src/stages/stage2.py)
    Stage3DeepReason      (src/stages/stage3.py)
    Stage4ExpertSelection (src/stages/stage4.py)

    Contract
    --------
    - generate() must return an Answer whose `stage` field matches
      self.stage_id.
    - generate() must NOT modify the Query object.
    - energy_joules and latency_ms on the returned Answer are filled
      in by the EnergyMeter wrapper, not by the stage itself.
    """

    @property
    @abstractmethod
    def stage_id(self) -> StageID:
        """Which tier this stage represents."""
        ...

    @abstractmethod
    def generate(self, query: Query) -> Answer:
        """
        Produce a candidate answer for the given query.

        Parameters
        ----------
        query:
            The input request. Do not mutate it.

        Returns
        -------
        Answer
            A candidate response. The `confidence` field will be None
            here — it is filled in by the ThresholdGate after the
            verifier scores the (query, answer) pair.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(stage={self.stage_id.name})"