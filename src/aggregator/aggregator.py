"""Aggregator is used to compose a series of local scorings to
a global scoring.
"""

from abc import ABC, abstractmethod
from typing import List, TypeVar
from registrable import Registrable


_T = TypeVar("_T")


class Aggregator(ABC, Registrable):
    def __init__(self):
        super().__init__()
        pass
    
    @abstractmethod
    def _aggregate(self, scores: List[_T]) -> _T:
        raise NotImplementedError("Override the aggregation to get proper aggregation.")
    
    def __call__(self, scores: List[_T]) -> _T:
        """
        """
        return self._aggregate(scores)