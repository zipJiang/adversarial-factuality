"""
"""

from abc import ABC, abstractmethod
from ..utils.instances import ScorerInstance
from typing import Text, List
from overrides import overrides
from registrable import Registrable


class Decomposer(ABC, Registrable):
    def __init__(self):
        super().__init__()
        
    def __call__(self, instance: ScorerInstance) -> List[ScorerInstance]:
        """
        """
        return self._decompose(instance)
        
    @abstractmethod
    def _decompose(self, instance: ScorerInstance) -> List[ScorerInstance]:
        raise NotImplementedError("Override the decomposition to get proper decomposition.")