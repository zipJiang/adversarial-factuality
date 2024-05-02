"""
"""

from abc import ABC, abstractmethod
from typing import Text
from registrable import Registrable


class AbstentionDetector(Registrable, ABC):
    """A detector that returns True if the response is abstained.
    """
    def __init__(self):
        super().__init__()

    def __call__(self, response: Text) -> bool:
        """Take a response and return whether the response is abstained.
        """
        
        return self._detect_abstention(response)
    
    @abstractmethod
    def _detect_abstention(self, response: Text) -> bool:
        """Take a response and return whether the response is abstained.
        """
        
        raise NotImplementedError("The method _detect_abstention must be implemented.")