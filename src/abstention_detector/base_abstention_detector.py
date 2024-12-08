"""
"""

from abc import ABC, abstractmethod
from typing import Text, List, Union
from registrable import Registrable


class BaseAbstentionDetector(Registrable, ABC):
    """A detector that returns True if the response is abstained.
    """
    def __init__(self):
        super().__init__()

    def __call__(self, response: Union[Text, List[Text]]) -> bool:
        """Take a response and return whether the response is abstained.
        """
        
        if isinstance(response, list):
            return self._detect_abstention_batched(response)
        
        return self._detect_abstention(response)
    
    @abstractmethod
    def _detect_abstention(self, response: Text) -> bool:
        """Take a response and return whether the response is abstained.
        """
        
        raise NotImplementedError("The method _detect_abstention must be implemented.")
    
    def _detect_abstention_batched(self, responses: List[Text]) -> List[bool]:
        """Take a list of responses and return whether each response is abstained.
        """
        
        return [self._detect_abstention(response) for response in responses]