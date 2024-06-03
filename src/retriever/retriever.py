"""
"""

from abc import ABC, abstractmethod
from typing import Text, List, Dict, Any
from registrable import Registrable


class Retriever(Registrable, ABC):
    def __init__(self):
        """
        """
        super().__init__()
        
    @abstractmethod
    def get_passages(self, topic: Text, question: Text, k: int) -> List[Dict[Text, Any]]:
        """
        """
        raise NotImplementedError()
    
    def get_passages_batched(self, topics: List[Text], questions: List[Text], k: int) -> List[List[Dict[Text, Any]]]:
        """
        """
        
        return [self.get_passages(topic, question, k) for topic, question in zip(topics, questions)]