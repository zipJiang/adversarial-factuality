"""Example parser to be used in the ChatInterface class.
"""

from typing import Text, List
from registrable import Registrable
from abc import ABC, abstractmethod
from ..utils.instances import InContextExample


class ExampleParser(Registrable):
    """
    """
    def __init__(
        self,
    ):
        """
        """
        super().__init__()
        
    def __call__(self, filepath: Text) -> List[InContextExample]:
        """
        """
        return self._parse(filepath)
    
    def _parse(self, filepath: Text) -> List[InContextExample]:
        """
        """
        
        with open(filepath, 'r', encoding='utf-8') as file_:
            lines = file_.readlines()
        return [
            InContextExample(
                input_text=lines[i].strip(),
                generation=lines[i+1].strip()
            ) for i in range(0, len(lines), 2)
        ]
    
ExampleParser.register("example-parser")(ExampleParser)