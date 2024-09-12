"""Example parser for decomposition tasks.
"""

from typing import List, Text
from src.utils.instances import InContextExample
from overrides import overrides
from .example_parser import ExampleParser


@ExampleParser.register("decomp-parser")
class DecompositionParser(ExampleParser):
    """Parsing the decomposition tasks.
    """

    def __init__(
        self
    ):
        """
        """
        super().__init__()
        
    @overrides
    def _parse(self, filepath: str) -> List[InContextExample]:
        """
        """
        examples = []
        
        with open(filepath, "r", encoding="utf-8") as file_:
            items = file_.read().split("\n\n")
            for item in items:
                lines = item.split("\n")
                # example = {
                #     "input": lines[0],
                #     "output": "\n".join(lines[1:]),
                # }
                examples.append(InContextExample(
                    input_text=lines[0],
                    generation="\n".join(lines[1:])
                ))
                
        return examples