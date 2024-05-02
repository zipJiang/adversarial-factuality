"""
"""

from typing import Text
from .task import Task


class FilterRequiredEntities(Task):
    def __init__(
        self,
        popQA_path: Text,
        wikientity_path: Text,
        wikipedia_dump_path: Text,
        protected_entity_path: Text,
    ):
        super().__init__()
        self._popQA_path = popQA_path
        self._wikientity_path = wikientity_path
        self._wikipedia_dump_path = wikipedia_dump_path
        self._protected_entity_path = protected_entity_path