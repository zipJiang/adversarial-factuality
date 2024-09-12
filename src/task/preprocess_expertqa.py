"""
"""

import datasets
import ujson as json
from overrides import overrides
from typing import Optional, List, Text
from .task import Task


@Task.register("preprocess-expertqa")
class PreprocessExpertQATask(Task):
    """
    """
    __NAME__ = "preprocess-expertqa"
    
    def __init__(
        self,
        input_path: Text,
        write_to: Text,
    ):
        """
        """
        self._dataset = []

        with open(input_path, 'r', encoding='utf-8') as file_:
            for lidx, line in enumerate(file_):
                self._dataset.append({"_id": lidx, **json.loads(line)})
        self._write_to = write_to

    @overrides
    def run(self):
        """
        """
        with open(self._write_to, 'w', encoding='utf-8') as file_:
            for example in self._dataset:
                file_.write(json.dumps({"_id": example["_id"], "topic": example['question'], "category": example["metadata"]['field']}) + '\n')