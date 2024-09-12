"""
"""

import os
from overrides import overrides
import ujson as json
from typing import Text
from .task import Task


@Task.register("preprocess-longfact")
class PreprocessLongFactTask(Task):
    
    __NAME__ = "preprocess-longfact"
    
    def __init__(
        self,
        input_dir: Text,
        write_to: Text
    ):
        """
        """
        super().__init__()

        self._input_dir = input_dir
        self._write_to = write_to
        
        self._dataset = []
        
        for filename in os.listdir(input_dir):
            with open(os.path.join(input_dir, filename), 'r', encoding='utf-8') as file_:
                for lidx, line in enumerate(file_):
                    ldata = json.loads(line)
                    self._dataset.append({
                        "_id": lidx,
                        "topic": ldata['prompt'],
                        "category": filename
                    })
        
    @overrides
    def run(self):
        """
        """
        with open(self._write_to, 'w', encoding='utf-8') as file_:
            for example in self._dataset:
                file_.write(json.dumps(example) + '\n')