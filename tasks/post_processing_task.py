""" Apply post-processor in sequence to the decompmosition results. """

try:
    import ujson as json
except ImportError:
    import json
import os
from glob import glob
from overrides import overrides
from typing import (
    Text,
    List
)
from tasker import BaseTask
from src.data_reader import DecompositionDataReader
from src.post_processor import BasePostProcessor
from src.utils.common import (
    __MAX_LINE_PER_FILE__,
    paginate_func,
    stream_paginate_func
)


class PostProcessingTask(BaseTask):
    """ """
    
    __VERSION__ = "0.0.1"

    def __init__(
        self,
        input_dir: Text,
        output_dir: Text,
        post_processors: List[BasePostProcessor]
    ):
        super().__init__(output_dir=output_dir)
        self._input_dir = input_dir
        self._reader = DecompositionDataReader()
        self._post_processors = post_processors

    @overrides
    def _run(self):
        
        current_iterator = self._reader(glob(os.path.join(self._input_dir, "*.jsonl")))

        for post_processor in self._post_processors:
            current_iterator = post_processor(current_iterator)
            
        return current_iterator
    
    @overrides
    def _write(self, outputs):
        
        max_line_per_file = int(os.getenv("MAX_LINE_PER_FILE", __MAX_LINE_PER_FILE__))
        num_written_files = 0
        
        def counted_write(items):
            with open(os.path.join(self._output_dir, f"post-processed-{num_written_files:08d}.jsonl")) as file_:
                for item in items:
                    file_.write(
                        json.dumps(item.to_dict()) + "\n"
                    )
                    
            num_written_files += 1
        
        stream_paginate_func(
            items=outputs,
            func=counted_write,
            max_line_per_file=max_line_per_file
        )
        
        return num_written_files