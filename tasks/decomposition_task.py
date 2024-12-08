""" """

import os
try:
    import ujson as json
except ImportError:
    import json
from glob import glob
from typing import Text
from overrides import overrides
from tasker import BaseTask
from src.decomposer import BaseDecomposer
from src.data_reader import GenerationDataReader
from src.utils.common import (
    __MAX_LINE_PER_FILE__,
    paginate_func,
    stream_paginate_func
)


@BaseTask.register("decomposition")
class DecompositionTask(BaseTask):
    """ """
    
    __VERSION__ = "0.0.1"

    def __init__(self, input_dir: Text, output_dir: Text, decomposer: BaseDecomposer):
        super().__init__(output_dir=output_dir)
        self._input_dir = input_dir
        self._reader = GenerationDataReader()
        self._decomposer = decomposer

    @overrides
    def _run(self):
        """ """

        outputs = self._decomposer(
            instance=self._reader(
                file_paths=glob(os.path.join(self._input_dir, "*.jsonl"))
            )
        )

        return outputs
    
    @overrides
    def _write(self, outputs):
        # if MAX_LINE_PER_FILE is set as environment variable, use that value
        max_line_per_file = int(os.getenv("MAX_LINE_PER_FILE", __MAX_LINE_PER_FILE__))
        num_written_files = 0
        
        def counted_write(items):
            with open(os.path.join(self._output_dir, f"decomposition-{num_written_files:08d}.jsonl")) as file_:
                for item in items:
                    file_.write(
                        json.dumps(item.to_dict()) + "\n"
                    )
                    
            num_written_files += 1
        
        stream_paginate_func(
            items=outputs,
            page_size=max_line_per_file,
            func=counted_write,
        )