""" """

try:
    import ujson as json
except ImportError:
    import json
import os
import numpy as np
from glob import glob
from overrides import overrides
from typing import (
    Text,
    List
)
from tasker import BaseTask
from src.data_reader import DecompositionDataReader
from src.verifier import BaseVerifier
from src.utils.common import (
    __MAX_LINE_PER_FILE__,
    paginate_func,
    stream_paginate_func
)


@BaseTask.register("verification")
class VerificationTask(BaseTask):
    """ """
    __VERSION__ = "0.0.1"
    
    def __init__(
        self,
        input_dir: Text,
        output_dir: Text,
        verifier: BaseVerifier
    ):
        super().__init__(output_dir=output_dir)
        self._input_dir = input_dir
        self._reader = DecompositionDataReader()
        self._verifier = verifier
        
    @overrides
    def _run(self):
        """ """
        return self._verifier(self._reader(glob(os.path.join(self._input_dir, "*.jsonl"))))
    
    @overrides
    def _write(self, outputs):
        """ """
        
        max_line_per_file = int(os.getenv("MAX_LINE_PER_FILE", __MAX_LINE_PER_FILE__))
        num_written_files = 0
        agg_scores = []
        
        def counted_write(items):
            nonlocal num_written_files
            nonlocal agg_scores
            with open(os.path.join(self._output_dir, f"decomposition-{num_written_files:08d}.jsonl"), 'w') as file_:
                for item in items:
                    agg_scores.append(item.aggregated_score)
                    file_.write(
                        json.dumps(item.to_dict()) + "\n"
                    )
                    
            num_written_files += 1
        
        list(stream_paginate_func(
            items=outputs,
            page_size=max_line_per_file,
            func=counted_write,
        ))

        # also calculate the average score to write to a separate file
        with open(os.path.join(self._output_dir, "agg_scores.json"), "w") as file_:
            json.dump({
                "average_score": np.mean(agg_scores),
                "breakdown": agg_scores
            }, file_, indent=4)