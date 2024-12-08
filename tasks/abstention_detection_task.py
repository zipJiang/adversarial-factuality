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
from src.utils.instances import LLMGenerationInstance
from src.abstention_detector import BaseAbstentionDetector
from src.data_reader import GenerationDataReader
from src.utils.common import (
    __MAX_LINE_PER_FILE__,
    paginate_func,
    stream_paginate_func
)


class AbstentionDetectionTask(BaseTask):
    """ """
    def __init__(self, input_dir: Text, output_dir: Text, abstention_detector: BaseAbstentionDetector):
        super().__init__(output_dir=output_dir)
        self._input_dir = input_dir
        self._reader = GenerationDataReader()
        self._abstention_detector = abstention_detector

    @overrides
    def _run(self):
        
        
        # TODO: If some methods actually utilize the batch processing
        # of abstention_detection, then the following code should be
        # modified to improve the performance.
        outputs = [
            LLMGenerationInstance(
                id_=instance.id_,
                generation=instance.generation,
                meta={
                    **instance.meta,
                    "is_abstained": self._abstention_detector(response=instance.generation)
                }
            )
            for instance in self._reader(
                instance=self._reader(
                    file_paths=glob(os.path.join(self._input_dir, "*.jsonl"))
                )
            )
        ]
        
        return outputs
    
    @overrides
    def _write(self, outputs):
        
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