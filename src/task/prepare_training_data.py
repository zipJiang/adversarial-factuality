"""Prepare training data.
"""

import ujson as json
from random import Random
from typing import Text, Optional, Tuple, List
import os
import logging
from overrides import overrides
from .task import Task
from ..utils.common import batched


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
handler.setLevel(logging.INFO)
logger.addHandler(handler)


@Task.register("prepare-training-data")
class PrepareTrainingDataTask(Task):
    """
    """

    __NAME__ = "prepare-training-data"

    def __init__(
        self,
        data_path: Text,
        generation_path: Text,
        is_question: bool,
        allowed_categories: List[Text],
        write_dir: Text,
        batch_size: Optional[int] = 5,
        split_size: Optional[Tuple[int, int, int]] = (0.8, 0.1, 0.1),
        seed: Optional[int] = 42
    ):
        super().__init__()
        self._data_path = data_path
        self._generation_path = generation_path
        self._is_question = is_question
        self._batch_size = batch_size
        self._split_size = split_size
        self._allowed_categories = allowed_categories
        self._write_dir = write_dir
        self._seed = seed
        self._random_obj = Random(seed)

    @overrides
    def run(self):
        """
        """
        
        with open(self._data_path, 'r', encoding='utf-8') as file_:
            data = [json.loads(line.strip()) for line in file_]

        with open(self._generation_path, 'r', encoding='utf-8') as file_:
            generations = batched([json.loads(line.strip()) for line in file_], batch_size=self._batch_size)
            
        assert len(data) == len(generations), "Data and generations should be of the same length."

        cat_set = list({orig['category'] for orig in data})
        logger.info(f"Categories found: {cat_set}")
        
        selected_data = [
            (orig, gen_group)
            for orig, gen_group in zip(data, generations) if orig['category'] in self._allowed_categories
        ]
        
        logger.info(f"Selected data size: {len(selected_data)}")

        self._random_obj.shuffle(selected_data)
        
        size_of_data = len(selected_data)
        size_of_train = int(size_of_data * self._split_size[0])
        size_of_validation = int(size_of_data * self._split_size[1])
        size_of_test = size_of_data - size_of_train - size_of_validation

        train_data = selected_data[:size_of_train]
        validation_data = selected_data[size_of_train:size_of_train+size_of_validation]
        test_data = selected_data[-size_of_test:]
        
        os.makedirs(self._write_dir, exist_ok=True)
        
        with open(os.path.join(self._write_dir, "train.jsonl"), 'w', encoding='utf-8') as file_:
            for orig, gen_group in train_data:
                for gen in gen_group:
                    file_.write(json.dumps({
                        "_id": orig['_id'],
                        "category": orig['category'],
                        "topic": orig['topic'],
                        "output": gen['output'],
                        "is_question": self._is_question
                    }) + '\n')
                
        with open(os.path.join(self._write_dir, "validation.jsonl"), 'w', encoding='utf-8') as file_:
            for orig, gen_group in validation_data:
                for gen in gen_group:
                    file_.write(json.dumps({
                        "_id": orig['_id'],
                        "category": orig['category'],
                        "topic": orig['topic'],
                        "output": gen['output'],
                        "is_question": self._is_question
                    }) + '\n')
                
        with open(os.path.join(self._write_dir, "test.jsonl"), 'w', encoding='utf-8') as file_:
            for orig, gen_group in test_data:
                for gen in gen_group:
                    file_.write(json.dumps({
                        "_id": orig['_id'],
                        "category": orig['category'],
                        "topic": orig['topic'],
                        "output": gen['output'],
                        "is_question": self._is_question
                    }) + '\n')