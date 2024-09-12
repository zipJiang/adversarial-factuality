"""
"""

import datasets
import ujson as json
from overrides import overrides
from typing import Optional, List, Text
from .task import Task


@Task.register("preprocess-wildhallucination")
class PreprocessWildHallucinationTask(Task):
    """
    """
    __NAME__ = "preprocess-wildhallucination"
    
    def __init__(
        self,
        write_to: Text,
        allowed_categories: List[Text] = None,
        not_allowed_categories: List[Text] = None,
        with_wiki: Optional[bool] = True,
    ):
        """
        """
        
        self._with_wiki = with_wiki
        self._categories = allowed_categories
        self._not_allowed_categories = not_allowed_categories
        self._dataset = datasets.load_dataset("wentingzhao/WildHallucinations", split='train')
        self._dataset = self._dataset.map(lambda _, i: {"_id": i}, with_indices=True)
        if self._with_wiki:
            self._dataset = self._dataset.filter(lambda x: x['wiki'] == 1, load_from_cache_file=False)
            
        if self._categories is not None or self._not_allowed_categories is not None:
            category_set = set(self._categories) if self._categories is not None else set(self._dataset.unique('category'))
            not_allowed_categories = set(self._not_allowed_categories) if self._not_allowed_categories is not None else set()
            self._dataset = self._dataset.filter(lambda x: x['category'] in category_set and x['category'] not in not_allowed_categories, load_from_cache_file=False)

        self._write_to = write_to

    @overrides
    def run(self):
        """
        """
        with open(self._write_to, 'w', encoding='utf-8') as file_:
            for example in self._dataset:
                file_.write(json.dumps({"_id": example["_id"], "topic": example['entity'], "category": example["category"]}) + '\n')