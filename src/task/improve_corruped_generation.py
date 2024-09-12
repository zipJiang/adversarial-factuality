"""Combine the corrupted generation with less-informative features to improve the evaluation of the score.
"""

import json
import spacy
import os
from random import Random
from copy import deepcopy
from .task import Task
from typing import List, Text, Dict, Any


@Task.register("improve-corrupted-generation")
class ImproveCorruptedGenerationTask(Task):
    
    def __init__(
        self,
        num_facts_to_consider: List[int],
        generation_dir: Text,
        informative_path: Text,
        repetition_path: Text,
        seed: int = 42
    ):
        super().__init__()
        self._num_facts_to_consider = num_facts_to_consider
        self._generation_dir = generation_dir
        self._informative_path = informative_path
        self._repetition_path = repetition_path

        self._nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
        self._nlp.add_pipe("sentencizer")
        
        self._random_obj = Random(seed)

    def run(self):
        """
        """
        
        with open(os.path.join(self._generation_dir, "corrupted.jsonl"), 'r', encoding='utf-8') as file_:
            data = [json.loads(line) for line in file_]
            
        with open(self._informative_path, 'r', encoding='utf-8') as file_:
            informative_data = [json.loads(line) for line in file_]
            
        with open(self._repetition_path, 'r', encoding='utf-8') as file_:
            repetition_data = [json.loads(line) for line in file_]
            
        pass
            
        def _improve_instance(instance, improve_base, k: int) -> Dict[Text, Any]:
            instance = deepcopy(instance)
            candidates = [sent.text for sent in self._nlp(improve_base['output']['raw']).sents]
            if len(candidates) < k:
                candidates = candidates * (k // len(candidates) + 1)
            selected = self._random_obj.sample(candidates, k)

            instance['output']['raw'] = ' '.join([instance['output']['raw']] + selected)
            instance['output']['parsed'] = instance['output']['raw'].strip()
            
            return instance

        # now we will iterate over the data and improve the generation
        for num_facts in self._num_facts_to_consider:
            iimprove_list = []
            rimprove_list = []
            for idx, instance in enumerate(data):
                informative_instance = informative_data[idx]
                repetition_instance = repetition_data[idx]
                
                iimproved = _improve_instance(instance, informative_instance, k=num_facts)
                rimproved = _improve_instance(instance, repetition_instance, k=num_facts)

                iimprove_list.append(iimproved)
                rimprove_list.append(rimproved)
                
            # now we save the improved data to the disk
            with open(os.path.join(self._generation_dir, f"corrupted-informative-{num_facts}.jsonl"), 'w', encoding='utf-8') as file_:
                for instance in iimprove_list:
                    file_.write(json.dumps(instance, ensure_ascii=False) + '\n')

            with open(os.path.join(self._generation_dir, f"corrupted-repetitive-{num_facts}.jsonl"), 'w', encoding='utf-8') as file_:
                for instance in rimprove_list:
                    file_.write(json.dumps(instance, ensure_ascii=False) + '\n')