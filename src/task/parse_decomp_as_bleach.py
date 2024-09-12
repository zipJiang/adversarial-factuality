"""
"""

from typing import Text, List, Dict, Iterable
from overrides import overrides
from tqdm import tqdm
import ujson as json
import re
import os
from .task import Task


@Task.register("parse-decomp-as-bleach")
class ParseDecompAsBleachTask(Task):
    __NAME__ = "parse-decomp-as-bleach"
    def __init__(
        self,
        decomposition_path: Text,
        output_path: Text
    ):
        """
        """
        super().__init__()
        self._decomposition_path = decomposition_path
        self._output_path = output_path
        self._topic_to_bleached_context_dict: Dict[Text, List] = {}
        
    def _parse_sentence(self, sentence: Text) -> Text | None:
        """
        """
        sentence = sentence.strip()
        # if sentence matches [0-9]+\. (.*), 
        matched = re.match(r'[0-9]+\. (.*)', sentence)
        if matched:
            return matched.group(1)
        
        # else if match with leading - or *, remove it
        matched = re.match(r'[-\*](.*)', sentence)
        if matched:
            return matched.group(1)
        
        return None
        
    def _parse_generation(self, generation: Text) -> Iterable[Text]:
        """
        """
        sentences = generation.split('\n')
        return filter(lambda x: x is not None, [self._parse_sentence(sent) for sent in sentences])
        
    @overrides
    def run(self):
        """
        """
        
        with open(self._decomposition_path, 'r', encoding='utf-8') as file_:
            for line in tqdm(file_):
                ldata = json.loads(line)
                topic = ldata['input']
                if topic not in self._topic_to_bleached_context_dict:
                    self._topic_to_bleached_context_dict[topic] = []
                generation = ldata['output']['raw']
                self._topic_to_bleached_context_dict[topic].extend(self._parse_generation(generation))

        os.makedirs(os.path.dirname(self._output_path), exist_ok=True)

        with open(self._output_path, 'w', encoding='utf-8') as file_:
            json.dump(self._topic_to_bleached_context_dict, file_, indent=4)