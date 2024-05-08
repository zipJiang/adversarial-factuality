""""""

import json
import os
from tqdm import tqdm
from overrides import overrides
from typing import Text
from .task import Task
from src.scorer.scorer import Scorer
from src.utils.instances import ScorerInstance


@Task.register("score-generation")
class ScoreGenerationTask(Task):
    def __init__(
        self,
        scorer: Scorer,
        input_path: Text,
        output_path: Text
    ):
        super().__init__()
        self.scorer = scorer
        self.input_path = input_path
        self.output_path = output_path
        
    @overrides
    def run(self):
        """
        """
        
        with open(self.input_path, "r", encoding='utf-8') as file_:
            data = json.load(file_)
            
        results = []
        
        inputs = [ScorerInstance(text=item['output']['parsed'], topic=item['topic']) for item in tqdm(data)]
            # score = self.scorer(ScorerInstance(
            #     text=item['output']['parsed'],
            #     topic=item['topic'],
            # ))
            
            # results.append({
            #     **item,
            #     "score": score
            # })
            
        results = self.scorer(inputs)
        results = [{**item, "score": result} for result, item in zip(results, data)]

        with open(self.output_path, "w", encoding='utf-8') as file_:
            json.dump(results, file_, indent=4)