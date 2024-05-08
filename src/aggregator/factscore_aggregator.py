"""This replicates the FActScore aggregator, which will take a list of
scores, binarize, and then aggregate them to a single score.
"""

import numpy as np
from typing import List, Dict
from .aggregator import Aggregator, _T


@Aggregator.register("factscore")
class FActScoreAggregator(Aggregator):
    def __init__(
        self,
        gamma: int = 10
    ):
        """
        gamma: int --- the threshold for length penalty application.
        """
        super().__init__()
        self._gamma = gamma
    
    def _aggregate(self, scores: List[_T]) -> _T:
        """Take a list of scorings, and aggregate them to a single score.
        """

        score = None

        if not scores:
            return 0.0
            
        means = np.mean(scores)
        num_facts = len(scores)

        if num_facts < self._gamma:
            penalty = np.exp(1 - self._gamma / num_facts)
            score = (means * penalty).item()
        else:
            score = means.item()
            
        return score