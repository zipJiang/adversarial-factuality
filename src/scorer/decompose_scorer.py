"""This type of factuality scorer willl take a summary
(or other knowledge intensive paragraph) and score the
output of this summary by decomposing the summary into
individual facts and scoring each fact individually.
"""

from registrable import Lazy
from langchain_interface.interfaces import ChatInterface
from ..aggregator.aggregator import Aggregator
from ..decomposer.decomposer import Decomposer
from ..utils.instances import ScorerInstance
from typing import Text, List, Dict, Union
from overrides import overrides
from .scorer import Scorer


@Scorer.register("decompose")
class DecomposeScorer(Scorer):
    
    __NAME__ = "decompose"

    def __init__(
        self,
        decomposer: Decomposer,
        base_scorer: Scorer,
        aggregator: Aggregator
    ):
        super().__init__()
        self.decomposer = decomposer
        self.base_scorer = base_scorer
        self.aggregator = aggregator
        
    @overrides
    def _score(self, instance: ScorerInstance) -> Dict[Text, Union[Text, float]]:
        """
        """
        
        decomposed_instances: List[ScorerInstance] = self.decomposer(instance)
        print([item.text for item in decomposed_instances])
        scores = [self.base_scorer(dt, return_raw=True) for dt in decomposed_instances]

        parsed_scores = [s['parsed'] for s in scores]

        agg_score = self.aggregator(parsed_scores)
        
        return {
            "parsed": agg_score,
            "raw": " ## ".join([str(s['raw']) for s in scores])
        }