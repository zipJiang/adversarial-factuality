"""This type of factuality scorer willl take a summary
(or other knowledge intensive paragraph) and score the
output of this summary by decomposing the summary into
individual facts and scoring each fact individually.
"""

from registrable import Lazy
from dataclasses import dataclass, asdict
from langchain_interface.interfaces import ChatInterface
from ..abstention_detector.abstention_detector import AbstentionDetector
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
        abstention_detector: AbstentionDetector,
        decomposer: Decomposer,
        base_scorer: Scorer,
        aggregator: Aggregator
    ):
        super().__init__()
        self.abstention_detector = abstention_detector
        self.decomposer = decomposer
        self.base_scorer = base_scorer
        self.aggregator = aggregator
        
    @overrides
    def _score(self, instance: ScorerInstance) -> Dict[Text, Union[Text, float]]:
        """
        """
        
        if self.abstention_detector(instance.text):
            return {
                "parsed": 0.0,
                "raw": f"[[Abstained Response Detected]]: {instance.text}"
            }
        
        decomposed_instances: List[ScorerInstance] = self.decomposer(instance)
        scores = [self.base_scorer(dt, return_raw=True) for dt in decomposed_instances]

        parsed_scores = [s['parsed'] for s in scores]

        agg_score = self.aggregator(parsed_scores)
        
        return {
            "parsed": agg_score,
            "raw": " ## ".join([str(s['raw']) for s in scores])
        }
    
    @overrides
    def _batch_score(self, instances: List[ScorerInstance]) -> List[Dict[Text, Text | float]]:
        """We cascade over all the components instead of running components one by one
        on instance.
        """

        results = {}
        
        instance_needs_process = []

        for idx, instance in enumerate(instances):
            if self.abstention_detector(instance.text):
                results[idx] = {
                    "parsed": 0.0,
                    "raw": f"[[Abstained Response Detected]]: {instance.text}"
                }
                continue
            else:
                instance_needs_process.append(idx)
                
        # now we have a list of instances that need processing
        input_instances = [instances[idx] for idx in instance_needs_process]
        decomposed_instance_chunks: List[List[ScorerInstance]] = self.decomposer(input_instances)
        
        # extend them to tuples with index
        decomposed_instance_tuples = [(idx, dt) for idx, dts in zip(instance_needs_process, decomposed_instance_chunks) for dt in dts]
        raw_scores = self.base_scorer([dt[1] for dt in decomposed_instance_tuples], return_raw=True)
        # parsed_scores = [s['parsed'] for s in scores]
        
        # print(decomposed_instance_tuples)
        
        # grouped parsed scores by index
        grouped_parsed_scores = {idx: [] for idx in range(len(instances))}
        for (idx, di), score_dict in zip(decomposed_instance_tuples, raw_scores):
            grouped_parsed_scores[idx].append({**score_dict, **asdict(di)})
            
        for idx, score_dicts in grouped_parsed_scores.items():
            agg_score = self.aggregator([s['parsed'] for s in score_dicts])
            assert idx not in results, f"Index already exists in results {idx}."
            results[idx] = {
                "parsed": agg_score,
                "raw": " ## ".join([str(s['raw']) for s in score_dicts]),
                "claims": [s['text'] for s in score_dicts]
            }
            
        return [results[index] for index in range(len(instances))]