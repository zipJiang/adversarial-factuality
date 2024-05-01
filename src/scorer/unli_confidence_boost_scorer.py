""" This score scores a given claim by how much confidence boost stating that claim
explicitly gives to UNLI being able to hypothetically predict the claim. """

from dataclasses import dataclass, field
import string
from typing import Text, Dict, List, Union, Optional
import os
from overrides import overrides
from ..entailer.entailer import Entailer, EntailerInstance
from langchain_interface.instances import LLMQueryInstance
from langchain_interface.interfaces import ChatInterface
from ..utils.instances import ScorerInstance
from .scorer import Scorer
from ..retriever.retriever import Retriever


@Scorer.register("unli-confidence-boost")
class UNLIConfidenceBoostScorer(Scorer):
    """1 - p(claim | bleached_context)"""

    __NAME__ = "unli-confidence-boost"

    def __init__(
        self,
        bleached_templates: List[Text],
        entailer: Entailer,
        cap_entailer: Optional[Entailer] = None,
        epsilon: float = 1e-7,
    ):
        """We don't explicitly require the entailer to
        be soft, but practically people should always use a
        soft entailer for proper tie-breaking.
        """

        super().__init__()
        self._bleached_templates = bleached_templates
        self._entailer = entailer
        self._cap_entailer = cap_entailer
        self._epsilon = epsilon

    @overrides
    def _score(self, instance: ScorerInstance) -> Dict[Text, Union[Text, float]]:
        """Here the scorer will score the instance into
        result dict
        """

        bleached_context = [bt.format(topic=instance.topic) for bt in self._bleached_templates]

        # pair each bleached_context with the claim
        # and score each pair

        inputs = [
            EntailerInstance(premise=bc, hypothesis=instance.text)
            for bc in bleached_context
        ]

        scores = self._entailer(inputs)
        score = max(scores)

        cap_entailer_outputs = {}

        if self._cap_entailer is not None:
            # if cap_score == 1, then the claim needs to be
            # capped regardless of the score
            cap_scores = self._cap_entailer(inputs)
            cap_score = max(cap_scores)
            score = max(score, cap_score)

            cap_entailer_outputs["cap_scores"] = cap_scores

        # The reason for subtracting epsilon from the score is to not select
        # already entailed claims (from bleached context).
        parsed_score = 1 - self._epsilon - score

        return {
            "premises": bleached_context,
            "hypothesis": instance.text,
            "parsed": parsed_score,
            "scores": scores,
            **cap_entailer_outputs,
        }

    @overrides
    def _batch_score(
        self, instances: List[ScorerInstance]
    ) -> List[Dict[Text, Text | float]]:
        """Run scores in batch."""
        
        inputs = [
            EntailerInstance(premise=bt.format(topic=instance.topic), hypothesis=instance.text)
            for instance in instances
            for bt in self._bleached_templates
        ]

        all_scores = self._entailer(inputs)
        all_scores = [
            all_scores[
                i * len(self._bleached_templates) : (i + 1) * len(self._bleached_templates)
            ]
            for i in range(len(instances))
        ]

        all_cap_entailer_outputs = [{}] * len(instances)

        if self._cap_entailer is not None:
            cap_scores = self._cap_entailer(inputs)
            all_cap_entailer_outputs = [
                {
                    "cap_scores": cap_scores[
                        i
                        * len(self._bleached_templates) : (i + 1)
                        * len(self._bleached_templates)
                    ]
                }
                for i in range(len(instances))
            ]

        return [
            {
                "premises": [bt.format(topic=instance.topic) for bt in self._bleached_templates],
                "hypothesis": instance.text,
                "parsed": (
                    1 - self._epsilon - max(scores)
                    if "cap_scores" not in cap_entailer_outputs
                    else 1
                    - self._epsilon
                    - max(max(scores), max(cap_entailer_outputs["cap_scores"]))
                ),
                "scores": scores,
                **cap_entailer_outputs,
            }
            for instance, scores, cap_entailer_outputs in zip(
                instances, all_scores, all_cap_entailer_outputs
            )
        ]
