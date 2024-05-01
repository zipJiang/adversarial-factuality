""" Always output constant scoring """


from abc import ABC, abstractmethod
from dataclasses import dataclass
from overrides import overrides
from .scorer import Scorer
from typing import Text, List, Union, Dict
from registrable import Registrable
from ..utils.instances import ScorerInstance


@Scorer.register("constant")
class ConstantScorer(Scorer):
    """ """
    
    __NAME__ = "constant"

    def __init__(self, constant_score: float):
        super().__init__()
        self._constant_score = constant_score

    @overrides
    def _score(self, instance: ScorerInstance) -> Dict[Text, Union[Text, float]]:
        """ """
        return {"raw": self._constant_score, "parsed": self._constant_score}