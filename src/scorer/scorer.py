"""
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from overrides import overrides
from typing import Text, List, Union, Dict, AsyncGenerator, Tuple
from registrable import Registrable
from ..utils.instances import ScorerInstance


class Scorer(ABC, Registrable):
    """ """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def _score(self, instance: ScorerInstance) -> Dict[Text, Union[Text, float]]:
        """ """
        raise NotImplementedError("Override the scoring to get proper scoring.")

    def __call__(
        self,
        instance: Union[ScorerInstance, List[ScorerInstance]],
        return_raw: bool = False,
    ) -> Union[
        Dict[Text, Union[Text, float]],
        float,
        List[Dict[Text, Union[Text, float]]],
        List[float],
    ]:
        """ """
        
        if not isinstance(instance, ScorerInstance):
            results = self._batch_score(instances=instance)
            if not return_raw:
                return [r["parsed"] for r in results]
            return results

        result = self._score(instance)
        if not return_raw:
            return result["parsed"]
        return result

    def _batch_score(self, instances: List[ScorerInstance]) -> List[Dict[Text, Union[Text, float]]]:
        """ """
        return [self._score(instance) for instance in instances]
    
    async def _async_batch_score(self, instances: List[ScorerInstance]) -> AsyncGenerator[Tuple[int, Dict[Text, Union[Text, float]]], None]:
        """ """
        raise NotImplementedError("Override the async scoring to get proper async scoring.")