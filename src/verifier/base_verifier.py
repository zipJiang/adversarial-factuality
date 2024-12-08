""" """

from abc import ABC, abstractmethod
from typing import (
    Union,
    Iterable
)
from registrable import Registrable
from ..utils.instances import (
    DecomposedLLMGenerationInstance,
    VerifiedLLMGenerationInstance
)


class BaseVerifier(ABC, Registrable):
    def __init__(self):
        super().__init__()
        
    def __call__(
        self,
        instance: Union[DecomposedLLMGenerationInstance, Iterable[DecomposedLLMGenerationInstance]]
    ) -> Union[VerifiedLLMGenerationInstance, Iterable[VerifiedLLMGenerationInstance]]:
        """ """
        
        if isinstance(instance, DecomposedLLMGenerationInstance):
            if instance.meta.get("is_abstention", False):
                return VerifiedLLMGenerationInstance(
                    id_=instance.id_,
                    generation=instance.generation,
                    atomic_claims=[],
                    meta=instance.meta,
                    aggregated_score=0.0
                )
            return self._verify(instance)
        else:
            selections = [idx for idx in range(len(instance)) if not instance[idx].meta.get("is_abstention", False)]
            input_instances = [instance[idx] for idx in selections]
            
            verified_instances = self._batch_verify(input_instances)
            placeholders = [
                VerifiedLLMGenerationInstance(
                    id_=ins.id_,
                    generation=ins.generation,
                    atomic_claims=[],
                    meta=ins.meta,
                    aggregated_score=0.0
                )
                for ins in instance
            ]
            
            for idx, selection in enumerate(selections):
                placeholders[selection] = verified_instances[idx]
                
            return placeholders
        
    @abstractmethod
    def _verify(self, instance: DecomposedLLMGenerationInstance) -> VerifiedLLMGenerationInstance:
        """ """
        
        raise NotImplementedError("Subclass must implement this method.")
    
    def _batch_verify(self, instances: Iterable[DecomposedLLMGenerationInstance]) -> Iterable[VerifiedLLMGenerationInstance]:
        """ """
        return [self._verify(instance) for instance in instances]