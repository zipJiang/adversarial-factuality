"""
"""

from abc import ABC, abstractmethod
from ..utils.instances import (
    LLMGenerationInstance,
    DecomposedLLMGenerationInstance,
)
from typing import Text, List, Iterable, Tuple, Union
from overrides import overrides
from registrable import Registrable


class BaseDecomposer(ABC, Registrable):
    def __init__(self):
        super().__init__()

    def __call__(
        self, instance: Union[Iterable[LLMGenerationInstance], LLMGenerationInstance]
    ) -> Union[Iterable[DecomposedLLMGenerationInstance], DecomposedLLMGenerationInstance]:
        """ """
        if not isinstance(instance, LLMGenerationInstance):
            selections = [
                idx for idx in range(len(instance)) if not instance[idx].meta.get("is_abstention", False)
            ]
            input_instances = [instance[idx] for idx in selections]
            decomposed_instances = self._batch_decompose(input_instances)
            
            placeholders = [
                DecomposedLLMGenerationInstance(
                    id_=ins.id_,
                    generation=ins.generation,
                    atomic_claims=[],
                    meta=ins.meta,
                )
                for ins in instance
            ]
            
            for idx, selection in enumerate(selections):
                placeholders[selection] = decomposed_instances[idx]
                
            return placeholders
        
        if instance.meta.get("is_abstention", False):
            return DecomposedLLMGenerationInstance(
                id_=instance.id_,
                generation=instance.generation,
                atomic_claims=[],
                meta=instance.meta,
            )

    @abstractmethod
    def _decompose(self, instance: LLMGenerationInstance) -> DecomposedLLMGenerationInstance:
        raise NotImplementedError(
            "Override the decomposition to get proper decomposition."
        )

    def _batch_decompose(
        self, instances: Iterable[LLMGenerationInstance]
    ) -> Iterable[DecomposedLLMGenerationInstance]:
        """We batch whatever way we like, and try to generate a list of decomposed items,
        with indices indicating from which instance the decomposed instance comes from.
        """

        for instance in instances:
            yield self._decompose(instance)
