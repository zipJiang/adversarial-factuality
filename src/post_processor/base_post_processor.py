""" """

from registrable import Registrable
from ..utils.instances import DecomposedLLMGenerationInstance
from typing import Union, List, Iterable, Text
from abc import ABC, abstractmethod


class BasePostProcessor(ABC, Registrable):
    def __init__(self, namespace: Text):
        super().__init__()
        self._namespace = namespace

    def __call__(
        self,
        instance: Union[
            DecomposedLLMGenerationInstance, Iterable[DecomposedLLMGenerationInstance]
        ],
    ) -> Union[
        DecomposedLLMGenerationInstance, Iterable[DecomposedLLMGenerationInstance]
    ]:
        """ """

        if isinstance(instance, DecomposedLLMGenerationInstance):
            if instance.meta.get("is_abstention", False):
                return self._process(instance)
            return instance

        selections = [
            idx
            for idx in range(len(instance))
            if not instance[idx].meta.get("is_abstention", False)
        ]
        input_instances = [instance[idx] for idx in selections]
        processed = self._batch_process(input_instances)
        
        placeholders = instance
        
        for idx, selection in enumerate(selections):
            placeholders[selection] = processed[idx]
            
        return placeholders

    @abstractmethod
    def _process(
        self, instance: DecomposedLLMGenerationInstance
    ) -> DecomposedLLMGenerationInstance:
        """ """
        raise NotImplementedError(
            "Override the process method to get proper processing."
        )

    @abstractmethod
    def _batch_process(
        self, instances: Iterable[DecomposedLLMGenerationInstance]
    ) -> Iterable[DecomposedLLMGenerationInstance]:
        """ """
        raise NotImplementedError(
            "Override the batch process method to get proper batch processing."
        )
