"""A decontextualizer that refines the text to be
self-contained and context-free.
"""

import logging
from overrides import overrides
from registrable import Registrable
from typing import Text, Optional, List, Union, Dict, Any, Iterable
from langchain_core.runnables.config import RunnableConfig
from langchain_interface.models import ChatOpenAIWithBatchAPI
from langchain_interface.steps.decontextualization_step import (
    DecontextualizationStep,
    DecontextualizationResponse
)
from ..utils.instances import (
    AtomicClaim,
    DecomposedLLMGenerationInstance
)
from .base_post_processor import BasePostProcessor


logger = logging.getLogger(__name__)


class DecontextualizationPostProcessor(BasePostProcessor):
    """Take a sentence and make it standalone."""

    def __init__(
        self,
        model_name: Text,
        # example_path: Text,
        base_url: Optional[Text] = None,
        api_key: Optional[Text] = None,
    ):
        """ """
        super().__init__(namespace="__decontextualization")
        # self._example_path = example_path
        self._model_name = model_name
        self._base_url = base_url
        self._api_key = api_key

        self._llm = ChatOpenAIWithBatchAPI(
            model_name=self._model_name,
            base_url=self._base_url,
            api_key=self._api_key,
            # top_p=0.98,
            model_kwargs={"top_p": 0.98},
            temperature=0.0,
            max_tokens=512,
        )
        self._runnable_config = RunnableConfig(
            max_concurrency=32,
        )

        self._agent = DecontextualizationStep().chain_llm(self._llm)

    def __call__(
        self,
        instance: Union[DecomposedLLMGenerationInstance, Iterable[DecomposedLLMGenerationInstance]],
    ) -> Union[DecomposedLLMGenerationInstance, Iterable[DecomposedLLMGenerationInstance]]:
        """This will decontextualize the instances, and return
        the decontextualized version paired with the correct information.
        """
        
        if not isinstance(instance, DecomposedLLMGenerationInstance):
            result = self._batch_process(instances=instance)
        else:
            result = self._process(instance=instance)
                
        return result
        
    @overrides
    def _process(self, instance: DecomposedLLMGenerationInstance) -> DecomposedLLMGenerationInstance:
        """ """
        
        inputs = [
            {
                "input": claim.claim,
                "context": claim.meta['source_text'],
            } for claim in instance.claims
        ]
        
        results = self._agent.batch(inputs, config=self._runnable_config)
        
        return DecomposedLLMGenerationInstance(
            id_=instance.id_,
            generation=instance.generation,
            meta=instance.meta,
            claims=[
                AtomicClaim(
                    claim=result.revised if result.revised is not None else claim.claim,
                    meta={
                        **claim.meta,
                        f"{self._namespace}": {
                            "raw": result.messages,
                            "original": claim.claim,
                        }
                    },
                )
                for claim, result in zip(instance.claims, results)
            ]
        )
    
    @overrides
    def _batch_decontextualize(
        self,
        instances: Iterable[DecomposedLLMGenerationInstance]
    ) -> Iterable[DecomposedLLMGenerationInstance]:
        """ """
        
        iidx_cidx_to_id = {}
        inputs = []
        
        for idx, instance in enumerate(instances):
            for cidx, claim in enumerate(instance.claims):
                iidx_cidx_to_id[(idx, cidx)] = len(inputs)
                inputs.append({
                    "input": claim.claim,
                    "context": claim.meta['source_text'],
                })
                
        results = self._agent.batch(inputs, config=self._runnable_config)
        
        return [
            DecomposedLLMGenerationInstance(
                id_=instance.id_,
                generation=instance.generation,
                meta=instance.meta,
                claims=[
                    AtomicClaim(
                        claim=results[iidx_cidx_to_id[(idx, cidx)]].revised if results[iidx_cidx_to_id[(idx, cidx)]].revised is not None else claim.claim,
                        meta={
                            **claim.meta,
                            f"{self._namespace}": {
                                "raw": results[iidx_cidx_to_id[(idx, cidx)]]["messages"],
                                "original": claim.claim,
                            }
                        },
                    )
                    for cidx, claim in enumerate(instance.claims)
                ]
            )
            for idx, instance in enumerate(instances)
        ]