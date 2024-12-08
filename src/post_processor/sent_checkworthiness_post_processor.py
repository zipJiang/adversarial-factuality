""" """

import numpy as np
from typing import Text, Optional, Dict, Tuple, List, Iterable
from overrides import overrides
from langchain_interface.models import ChatOpenAIWithBatchAPI
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough
)
from langchain_core.runnables.config import RunnableConfig
from ..langchain_step.sent_checkworthiness_step import SentCheckworthinessStep
from ..utils.instances import DecomposedLLMGenerationInstance, AtomicClaim
from ..utils.common import path_index, solve_milp, paginate_func
from .base_post_processor import BasePostProcessor
import logging


logger = logging.getLogger(__name__)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger.setLevel(logging.INFO)
handler = logging.FileHandler("sent_checkworthiness_post_processor.log")
handler.setFormatter(formatter)
logger.addHandler(handler)


@BasePostProcessor.register("sent-checkworthiness")
class SentCheckworthinessPostProcessor(BasePostProcessor):
    def __init__(
        self,
        model_name: Text,
        in_batch_num: int = 4,
        base_url: Optional[Text] = None,
        api_key: Optional[Text] = None
    ):
        super().__init__(namespace="sent-checkworthiness")
        self._model_name = model_name
        self._base_url = base_url
        self._api_key = api_key

        self._in_batch_num = in_batch_num
        
        self._llm = ChatOpenAIWithBatchAPI(
            model_name=self._model_name,
            base_url=self._base_url,
            api_key=self._api_key,
            # top_p=0.98,
            model_kwargs={"top_p": 0.98},
            temperature=0.0,
            max_tokens=128,
        )
        self._runnable_config = RunnableConfig(max_concurrency=32)
        self._agent = RunnableParallel(
            passthrough=RunnablePassthrough(),
            generation=RunnableLambda(
                lambda x: {
                    "texts": x['texts']
                } | SentCheckworthinessStep().chain_llm(self._llm)
            )
        ) | RunnableLambda(
            lambda x: {
                sent: x['generation'].parsed[sidx] if sidx < len(x['generation'].parsed) else x['generation'].parsed[-1] 
                for sidx, sent in enumerate(x['passthrough']['sents'])
            }
        )
        
    @overrides
    def _process(self, instance: DecomposedLLMGenerationInstance) -> DecomposedLLMGenerationInstance:
        """ """
        
        dedup_inputs = set()
        
        for claim in instance.claims:
            source_text = claim.meta['source_text']
            dedup_inputs.add(source_text)

        dedup_inputs = list(dedup_inputs)
        
        inputs = paginate_func(
            items=dedup_inputs,
            page_size=self._in_batch_num,
            func=lambda x: {
                "texts": ', '.join(['"{}"'.format(text) for text in x]),
                "sents": x
            },
            combination=None,
            silent=True,
        )
        
        results = self._agent.batch(inputs, config=self._runnable_config)
        agg_result_dict = {}
        
        for rdict in results:
            agg_result_dict.update(rdict)
            
        return DecomposedLLMGenerationInstance(
            id_=instance.id_,
            generation=instance.generation,
            meta=instance.meta,
            claims=[
                AtomicClaim(
                    claim=claim.claim,
                    meta={
                        **claim.meta,
                        f"{self._namespace}": {"score": agg_result_dict[claim.meta['source_text']]}
                    }
                )
                for claim in instance.claims
            ]
        )
        
    @overrides
    def _batch_process(
        self,
        instances: List[DecomposedLLMGenerationInstance]
    ) -> List[DecomposedLLMGenerationInstance]:
        """ """
        
        dedup_inputs = set()
        
        for instance in instances:
            for claim in instance.claims:
                source_text = claim.meta['source_text']
                dedup_inputs.add(source_text)
                
        dedup_inputs = list(dedup_inputs)
        
        inputs = paginate_func(
            items=dedup_inputs,
            page_size=self._in_batch_num,
            func=lambda x: {
                "texts": ', '.join(['"{}"'.format(text) for text in x]),
                "sents": x
            },
            combination=None,
            silent=True,
        )
        
        results = self._agent.batch(inputs, config=self._runnable_config)
        agg_result_dict = {}
        
        for rdict in results:
            agg_result_dict.update(rdict)
            
        return [
            DecomposedLLMGenerationInstance(
                id_=instance.id_,
                generation=instance.generation,
                meta=instance.meta,
                claims=[
                    AtomicClaim(
                        claim=claim.claim,
                        meta={
                            **claim.meta,
                            f"{self._namespace}": {"score": agg_result_dict[claim.meta['source_text']]}
                        }
                    )
                    for claim in instance.claims
                ]
            )
            for instance in instances
        ]