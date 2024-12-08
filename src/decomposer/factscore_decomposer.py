"""Replicate the FactScore Decomposer
that takes in a summary and decomposes it into
atomic facts.
"""

import spacy
import json
from overrides import overrides
from typing import List, Text, Optional, Tuple, Iterable
from langchain_openai import ChatOpenAI
from langchain_core.runnables.config import RunnableConfig
from langchain_core.runnables import (
    RunnableBranch,
    RunnablePassthrough,
    RunnableParallel,
    RunnableLambda
)
from langchain_interface.example_selectors import ConstantExampleSelector
from ..utils.instances import (
    LLMGenerationInstance,
    DecomposedLLMGenerationInstance,
    AtomicClaim,
)

# from ..langchain_step.factscore_evidential_support_step import (
#     FActScoreEvidentialSupportStep,
#     FActScoreEvidentialSupportResponse
# )
from langchain_interface.steps.decomposition_step import (
    DecompositionStep,
    DecompositionResponse,
)
from .base_decomposer import BaseDecomposer


@BaseDecomposer.register("factscore")
class FActScoreDecomposer(BaseDecomposer):

    __NAME__ = "factscore"

    def __init__(
        self,
        model_name: Text,
        nlp_model_name: Text = "en_core_web_sm",
        sentencize: bool = True,
        base_url: Optional[Text] = None,
        api_key: Optional[Text] = None,
        example_path: Optional[Text] = None,
    ):
        """In general, this decomposer runs a sentence splitter,
        and then a atomic fact extractor to get the atomic facts.
        """

        super().__init__()
        self._example_path = example_path
        self._model_name = model_name
        self._base_url = base_url
        self._api_key = api_key
        self._nlp = spacy.load(nlp_model_name, disable=["ner", "parser"])
        self._nlp.add_pipe("sentencizer")
        self._sentencize = sentencize

        self._llm = ChatOpenAI(
            model_name=self._model_name,
            base_url=self._base_url,
            api_key=self._api_key,
            max_tokens=512,
            # top_p=0.98,
            model_kwargs={"top_p": 0.98},
            temperature=0.0,
        )

        example_selector = None
        if example_path is not None:

            example_selector = ConstantExampleSelector()
            with open(example_path, "r", encoding="utf-8") as file_:
                items = file_.read().split("\n\n")
                for item in items:
                    lines = item.split("\n")
                    example = {
                        "input": lines[0],
                        "output": "\n".join(lines[1:]),
                    }
                    example_selector.add_example(example)

        self._agent = DecompositionStep(example_selector=example_selector).chain_llm(
            self._llm
        )
        self._runnable_config = RunnableConfig(max_concurrency=32)

    @overrides
    def _decompose(
        self, instance: LLMGenerationInstance
    ) -> DecomposedLLMGenerationInstance:
        """ """

        instance_text = instance.generation
        # previously we explicitly read topic, but now it will be packed
        # into the instance meta

        outputs = []
        inputs = [{
            "input": instance_text,
            # "instance_id": instance.id_,
            # "generation": instance.generation,
            # "meta": instance.meta,
            # "in_instance_id": 0,
        }]

        if self._sentencize:
            inputs = [
                {
                    "input": sentence.text,
                    # "instance_id": instance.id_,
                    # "generation": instance.generation,
                    # "meta": instance.meta,
                } for sidx, sentence in enumerate(self._nlp(instance_text).sents)
            ]

        responses = self._agent.batch(inputs, config=self._runnable_config)
        atomic_claims = []

        for ipt, response in zip(inputs, responses):
            for claim in response.claims:
                claim_index = len(atomic_claims)
                atomic_claims.append(AtomicClaim(
                    claim=claim,
                    meta={
                        "source_text": ipt['input'],
                        "claim_index": claim_index,
                    }
                ))
                
        return DecomposedLLMGenerationInstance(
            id_=instance.id_,
            generation=instance.generation,
            meta=instance.meta,
            claims=atomic_claims,
        )

    @overrides
    def _batch_decompose(
        self, instances: Iterable[LLMGenerationInstance]
    ) -> Iterable[DecomposedLLMGenerationInstance]:
        """ """
        
        inputs = []
        iidx_sidx_to_id = {}
        lengths = []

        if not self._sentencize:
            for iidx, instance in enumerate(instances):
                iidx_sidx_to_id[(iidx, 0)] = len(inputs)
                inputs.append({
                    "input": instance.generation,
                })
                lengths.append(1)
        else:
            for iidx, instance in enumerate(instances):
                sents = list(self._nlp(instance.generation).sents)
                lengths.append(len(sents))
                for sidx, sentence in enumerate(sents):
                    iidx_sidx_to_id[(iidx, sidx)] = len(inputs)
                    inputs.append({
                        "input": sentence.text,
                    })

        responses = self._agent.batch(inputs, config=self._runnable_config)
        
        outputs = []
        
        for iidx, (instance, lil) in enumerate(zip(instances, lengths)):
            atomic_claims = []
            for sidx in range(lil):
                response = responses[iidx_sidx_to_id[(iidx, sidx)]]
                for claim in response.claims:
                    claim_index = len(atomic_claims)
                    atomic_claims.append(AtomicClaim(
                        claim=claim,
                        meta={
                            "source_text": inputs[iidx_sidx_to_id[(iidx, sidx)]]['input'],
                            "claim_index": claim_index,
                        }
                    ))
            outputs.append(
                DecomposedLLMGenerationInstance(
                    id_=instance.id_,
                    generation=instance.generation,
                    meta=instance.meta,
                    claims=atomic_claims,
                )
            )
            
        return outputs