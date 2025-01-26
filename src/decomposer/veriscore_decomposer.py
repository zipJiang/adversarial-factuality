"""Implement a VeriScore Decomposer that takes in
generation and exetract "Verifiable" claims.
"""

import spacy
import re
import json
from overrides import overrides
from typing import List, Text, Optional, Tuple, Iterable
from langchain_openai import OpenAI, ChatOpenAI
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
from ..langchain_step.veriscore_extraction_step import (
    VeriScoreExtractionStep,
    VeriScoreExtractionResponse
)
from .base_decomposer import BaseDecomposer


@BaseDecomposer.register("veriscore")
class VeriScoreDecomposer(BaseDecomposer):
    """Notice that this decomposer uses a OpenAI Completion
    instead of ChatOpenAI.
    """
    __NAME__ = "veriscore"
    
    def __init__(
        self,
        model_name: Text,
        question_template: Text,
        nlp_model_name: Text = "en_core_web_sm",
        sentencize: bool = True,
        base_url: Optional[Text] = None,
        api_key: Optional[Text] = None,
        example_path: Optional[Text] = None,
    ):
        """ """
        super().__init__()
        self._example_path = example_path
        self._model_name = model_name
        self._base_url = base_url
        self._api_key = api_key
        self._nlp = spacy.load(nlp_model_name, disable=["ner", "parser"])
        self._nlp.add_pipe("sentencizer")
        self._sentencize = sentencize
        
        self._llm = OpenAI(
            model=self._model_name,
            base_url=self._base_url,
            api_key=self._api_key,
            max_tokens=1500,
            # top_p=0.98,
            model_kwargs={"top_p": 0.98},
            temperature=0.0,
        )

        self._agent = VeriScoreExtractionStep().chain_llm(self._llm)
        self._runnable_config = RunnableConfig(max_concurrency=32)

        self._question_template = question_template
        
    @overrides
    def _decompose(
        self, instance: LLMGenerationInstance
    ) -> DecomposedLLMGenerationInstance:
        """ One major differences is that we need sentence before and after. """
        
        instance_text = instance.generation
        # previously we explicitly read topic, but now it will be packed
        # into the instance meta

        outputs = []
        inputs = [{
            "input": instance_text,
        }]
        
        sents = self._nlp(instance_text).sents
        
        if self._sentencize:
            inputs = [
                {
                    "input": instance_text.replace(sentence.text, f"<SOS>{sentence.text}<EOS>"),
                    "question": self.question_template.format(topic=instance.meta["topic"]),
                } for sidx, sentence in enumerate(sents)
            ]
            
        else:
            # Since we need sentencizing information to proceed
            raise NotImplementedError("Not implemented yet.")

        responses = self._agent.batch(inputs, config=self._runnable_config)
        atomic_claims = []

        for ipt, response in zip(inputs, responses):
            for claim in response.claims:
                claim_index = len(atomic_claims)
                atomic_claims.append(AtomicClaim(
                    claim=claim,
                    meta={
                        "source_text": re.search(r"<SOS>(.*)<EOS>", ipt['input'], flags=re.DOTALL).group(1),
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
            raise NotImplementedError("Not implemented yet.")
        else:
            for iidx, instance in enumerate(instances):
                sents = list(self._nlp(instance.generation).sents)
                lengths.append(len(sents))
                for sidx, sentence in enumerate(sents):
                    iidx_sidx_to_id[(iidx, sidx)] = len(inputs)
                    inputs.append({
                        "input": instance.generation.replace(sentence.text, f"<SOS>{sentence.text}<EOS>"),
                        "question": self._question_template.format(topic=instance.meta["topic"]),
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
                            "source_text": re.search(r"<SOS>(.*)<EOS>", inputs[iidx_sidx_to_id[(iidx, sidx)]]['input'], flags=re.DOTALL).group(1),
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