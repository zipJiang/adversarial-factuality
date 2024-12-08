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

        self._agent = RunnableParallel(
            passthrough=RunnablePassthrough(),
            decomposition=RunnableLambda(lambda x: {"input": x["input"]}) | DecompositionStep(example_selector=example_selector).chain_llm(
                self._llm
            )
        ) | RunnableLambda(lambda x: DecomposedLLMGenerationInstance(
                id_=x['passthrough']['instance_id'],
                generation=x['passthrough']['generation'],
                meta=x['passthrough']['meta'],
                claims=[
                    AtomicClaim(
                        claim=atom,
                        meta={
                            "claim_index": aidx,
                            "source_text": x["passthrough"]["input"],
                            "source_text_index": x['passthrough']['in_instance_id'],
                        },
                    ) for aidx, atom in enumerate(x['decomposition'].claims)
                ]
            )
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
            "instance_id": instance.id_,
            "generation": instance.generation,
            "meta": instance.meta,
            "in_instance_id": 0,
        }]

        if self._sentencize:
            inputs = [
                {
                    "input": sentence.text,
                    "instance_id": instance.id_,
                    "generation": instance.generation,
                    "meta": instance.meta,
                    "in_instance_id": sidx,
                } for sidx, sentence in enumerate(self._nlp(instance_text).sents)
            ]

        return self._agent.batch(inputs, config=self._runnable_config)

    @overrides
    def _batch_decompose(
        self, instances: Iterable[LLMGenerationInstance]
    ) -> Iterable[DecomposedLLMGenerationInstance]:
        """ """

        if not self._sentencize:
            # inputs = [
            #     {"input": instance.generation} for idx, instance in enumerate(instances)
            # ]
            # num_sents = [1 for _ in instances]
            def _naiive_generator():
                for ipt in instances:
                    yield {
                        "generation": ipt.generation,
                        "meta": ipt.meta,
                        "input": ipt.generation,
                        "instance_id": ipt.id_,
                        "in_instance_id": None,
                    }
            inputs = list(_naiive_generator())
            
        else:
            
            def _sentencized_generator():
                for idx, instance in enumerate(instances):
                    for sidx, sentence in enumerate(self._nlp(instance.generation).sents):
                        yield {
                            "generation": instance.generation,
                            "meta": instance.meta,
                            "input": sentence.text,
                            "instance_id": idx,
                            "in_instance_id": sidx,
                        }
                        
            inputs = list(_sentencized_generator())

        # TODO: have tasker to support streaming outputs
        return self._agent.batch(inputs, config=self._runnable_config)