"""Replicate the FactScore Decomposer
that takes in a summary and decomposes it into
atomic facts.
"""

import spacy
from overrides import overrides
from typing import List, Text, Optional, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.runnables.config import RunnableConfig
from langchain_interface.example_selectors import ConstantExampleSelector
from ..utils.instances import ScorerInstance
# from ..langchain_step.factscore_evidential_support_step import (
#     FActScoreEvidentialSupportStep,
#     FActScoreEvidentialSupportResponse
# )
from langchain_interface.steps.decomposition_step import (
    DecompositionStep,
    DecompositionResponse
)
from .decomposer import Decomposer


@Decomposer.register("factscore")
class FActScoreDecomposer(Decomposer):
    
    __NAME__ = "factscore"
    
    def __init__(
        self,
        model_name: Text,
        example_path: Text,
        nlp_model_name: Text = "en_core_web_sm",
        sentencize: bool = True,
        base_url: Optional[Text] = None,
        api_key: Optional[Text] = None
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

        # def _split_atomic_facts(output: Text) -> List[Text]:
        #     """Split the atomic facts from the output.
        #     each line is an atomic fact start with '- ',
        #     need to remove the '- ' from the start of the line.
        #     """

        #     return [
        #         line[2:].strip() for line in output.split("\n") if line.startswith("- ")
        #     ]

        # self._example_selector = ConstantExampleSelector()
        # with open(self._example_path, "r", encoding="utf-8") as file_:
        #     items = file_.read().split("\n\n")
        #     for item in items:
        #         lines = item.split("\n")
        #         example = {
        #             "input": lines[0],
        #             "output": "\n".join(lines[1:]),
        #         }
        #         self._example_selector.add_example(example)

        # self._agent = ChatInterface(
        #     model_name=self._model_name,
        #     batch_size=32,
        #     max_tokens=512,
        #     system_message=None,
        #     instruction_prompt=[],
        #     input_example_prompt="Please breakdown the following sentence into independent facts: {input}",
        #     output_example_prompt="{output}",
        #     output_parser=_split_atomic_facts,
        #     example_selector=self._example_selector,
        #     base_url=self._base_url,
        #     api_key=self._api_key,
        #     max_concurrency=32,
        # )
        
        self._llm = ChatOpenAI(
            model_name=self._model_name,
            base_url=self._base_url,
            api_key=self._api_key,
            max_tokens=512,
            # top_p=0.98,
            model_kwargs={"top_p": 0.98},
            temperature=0.0,
        )

        self._agent = DecompositionStep().chain_llm(self._llm)
        self._runnable_config = RunnableConfig(max_concurrency=32)
        
    @overrides
    def _decompose(self, instance: ScorerInstance) -> List[ScorerInstance]:
        """ """

        instance_text = instance.text
        topic = instance.topic

        # return [
        #     ScorerInstance(text=atom, topic=topic)
        #     for sidx, sentence in enumerate(self._nlp(instance_text).sents)
        #     for atom in self._agent([LLMQueryInstance(id=sidx, input=sentence.text)])[0]
        # ]

        outputs = []
        
        if not self._sentencize:
            outputs = self._agent.batch([{"input": instance_text}], config=self._runnable_config)
        else:
            outputs = self._agent.batch([{"input": sentence.text} for sentence in self._nlp(instance_text).sents], config=self._runnable_config)
        
        return [ScorerInstance(text=atom, topic=topic, source_text=instance.source_text) for otp in outputs for atom in otp.claims]
    
    @overrides
    def _batch_decompose(self, instances: List[ScorerInstance]) -> List[List[ScorerInstance]]:
        """
        """

        if not self._sentencize:
            inputs = [{"input": instance.text} for idx, instance in enumerate(instances)]
            num_sents = [1 for _ in instances]
        else:
            inputs = []
            num_sents = []
            
            for idx, instance in enumerate(instances):
                sents = self._nlp(instance.text).sents
                num_sents.append(len(sents))
                for sentence in sents:
                    # inputs.append(LLMQueryInstance(id=idx, input=sentence.text))
                    inputs.append({"input": sentence.text})

        outputs = self._agent.batch(inputs, config=self._runnable_config)
        
        # now since we are getting all outputs
        results = []
        # for ipt, opt in zip(inputs, outputs):
        #     if idx + 1 > len(results):
        #         results.append([])
        #     results[ipt.id].extend([ScorerInstance(text=atom, topic=instances[ipt.id].topic, source_text=instances[ipt.id].source_text) for atom in opt.claims])
        
        for nidx, ns in enumerate(num_sents):
            if ns == 0:
                results.append([])
            else:
                slice_ = outputs[:ns]
                results.append([
                    ScorerInstance(text=atom, topic=instances[nidx].topic, source_text=instances[nidx].source_text)
                    for otp in slice_
                    for atom in otp.claims
                ])
                outputs = outputs[ns:]
                
        assert len(outputs) == 0, "Outputs should be empty at the end of the loop."
            
        return results