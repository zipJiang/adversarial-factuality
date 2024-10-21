"""Basically the same scorer with a different evidential support
rating prompt.
"""

from dataclasses import dataclass, field
import re
import string
from typing import Text, Dict, List, Union, Optional, AsyncGenerator, Tuple
import os
from langchain_openai import ChatOpenAI
from langchain_core.runnables.config import RunnableConfig
from overrides import overrides
from ..langchain_step.safe_evidential_support_step import (
    SAFEEvidentialSupportStep,
    SAFEEvidentialSupportResponse
)
from ..utils.instances import ScorerInstance
from ..utils.prompts import SAFE_RATING_PROMPT
from .scorer import Scorer
from ..retriever.retriever import Retriever


# @dataclass(frozen=True, eq=True)
# class SAFEQueryInstance(LLMQueryInstance):
#     topic: Text = field(default=None)
#     passages: List[Dict[Text, Text]] = field(default_factory=list)
@Scorer.register("safe-support")
class SAFESupportScorer(Scorer):
    """
    """
    
    __NAME__ = "safe-support"
    
    def __init__(
        self,
        model_name: Text,
        retriever: Retriever,
        base_url: Optional[Text] = None,
        api_key: Optional[Text] = None,
        # retriever_batch_size: int = 256
    ):
        """
        """
        super().__init__()
        
        self._model_name = model_name
        self._base_url = base_url
        self._api_key = api_key
        # self._retriever_batch_size = retriever_batch_size
        
        # def _parse_input(instance: SAFEQueryInstance) -> Dict[Text, Text]:
        #     """Generate the input dictionary for the LLM.
        #     """
        #     return {
        #         "parsed_passages": '\n\n'.join([passage['text'] for passage in instance.passages]),
        #         "input": instance.input
        #     }
            
        # def _parse_output(output: Text) -> float:
        #     """Parse the output of the LLM.
        #     """
        #     generated_answer = output.strip().lower()
            
        #     # extract the answer within the bracket if exists
        #     answer_tag = re.search(r"\[(.*)\]", generated_answer)

        #     if answer_tag is None:
        #         return 0.0
            
        #     answer = answer_tag.group(1).strip()
        #     is_supported = 0.0
            
        #     if answer == "supported" or answer == "\"supported\"":
        #         is_supported = 1.0
            
        #     return is_supported
        
        # self._agent = ChatInterface(
        #     model_name=self._model_name,
        #     batch_size=32,
        #     max_tokens=512,
        #     system_message=None,
        #     instruction_prompt=[],
        #     input_example_prompt=SAFE_RATING_PROMPT,
        #     output_example_prompt="",
        #     input_parser=_parse_input,
        #     output_parser=_parse_output,
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
        self._runnable_config = RunnableConfig(max_concurrency=32)
        self._agent = SAFEEvidentialSupportStep().chain_llm(self._llm)
        
        self._retriever = retriever
        
    @overrides
    def _score(self, instance: ScorerInstance) -> Dict[Text, Union[Text, float]]:
        """
        """
        
        # first convert the input instance into the SAFEQueryInstance
        # by retrieve from the database (we set the retrieval number to 5)
        passages = self._retriever.get_passages(instance.topic, instance.text, 3)
        
        # input_instance = SAFEQueryInstance(
        #     id=0,
        #     topic=instance.topic,
        #     passages=passages,
        #     input=instance.text
        # )
        
        input_instance = {
            "parsed_passages": '\n\n'.join([passage['text'] for passage in instance.passages]),
            "input": instance.text
        }
        
        result = self._agent.invoke(input_instance, config=self._runnable_config)
        
        return {
            "raw": result.messages,
            "parsed": result.evidential_support
        }
    
    @overrides
    def _batch_score(self, instances: List[ScorerInstance]) -> List[Dict[Text, Text | float]]:
        """Now we will first retrieve for all the instances.
        """
        
        topics = [instance.topic for instance in instances]
        texts = [instance.text for instance in instances]

        passage_chunks = self._retriever.get_passages_batched(topics=topics, questions=texts, k=5)

        assert len(passage_chunks) == len(instances)

        input_instances = [
            # SAFEQueryInstance(
            #     id=idx,
            #     topic=instance.topic,
            #     passages=passages,
            #     input=instance.text
            {
                "parsed_passages": '\n\n'.join([passage['text'] for passage in passages]),
                "input": instance.text
            } for instance, passages in zip(instances, passage_chunks)
        ]
        
        responses = self._agent.batch(input_instances, config=self._runnable_config)
        
        return [
            {
                "raw": response.messages,
                "parsed": response.evidential_support
            } for response in responses
        ]