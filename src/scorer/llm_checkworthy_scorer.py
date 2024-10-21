""" A score used to evaluate the checkworthiness of a sentence (possibly atomic facts). """

import asyncio
from dataclasses import dataclass, field
import string
from typing import AsyncGenerator, Coroutine, Text, Dict, List, Union, Optional, Tuple
import os
import re
from overrides import overrides
from langchain_openai import ChatOpenAI
from langchain_core.runnables.config import RunnableConfig
from ..langchain_step.sent_checkworthiness_step import (
    SentCheckworthinessStep,
    SentCheckworthinessResponse,
)
from ..langchain_step.claim_checkworthiness_step import (
    ClaimCheckworthinessStep,
    ClaimCheckworthinessResponse,
)
from ..utils.common import paginate_func
from ..utils.instances import ScorerInstance
from .scorer import Scorer
from ..retriever.retriever import Retriever


# # TODO: Potentially need access to logits to make soft scorings.

# @dataclass(frozen=True, eq=True)
# class BatchedLLMQueryInstance(LLMQueryInstance):
#     num_instances: int = field(default=1)

@Scorer.register("llm-checkworthy-general")
class LLMGeneralCheckWorthyScorer(Scorer):
    """In the original paper this scorer is
    called on the sentence level to identify
    whether a sentence is checkworthy (has checkworthy statements
    or not).
    """

    __NAME__ = "llm-checkworthy-general"

    def __init__(
        self,
        model_name: Text,
        in_batch_num: int = 4,
        base_url: Optional[Text] = None,
        api_key: Optional[Text] = None
    ):
        super().__init__()
        self._model_name = model_name
        self._base_url = base_url
        self._api_key = api_key

        # def _parse_input(instance: LLMQueryInstance) -> Dict[Text, Text]:
        #     """Generate the input dictionary for the LLM."""
        #     return {"texts": instance.input}

        # def _parse_output(output: Text) -> List[float]:
        #     # return float(output.strip() == "Yes")
        #     # The model will output a list
        #     # "["Yes", "No"]", we need to convert it to a list of floats
        #     output = output.strip().lower()
        #     return [1.0 if re.search("yes", rsb) is not None else 0.0 for rsb in output.split(",")]
        
        self._in_batch_num = in_batch_num

        # self._agent = ChatInterface(
        #     model_name=self._model_name,
        #     batch_size=32,
        #     max_tokens=64,
        #     system_message="You are a helpful factchecker assistant." if self._base_url is None else None,
        #     instruction_prompt=[],
        #     input_example_prompt=CHECKWORTHY_PROMPT,
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
            # top_p=0.98,
            model_kwargs={"top_p": 0.98},
            temperature=0.0,
            max_tokens=128,
        )
        self._runnable_config = RunnableConfig(max_concurrency=32)
        self._agent = SentCheckworthinessStep().chain_llm(self._llm)

    @overrides
    def _score(self, instance: ScorerInstance) -> Dict[Text, Union[Text, float]]:

        # construct the input
        # input_instance = LLMQueryInstance(
        #     id=0,
        #     input=f"[\"{instance.text}\"]",
        # )
        
        input_instance = {
            "texts": f"[\"{instance.text}\"]"
        }
        
        return self._agent.invoke(input_instance, config=self._runnable_config)

    @overrides
    def _batch_score(self, instances: List[ScorerInstance]) -> List[Dict[Text, Union[Text, float]]]:

        # input_instances = [
        #     LLMQueryInstance(id=idx, input=instance.text)
        #     for idx, instance in enumerate(instances)
        # ]

        # we can chunk items into batches
        chunking = lambda xs: {
            "texts": "[" + ', '.join(['"{}"'.format(x.text) for x in xs]) + "]",
        }
        
        chunked_instances = []
        chunked_sizes = []

        for i in range(0, len(instances), self._in_batch_num):
            slice_ = instances[i:i + self._in_batch_num]
            chunked_instances.append(chunking(slice_))
            chunked_sizes.append(len(slice_))

        chunked_results = self._agent.batch(chunked_instances, config=self._runnable_config)
        separated = []
        
        # print(chunked_instances[0])
        # print(chunked_results)
        
        for ridx, (result, num_instances) in enumerate(zip(chunked_results, chunked_sizes)):
            parsed = result.checkworthiness
            # original_chunk = chunked_instances[ridx]
            if len(parsed) != num_instances:
                # We'll need to duplicate the last judgments,
                # as sometimes the llm gives less judgments
                # than the number of instances
                parsed = (parsed + [parsed[-1]] * num_instances)[:num_instances]
            for idx, r in enumerate(parsed):
                separated.append({
                    "raw": result.messages + f"[{idx}]",
                    "parsed": r
                })

        return separated
    
    
@Scorer.register("llm-checkworthy-specific")
class LLMSpecificCheckWorthyScorer(Scorer):
    """In contrast, this checker is mainly called on the atomic fact level"""

    __NAME__ = "llm-checkworthy-specific"

    def __init__(
        self,
        model_name: Text,
        base_url: Optional[Text] = None,
        api_key: Optional[Text] = None
    ):

        super().__init__()
        self._model_name = model_name
        self._base_url = base_url
        self._api_key = api_key

        # def _parse_input(instance: LLMQueryInstance) -> Dict[Text, Text]:
        #     """Generate the input dictionary for the LLM."""
        #     return {"sentence": instance.input}

        # def _parse_output(output: Text) -> float:
        #     # TODO: find a better scaling factor
        #     return 1.0 if abs(float(output.strip()) - 1.0) < 1e-6 else 0.0

        # self._agent = ChatInterface(
        #     model_name=self._model_name,
        #     batch_size=16,
        #     max_tokens=10,
        #     system_message="You are a helpful factchecker assistant." if self._base_url is None else None,
        #     instruction_prompt=[],
        #     input_example_prompt=SPECIFY_CHECKWORTHY_CATEGORY_PROMPT,
        #     output_example_prompt="",
        #     input_parser=_parse_input,
        #     output_parser=_parse_output,
        #     base_url=self._base_url,
        #     api_key=self._api_key,
        # )
        
        self._llm = ChatOpenAI(
            model_name=self._model_name,
            base_url=self._base_url,
            api_key=self._api_key,
            # top_p=0.98,
            model_kwargs={"top_p": 0.98},
            temperature=0.0,
            max_tokens=10,
        )
        self._runnable_config = RunnableConfig(max_concurrency=16)
        self._agent = ClaimCheckworthinessStep().chain_llm(self._llm)

    @overrides
    def _score(self, instance: ScorerInstance) -> Dict[Text, Union[Text, float]]:

        # construct the input
        # input_instance = LLMQueryInstance(
        #     id=0,
        #     input=instance.text,
        # )
        input_instance = {
            "sentence": instance.text
        }
        
        response = self._agent.invoke(input_instance, config=self._runnable_config)

        return {
            "raw": response.messages,
            "parsed": response.checkworthiness
        }

    @overrides
    def _batch_score(self, instances: List[ScorerInstance]) -> List[Dict[Text, Union[Text, float]]]:

        input_instances = [
            # LLMQueryInstance(id=idx, input=instance.text)
            {"sentence": instance.text}
            for instance in instances
        ]

        responses = self._agent.batch(input_instances, config=self._runnable_config)
        
        return [
            {
                "raw": response.messages,
                "parsed": response.checkworthiness
            }
            for response in responses
        ]