""" A score used to evaluate the checkworthiness of a sentence (possibly atomic facts). """

import asyncio
from dataclasses import dataclass, field
import string
from typing import AsyncGenerator, Coroutine, Text, Dict, List, Union, Optional, Tuple
import os
import re
from overrides import overrides
from langchain_interface.instances import LLMQueryInstance
from langchain_interface.interfaces import ChatInterface
from ..utils.common import paginate_func
from ..utils.instances import ScorerInstance
from .scorer import Scorer
from ..retriever.retriever import Retriever
from ..utils.prompts import CHECKWORTHY_PROMPT, SPECIFY_CHECKWORTHY_CATEGORY_PROMPT


# TODO: Potentially need access to logits to make soft scorings.
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

        def _parse_input(instance: LLMQueryInstance) -> Dict[Text, Text]:
            """Generate the input dictionary for the LLM."""
            return {"texts": instance.input}

        def _parse_output(output: Text) -> List[float]:
            # return float(output.strip() == "Yes")
            # The model will output a list
            # "["Yes", "No"]", we need to convert it to a list of floats
            output = output.strip().lower()
            return [1.0 if re.search("yes", rsb) is not None else 0.0 for rsb in output.split(",")]
        
        self._in_batch_num = in_batch_num

        self._agent = ChatInterface(
            model_name=self._model_name,
            batch_size=16,
            max_tokens=32,
            system_message="You are a helpful factchecker assistant." if self._base_url is None else None,
            instruction_prompt=[],
            input_example_prompt=CHECKWORTHY_PROMPT,
            output_example_prompt="",
            input_parser=_parse_input,
            output_parser=_parse_output,
            base_url=self._base_url,
            api_key=self._api_key,
        )

    @overrides
    def _score(self, instance: ScorerInstance) -> Dict[Text, Union[Text, float]]:

        # construct the input
        input_instance = LLMQueryInstance(
            id=0,
            input=f"[\"{instance.text}\"]",
        )
        
        return self._agent([input_instance])[0]

    @overrides
    def _batch_score(self, instances: List[ScorerInstance]) -> List[Dict[Text, Union[Text, float]]]:

        # input_instances = [
        #     LLMQueryInstance(id=idx, input=instance.text)
        #     for idx, instance in enumerate(instances)
        # ]

        # we can chunk items into batches
        chunking = lambda xs: LLMQueryInstance(
            id=0,
            input="[" + ', '.join(['"{}"'.format(x.text) for x in xs]) + "]",
        )
        
        chunked_instances = []

        for i in range(0, len(instances), self._in_batch_num):
            chunked_instances.append(chunking(instances[i:i + self._in_batch_num]))

        chunked_results = self._agent(chunked_instances, silence=False)
        separated = []
        
        # print(chunked_instances[0])
        # print(chunked_results)
        
        for result in chunked_results:
            for idx, r in enumerate(result['parsed']):
                separated.append({
                    "raw": result['raw'] + f"[{idx}]",
                    "parsed": r
                })

        return separated
    
    @overrides
    async def _async_batch_score(self, instances: List[ScorerInstance]) -> AsyncGenerator[Tuple[int, Dict[Text, Union[Text, float]]], None]:
        """We can utilize the async_call function of the interface to make async calls.
        """
        
        chunking = lambda xs: LLMQueryInstance(
            id=0,
            input="[" + ', '.join(['"{}"'.format(x.text) for x in xs]) + "]",
        )
        
        chunked_instances = []

        for i in range(0, len(instances), self._in_batch_num):
            chunked_instances.append(chunking(instances[i:i + self._in_batch_num]))
            
        async for index, result in self._agent.async_call(chunked_instances, silence=True):
            for idx, r in enumerate(result['parsed']):
                yield index * len(chunked_instances) + idx, {
                    "raw": result['raw'] + f"[{idx}]",
                    "parsed": r
                }

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

        def _parse_input(instance: LLMQueryInstance) -> Dict[Text, Text]:
            """Generate the input dictionary for the LLM."""
            return {"sentence": instance.input}

        def _parse_output(output: Text) -> float:
            # TODO: find a better scaling factor
            return 1.0 if abs(float(output.strip()) - 1.0) < 1e-6 else 0.0

        self._agent = ChatInterface(
            model_name=self._model_name,
            batch_size=16,
            max_tokens=10,
            system_message="You are a helpful factchecker assistant." if self._base_url is None else None,
            instruction_prompt=[],
            input_example_prompt=SPECIFY_CHECKWORTHY_CATEGORY_PROMPT,
            output_example_prompt="",
            input_parser=_parse_input,
            output_parser=_parse_output,
            base_url=self._base_url,
            api_key=self._api_key,
        )

    @overrides
    def _score(self, instance: ScorerInstance) -> Dict[Text, Union[Text, float]]:

        # construct the input
        input_instance = LLMQueryInstance(
            id=0,
            input=instance.text,
        )

        return self._agent([input_instance])[0]

    @overrides
    def _batch_score(self, instances: List[ScorerInstance]) -> List[Dict[Text, Union[Text, float]]]:

        input_instances = [
            LLMQueryInstance(id=idx, input=instance.text)
            for idx, instance in enumerate(instances)
        ]

        return self._agent(input_instances, silence=False)


    @overrides
    async def _async_batch_score(self, instances: List[ScorerInstance]) -> AsyncGenerator[Tuple[int, Dict[Text, Union[Text, float]]], None]:
        input_instances = [
            LLMQueryInstance(id=idx, input=instance.text)
            for idx, instance in enumerate(instances)
        ]
        
        async for index, result in self._agent.async_call(input_instances, silence=True):
            yield index, result