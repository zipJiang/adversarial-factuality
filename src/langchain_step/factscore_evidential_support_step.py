""" """

import re
import string
from dataclasses import dataclass
from overrides import overrides
from typing import Union, Text, List, Dict, Optional, Callable, Any

from langchain_core.runnables.config import RunnableConfig
from langchain_core.runnables.base import Runnable
from langchain_core.outputs import Generation, ChatGeneration
from langchain.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)
from langchain_core.output_parsers import BaseOutputParser

from langchain_interface.example_selectors import ConstantExampleSelector
from langchain_interface.steps.step import Step
from langchain_interface.instances.instance import LLMResponse, Instance


@dataclass(frozen=True, eq=True)
class FActScoreEvidentialSupportResponse(LLMResponse):
    evidential_support: float
    # We add reasoning chain here if applicable (no default to maintain easy extensibility)
    reasoning_content: Optional[List[Dict]]

class FActScoreEvidentialSupportOutputParser(BaseOutputParser[FActScoreEvidentialSupportResponse]):

    # TODO: making reasoning chain parsing a base operation that can be inherited
    # For example, making this a mixin
    
    @overrides
    def parse_result(self, result: list[Generation], *, partial = False):
        """We should have access to the reasoning chain here.
        """
        
        # I'm not sure how important this assumption is
        result_processing: List[ChatGeneration] = result
        reasoning_content = result_processing[0].message.additional_kwargs.get("reasoning_content", None)
        
        return self._parse_with_reasoning(
            result_processing[0].message.content,
            reasoning_content
        )

    @overrides
    
    def parse(
        self,
        text: Text
    ) -> Dict:
        generated_answer = text.strip().lower()
        is_supported = 0.0
        
        if "true" in generated_answer or "false" in generated_answer:
            if "true" in generated_answer and "false" not in generated_answer:
                is_supported = 1.0
            elif "false" in generated_answer and "true" not in generated_answer:
                is_supported = 0.0
            else:
                # I feel this is random tie breaking
                is_supported = generated_answer.index("true") > generated_answer.index("false")
                is_supported = 1.0 if is_supported else 0.0
        else:
            generated_answer = generated_answer.translate(str.maketrans("", "", string.punctuation)).split()
            is_supported = all([keyword not in generated_answer for keyword in ["not", "cannot", "unknown", "information"]])
            is_supported = 1.0 if is_supported else 0.0
            
        return FActScoreEvidentialSupportResponse(
            messages=text,
            evidential_support=is_supported,
            reasoning_content=None
        )
        
    def _parse_with_reasoning(
        self,
        text: Text,
        reasoning_content: Optional[List[Dict]]
    ) -> FActScoreEvidentialSupportResponse:
        generated_answer = text.strip().lower()
        is_supported = 0.0
        
        if "true" in generated_answer or "false" in generated_answer:
            if "true" in generated_answer and "false" not in generated_answer:
                is_supported = 1.0
            elif "false" in generated_answer and "true" not in generated_answer:
                is_supported = 0.0
            else:
                # I feel this is random tie breaking
                is_supported = generated_answer.index("true") > generated_answer.index("false")
                is_supported = 1.0 if is_supported else 0.0
        else:
            generated_answer = generated_answer.translate(str.maketrans("", "", string.punctuation)).split()
            is_supported = all([keyword not in generated_answer for keyword in ["not", "cannot", "unknown", "information"]])
            is_supported = 1.0 if is_supported else 0.0
            
        return FActScoreEvidentialSupportResponse(
            messages=text,
            evidential_support=is_supported,
            reasoning_content=reasoning_content
        )
    
    @property
    def _type(self) -> Text:
        return "factscore-evidential-support"
    
    
@Step.register("factscore-evidential-support")
class FActScoreEvidentialSupportStep(Step):
    @overrides
    def get_prompt_template(self) -> Runnable:
        
        return ChatPromptTemplate.from_messages(
            [
                ("human", (
                    "Answer the question about {topic} based on the given context.\n\n"
                    "{parsed_passages}\n\n"
                    "Input: {input} True or False?"
                ))
            ]
        )
        
    @overrides
    def get_output_parser(self) -> Runnable:
        return FActScoreEvidentialSupportOutputParser()
