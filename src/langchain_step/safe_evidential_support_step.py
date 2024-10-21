""" Evidential support used by SAFE """

import re
from dataclasses import dataclass
from overrides import overrides
from typing import Union, Text, List, Dict, Optional, Callable, Any

from langchain_core.runnables.config import RunnableConfig
from langchain_core.runnables.base import Runnable
from langchain.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)
from langchain_core.output_parsers import BaseOutputParser

from langchain_interface.example_selectors import ConstantExampleSelector
from langchain_interface.steps.step import Step
from langchain_interface.instances.instance import LLMResponse, Instance

from ..utils.prompts import SAFE_RATING_PROMPT


@dataclass(frozen=True, eq=True)
class SAFEEvidentialSupportResponse(LLMResponse):
    evidential_support: float
    
    
class SAFEEvidentialSupportOutputParser(BaseOutputParser[SAFEEvidentialSupportResponse]):

    def parse(self, text: Text) -> Dict:
        generated_answer = text.strip().lower()
        
        # extract the answer within the bracket if exists
        answer_tag = re.search(r"\[(.*)\]", generated_answer)

        if answer_tag is None:
            return 0.0
        
        answer = answer_tag.group(1).strip()
        is_supported = 0.0
        
        if answer == "supported" or answer == "\"supported\"":
            is_supported = 1.0
        
        return is_supported
    
    @property
    def _type(self) -> Text:
        return "safe-evidential-support"
    
    
@Step.register("safe-evidential-support")
class SAFEEvidentialSupportStep(Step):
    @overrides
    def get_prompt_template(self) -> Runnable:
        
        return ChatPromptTemplate.from_messages(
            [
                ("human", SAFE_RATING_PROMPT)
            ]
        )
    
    @overrides
    def get_output_parser(self) -> Runnable:
        return SAFEEvidentialSupportOutputParser()