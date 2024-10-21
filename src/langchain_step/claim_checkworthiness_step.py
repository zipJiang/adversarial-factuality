""" """

import re
import string
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

from ..utils.prompts import SPECIFY_CHECKWORTHY_CATEGORY_PROMPT


@dataclass(frozen=True, eq=True)
class ClaimCheckworthinessResponse(LLMResponse):
    checkworthiness: float

    
@dataclass(frozen=True, eq=True)
class ClaimCheckworthinessOutputParser(BaseOutputParser[ClaimCheckworthinessResponse]):
    def parse(self, text: Text) -> Dict:
        return ClaimCheckworthinessResponse(
            messages=text,
            checkworthiness=1.0 if abs(float(text.strip()) - 1.0) < 1e-6 else 0.0,
        )
    
    @property
    def _type(self) -> Text:
        return "claim-checkworthiness"

    
@Step.register("claim-checkworthiness")
class ClaimCheckworthinessStep(Step):
    @overrides
    def get_prompt_template(self) -> Runnable:
        return ChatPromptTemplate.from_messages(
            [
                ("human", SPECIFY_CHECKWORTHY_CATEGORY_PROMPT)
            ]
        )
    
    @overrides
    def get_output_parser(self) -> Runnable:
        return ClaimCheckworthinessOutputParser()