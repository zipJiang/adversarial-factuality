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

from ..utils.prompts import CHECKWORTHY_PROMPT


@dataclass(frozen=True, eq=True)
class SentCheckworthinessResponse(LLMResponse):
    check_worthiness: List[float]

    
class SentCheckworthinessOutputParser(BaseOutputParser[SentCheckworthinessResponse]):
    def parse(self, text: Text) -> Dict:
        output = text.strip().lower()
        return [1.0 if re.search("yes", rsb) is not None else 0.0 for rsb in output.split(",")]
    
    def _type(self) -> Text:
        return "sent-checkworthiness"
    
    
@Step.register("sent-checkworthiness")
class SentCheckworthinessStep(Step):
    """ """
    @overrides
    def get_prompt_template(self) -> Runnable:
        """ """
        return ChatPromptTemplate.from_messages(
            [
                ("human", CHECKWORTHY_PROMPT)
            ]
        )
        
    @overrides
    def get_output_parser(self) -> BaseOutputParser:
        """ """
        return SentCheckworthinessOutputParser()