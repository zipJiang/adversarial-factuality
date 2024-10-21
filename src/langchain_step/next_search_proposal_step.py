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

from ..utils.prompts import NEXT_SEARCH_PROMPT


@dataclass(frozen=True, eq=True)
class NextSearchProposalResponse(LLMResponse):
    next_query: Text

    
class NextSearchProposalOutputParser(BaseOutputParser[NextSearchProposalResponse]):
    @overrides
    def parse(self, text: Text) -> NextSearchProposalResponse:
        def _parse(output: Text) -> Text:
            _parse_output = lambda x: re.search(r"```(.+)```", x, re.DOTALL).group(1).strip()

            # define a second strip function that extract the content within the double quote
            _second_strip = lambda x: re.search(r"\"(.+)\"", x).group(1).strip()
            
            try:
                first_strip = _parse_output(output)
                if first_strip.startswith("\"") or first_strip.startswith("markdown"):
                    return _second_strip(first_strip)
                return first_strip
            except AttributeError:
                return "N/A"

        return NextSearchProposalResponse(
            messages=text,
            next_query=_parse(text)
        )
    
    
@Step.register("next-search-proposal")
class NextSearchProposalStep(Step):

    @overrides
    def get_prompt_template(self) -> Runnable:
        return ChatPromptTemplate.from_messages([
            ("human", NEXT_SEARCH_PROMPT)
        ])
    
    @overrides
    def get_output_parser(self) -> BaseOutputParser:
        return NextSearchProposalOutputParser()