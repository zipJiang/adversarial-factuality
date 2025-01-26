"""This is the claim extractor
following the VeriScore extraction pattern.
"""

import re
import string
from dataclasses import dataclass
from overrides import overrides
from typing import Union, Text, List, Dict, Optional, Callable, Any

from langchain_core.runnables.config import RunnableConfig
from langchain_core.runnables.base import Runnable
from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)
from langchain_core.output_parsers import BaseOutputParser

from langchain_interface.example_selectors import ConstantExampleSelector
from langchain_interface.steps.step import Step
from langchain_interface.instances.instance import LLMResponse, Instance


@dataclass(frozen=True, eq=True)
class VeriScoreExtractionResponse(LLMResponse):
    claims: List[Text]
    

class VeriScoreExtractionOutputParser(BaseOutputParser[VeriScoreExtractionResponse]):
    """Parse the output of the decomposition model.
    """
    def parse(self, text: Text):
        # I don't understand this but it is what is used in VeriScore
        try:
            text.index("No verifiable claim.")
            return VeriScoreExtractionResponse(
                messages=text,
                claims=[]
            )
        except ValueError:
            clean_output = text.split("### Response:")[-1].strip().replace("</s>", "") 
            return VeriScoreExtractionResponse(
                messages=text,
                claims=[c.strip() for c in clean_output.split("\n")]
            )
    
    @property
    def _type(self) -> str:
        return "veriscore-extraction"
    
    
@Step.register("veriscore-extraction")
class VeriScoreExtractionStep(Step):
    """When using the tuned model, the decomposition does
    not use fewshot examples. (And no system prompt as well)

    And it is important to notice that the model is not used through the chat.completion API.
    """
    
    @overrides
    def get_prompt_template(self) -> Runnable:
        """ """
        return PromptTemplate.from_template(
            "Below is an instruction that describes a task, "
            "paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n"
            "You are trying to verify how factual a piece of text is. "
            "To do so, you need to break down a sentence and extract as "
            "many fine-grained facts mentioned in the sentence as possible. "
            "Each of these fine-grained facts should be verifiable against "
            "reliable external world knowledge (e.g., via Wikipedia). "
            "Any story, personal experiences, hypotheticals (e.g., \"would be\" or subjunctive), "
            "subjective statements (e.g., opinions), suggestions, "
            "advice, instructions, and other such content "
            "should not be included in the list. Biographical, historical, scientific, "
            "and other such texts are not personal experiences or stories. "
            "You should extract verifiable facts from them. Each fact should also be "
            "describing either one single event (e.g., \"Nvidia is founded in 1993 in Sunnyvale, California, U.S.\") "
            "or single state (e.g., \"UMass Amherst has existed for 161 years.\") "
            "with necessary time and location information. Quotations should be extracted verbatim with the source when available. "
            "Listed references should be ignored.\n\n"
            "Extract fine-grained facts from the sentence marked between <SOS> and <EOS>. "
            "You should focus on the named entities and numbers in the sentence and extract relevant information from the sentence. "
            "Other sentences are only context for you to recover pronouns, definite phrases (e.g., \"the victims\" or \"the pope\"), "
            "and so on. Each fact should be understandable on its own and require no additional context. "
            "This means that all entities must be referred to by name but not pronoun. "
            "Use the name of entities rather than definite noun phrases (e.g., 'the teacher') whenever possible. "
            "If a definite noun phrase is used, be sure to add modifiers (e.g., a embedded clause, a prepositional phrase, etc.). "
            "Each fact must be situated within relevant temporal and location whenever needed. "
            "Keep each fact to one sentence with zero or at most one embedded clause.\n\n"
            "If there is no verifiable fact in the sentence, please write \"No verifiable claim.\"\n\n"
            "### Question:\n"
            # "{before}<SOS>{sentence}<EOS>{after}\n\n"
            "{question}\n\n"
            "### Response:\n"
            "{input}\n\n"
            "### Facts:\n"
        )
        
    def get_output_parser(self):
        return VeriScoreExtractionOutputParser()