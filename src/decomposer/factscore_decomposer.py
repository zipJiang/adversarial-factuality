"""Replicate the FactScore Decomposer
that takes in a summary and decomposes it into
atomic facts.
"""

import spacy
from overrides import overrides
from typing import List, Text, Optional
from langchain_interface.instances import LLMQueryInstance
from langchain_interface.example_selectors import ConstantExampleSelector
from langchain_interface.interfaces import ChatInterface
from ..utils.instances import ScorerInstance
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

        def _split_atomic_facts(output: Text) -> List[Text]:
            """Split the atomic facts from the output.
            each line is an atomic fact start with '- ',
            need to remove the '- ' from the start of the line.
            """

            return [
                line[2:].strip() for line in output.split("\n") if line.startswith("- ")
            ]

        # TODO: include all the examples, and try retrieving
        self._example_selector = ConstantExampleSelector()
        with open(self._example_path, "r", encoding="utf-8") as file_:
            items = file_.read().split("\n\n")
            for item in items:
                lines = item.split("\n")
                example = {
                    "input": lines[0],
                    "output": "\n".join(lines[1:]),
                }
                self._example_selector.add_example(example)

        self._agent = ChatInterface(
            model_name=self._model_name,
            batch_size=4,
            max_tokens=512,
            system_message=None,
            instruction_prompt=[],
            input_example_prompt="Please breakdown the following sentence into independent facts: {input}",
            output_example_prompt="{output}",
            output_parser=_split_atomic_facts,
            example_selector=self._example_selector,
            base_url=self._base_url,
            api_key=self._api_key
        )

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
            outputs = self._agent([LLMQueryInstance(id=0, input=instance_text)], silence=True)
        else:
            outputs = self._agent([LLMQueryInstance(id=sidx, input=sentence.text) for sidx, sentence in enumerate(self._nlp(instance_text).sents)], silence=True)
        
        return [ScorerInstance(text=atom, topic=topic) for otp in outputs for atom in otp['parsed']]