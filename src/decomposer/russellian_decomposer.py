"""Similar to FactScore Decomposer, but adopt
a different prompt and examples.
"""

import spacy
from overrides import overrides
from typing import List, Text, Optional, Tuple
from langchain_interface.instances import LLMQueryInstance
from langchain_interface.example_selectors import ConstantExampleSelector
from langchain_interface.interfaces import ChatInterface
from ..utils.instances import ScorerInstance
from .factscore_decomposer import FActScoreDecomposer
from .decomposer import Decomposer


@Decomposer.register("russellian")
class RussellianDecomposer(FActScoreDecomposer):
    
    __NAME__ = "russellian"
    
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
        
        super().__init__(
            model_name=model_name,
            example_path=example_path,
            nlp_model_name=nlp_model_name,
            sentencize=sentencize,
            base_url=base_url,
            api_key=api_key
        )
        
        # update agent
        
        def _split_atomic_facts(output: Text) -> List[Text]:
            """Split the atomic facts from the output.
            each line is an atomic fact start with '- ',
            need to remove the '- ' from the start of the line.
            """

            return [
                line[2:].strip() for line in output.split("\n") if line.startswith("- ")
            ]
        
        self._agent = ChatInterface(
            model_name=self._model_name,
            batch_size=32,
            max_tokens=512,
            system_message=None,
            instruction_prompt=[],
            input_example_prompt="Please decompose the following sentence into individual facts: {input}",
            output_example_prompt="{output}",
            output_parser=_split_atomic_facts,
            example_selector=self._example_selector,
            base_url=self._base_url,
            api_key=self._api_key,
            max_concurrency=32
        )