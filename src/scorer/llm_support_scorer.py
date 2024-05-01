"""Adopting an LLM to do Evidential support scoring.
"""

from dataclasses import dataclass, field
import string
from typing import Text, Dict, List, Union
import os
from overrides import overrides
from langchain_interface.instances import LLMQueryInstance
from langchain_interface.interfaces import ChatInterface
from ..utils.instances import ScorerInstance
from .scorer import Scorer
from ..retriever.retriever import Retriever


@dataclass(frozen=True, eq=True)
class FActScoreQueryInstance(LLMQueryInstance):
    topic: Text = field(default=None)
    passages: List[Dict[Text, Text]] = field(default_factory=list)


@Scorer.register("llm-support")
class LLMSupportScorer(Scorer):
    
    __NAME__ = "llm-support"

    def __init__(
        self,
        db_path: Text,
        cache_dir: Text
    ):
        """
        """
        super().__init__()
        
        def _parse_input(instance: FActScoreQueryInstance) -> Dict[Text, Text]:
            """Generate the input dictionary for the LLM.
            """
            return {
                "parsed_passages": '\n\n'.join([f"Title: {passage['title']} Text: {passage['text']}" for passage in instance.passages]) + '\n\n',
                "topic": instance.topic,
                "input": instance.input
            }
            
        def _parse_output(output: Text) -> float:
            """Parse the output of the LLM.
            """
            generated_answer = output.strip().lower()
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
                
            return is_supported
        
        self._agent = ChatInterface(
            model_name="gpt-3.5-turbo",
            batch_size=1,
            max_tokens=10,
            system_message="",
            input_variables=["input", "topic", "parsed_passages"],
            instruction_prompt=[],
            input_example_prompt="".join([
                "Answer the question about {topic} based on the given context.\n\n",
                "{parsed_passages}\n\n",
                "Input: {input} True or False?\nOutput:"
            ]),
            output_example_prompt="",
            input_parser=_parse_input,
            output_parser=_parse_output,
        )
        
        self._db_path = db_path
        self._cache_dir = cache_dir
        self._retriever = Retriever(
            db_path=self._db_path,
            cache_path=os.path.join(self._cache_dir, "retriever-cache.json"),
            embed_cache_path=os.path.join(self._cache_dir, "retriever-embed-cache.pkl"),
            batch_size=256,
        )
        
    @overrides
    def _score(self, instance: ScorerInstance) -> Dict[Text, Union[Text, float]]:
        """
        """
        
        # first convert the input instance into the FActScoreQueryInstance
        # by retrieve from the database
        passages = self._retriever.get_passages(instance.topic, instance.text, 5)
        
        input_instance = FActScoreQueryInstance(
            id=0,
            topic=instance.topic,
            passages=passages,
            input=instance.text
        )
        
        result = self._agent([input_instance])[0]
        
        return result