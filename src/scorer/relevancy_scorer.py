"""
"""

from typing import Text, Dict, List, Union, Optional
import re
from overrides import overrides
from langchain_interface.instances import LLMQueryInstance
from langchain_interface.example_selectors import ConstantExampleSelector
from langchain_interface.interfaces import ChatInterface
from ..utils.instances import ScorerInstance, RelevancyInstance, RelevancyScorerInstance
from ..utils.prompts import RELEVANCY_PROMPT
from .scorer import Scorer


@Scorer.register("relevancy-scorer")
class RelevancyScorer(Scorer):
    """ """
    
    __NAME__ = "relevancy-scorer"

    def __init__(
        self,
        generation_prompt_template: Text,
        model_name: Text,
        example_path: Text,
        base_url: Optional[Text] = None,
        api_key: Optional[Text] = None,
    ):
        """ """

        super().__init__()
        self._generation_prompt_template = generation_prompt_template
        self._model_name = model_name
        self._base_url = base_url
        self._api_key = api_key

        self._example_path = example_path
        self._example_selector = ConstantExampleSelector()

        with open(example_path, "r", encoding="utf-8") as file_:
            text = file_.read()
            examples = text.split("#####")
            examples = [
                [item for item in example.strip().split("-----")]
                for example in examples
            ]
            for example in examples:
                assert len(example) == 4, f"Invalid example format: {example}"
                self._example_selector.add_example(
                    {
                        "question": example[0],
                        "response": example[1],
                        "statement": example[2],
                        "solution": example[3],
                    }
                )
                
        def _parse_output(output: Text) -> float:
            search_result = re.search(r"\[(.*?)\]", output)
            if search_result is None:
                return 0.0
            return 1.0 if search_result.group(1).strip() == "Foo" else 0.0

        self._agent = ChatInterface(
            model_name=self._model_name,
            batch_size=32,
            max_tokens=512,
            system_message=None,
            instruction_prompt=[
                RELEVANCY_PROMPT,
                "Sure, please provide me with a statement and a question you want me to check relevancy for.",
            ],
            example_selector=self._example_selector,
            input_example_prompt="QUESTION:\n{question}\n\nRESPONSE:\n{response}\n\nSTATEMENT:\n{statement}",
            output_example_prompt="SOLUTION:\n{solution}",
            input_parser=lambda x: {
                "question": x.question,
                "response": x.sentence,
                "statement": x.input,
            },
            # extract the text inside [].
            output_parser=_parse_output,
            temperature=0.0,
            base_url=self._base_url,
            api_key=self._api_key,
            max_concurrency=32,
        )

    @overrides
    def _score(self, instance: ScorerInstance) -> Dict[Text, Union[Text, float]]:

        instance: RelevancyScorerInstance

        # construct the input
        # TODO: Notice that this is very restricted,
        # but we do need to have access to the question prompt.
        input_instance = RelevancyInstance(
            id=0,
            input=instance.text,
            question=self._generation_prompt_template.format(topic=instance.topic),
            sentence=instance.sent,
        )

        return self._agent([input_instance])[0]

    @overrides
    def _batch_score(
        self, instances: List[ScorerInstance]
    ) -> List[Dict[Text, Union[Text, float]]]:

        input_instances = [
            RelevancyInstance(
                id=idx,
                input=instance.text,
                question=self._generation_prompt_template.format(topic=instance.topic),
                sentence=instance.sent,
            )
            for idx, instance in enumerate(instances)
        ]
        
        # for ins in input_instances:
        #     print("=" * 20)
        #     print(ins.input)
        #     print(ins.question)
        #     print(ins.sentence)
        #     print("=" * 20)

        return self._agent(input_instances, silence=False)
