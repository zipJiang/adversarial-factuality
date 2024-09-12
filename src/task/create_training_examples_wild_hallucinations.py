"""Generate training examples from the given prompt.
"""
import ujson as json
from .task import Task
from overrides import overrides
from typing import List, Text, Optional
from ..example_parsers.example_parser import ExampleParser
from langchain_interface.instances import LLMQueryInstance
from langchain_interface.interfaces import ChatInterface
from langchain_interface.example_selectors import ConstantExampleSelector


@Task.register("create-training-examples-wild-hallucinations")
class CreateTraniningExamplesWildHallucinations(Task):
    """
    """

    __NAME__ = "create-training-examples-wild-hallucinations"
    
    def __init__(
        self,
        model_name: Text,
        input_example_prompt: Text,
        output_example_prompt: Text,
        num_samples_per_topic: int,
        example_path: Text,
        input_path: Text,
        output_path: Text,
        instruction_prompts: Optional[List[Text]] = [],
        example_parser: Optional[ExampleParser] = ExampleParser(),
        api_key: Optional[Text] = None,
        base_url: Optional[Text] = None,
    ):
        """
        """
        super().__init__()
        
        self._example_selector = ConstantExampleSelector()
        self._example_parser = example_parser

        # with open(example_path, 'r', encoding='utf-8') as file_:
        #     lines = file_.readlines()
        #     for i in range(0, len(lines), 2):
        #         self._example_selector.add_example({
        #             "topic": lines[i].strip(),
        #             "generation": lines[i+1].strip()
        #         })
        examples = self._example_parser(example_path)
        for example in examples:
            self._example_selector.add_example({"topic": example.input_text, "generation": example.generation})

        self._agent = ChatInterface(
            model_name=model_name,
            batch_size=64,
            max_tokens=512,
            instruction_prompt=instruction_prompts,
            example_selector=self._example_selector,
            input_example_prompt=input_example_prompt,
            output_example_prompt=output_example_prompt,
            input_parser=lambda x: {"topic": x.input},
            output_parser=lambda x: x,
            temperature=0.7,
            top_p=0.9,
            api_key=api_key,
            base_url=base_url,
        )
        
        self.input_path = input_path
        self.output_path = output_path
        self.num_samples_per_topic = num_samples_per_topic
        
    @overrides
    def run(self):
        """
        """

        with open(self.input_path, 'r', encoding='utf-8') as file_:
            # topics = [line.strip() for line in file_]
            data = [json.loads(line) for line in file_]
            topics = [f"{ldata['topic']} in {ldata['category']}" for ldata in data]
            inputs = [LLMQueryInstance(id=0, input=topic, output=None) for topic in topics for _ in range(self.num_samples_per_topic)]

            outputs = self._agent(inputs, silence=False,)
            
        with open(self.output_path, 'w', encoding='utf-8') as file_:
            for ipt, opt in zip(inputs, outputs):
                file_.write(json.dumps({"input": ipt.input, "output": opt}) + '\n')