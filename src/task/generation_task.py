"""Running a topic summarization generation.
"""

import json
import os
from overrides import overrides
from typing import Text
from langchain_interface.instances import LLMQueryInstance
from langchain_interface.interfaces import ChatInterface
from .task import Task


@Task.register("generation")
class GenerationTask(Task):
    def __init__(self, topic_path: Text, output_path: Text, prompt: Text):
        """ """
        super().__init__()

        self.prompt = prompt
        self.output_path = output_path

        with open(topic_path, "r", encoding="utf-8") as file_:
            self.topics = [
                LLMQueryInstance(id=lidx, input=line.strip(), output=None)
                for lidx, line in enumerate(file_)
            ]

        # TODO: process bad outputs
        self.agent = ChatInterface(
            model_name="gpt-3.5-turbo",
            batch_size=4,
            max_tokens=1024,
            system_message="",
            input_variables=["input"],
            instruction_prompt=[],
            input_example_prompt=self.prompt,
            output_parser=lambda x: x,
        )

    @overrides
    def run(self):

        outputs = self.agent(self.topics)

        with open(self.output_path, "w", encoding="utf-8") as file_:
            data = [
                {"topic": topic.input, "output": output}
                for topic, output in zip(self.topics, outputs)
            ]

            json.dump(data, file_, indent=4)
