"""A decontextualizer that refines the text to be
self-contained and context-free.
"""

import logging
from registrable import Registrable
from typing import Text, Optional, List, Union, Dict, Any
from langchain_core.runnables.config import RunnableConfig
from langchain_openai import ChatOpenAI
from langchain_interface.steps.decontextualization_step import (
    DecontextualizationStep,
    DecontextualizationResponse
)
from ..utils.instances import ScorerInstance, DecontextInstance, DecontextScorerInstance


logger = logging.getLogger(__name__)


class Decontextualizer(Registrable):
    """Take a sentence and make it standalone."""
    
    __NAME__ = "default"

    def __init__(
        self,
        model_name: Text,
        # example_path: Text,
        base_url: Optional[Text] = None,
        api_key: Optional[Text] = None,
    ):
        """ """
        super().__init__()
        # self._example_path = example_path
        self._model_name = model_name
        self._base_url = base_url
        self._api_key = api_key

        # self._example_selector = ConstantExampleSelector()
        # with open(example_path, "r", encoding="utf-8") as file_:
        #     text = file_.read()
        #     examples = text.split("#####")
        #     examples = [
        #         [item for item in example.strip().split("-----")]
        #         for example in examples
        #     ]
        #     for example in examples:
        #         assert len(example) == 3, f"Invalid example format: {example}"
        #         self._example_selector.add_example(
        #             {
        #                 "statement": example[0],
        #                 "response": example[1],
        #                 "revised": example[2],
        #             }
        #         )
                
        # def _parse_output(output: Text) -> Text:
        #     apl = lambda x: x.strip().split("```")[1].strip()
        #     result = ""
        #     try:
        #         result = apl(output)
        #     except IndexError:
        #         logger.warning(f"Failed to parse Decontextualizer output: \"{output}\"")
                
        #     return result

        # self._agent = ChatInterface(
        #     model_name=self._model_name,
        #     batch_size=32,
        #     max_tokens=512,
        #     system_message=None,
        #     instruction_prompt=[
        #         DECONTEXTUALIZE_PROMPT,
        #         "Sure, please provide me with statements you want me to revise.",
        #     ],
        #     example_selector=self._example_selector,
        #     input_example_prompt="STATEMENT:\n{statement}\n\nRESPONSE:\n{response}",
        #     output_example_prompt="REVISED STATEMENT:\n{revised}",
        #     input_parser=lambda x: {"statement": x.input, "response": x.sentence},
        #     # Should extract the one within wrapping ``` and strip
        #     output_parser=_parse_output,
        #     temperature=0.0,
        #     base_url=self._base_url,
        #     api_key=self._api_key,
        #     max_concurrency=32,
        # )

        self._llm = ChatOpenAI(
            model_name=self._model_name,
            base_url=self._base_url,
            api_key=self._api_key,
            # top_p=0.98,
            model_kwargs={"top_p": 0.98},
            temperature=0.0,
            max_tokens=512,
        )
        self._runnable_config = RunnableConfig(
            max_concurrency=32,
        )

        self._agent = DecontextualizationStep().chain_llm(self._llm)

    def __call__(
        self, instance: Union[DecontextScorerInstance, List[DecontextScorerInstance]],
        return_raw: bool = False
    ) -> Union[ScorerInstance, List[ScorerInstance]]:
        """This will decontextualize the instances, and return
        the decontextualized version paired with the correct information.
        """
        
        if isinstance(instance, list):
            result = self._batch_decontextualize(instances=instance)
            if not return_raw:
                result = [ScorerInstance(text=r["parsed"], topic=ins.topic, source_text=ins.source_text) for r, ins in zip(result, instance)]
        else:
            result = self._decontextualize(instance=instance)
            if not return_raw:
                result = ScorerInstance(text=result["parsed"], topic=instance.topic, source_text=instance.source_text)
                
        return result
        
    def _decontextualize(self, instance: DecontextScorerInstance) -> Dict[Text, Any]:
        """ """
        # instance = DecontextInstance(
        #     id=0,
        #     input=instance.text,
        #     sentence=instance.sent,
        # )
        
        instance = {
            "input": instance.text,
            "context": instance.sent
        }
        
        result = self._agent.invoke(instance, config=self._runnable_config)
        
        # if result["parsed"] == "":
        #     result['parsed'] = instance.text
            
        return {
            "raw": result.messages,
            "parsed": result.revised if result.revised is not None else instance.text,
        }
    
    def _batch_decontextualize(self, instances: List[DecontextScorerInstance]) -> List[Dict[Text, Any]]:
        """ """
        instances = [
            # DecontextInstance(
            #     id=i,
            #     input=instance.text,
            #     sentence=instance.sent,
            # )
            {
                "input": instance.text,
                "context": instance.sent,
            }
            for instance in instances
        ]
        
        results = self._agent.batch(instances, config=self._runnable_config)
        
        # If the decontextualizer fails to decontextualize, return the original text
        [
            {
                "raw": result.messages,
                "parsed": result.revised if result.revised is not None else instances[i].input,
            } for i, result in enumerate(results)
        ]
            # if result["parsed"] == "":
            #     result['parsed'] = instances[i].input
                
        return results


Decontextualizer.register("default")(Decontextualizer)