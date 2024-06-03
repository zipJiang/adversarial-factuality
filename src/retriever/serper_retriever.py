"""A retriever that retrieves with search result from SerperAPI.
"""

import re
import numpy as np
from .retriever import Retriever
from typing import Dict, List, Text, Tuple, Any
from overrides import overrides
from langchain_interface.interfaces import ChatInterface
from ..utils.prompts import NEXT_SEARCH_PROMPT
from ..utils.query_serper import CachedSerperAPI
from ..utils.instances import NextSearchInstance


@Retriever.register("serper")
class SerperRetriever(Retriever):

    __NAME__ = "serper"
    
    def __init__(
        self,
        model_name: Text,
        base_url: Text,
        api_key: Text,
        serper_api_key: Text,
        cache_path: Text,
        top_k: int = 3
    ):
        """
        """
        self._cached_serper_api = CachedSerperAPI(
            serper_api_key=serper_api_key,
            cache_path=cache_path,
            k=top_k
        )
        
        def _parse_output_with_exception_handler(output: Text) -> Text:
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
        
        self._agent = ChatInterface(
            model_name=model_name,
            batch_size=32,
            max_tokens=512,
            system_message=None,
            instruction_prompt=None,
            input_example_prompt=NEXT_SEARCH_PROMPT,
            input_parser=lambda x: {"input": x.input.strip(), "knowledge": '\n'.join(x.knowledge) if x.knowledge else 'N/A'},
            # Extract from ``` ```
            output_parser=_parse_output_with_exception_handler,
            max_concurrency=32,
            base_url=base_url,
            api_key=api_key,
        )
        
    @overrides
    def get_passages(self, topic: Text, question: Text, k: int) -> List[Dict[Text, Any]]:
        """Retrieve passages from maybe search.
        """
        
        maximum_attempts = k
        
        def maybe_search(search_instance: NextSearchInstance) -> Tuple[NextSearchInstance, bool]:
            """The bool return value indicates whether this instance needs t
            be further extended.
            """
            
            needs_further_search = False

            query = self._agent([search_instance], silence=True)[0]['parsed']
            
            if query == "N/A":
                return search_instance, False
            
            knowledge_paragraph = self._cached_serper_api(query)
            
            if len(knowledge_paragraph) > 0:
                search_instance = NextSearchInstance(
                    id=search_instance.id,
                    input=search_instance.input,
                    output=search_instance.output,
                    knowledge=search_instance.knowledge + [knowledge_paragraph],
                )
                needs_further_search = True

            return search_instance, needs_further_search
        
        cinstance = NextSearchInstance(
            id=0,
            input=f"{topic}: {question}",
            output="",
            knowledge=[],
        )
        
        for _ in range(maximum_attempts):
            cinstance, needs_further_search = maybe_search(cinstance)
            
            if not needs_further_search:
                break
            
        return [{"title": topic, "text": '\n\n'.join(cinstance.knowledge)}]
    
    @overrides
    def get_passages_batched(self, topics: List[Text], questions: List[Text], k: int) -> List[List[Dict[Text, Any]]]:
        """Run the similar retrieval logic over multiple instances.
        """
        
        maximum_attempts = k
        
        def maybe_search_batched(search_instance_tuples: List[Tuple[NextSearchInstance, bool]]) -> List[Tuple[NextSearchInstance, bool]]:
            """
            """
            
            round_indices = [idx for idx, (_, init_status) in enumerate(search_instance_tuples) if init_status]

            if not round_indices:
                return search_instance_tuples
            
            search_instances = [search_instance_tuples[index][0] for index in round_indices]
            status_indicator = np.zeros(len(search_instances), dtype=np.bool_)
            
            # first query the agent for query updates
            queries = [item['parsed'] for item in self._agent(search_instances, silence=False)]
            
            status_updates = np.array([query != "N/A" for query in queries], dtype=np.bool_)
            status_indicator = (status_indicator | status_updates).tolist()
            
            # select queries and indices
            queries = [query for query, status in zip(queries, status_indicator) if status]

            round_indices = [index for index, status in zip(round_indices, status_indicator) if status]

            # now we asynchronously query the serper api cached
            results: List[Text] = self._cached_serper_api.batched_call(queries)

            # negates all the status indicators
            search_instance_tuples = [
                (sinstance, False) for sinstance, _ in search_instance_tuples
            ]
            
            for index, result in zip(round_indices, results):
                if len(result) > 0:
                    search_instance_tuples[index] = (
                        NextSearchInstance(
                            id=search_instance_tuples[index][0].id,
                            input=search_instance_tuples[index][0].input,
                            output=search_instance_tuples[index][0].output,
                            knowledge=search_instance_tuples[index][0].knowledge + [result],
                        ), True
                    )
                else:
                    search_instance_tuples[index] = (search_instance_tuples[index][0], False)

            return search_instance_tuples
                    
        # first create init search_next instances
        search_instance_tuples = [
            (NextSearchInstance(id=idx, input=f"{topic}: {question}", output="", knowledge=[]), True)
            for idx, (topic, question) in enumerate(zip(topics, questions))
        ]

        for _ in range(maximum_attempts):
            search_instance_tuples = maybe_search_batched(search_instance_tuples)
            
        return [[{"title": topic, "text": '\n\n'.join(search_instance.knowledge)}] for topic, (search_instance, _) in zip(topics, search_instance_tuples)]