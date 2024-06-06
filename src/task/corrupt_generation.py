"""A task that takes decomposed text, corrupts some of the
facts, and merge back with extant merging prompt.
"""

import json
import os
from overrides import overrides
from dataclasses import dataclass
from random import Random
from .task import Task
from typing import Text, Dict, List, Optional
from langchain_interface.interfaces import ChatInterface
from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
from langchain_interface.instances import LLMQueryInstance
from src.utils.instances import ScorerInstance
from src.scorer import Scorer


@dataclass(frozen=True)
class MergeQueryInstance(LLMQueryInstance):
    """ """

    topic: Optional[Text] = None


@Task.register("corrupt-generation")
class CorruptGenerationTask(Task):
    """ """

    def __init__(
        self,
        fact_checker: Scorer,
        generation_path: Text,
        corruption_ratio: float,
        output_dir: Text,
        cache_path_overwrite: Optional[Text] = None,
        merge_cache_path_overwrite: Optional[Text] = None,
        seed: int = 42,
    ):
        """ """
        
        super().__init__()
        
        self._generation_path = generation_path
        self._fact_checker = fact_checker
        self._corruption_ratio = corruption_ratio
        self._cache_path_overwrite = cache_path_overwrite
        self._merge_cache_path_overwrite = merge_cache_path_overwrite
        self._random_obj = Random(seed)
        self._output_dir = output_dir

        self._corrupt_agent = ChatInterface(
            model_name="gpt-3.5-turbo-0125",
            batch_size=4,
            max_tokens=128,
            temperature=0.0,
            instruction_prompt=[
                "You are given some factually correct statements. Your task is to modify the statements to make them factually incorrect. Try to make the edits atomic without any additional output.",
                "Sure, please provide the factually correct statements that you would like me to modify.",
            ],
            input_example_prompt="{input}",
            input_parser=lambda x: x.input,
            output_parser=lambda x: x,
            max_concurrency=4,
        )

        self._merge_agent = ChatInterface(
            model_name="gpt-4-turbo-2024-04-09",
            batch_size=4,
            max_tokens=512,
            temperature=0.0,
            input_example_prompt="You will get an instruction and a set of facts that are true. Construct an answer using ONLY the facts provided, and try to use all facts as long as its possible. If no facts are given, reply to the instruction incorporating the fact that you dont know enough to fully respond. \n\nThe facts:\n {claim_string}\n\nThe instruction:\n{prompt}",
            input_parser=lambda x: {
                "prompt": f"Tell me a bio of {x.topic}.",
                "claim_string": {x.input},
            },
            output_parser=lambda x: x,
            max_concurrency=4,
        )

    @overrides
    def run(self):
        """ """

        with open(self._generation_path, "r", encoding="utf-8") as file_:
            generation = json.load(file_)

        all_claims = []

        for entry in generation:
            topic = entry["topic"]
            raw = entry["output"]["raw"]
            claims = entry["meta"]["claims"]

            # create scorer instance
            all_claims.extend(
                [
                    ScorerInstance(text=claim, source_text=raw, topic=topic)
                    for claim in claims
                ]
            )

        scores = self._fact_checker(all_claims)
        
        os.makedirs(self._output_dir, exist_ok=True)

        set_llm_cache(SQLiteCache(self._cache_path_overwrite))

        corruption_input = [
            LLMQueryInstance(id=idx, input=claim.text)
            for idx, (claim, score) in enumerate(zip(all_claims, scores))
            if score > 0.5 and self._random_obj.random() < self._corruption_ratio
        ]

        corrupted = self._corrupt_agent(corruption_input)

        for ipt, opt in zip(corruption_input, corrupted):
            all_claims[ipt.id] = ScorerInstance(
                text=opt["parsed"],
                source_text=all_claims[ipt.id].source_text,
                topic=all_claims[ipt.id].topic,
            )

        claim_combinations = {}

        for claim in all_claims:
            if claim.topic not in claim_combinations:
                claim_combinations[claim.topic] = []
            claim_combinations[claim.topic].append(claim.text)

        with open(os.path.join(self._output_dir, "corrupted_claims.json"), "w", encoding="utf-8") as file_:
            json.dump(claim_combinations, file_, ensure_ascii=False, indent=4)

        merge_inputs = [
            MergeQueryInstance(
                id=idx,
                input="\n".join([f"{cidx + 1}. {c}" for cidx, c in enumerate(claims)]),
                topic=topic,
            )
            for idx, (topic, claims) in enumerate(claim_combinations.items())
        ]
        
        set_llm_cache(SQLiteCache(self._merge_cache_path_overwrite))
        merge_outputs = self._merge_agent(merge_inputs)
        
        # for mo in merge_outputs:
        #     print("=" * 20)
        #     print(mo['parsed'])
        #     print("=" * 20)
        
        result = [{"topic": mi.topic, "output": mo} for mi, mo in zip(merge_inputs, merge_outputs)]
        
        with open(os.path.join(self._output_dir, "corrupted.jsonl"), "w", encoding="utf-8") as file_:
            for entry in result:
                file_.write(json.dumps(entry, ensure_ascii=False) + "\n")