""""""
from unittest import TestCase
from src.utils.instances import ScorerInstance
from src.decomposer.factscore_decomposer import FActScoreDecomposer
from src.decomposer.deduplicated_decompser import DeduplicatedDecomposer
from src.scorer.llm_checkworthy_scorer import (
    LLMGeneralCheckWorthyScorer,
    LLMSpecificCheckWorthyScorer
)
from src.entailer.entailer import Entailer


class TestDecomposition(TestCase):
    def setUp(self) -> None:
        
        from langchain.globals import set_llm_cache
        from langchain.cache import SQLiteCache
        set_llm_cache(SQLiteCache(".cache/.gpt-3.5-turbo-cache.db"))
        self.test_cases = [
            ScorerInstance(
                text="There does not appear to be solid consensus on how best to do few-shot prompting, and the optimal prompt compilation will likely vary by model.",
                topic="LangChain"),
            ScorerInstance(
                text="In addition to his acting roles, Bateman has written and directed two short films and is currently in development on his feature debut.",
                topic="Bateman"),
            ScorerInstance(
                text="Emmett Skilton (born 23 September 1987) is a New Zealand actor and director.",
                topic="Emmett Skilton"),
            ScorerInstance(
                text="Toyoko Tokiwa is a Japanese author. \nToyoko Tokiwa is a well-known writer from Japan. \nToyoko Tokiwa is a respected literary figure in Japan. \nToyoko Tokiwa is a renowned Japanese novelist. \nToyoko Tokiwa is a celebrated author in the Japanese literary world.",
                topic="Toyoko Tokiwa"),
        ]
    
    def test_fact_score_decomposer(self):
        decomposer = FActScoreDecomposer(
            example_path="data/factscore_decomp_examples.txt",
        )
        
        for case in self.test_cases:
            results = [r.text for r in decomposer(case)]
            
            print("=" * 20)
            print(f"Decomposed by: {decomposer.__NAME__}")
            print('\n'.join(results))
            print("=" * 20)
            
    def test_filtered_fact_score_decomposer(self):
        decomposer = DeduplicatedDecomposer(
            base_decomposer=FActScoreDecomposer(
                example_path="data/factscore_decomp_examples.txt",
                sentencize=False
            ),
            sentence_level_checkworthy_scorer=LLMGeneralCheckWorthyScorer(),
            claim_level_checkworthy_scorer=LLMSpecificCheckWorthyScorer(),
            entailer=Entailer(
                model_name="ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli",
                internal_batch_size=16,
                max_length=256,
            ),
            sentencize=True,
        )
        
        for case in self.test_cases:
            results = [r.text for r in decomposer(case)]
            
            print("=" * 20)
            print(f"Decomposed by: {decomposer.__NAME__}")
            print('\n'.join(results))
            print("=" * 20)