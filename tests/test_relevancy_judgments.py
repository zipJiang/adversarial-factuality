""""""

import unittest
from src.scorer import RelevancyScorer
from src.utils.instances import RelevancyScorerInstance


class TestRelevancyScorer(unittest.TestCase):
    def setUp(self):
        self._relevancy_scorer = RelevancyScorer(
            generation_prompt_template="tell me a bio about {topic}.",
            model_name="mistralai/Mistral-7B-Instruct-v0.2",
            example_path="relevancy_examples.txt",
            base_url="http://localhost:9871/v1",
            api_key="token-abc123",
        )
        
        self._test_cases = [
            RelevancyScorerInstance(
                text="Geofrey Chaucer is a poet.",
                sent="Kalki Koechlin is an Indian actress and writer known for her work in Hindi films. Geofrey Chaucer is a poet.",
                topic="Kalki Koechlin",
            ),
            RelevancyScorerInstance(
                text="France is a country in Europe.",
                sent="Adil Rami is a professional French footballer who was born on December 27, 1985, in Bastia, France.",
                topic="Adil Rami",
            ),
            RelevancyScorerInstance(
                text="Miroslav Djukic once coached AC Milan.",
                sent="Adil Rami is a professional French footballer once played for AC Milan, coached by Miroslav Djukic.",
                topic="Adil Rami",
            )
        ]
        
    def test_score(self):
        
        print(self._relevancy_scorer(self._test_cases, return_raw=True))