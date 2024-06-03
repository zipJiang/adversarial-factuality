"""Test the retriever with serper cached.
"""

import os
from unittest import TestCase
from src.retriever.serper_retriever import SerperRetriever


class TestSerperRetrieval(TestCase):
    """
    """

    def setUp(self):
        
        self._retriever = SerperRetriever(
            model_name="mistralai/Mistral-7B-Instruct-v0.2",
            base_url="http://localhost:9871/v1",
            api_key="token-abc123",
            serper_api_key=os.environ["SERPER_API_KEY"],
            cache_path=".cache/serper/test_serper_cache.db",
        )
        
        self.test_decomposed_cases_raw = [
            [
                "Kalki Koechlin is an Indian actress.",
                "Kalki Koechlin is a writer.",
                "Kalki Koechlin is known for her work in Hindi films.",
                "Kalki was born on January 10, 1984.",
                "D\" was released in 2009.",
                "She received an award.",
                "She received the Filmfare Award.",
                "Kalki has showcased her versatility as an actress in films.",
            ],
            [
                "Adil Rami is a professional French footballer.",
                "Rami began his professional career at Lille OSC in 2006.",
                "Rami quickly established himself as a key player in the team's defense at Lille OSC in 2006.",
                "Rami made a move to Valencia CF in 2011.",
                "Rami continued to impress with his performances.",
                "Rami made a move to Valencia CF in 2011 and continued to impress with his performances.",
            ],
            [
                "Song Kang is a South Korean actor.",
                "Song Kang was born on April 23, 1994.",
                "Song Kang is one of the most sought-after young actors in the Korean entertainment industry.",
            ],
        ]
        
        self.topics = ["Kalki Koechlin", "Adil Rami", "Song Kang"]

    def test_retrieval(self):
        """
        """
        
        inputs = [(topic, test_case) for topic, test_cases in zip(self.topics, self.test_decomposed_cases_raw) for test_case in test_cases]
        topics = [ipt[0] for ipt in inputs]
        questions = [ipt[1] for ipt in inputs]

        results = self._retriever.get_passages_batched(topics, questions, k=3)
        
        for result, topic, question in zip(results, topics, questions):
            print('-' * 20)
            print(f"Topic: {topic}")
            print(f"Question: {question}")
            print(f"Results: {result[0]['text']}")
            print('-' * 20)