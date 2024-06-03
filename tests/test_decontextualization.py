"""Test decontextualizer to see if it gives proper
results.
"""

from unittest import TestCase
from src.utils.instances import DecontextScorerInstance
from src.decontextualizer.decontextualizer import Decontextualizer


class TestDecontextualizer(TestCase):
    def setUp(self):
        """
        """
        
        self.decontextualizer = Decontextualizer(
            model_name="mistralai/Mistral-7B-Instruct-v0.2",
            example_path="decontextualize_examples.txt",
            base_url="http://localhost:9871/v1",
            api_key="token-abc123"
        )
        
        self.test_cases = [
            DecontextScorerInstance(
                topic="Kalki Koechlin",
                text="She's an actress.",
                sent="Kalki Koechlin is an Indian actress and writer known for her work in Hindi films.",
                source_text=None
            )
        ]
        
    def test_decontextualization(self):
        """
        """
        
        print(self.decontextualizer(self.test_cases))