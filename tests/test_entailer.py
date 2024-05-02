"""Try to understand how the entailer works.
"""


from unittest import TestCase
from src.entailer import Entailer, EntailerInstance


class TestEntailer(TestCase):
    def setUp(self):
        """
        """
        self.entailer = Entailer(
            model_name="ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli",
            internal_batch_size=16,
            max_length=512,
            device="cuda",
        )
        
        self.test_cases = [
            {"premise": "This is a lot of instance to test.", "hypothesis": "This is a lot of instance to test."}
        ] * 200
        
    def test_entailer(self):
        
        print(self.entailer._model.device)
        self.entailer([EntailerInstance(**case) for case in self.test_cases])