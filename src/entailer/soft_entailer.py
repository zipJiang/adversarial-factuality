""" Soft Entailer calculates p(h|p) instead of ternary classification. """


from .entailer import Entailer, EntailerInstance
from overrides import overrides
import torch
from typing import Text, List
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@Entailer.register("soft-entailer")
class SoftEntailer(Entailer):
    def __init__(
        self,
        model_name: Text,
        device: Text = "cuda",
        internal_batch_size: int = 16,
        max_length: int = 512,
    ):
        super().__init__(
            model_name=model_name,
            device=device,
            internal_batch_size=internal_batch_size,
            max_length=max_length,
        )

    @overrides
    def _call_batch(self, instances: List[EntailerInstance]) -> List[float]:
        """This is the actual calling function of the model."""

        assert len(instances) <= self._internal_batch_size, "Batch size is too large."

        with torch.no_grad():
            inputs = self._collate_fn(instances)
            outputs = self._model(**inputs)

        # indices = torch.argmax(outputs.logits, dim=1).int().cpu().numpy().tolist()
        probs = torch.sigmoid(outputs.logits).squeeze(-1).cpu().numpy().tolist()

        return probs