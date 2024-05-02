"""Entailer will be used to judge the entailment relationship between two setences"""

from dataclasses import dataclass
from registrable import Registrable
from typing import Text
import torch
from timeit import timeit
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from typing import Text, Dict, List, Union
from ..utils.common import paginate_func


@dataclass
class EntailerInstance:
    premise: Text
    hypothesis: Text


# TODO: add cache for the entailer calculation (probably use flaiss? or just hash search).
class Entailer(Registrable):
    __LABEL_MAP__ = [1, 0, 0]

    def __init__(
        self,
        model_name: Text,
        device: Text = "cuda",
        internal_batch_size: int = 16,
        max_length: int = 512,
    ):
        super().__init__()
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name).to(
            device
        )
        self._device = device
        self._model.eval()
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self._internal_batch_size = internal_batch_size
        self._max_length = max_length

    def _collate_fn(
        self, instances: List[EntailerInstance]
    ) -> Dict[Text, torch.Tensor]:
        """Notice that we are requiring this to run entailer instances."""

        premises = [instance.premise for instance in instances]
        hypotheses = [instance.hypothesis for instance in instances]

        tokenized = self._tokenizer(
            text=premises,
            text_pair=hypotheses,
            padding=True,
            truncation=True,
            max_length=self._max_length,
            return_tensors="pt",
            add_special_tokens=True,
            return_attention_mask=True,
            return_token_type_ids=True,
        )

        return {
            "input_ids": tokenized["input_ids"].to(self._device),
            "attention_mask": tokenized["attention_mask"].to(self._device),
            "token_type_ids": (
                tokenized["token_type_ids"].to(self._device)
                if "token_type_ids" in tokenized
                else None
            ),
        }

    def __call__(
        self,
        instances: List[EntailerInstance],
    ) -> List[float]:
        """ """
        return paginate_func(
            items=instances,
            page_size=self._internal_batch_size,
            func=self._call_batch,
            combination=lambda x: [xxx for xx in x for xxx in xx],
        )

    def _call_batch(self, instances: List[EntailerInstance]) -> List[float]:
        """This is the actual calling function of the model."""

        assert len(instances) <= self._internal_batch_size, "Batch size is too large."

        with torch.no_grad():
            inputs = self._collate_fn(instances)
            outputs = self._model(**inputs)

        indices = torch.argmax(outputs.logits, dim=1).int().cpu().numpy().tolist()

        return [float(self.__LABEL_MAP__[index]) for index in indices]


Entailer.register("default")(Entailer)
