"""This is a simmple task that trains a UNLI model from ynie model with huggingface trainer."""

import datasets
import torch
import numpy as np
from scipy.special import expit
from scipy.stats import pearsonr, spearmanr
from typing import Dict, Text, Any, Optional
from transformers import Trainer, TrainingArguments
from transformers import EarlyStoppingCallback
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from .task import Task


@Task.register("train-unli")
class TrainUNLITask(Task):
    """ """

    def __init__(
        self,
        output_dir: Text,
        gradient_accumulation_steps: Optional[int] = 1,
        learning_rate: Optional[float] = 5e-6,
        num_train_epochs: Optional[int] = 6,
    ):
        """ """
        super().__init__()

        self._config = AutoConfig.from_pretrained(
            "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli",
            problem_type="multi_label_classification",
            num_labels=1,
        )
        
        # TODO: Rewrite part of the configurations of the preset model configuration

        self._model = AutoModelForSequenceClassification.from_pretrained(
            "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli",
            ignore_mismatched_sizes=True,
            config=self._config
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli", use_fast=True
        )

        train_dataset = datasets.load_dataset("Zhengping/UNLI", split="train")
        eval_dataset = datasets.load_dataset("Zhengping/UNLI", split="validation")
        test_dataset = datasets.load_dataset("Zhengping/UNLI", split="test")

        # now preprocess with the tokenizer
        tokenize = (
            lambda x: self._tokenizer(
                x["premise"],
                x["hypothesis"],
                truncation=True,
                max_length=256,
                return_attention_mask=True,
                return_token_type_ids=True,
            )
        )

        self._train_dataset = train_dataset.map(tokenize, batched=True)
        self._eval_dataset = eval_dataset.map(tokenize, batched=True)
        self._test_dataset = test_dataset.map(tokenize, batched=True)
        
        # process 'labels' from float to List[float]
        wrap_label = lambda x: {"label": [[xx] for xx in x['label']]}
        self._train_dataset = self._train_dataset.map(
            wrap_label,
            batched=True,
        )
        self._eval_dataset = self._eval_dataset.map(
            wrap_label,
            batched=True,
        )
        self._test_dataset = self._test_dataset.map(
            wrap_label,
            batched=True,
        )
        
        # define some advanced metric computation

        def compute_metrics(eval_pred) -> Dict[Text, Any]:
            """
            """
            predictions, labels = eval_pred
            
            # predictions of shape [batch_size, 1]
            # labels of shape [batch_size, 1]
            
            predictions: np.ndarray = expit(predictions)
            predictions = np.squeeze(predictions, -1)
            labels = np.squeeze(labels, -1)
            
            return {
                "pearson": pearsonr(predictions, labels)[0],
                "spearman": spearmanr(predictions, labels)[0],
                "mse": ((predictions - labels) ** 2).mean(),
            }
            
        self._output_dir = output_dir
        self._gradient_accumulation_steps = gradient_accumulation_steps
        self._learning_rate = learning_rate
        self._num_train_epochs = num_train_epochs
        
        self._training_args = TrainingArguments(
            output_dir=self._output_dir,
            do_train=True,
            do_eval=True,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            per_device_eval_batch_size=16,
            per_device_train_batch_size=16,
            gradient_accumulation_steps=self._gradient_accumulation_steps,
            learning_rate=self._learning_rate,
            num_train_epochs=self._num_train_epochs,
            load_best_model_at_end=True,
            metric_for_best_model="pearson",
            greater_is_better=True,
        )

        self._trainer = Trainer(
            model=self._model,
            args=self._training_args,
            tokenizer=self._tokenizer,
            train_dataset=self._train_dataset,
            eval_dataset=self._eval_dataset,
            compute_metrics=compute_metrics,
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=2, early_stopping_threshold=0.01)
            ],
        )

    def run(self):
        """ """

        self._trainer.train()
