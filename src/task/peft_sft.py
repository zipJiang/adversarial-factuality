"""We peft tune the LLM model with the SFT methods,
and try to find the model that optimize for FActScore
with Cooperative Principle Violation.
"""

from overrides import overrides
from datetime import datetime
import math
import torch
import os
from datasets import Dataset
from typing import List, Any, Dict, Text, Optional
from peft import LoraConfig, TaskType, prepare_model_for_kbit_training, get_peft_model, get_peft_config
import transformers
from transformers import Trainer, TrainingArguments
from transformers import SchedulerType
from transformers import DataCollatorForLanguageModeling
from transformers import EarlyStoppingCallback
from transformers import AutoModelForCausalLM, AutoTokenizer
from .task import Task


os.environ['WANDB_PROJECT'] = "peft-sft-adverserial-factuality"
os.environ['WANDB_LOG_MODEL'] = "checkpoint"


@Task.register('peft-sft')
class PeftSFTTask(Task):
    """
    """

    __NAME__ = 'peft-sft'
    
    def __init__(
        self,
        model_name: Text,
        peft_config: Dict[Text, Any],
        output_dir: Text,
        batch_size: int,
        learning_rate: float,
        num_train_epochs: int,
        train_data_path: Text,
        eval_data_path: Text,
        test_data_path: Text,
        is_chat_model: bool,
        gradient_accumulation_steps: Optional[int] = 1,
        load_in_8bit: Optional[bool] = True
    ):
        """
        A potentially possible peft_config:
        
        peft_config: {
            "r": int,
            "lora_alpha": float,
            "target_modules": List[Text],
            "fan_in_fan_out": bool,
            "lora_dropout": float,
            ...
        }
        """
        super().__init__()
        self._peft_config = peft_config
        self._model_name = model_name
        
        # training args
        self._output_dir = output_dir
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._num_train_epochs = num_train_epochs
        self._gradient_accumulation_steps = gradient_accumulation_steps

        self._train_dataset = Dataset.from_json(train_data_path)
        self._eval_dataset = Dataset.from_json(eval_data_path)
        self._test_dataset = Dataset.from_json(test_data_path)
        
        self._is_chat_model = is_chat_model
        self._load_in_8bit = load_in_8bit

        self._run_stem = '-'.join([
            os.path.basename(train_data_path).split('.')[0].split("-")[-1],
            os.path.basename(model_name)
        ])

    @overrides
    def run(self):
        """
        """
        
        # create peft_config
        # TODO: add QLoRA quantization
        peft_config = LoraConfig(
            # TODO: Potentially need to extend to seq2seqlm
            task_type=TaskType.CAUSAL_LM,
            **self._peft_config,
        )
        
        quantization_config = transformers.BitsAndBytesConfig(load_in_8bit=self._load_in_8bit) if self._load_in_8bit else None
        quantization_dicts = {}
        
        if self._load_in_8bit:
            quantization_dicts = {
                "torch_dtype": torch.float16,
                "quantization_config": quantization_config,
            }
        
        # load the base model
        tokenizer = AutoTokenizer.from_pretrained(
            self._model_name,
        )
        tokenizer.pad_token = tokenizer.eos_token
        base_model = AutoModelForCausalLM.from_pretrained(
            self._model_name,
            pad_token_id=tokenizer.eos_token_id,
            **quantization_dicts,
            # torch_dtype=torch.float16,
            # quantization_config=quantization_config,
            # TODO: consider other required parameters
            # For example, whether to load in 8 bit
        )
        
        if self._load_in_8bit:
            base_model = prepare_model_for_kbit_training(base_model)
            
        lora_model = get_peft_model(base_model, peft_config)

        train_args = TrainingArguments(
            output_dir=self._output_dir,
            do_train=True,
            do_eval=True,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            per_device_eval_batch_size=self._batch_size,
            per_device_train_batch_size=self._batch_size,
            gradient_accumulation_steps=self._gradient_accumulation_steps,
            learning_rate=self._learning_rate,
            lr_scheduler_type=SchedulerType.CONSTANT_WITH_WARMUP,
            warmup_steps=100,
            num_train_epochs=self._num_train_epochs,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="wandb",
            run_name=f"{self._run_stem}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}",
            logging_steps=20,
            save_total_limit=1
        )
        
        # need to create data with the tokenizer
        def create_input_and_tokenize(examples: Dict[Text, List[Any]]) -> Dict[Text, torch.Tensor]:
            """We need to create input data with the inst model template.
            """
            if self._is_chat_model:
                # use apply_chat_template of the tokenizer
                messages = [
                        [
                        {
                            "role": "user",
                            "content": f"Tell me a bio of {topic}."
                        },
                        {
                            "role": "assistant",
                            "content": output['parsed']
                        }
                    ]
                    for topic, output in zip(examples['topic'], examples['output'])
                ]
                return {
                    "input_ids": tokenizer.apply_chat_template(
                        messages,
                    )
                }
            else:
                messages = [
                    f"Tell me a bio of {topic}. {output['parsed']}"
                    for topic, output in zip(examples['topic'], examples['output'])
                ]
                
                return tokenizer(
                    messages,
                    max_length=512,
                )
                
        train_dataset = self._train_dataset.map(create_input_and_tokenize, batched=True, remove_columns=self._train_dataset.column_names)
        eval_dataset = self._eval_dataset.map(create_input_and_tokenize, batched=True, remove_columns=self._eval_dataset.column_names)
        test_dataset = self._test_dataset.map(create_input_and_tokenize, batched=True, remove_columns=self._test_dataset.column_names)
        
        collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            pad_to_multiple_of=8 if self._load_in_8bit else None,
        )
        
        trainer = Trainer(
            model=lora_model,
            args=train_args,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=1, early_stopping_threshold=0.01)
            ],
            data_collator=collator,
        )
        
        trainer.train()
        eval_result = trainer.evaluate(eval_dataset=test_dataset)
        print(f"Perplexity: {math.exp(eval_result['eval_loss']):.2f}")