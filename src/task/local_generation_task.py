"""
"""

import os
from overrides import overrides
import json
from tqdm import tqdm
from typing import Text
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftModel
from .task import Task


@Task.register("local-generation")
class LocalGenerationTask(Task):
    """
    """
    
    __NAME__ = "local-generation"
    
    def __init__(
        self,
        model_dir: Text,
        topic_path: Text,
        output_path: Text,
        prompt: Text,
        is_chat: bool
    ):
        """
        """
        super().__init__()
        
        with open(os.path.join(model_dir, "config.json"), "r", encoding="utf-8") as file_:
            config = json.load(file_)
            model_name = config['_name_or_path']
            self._model_name = model_name

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
        
        self._model_dir = model_dir
        self._topic_path = topic_path
        self._output_path = output_path
        self._prompt = prompt
        self._is_chat = is_chat and not self._model_name.endswith("gpt2")
        
    @overrides
    def run(self):
        """
        """
        
        generation_config = GenerationConfig.from_pretrained(self._model_name, max_new_tokens=256)
        generation_config.pad_token_id = self._tokenizer.eos_token_id
        model = AutoModelForCausalLM.from_pretrained(self._model_dir)
        # model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
        # model = PeftModel.from_pretrained(model, "ckpt/peft-gpt2-info/checkpoint-315/")
        model.eval()
        model = model.to("cuda:0")
        
        with open(self._topic_path, "r", encoding="utf-8") as file_:
            topics = [
                line.strip()
                for line in file_
            ]

        os.makedirs(os.path.dirname(self._output_path), exist_ok=True)
        with open(self._output_path, "w", encoding="utf-8") as file_:
            for topic in tqdm(topics):
                formatted = self._prompt.format(input=topic)
                
                if self._is_chat:
                    input_ids = self._tokenizer.apply_chat_template([
                        {
                            "role": "user",
                            "content": formatted
                        },
                    ], return_tensors="pt")
                    input_ids = {
                        "input_ids": input_ids.to("cuda:0"),
                    }
                    seq_len = input_ids["input_ids"].shape[1]
                else:
                    input_ids = self._tokenizer(formatted, return_tensors="pt", return_attention_mask=True)
                    input_ids = {
                        "input_ids": input_ids.input_ids.to("cuda:0"),
                        "attention_mask": input_ids.attention_mask.to("cuda:0"),
                    }
                    seq_len = input_ids["input_ids"].shape[1]

                outputs = model.generate(
                    **input_ids,
                    return_dict_in_generate=True,
                    output_scores=True,
                    generation_config=generation_config
                )
                
                sequence = self._tokenizer.decode(outputs.sequences[0][seq_len:], skip_special_tokens=True)
                
                file_.write(json.dumps({
                    "topic": topic,
                    "output": {
                        "raw": sequence,
                        # remove last incomplete sentence (denoted by a period)
                        "parsed": sequence.rsplit(".", 1)[0] + "."
                    }
                }) + "\n")