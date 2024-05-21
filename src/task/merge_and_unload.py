"""Take an adaptor and merge it into the main model weights.
"""

from .task import Task
from typing import Text
from transformers import AutoModelForCausalLM, AutoTokenizer
from overrides import overrides
from peft import PeftModel


@Task.register("merge-and-unload")
class MergeAndUnload(Task):
    """
    """
    
    __NAME__ = "merge-and-unload"
    
    def __init__(
        self,
        model_name: Text,
        adaptor_path: Text,
        save_path: Text
    ):
        """This can successfully merge adaptor to model.
        """
        
        super().__init__()
        
        self._model_name = model_name
        self._adaptor_path = adaptor_path
        self._save_path = save_path
        
    @overrides
    def run(self):
        """
        """
        base_model = AutoModelForCausalLM.from_pretrained(self._model_name)
        # tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = PeftModel.from_pretrained(base_model, self._adaptor_path)
        merge_model = model.merge_and_unload()
        
        # save model to disk
        merge_model.save_pretrained(self._save_path)