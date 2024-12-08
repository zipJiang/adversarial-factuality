""" Data Reader for the raw generation from the dataset """

try:
    import ujson as json
except ImportError:
    import json
from typing import List, Dict, Text, Any, Union
from langchain_interface.instances import Instance
from ..utils.instances import LLMGenerationInstance
from overrides import overrides
from .base_data_reader import BaseDataReader


@BaseDataReader.register("generation")
class GenerationDataReader(BaseDataReader):
    """ The expected data format is as follows:
    {
        "id": "unique identifier of the instance",
        "generation": "the generation text",
        ... // Any other fields that will be packed into 'meta'
    }
    """

    @overrides
    def _parse(self, line: Text) -> Instance:
        """ """
        
        item = json.loads(line)
        
        generation = item.pop("generation", None)
        id_ = item.pop("id", None)
        
        assert generation is not None, "Generation text is not provided."
        assert id_ is not None, "ID is not provided."
        
        return LLMGenerationInstance(
            id_=id_,
            generation=generation,
            meta=item
        )