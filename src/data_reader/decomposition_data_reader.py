""" """

try:
    import ujson as json
except ImportError:
    import json
from typing import List, Dict, Text, Any, Union
from langchain_interface.instances import Instance
from ..utils.instances import (
    AtomicClaim,
    DecomposedLLMGenerationInstance
)
from overrides import overrides
from .base_data_reader import BaseDataReader


@BaseDataReader.register("decomposition")
class DecompositionDataReader(BaseDataReader):
    """ The expected data format is as follows:
    {
        "id": "unique identifier of the instance",
        "generation": "the generation text",
        "claims": [
            ...
        ] // The claims that are decomposed
        ... // Any other fields that will be packed into 'meta'
    }
    """
    
    @overrides
    def _parse(self, line: Text) -> Instance:
        """ """
        
        item = json.loads(line)
        generation = item.pop("generation", None)
        id_ = item.pop("id", None)
        claims = item.pop("claims", None)
        atomic_claims = []

        for claim in claims:
            claim_text = claim.pop("claim", None)
            assert claim_text is not None, "Claim text is not provided."
            atomic_claims.append(AtomicClaim(
                claim=claim_text,
                meta=claim
            ))
        
        assert generation is not None, "Generation text is not provided."
        assert id_ is not None, "ID is not provided."
        
        return DecomposedLLMGenerationInstance(
            id_=id_,
            generation=generation,
            meta=item,
            claims=atomic_claims
        )