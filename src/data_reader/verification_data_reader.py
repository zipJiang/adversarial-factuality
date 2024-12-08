""" Data Reader for Verification instance """

try:
    import ujson as json
except ImportError:
    import json
from overrides import overrides
from typing import Text
from .base_data_reader import BaseDataReader
from ..utils.instances import (
    VerifiedAtomicClaim,
    VerifiedLLMGenerationInstance
)


class VerificationDataReader(BaseDataReader):
    """ The expected data format is as follows:
    {
        "id_": "unique identifier of the instance",
        "generation": "the generation text",
        "claims": [
            ...
        ] // The claims that are decomposed
        ... // Any other fields that will be packed into 'meta'
        "aggregated_score": float
    }
    """

    @overrides
    def _parse(self, line: Text) -> VerifiedLLMGenerationInstance:
        """ """

        item = json.loads(line)
        generation = item.pop("generation", None)
        id_ = item.pop("id_", None)
        claims = item.pop("claims", None)
        meta = item.pop("meta", {})
        aggregated_score = item.pop("aggregated_score", None)
        verified_atomic_claims = []

        for claim in claims:
            claim_text = claim.pop("claim", None)
            claim_meta = claim.pop("meta", {})
            claim_score = claim.pop("factual_score", None)
            assert claim_score is not None, "Claim score is not provided."
            assert claim_text is not None, "Claim text is not provided."
            verified_atomic_claims.append(VerifiedAtomicClaim(
                claim=claim_text,
                factual_score=claim_score,
                meta={**claim, **claim_meta}
            ))

        assert generation is not None, "Generation text is not provided."
        assert id_ is not None, "ID is not provided."
        assert aggregated_score is not None, "Aggregated score is not provided."

        return VerifiedLLMGenerationInstance(
            id_=id_,
            generation=generation,
            meta={**item, **meta},
            claims=verified_atomic_claims,
            aggregated_score=aggregated_score
        )