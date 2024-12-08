""" """

import numpy as np
from typing import Text, Optional, Dict, Tuple, List, Iterable
from overrides import overrides
from ..entailer import BaseEntailer, EntailerInstance
from ..utils.template import Template
from ..utils.instances import DecomposedLLMGenerationInstance, AtomicClaim
from ..utils.common import path_index, solve_milp
from ..utils.template import BaseTemplate
from .base_post_processor import BasePostProcessor
import logging


logger = logging.getLogger(__name__)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger.setLevel(logging.INFO)
handler = logging.FileHandler("claim_checkworthiness_post_processor.log")
handler.setFormatter(formatter)
logger.addHandler(handler)


@BasePostProcessor.register("claim-checkworthiness")
class ClaimCheckworthinessPostProcessor(BasePostProcessor):
    """To successfully have this composer run,
    we need the claim checkworthiness and sent checkworthiness
    """

    def __init__(
        self,
        bleached_templates: List[BaseTemplate],
        entailer: BaseEntailer,
        cap_entailer: Optional[BaseEntailer] = None,
        epsilon: float = 1e-3,
    ):
        """To make this bleached_templates more versatile,
        we allow path indices to be used with BleachedTemplates
        """

        super().__init__(namespace="__claim-checkworthiness")

        self._bleached_templates = bleached_templates
        self._entailer = entailer
        self._cap_entailer = cap_entailer
        self._epsilon = epsilon

    @overrides
    def _process(
        self, instance: DecomposedLLMGenerationInstance
    ) -> DecomposedLLMGenerationInstance:
        """Here the scorer will score the instance into
        result dict
        """

        # pair each bleached_context with the claim
        # and score each pair
        inputs = [
            bleached_template(claim)
            for claim in instance.claims
            for bleached_template in self._bleached_templates
        ]
        scores = self._entailer(inputs, desc="UNLI Scoring")
        scores = np.array(scores).reshape(len(instance.claims), len(self._bleached_templates))
        scores = np.max(scores, axis=1)
        # Adding lowerbound to cap_score to avoid log(0)
        scores = np.maximum(scores, self._epsilon)

        cap_entailer_outputs = [{} for _ in range(len(instance.claims))]

        if self._cap_entailer is not None:
            # if cap_score == 1, then the claim needs to be
            # capped regardless of the score
            cap_scores = self._cap_entailer(inputs, desc="Cap Scoring")
            cap_scores = np.array(cap_scores).reshape(
                len(instance.claims), len(self._bleached_templates)
            )
            cap_scores = np.max(cap_scores, axis=1)
            cap_scores = np.maximum(cap_scores, self._epsilon)
            scores = np.maximum(scores, cap_scores)
            for i in range(len(instance.claims)):
                cap_entailer_outputs[i]["cap_score"] = cap_scores[i].item()

        # Zhengping 05/24/2025: Use - log(score) to align with CPMI
        parsed_scores = (-np.log(scores) - self._epsilon).tolist()

        # return {
        #     "premises": bleached_context,
        #     "hypothesis": instance.text,
        #     "parsed": parsed_score,
        #     "scores": scores,
        #     **cap_entailer_outputs,
        # }
        
        return DecomposedLLMGenerationInstance(
            id_=instance.id_,
            generation=instance.generation,
            meta=instance.meta,
            claims=[
                AtomicClaim(
                    claim=claim.claim,
                    meta={**claim.meta, f"{self._namespace}": {"score": score, **cap_score_dict}},
                )
                for claim, score, cap_score_dict in zip(instance.claims, parsed_scores, cap_entailer_outputs)
            ]
        )

    @overrides
    def _batch_process(
        self, instances: Iterable[DecomposedLLMGenerationInstance]
    ) -> Iterable[DecomposedLLMGenerationInstance]:
        """Run scores in batch."""
        
        iidx_cidx_to_id = {}
        inputs = []
        
        for iidx, instance in enumerate(instances):
            for cidx, claim in enumerate(instance.claims):
                iidx_cidx_to_id[(iidx, cidx)] = len(inputs) // len(self._bleached_templates)
                for bleached_template in self._bleached_templates:
                    inputs.append(bleached_template(claim))
                    
        scores = self._entailer(inputs, desc="UNLI Scoring")
        scores = np.array(scores).reshape(-1, len(self._bleached_templates))
        scores = np.max(scores, axis=1)
        # Adding lowerbound to cap_score to avoid log(0)
        scores = np.maximum(scores, self._epsilon)

        cap_entailer_outputs = [{} for _ in range(scores.shape(0))]
        
        if self._cap_entailer is not None:
            cap_scores = self._cap_entailer(inputs, desc="Cap Scoring")
            cap_scores = np.array(cap_scores).reshape(-1, len(self._bleached_templates))
            cap_scores = np.max(cap_scores, axis=1)
            cap_scores = np.maximum(cap_scores, self._epsilon)
            scores = np.maximum(scores, cap_scores)
            for i in range(len(cap_entailer_outputs)):
                cap_entailer_outputs[i]["cap_score"] = cap_scores[i].item()
                
        return [
            DecomposedLLMGenerationInstance(
                id_=instance.id_,
                generation=instance.generation,
                meta=instance.meta,
                claims=[
                    AtomicClaim(
                        claim=claim.claim,
                        meta={
                            **claim.meta,
                            f"{self._namespace}": {
                                "score": scores[iidx_cidx_to_id[(iidx, cidx)]],
                                **(cap_entailer_outputs[iidx_cidx_to_id[(iidx, cidx)]]),
                            }
                        },
                    )
                    for cidx, claim in enumerate(instance.claims)
                ]
            )
            for iidx, instance in enumerate(instances)
        ]