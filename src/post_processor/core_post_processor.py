""" The post-processor of Core """

import numpy as np
from typing import (
    Text, Optional,
    Dict, Tuple,
    List,
    Iterable
)
from overrides import overrides
from ..entailer import BaseEntailer, EntailerInstance
from ..utils.instances import DecomposedLLMGenerationInstance, AtomicClaim
from ..utils.common import path_index, solve_milp
from .base_post_processor import BasePostProcessor
import logging


logger = logging.getLogger(__name__)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger.setLevel(logging.INFO)
handler = logging.FileHandler("core_post_processor.log")
handler.setFormatter(formatter)
logger.addHandler(handler)


@BasePostProcessor.register("core")
class CorePostProcessor(BasePostProcessor):
    """To successfully have this composer run,
    we need the claim checkworthiness and sent checkworthiness
    """

    def __init__(
        self,
        entailer: BaseEntailer,
        claim_checkworthiness_path: Optional[Text],
        sent_checkworthiness_path: Optional[Text],
    ):
        super().__init__(namespace="__core")
        self._claim_checkworthiness_path = claim_checkworthiness_path
        self._sent_checkworthiness_path = sent_checkworthiness_path
        self._entailer = entailer

    @overrides
    def _process(
        self, instance: DecomposedLLMGenerationInstance
    ) -> DecomposedLLMGenerationInstance:
        """ """
        original_claims = instance.claims
        sent_entailment_scores = [1] * len(original_claims)

        if all("source_text" in claim.meta for claim in original_claims):
            sent_entailment_scores = self._entailer([
                EntailerInstance(
                    premise=claim.meta["source_text"], hypothesis=claim.claim
                )
                for claim in original_claims
            ], desc="StE Entailment")
        else:
            logger.warning(
                "Intend to calculate entailment scores from 'source_text' a claim comes from, "
                "but not all claims have 'source_text' in their meta. "
                "Using default sent_entailment_scores of 1."
            )

        claims = [claim for claim, score in zip(original_claims, sent_entailment_scores) if score > 0.5]
        claims = self._lexical_dedup(claims)
        
        pair_to_id = {}
        # id_to_pair = {}
        pairs = []

        for i, claim_i in enumerate(claims):
            for j, claim_j in enumerate(claims):
                if i != j:
                    pair_to_id[(i, j)] = len(pairs)
                    # id_to_pair[len(pairs)] = (i, j)
                    pairs.append(EntailerInstance(premise=claim_i.claim, hypothesis=claim_j.claim))
                    
        scores = self._entailer(pairs, desc="Pairwise Entailment")
        pair_to_scores = {p_: scores[id_] for p_, id_ in pair_to_id.items()}
        
        pairwise_scoring_mat = np.array([
            [
                0 if i == j else int(pair_to_scores[(i, j)] > 0.5)
                for j in range(len(claims))
            ]
            for i in range(len(claims))
        ], dtype=np.int16)
        
        weighting = np.array([
            path_index(claim, self._claim_checkworthiness_path, default=1 / len(claims), logger=logger) *
            path_index(claim, self._sent_checkworthiness_path, default=1 / len(claims), logger=logger)
            for claim in claims
        ], dtype=np.float32)
        
        selection, _ = solve_milp(pairwise_scoring_mat, weighting)
        non_zero_selection_indices = np.nonzero(selection)[0].tolist()

        return DecomposedLLMGenerationInstance(
            id_=instance.id_,
            generation=instance.generation,
            claims=[claims[i] for i in non_zero_selection_indices],
            meta={**instance.meta, f"{self._namespace}": {
                "ratio": len(non_zero_selection_indices) / len(instance.claims),
                "selected_indices": non_zero_selection_indices,
                "original_claims": [
                    {
                        # flatten
                        "index": claim.meta['claim_index'],
                        "text": claim.claim,
                    }
                    for claim in instance.claims
                ]
            }}
        )
        
    def _lexical_dedup(self, claims: List[AtomicClaim]) -> List[AtomicClaim]:
        """ Remove literal duplicates """

        saw_claim_texts = set()
        deduplication = []
        
        for claim in claims:
            if claim.claim not in saw_claim_texts:
                saw_claim_texts.add(claim.claim)
                deduplication.append(claim)
            
        return deduplication
    
    def _batch_process(
        self, instances: Iterable[DecomposedLLMGenerationInstance]
    ) -> Iterable[DecomposedLLMGenerationInstance]:
        """ """

        iidx_cidx_to_id = {}
        lexically_deduplicated_claims = []
        
        for instance in instances:
            claims = self._lexical_dedup(instance.claims)
            lexically_deduplicated_claims.append(claims)
            
        sent_entailment_inputs = []
        
        try:
            for iidx, claims in enumerate(lexically_deduplicated_claims):
                for cidx, claim in enumerate(claims):
                    iidx_cidx_to_id[(iidx, cidx)] = len(sent_entailment_inputs)
                    sent_entailment_inputs.append(EntailerInstance(
                        premise=claim.meta["source_text"], hypothesis=claim.claim
                    ))
        except KeyError:
            logger.warning(
                "Intend to calculate entailment scores from 'source_text' a claim comes from, "
                "but not all claims have 'source_text' in their meta. "
                "Using default sent_entailment_scores of 1."
            )
            
            sent_entailment_inputs = None
            
        if sent_entailment_inputs:
            # entailment filtering
            sent_entailment_scores = self._entailer(sent_entailment_inputs, desc="StE Entailment")
            ste_filtered_claims = [
                [claim for cidx, claim in enumerate(claims) if sent_entailment_scores[iidx_cidx_to_id[(iidx, cidx)]] > 0.5]
                for iidx, claims in enumerate(lexically_deduplicated_claims)
            ]
            
        else:
            ste_filtered_claims = lexically_deduplicated_claims
        
        iidx_pair_to_id = {}
        pairs = []
        
        for iidx, claims in enumerate(ste_filtered_claims):
            for i, claim_i in enumerate(claims):
                for j, claim_j in enumerate(claims):
                    if i != j:
                        iidx_pair_to_id[(iidx, i, j)] = len(pairs)
                        pairs.append(EntailerInstance(premise=claim_i.claim, hypothesis=claim_j.claim))
                        
        pairwise_scores = self._entailer(pairs, desc="Pairwise Entailment")
        
        # If we want to create a streaming version of this, we need to change the following
        # to stream the computation
        pairwise_mat = [
            np.array([
                [
                    0 if i == j else int(pairwise_scores[iidx_pair_to_id[(iidx, i, j)]] > 0.5)
                    for j in range(len(claims))
                ]
                for i in range(len(claims))
            ], dtype=np.int16)
            for iidx, claims in enumerate(ste_filtered_claims)
        ]
        
        weightings = [
            np.array([
                path_index(claim, self._claim_checkworthiness_path, default=1 / len(claims), logger=logger) *
                path_index(claim, self._sent_checkworthiness_path, default=1 / len(claims), logger=logger)
                for claim in claims
            ], dtype=np.float32)
            for claims in ste_filtered_claims
        ]
        
        selection_indices = [
            np.nonzero(
                solve_milp(pairwise_mat_, weightings_)[0]
            )[0].tolist()
            for pairwise_mat_, weightings_ in zip(pairwise_mat, weightings)
        ]
        
        # create new instances
        return [
            DecomposedLLMGenerationInstance(
                id_=instance.id_,
                generation=instance.generation,
                claims=[
                    claims[s] for s in sindices
                ],
                meta={**instance.meta, f"{self._namespace}": {
                    "ratio": len(claims) / len(instance.claims),
                    "selected_indices": sindices,
                    "original_claims": [
                        {
                            # flatten
                            "index": claim.meta['claim_index'],
                            "text": claim.claim,
                        }
                        for claim in instance.claims
                    ]
                }}
            )
            for instance, sindices, claims in zip(instances, selection_indices, ste_filtered_claims)
        ]