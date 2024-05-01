""" Mostly follows factscore_decomposer (but apply filterings before and after) """


import spacy
import numpy as np
from dataclasses import dataclass
from overrides import overrides
from typing import List, Text, Tuple, Dict
from ..entailer.entailer import Entailer, EntailerInstance
from scipy.optimize import milp, Bounds, LinearConstraint
from langchain_interface.instances import LLMQueryInstance
from langchain_interface.example_selectors import ConstantExampleSelector
from langchain_interface.interfaces import ChatInterface
from ..scorer.scorer import Scorer
from ..scorer.llm_checkworthy_scorer import (
    LLMGeneralCheckWorthyScorer,
    LLMSpecificCheckWorthyScorer
)
from ..utils.instances import ScorerInstance
from .decomposer import Decomposer


@dataclass(frozen=True, eq=True)
class DedupScoreInstance(ScorerInstance):
    in_sent_claim_idx: int
    from_sent_idx: int
    sent: Text
    sent_checkworthy: float
    claim_checkworthy: float


@Decomposer.register("deduplicated")
class DeduplicatedDecomposer(Decomposer):
    """Deduplicate the facts from the base decomposer.
    """
    
    __NAME__ = "deduplicated"
    
    def __init__(
        self,
        base_decomposer: Decomposer,
        sentence_level_checkworthy_scorer: Scorer,
        claim_level_checkworthy_scorer: Scorer,
        entailer: Entailer,
        sentencize: bool = True,
    ):
        """
        """
        super().__init__()
        self._base_decomposer = base_decomposer
        self._nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
        self._nlp.add_pipe("sentencizer")
        self._sentencize = sentencize
        
        self._sentence_level_checkworthy_scorer = sentence_level_checkworthy_scorer
        self._claim_level_checkworthy_scorer = claim_level_checkworthy_scorer
        self._entailer = entailer
    
    def _decompose(self, instance: ScorerInstance) -> List[ScorerInstance]:
        """
        """

        all_instances = []
        
        if self._sentencize:
            sent_seqs = [ScorerInstance(text=sent.text, topic=instance.topic) for sent in self._nlp(instance.text).sents]
            checkworthiness = self._sentence_level_checkworthy_scorer(sent_seqs, return_raw=True)
            checkworthiness = [c['parsed'] for c in checkworthiness]
            
            for idx, (sent, sent_checkworthy) in enumerate(zip(sent_seqs, checkworthiness)):
                # TODO: optimize the threshold selection
                if sent_checkworthy < 0.5:
                    # This sentence is not checkworthy in general
                    continue
                # decomposing_instance = ScorerInstance(sent, topic=instance.topic)
                decomposed: List[ScorerInstance] = self._base_decomposer(sent)

                # filter out duplicated claims (on sentence-level)
                filtered_decomposed = []
                extant_claims = set()
                
                for decomposed_instance in decomposed:
                    if decomposed_instance.text not in extant_claims:
                        extant_claims.add(decomposed_instance.text)
                        filtered_decomposed.append(decomposed_instance)
                
                claim_checkworthiness = self._claim_level_checkworthy_scorer(filtered_decomposed)
                
                all_instances.extend([
                    DedupScoreInstance(
                        text=decomposed_instance.text,
                        topic=decomposed_instance.topic,
                        in_sent_claim_idx=claim_idx,
                        from_sent_idx=idx,
                        sent=sent.text,
                        sent_checkworthy=sent_checkworthy,
                        claim_checkworthy=claim_checkworthy,
                    )
                    for claim_idx, (decomposed_instance, claim_checkworthy) in enumerate(zip(filtered_decomposed, claim_checkworthiness))
                ])
                
        # we already get all the instances from the base decomposer,
        # now we will run the deduplication
        
        if not all_instances:
            return []
        
        return self._deduplicate(all_instances)
    
    def _deduplicate(self, instances: List[DedupScoreInstance]) -> List[DedupScoreInstance]:
        """
        """
        
        sent_filter_instances = [EntailerInstance(premise=instance.sent, hypothesis=instance.text) for instance in instances]
        sent_ent_results = self._entailer(sent_filter_instances)

        # filter out claims that are not entailed
        instances = [
            instance
            for instance, entailed in zip(instances, sent_ent_results)
            if entailed > 0.5
        ]

        # create pairwise entailment instances
        # if not in the result is 1
        finding_pair: Dict[Tuple[int, int], int] = {}
        pairwise_entailment_instances = []

        for i in range(len(instances)):
            for j in range(len(instances)):
                if i == j:
                    continue
                finding_pair[(i, j)] = len(pairwise_entailment_instances)
                pairwise_entailment_instances.append(
                    EntailerInstance(
                        premise=instances[i].text,
                        hypothesis=instances[j].text
                    )
                )
                
        pairwise_entailment_scoring = self._entailer(pairwise_entailment_instances)
        
        intra_ent_mat = np.array([
            [
                pairwise_entailment_scoring[finding_pair[(i, j)]] > 0.5 if i != j else False
                for j in range(len(instances))
            ]
            for i in range(len(instances))
        ], dtype=np.int16)
        
        # also, create the weighting vector from the instance
        # TODO: check whether this setting is optimal
        weighting = np.array([instance.sent_checkworthy * instance.claim_checkworthy for instance in instances], np.float32)

        # solve the MILP problem
        def solve_milp(
            pairwise_entailment: np.ndarray,
            weighting: np.ndarray,
        ) -> Tuple[np.ndarray, float]:
            """
            """
            
            if pairwise_entailment.size == 0:
                return np.array([], np.int16), 0.0
            
            or_mat = np.tril(
                np.bitwise_or(
                    pairwise_entailment,
                    np.transpose(pairwise_entailment)
                ),
            )
            
            indices = np.nonzero(or_mat)

            # TODO: add logging information
            
            constraints = np.zeros(
                (len(indices[0]), len(weighting)),
                dtype=np.float32
            )
            
            constraints[np.arange(len(indices[0])), indices[0]] = 1
            constraints[np.arange(len(indices[1])), indices[1]] = 1
            
            res = milp(
                c=-weighting,
                integrality=np.ones_like(weighting),
                bounds=Bounds(
                    lb=np.zeros_like(weighting) - 1e-8,
                    ub=np.ones_like(weighting) + 1e-8
                ),
                constraints=(
                    LinearConstraint(
                        A=constraints,
                        ub=np.ones(len(indices[0])) + 1e-8,
                    ),
                )
            )

            selection = res.x
            result = res.fun

            return selection, result
        
        selection, result = solve_milp(
            pairwise_entailment=intra_ent_mat,
            weighting=weighting
        )
        
        non_zero_selection_indices = np.nonzero(selection)[0].tolist()
        
        return [instances[index] for index in non_zero_selection_indices]