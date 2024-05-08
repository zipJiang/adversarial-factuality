""" Mostly follows factscore_decomposer (but apply filterings before and after) """


import spacy
import numpy as np
from dataclasses import dataclass
from overrides import overrides
from typing import List, Text, Tuple, Dict, Any, Union
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
        
        # filter out duplicated claims (on sentence-level)
        filtered_decomposed = []
        claim_to_sent: Dict[Text, List[Dict[Text, Any]]] = {}
        extant_claims = set()

        if self._sentencize:
            sent_seqs = [ScorerInstance(text=sent, topic=instance.topic) for sent in self._to_sents(instance.text)]
        else:
            sent_seqs = [instance]

        checkworthiness = self._sentence_level_checkworthy_scorer(sent_seqs, return_raw=True)
        checkworthiness = [c['parsed'] for c in checkworthiness]
        
        for sent, sent_checkworthy in zip(sent_seqs, checkworthiness):
            # TODO: optimize the threshold selection
            if sent_checkworthy < 0.5:
                # This sentence is not checkworthy in general
                continue
            # decomposing_instance = ScorerInstance(sent, topic=instance.topic)
            decomposed: List[ScorerInstance] = self._base_decomposer(sent)

            # filter out duplicated claims (on sentence-level)
            # filtered_decomposed = []
            # extant_claims = set()
            
            for didx, decomposed_instance in enumerate(decomposed):
                if decomposed_instance.text not in extant_claims:
                    extant_claims.add(decomposed_instance.text)
                    filtered_decomposed.append(decomposed_instance)
                if decomposed_instance.text not in claim_to_sent:
                    claim_to_sent[decomposed_instance.text] = []
                claim_to_sent[decomposed_instance.text].append({"from_sent_idx": sent.text, "in_sent_claim_idx": didx, "sent": sent.text})
            
        claim_checkworthiness = self._claim_level_checkworthy_scorer(filtered_decomposed)
        all_instances = [
            DedupScoreInstance(
                text=decomposed_instance.text,
                topic=decomposed_instance.topic,
                # in_sent_claim_idx=claim_idx,
                # from_sent_idx=idx,
                # sent=sent,
                sent_checkworthy=sent_checkworthy,
                claim_checkworthy=claim_checkworthy,
                **sent_dict
            )
            for decomposed_instance, claim_checkworthy in zip(filtered_decomposed, claim_checkworthiness) for sent_dict in claim_to_sent[decomposed_instance.text]
        ]
                
        # we already get all the instances from the base decomposer,
        # now we will run the deduplication
        
        if not all_instances:
            return []
        
        return self._deduplicate(all_instances)
    
    @overrides
    def _batch_decompose(self, instances: List[ScorerInstance]) -> List[List[ScorerInstance]]:
        """
        """

        if self._sentencize:
            # need to first breakdown into sentences
            sent_tuples = [(idx, sent) for idx, instance in enumerate(instances) for sent in self._to_sents(instance.text)]
        else:
            sent_tuples = [(idx, instance.text) for idx, instance in enumerate(instances)]
            
        sent_instances = [ScorerInstance(text=sent, topic=instances[idx].topic) for idx, sent in sent_tuples]

        checkworthiness = self._sentence_level_checkworthy_scorer(sent_instances, return_raw=True)
        checkworthiness = [c['parsed'] for c in checkworthiness]
        
        # filter down to those checkworthy
        sent_tuples = [(idx, ckwt, sent) for (idx, sent), ckwt in zip(sent_tuples, checkworthiness) if ckwt > 0.5]
        sent_instances = [ScorerInstance(text=sent, topic=instances[idx].topic) for idx, _, sent in sent_tuples]
        
        atomic_facts = self._base_decomposer(sent_instances)
        
        # group atomic facts based on  sentence origins
        claim_inputs = []
        for sidx, (stuple, fact_sets) in enumerate(zip(sent_tuples, atomic_facts)):
            claim_inputs.extend([(stuple[0], sidx, aidx, atom) for aidx, atom in enumerate(fact_sets)])
            
        claim_checkworthiness = self._claim_level_checkworthy_scorer([c[3] for c in claim_inputs])
        
        # create inputs to deduplication tupled with the instance indices
        ccw_inputs = [
            (
                ci[0],
                DedupScoreInstance(
                    text=ci[3].text,
                    topic=ci[3].topic,
                    in_sent_claim_idx=ci[2],
                    from_sent_idx=ci[1],
                    sent=sent_tuples[ci[1]][2],
                    sent_checkworthy=sent_tuples[ci[1]][1],
                    claim_checkworthy=ccw
                )
            ) for ci, ccw in zip(claim_inputs, claim_checkworthiness)]
        
        deduplicated = self._batch_deduplicate(ccw_inputs)
        
        # group them by the original instance index
        grouped_deduplicated = {}
        for idx, instance in deduplicated:
            if idx not in grouped_deduplicated:
                grouped_deduplicated[idx] = []
            grouped_deduplicated[idx].append(instance)
            
        return [
            grouped_deduplicated[idx] if idx in grouped_deduplicated else []
            for idx in range(len(instances))
        ]
        
    def _batch_deduplicate(self, instance_tuples: List[Tuple[int, DedupScoreInstance]]) -> List[Tuple[int, DedupScoreInstance]]:
        """
        """
        
        # group instances by the first integer from the tuple
        sent_ent_results = self._entailer([EntailerInstance(premise=instance.sent, hypothesis=instance.text) for _, instance in instance_tuples])
        
        grouped_instances: Dict[Text, Dict[Text, Dict[Text, Union[int, float]]]] = {}

        for iidx, (er, (idx, instance)) in enumerate(zip(sent_ent_results, instance_tuples)):
            if idx not in grouped_instances:
                grouped_instances[idx] = {}
            if instance.text not in grouped_instances[idx]:
                grouped_instances[idx][instance.text] = {
                    "score": 0.0,
                    "from_index": iidx
                }
            
            grouped_instances[idx][instance.text] = {
                "score": max(grouped_instances[idx][instance.text]['score'], er),
                "from_index": iidx if er > grouped_instances[idx][instance.text]['score'] else grouped_instances[idx][instance.text]['from_index']
            }
            
        grouped_texts = {
            idx: [
                (score_dict['from_index'], text)
                for text, score_dict in instances.items() if score_dict['score'] > 0.5
            ]
            for idx, instances in grouped_instances.items()
        }
        
        finding_pair_dicts = {}
        pairwise_entailment_inputs = []
        
        for idx, texts in grouped_texts.items():
            for i in range(len(texts)):
                for j in range(len(texts)):
                    if i == j:
                        continue
                    if idx not in finding_pair_dicts:
                        finding_pair_dicts[idx] = {}
                    finding_pair_dicts[idx][(i, j)] = len(pairwise_entailment_inputs)
                    pairwise_entailment_inputs.append(EntailerInstance(premise=texts[i][1], hypothesis=texts[j][1]))

        parwise_entailment_scoring = self._entailer(pairwise_entailment_inputs)
        
        # create intra_entailment_matrix for each of the group
        intra_entailment_matrices = {}
        
        for idx, texts in grouped_texts.items():
            intra_entailment_matrices[idx] = np.array([
                [
                    parwise_entailment_scoring[finding_pair_dicts[idx][(i, j)]] > 0.5 if i != j else False
                    for j in range(len(texts))
                ]
                for i in range(len(texts))
            ], dtype=np.int16)
            
        # solve the MILP problem for each group
        MILP_results = {}
        
        for idx, texts in grouped_texts.items():
            selection, result = self._solve_milp(
                pairwise_entailment=intra_entailment_matrices[idx],
                weighting=np.array([instance_tuples[sidx][1].sent_checkworthy * instance_tuples[sidx][1].claim_checkworthy for sidx, _ in texts], np.float32)
            )

            non_zero_selection_indices = np.nonzero(selection)[0].tolist()
            MILP_results[idx] = non_zero_selection_indices
            
        # fetch back results using MILP_results
        return_results = []
        for idx, texts in grouped_texts.items():
            return_results.extend([instance_tuples[texts[selected_idx][0]] for selected_idx in MILP_results[idx]])
            
        return return_results
        
    def _to_sents(self, text: Text) -> List[Text]:
        """ """

        return [sentence.text for sentence in self._nlp(text).sents]
    
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
                elif instances[i].text == instances[j].text:
                    finding_pair[(i, j)] = -1  # automatic entailment
                else:
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
                (pairwise_entailment_scoring[finding_pair[(i, j)]] > 0.5 if finding_pair[(i, j)] >= 0 else True) if i != j else False
                for j in range(len(instances))
            ]
            for i in range(len(instances))
        ], dtype=np.int16)
        
        # also, create the weighting vector from the instance
        # TODO: check whether this setting is optimal
        weighting = np.array([instance.sent_checkworthy * instance.claim_checkworthy for instance in instances], np.float32)

        selection, result = self._solve_milp(
            pairwise_entailment=intra_ent_mat,
            weighting=weighting
        )
        
        non_zero_selection_indices = np.nonzero(selection)[0].tolist()
        
        return [instances[index] for index in non_zero_selection_indices]
    
    # solve the MILP problem
    def _solve_milp(
        self,
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
    