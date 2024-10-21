"""Apply SAFE refinement to the deocomposition process.
"""

import spacy
from overrides import overrides
from typing import List, Text, Optional, Tuple
from langchain_interface.example_selectors import ConstantExampleSelector
from ..utils.instances import (
    ScorerInstance,
    DecontextScorerInstance,
    RelevancyScorerInstance,
    RelevancyInstance,
)
from ..decontextualizer.decontextualizer import Decontextualizer
from ..scorer.scorer import Scorer
from .decomposer import Decomposer


@Decomposer.register("refined-decomposer")
class RefinedDecomposer(Decomposer):
    """Adding SAFE relevency judgments as well as
    decontextualization to the process.

    1. Decontextualization
    2. Relevency Judgments
    """

    __NAME__ = "refined-decomposer"

    def __init__(
        self,
        base_decomposer: Decomposer,
        decontextualizer: Optional[Decontextualizer] = None,
        relevancy_scorer: Optional[Scorer] = None,
        nlp_model_name: Text = "en_core_web_sm",
        sentencize: bool = True,
    ):
        """The reason we need sentencize specification
        at this level is that we want to only provide
        context of the sentence the fact comes from.
        """
        super().__init__()

        self._base_decomposer = base_decomposer
        self._decontextualizer = decontextualizer
        self._relevancy_scorer = relevancy_scorer

        self._nlp = spacy.load(nlp_model_name, disable=["ner", "parser"])
        self._nlp.add_pipe("sentencizer")

        self._sentencize = sentencize

    @overrides
    def _decompose(self, instance: ScorerInstance) -> List[ScorerInstance]:
        """ """
        instance_text = instance.text
        topic = instance.topic

        if not self._sentencize:
            base_inputs = [ScorerInstance(text=instance_text, topic=topic, source_text=instance.source_text)]
        else:
            base_inputs = [
                ScorerInstance(text=sentence.text, topic=topic, source_text=instance.source_text)
                for sentence in self._nlp(instance_text).sents
            ]

        base_outputs = self._base_decomposer(base_inputs)[0]

        if self._decontextualizer is not None:
            decontextualizer_inputs = [
                DecontextScorerInstance(text=otp.text, topic=topic, sent=instance.source_text, source_text=instance.source_text)
                for otp in base_outputs
            ]
            
            # print([ip.text for ip in decontextualizer_inputs])
            decontextualized = self._decontextualizer(
                decontextualizer_inputs, return_raw=False
            )
        else:
            decontextualized = base_outputs

        # We need to evaluate relevancy if the relevancy_scorer is presented
        if self._relevancy_scorer is not None:
            # construct RelevancyScorerInstance from ScorerInstance
            rs_inputs = [
                RelevancyScorerInstance(text=opt.text, topic=opt.topic, sent=ipt.source_text, source_text=ipt.source_text)
                for opt, ipt in zip(decontextualized, decontextualizer_inputs)
            ]
            rscores: List[float] = self._relevancy_scorer(rs_inputs, return_raw=False)

            # filter down to those relevant

            outputs = [
                dinstance
                for dinstance, score in zip(decontextualized, rscores)
                if score > 0.5
            ]

        else:
            outputs = decontextualized

        return outputs

    @overrides
    def _batch_decompose(
        self, instances: List[ScorerInstance]
    ) -> List[List[ScorerInstance]]:
        """ """

        base_input_list: Tuple[int, ScorerInstance] = []

        for idx, instance in enumerate(instances):
            instance_text = instance.text
            topic = instance.topic

            if not self._sentencize:
                base_inputs = [ScorerInstance(text=instance_text, topic=topic, source_text=instance.source_text)]
            else:
                base_inputs = [
                    ScorerInstance(text=sentence.text, topic=topic, source_text=instance.source_text)
                    for sentence in self._nlp(instance_text).sents
                ]

            base_input_list.extend([(idx, inp) for inp in base_inputs])

        base_outputs: List[List[ScorerInstance]] = self._base_decomposer(
            [inp for _, inp in base_input_list]
        )

        if self._decontextualizer is not None:
            decontextualizer_input_tuples = [
                (
                    index,
                    DecontextScorerInstance(
                        text=opt.text, topic=ipt.topic, sent=ipt.source_text, source_text=ipt.source_text
                    ),
                )
                for opt_list, (index, ipt) in zip(base_outputs, base_input_list)
                for opt in opt_list
            ]
            decontextualized = self._decontextualizer(
                [decontext_scorer_instance for _, decontext_scorer_instance in decontextualizer_input_tuples], return_raw=False
            )
            # tag-back and flat
            decontextualized_tuples = [(idx, dinstance) for dinstance, (idx, _) in zip(decontextualized, decontextualizer_input_tuples)]
            
            # for (_, dinstance), (_, dipt) in zip(decontextualized_tuples, decontextualizer_input_tuples):
            #     print("-" * 20)
            #     print(f"Original: {dipt.text}")
            #     print(f"Decontextualized: {dinstance.text}")
            #     print("-" * 20)
            
        else:
            decontextualized_tuples = [
                (
                    index,
                    ScorerInstance(
                        text=opt.text, topic=ipt.topic, source_text=ipt.source_text
                    ),
                )
                for opt_list, (index, ipt) in zip(base_outputs, base_input_list)
                for opt in opt_list
            ]

        # We need to evaluate relevancy if the relevancy_scorer is presented
        if self._relevancy_scorer is not None:
            rs_inputs = [
                RelevancyScorerInstance(text=opt.text, topic=opt.topic, sent=opt.source_text, source_text=opt.source_text)
                for index, opt in decontextualized_tuples
            ]
            
            rscores: List[float] = self._relevancy_scorer(rs_inputs, return_raw=False)

            output_tuple_list = [
                dtuple
                for dtuple, score in zip(
                    decontextualized_tuples, rscores
                )
                if score > 0.5
            ]
        else:
            output_tuple_list = decontextualized_tuples

        # group by the original index
        return [
            list(map(lambda x: x[1], filter(lambda x: x[0] == idx, output_tuple_list)))
            for idx in range(len(instances))
        ]
