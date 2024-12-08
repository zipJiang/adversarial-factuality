""" """

from typing import (
    Text,
    Optional,
    Iterable
)
from overrides import overrides
from langchain_interface.models import ChatOpenAIWithBatchAPI
from langchain_core.runnables.config import RunnableConfig
from ..langchain_step.factscore_evidential_support_step import (
    FActScoreEvidentialSupportStep,
)
from ..utils.instances import (
    DecomposedLLMGenerationInstance,
    VerifiedLLMGenerationInstance,
    VerifiedAtomicClaim
)
from ..aggregator import Aggregator, FActScoreAggregator
from .base_verifier import BaseVerifier
from ..retriever import Retriever


@BaseVerifier.register("factscore-verifier")
class FActScoreVerifier(BaseVerifier):
    def __init__(
        self,
        model_name: Text,
        retriever: Retriever,
        aggregator: Aggregator,
        base_url: Optional[Text] = None,
        api_key: Optional[Text] = None,
    ):
        """ """
        super().__init__()
        
        self._model_name = model_name
        self._base_url = base_url
        self._api_key = api_key
        
        self._llm = ChatOpenAIWithBatchAPI(
            model_name=self._model_name,
            base_url=self._base_url,
            api_key=self._api_key,
            max_tokens=128,
            # top_p=0.98,
            model_kwargs={"top_p": 0.98},
            temperature=0.0,
        )
        
        self._runnable_config = RunnableConfig(max_concurrency=32)
        self._agent = FActScoreEvidentialSupportStep().chain_llm(self._llm)
        self._aggregator = aggregator
        self._retriever = retriever
        
    @overrides
    def _verify(
        self, instance: DecomposedLLMGenerationInstance
    ) -> VerifiedLLMGenerationInstance:
        """ """
        
        inputs = []
        assert 'topic' in instance.meta, "The instance must have a topic."
        
        for claim in instance.claims:
            passages = self._retriever.get_passages(topic=instance.meta['topic'], text=claim.claim, n=5)
            
            input_instance = {
                "topic": instance.meta['topic'],
                "parsed_passages": "\n\n".join(
                    [
                        f"Title: {passage['title']} Text: {passage['text']}"
                        for passage in passages
                    ]
                )
                + "\n\n",
                "input": instance.text,
            }
            
            inputs.append(input_instance)
            
        responses = self._agent.batch(inputs, config=self._runnable_config)
        
        return VerifiedLLMGenerationInstance(
            id_=instance.id_,
            generation=instance.generation,
            meta=instance.meta,
            claims=[
                VerifiedAtomicClaim(
                    claim=claim.claim,
                    meta=claim.meta,
                    factual_score=response.evidential_support
                )
                for claim, response in zip(instance.claims, responses)
            ],
            aggregated_score=self._aggregator([response.evidential_support for response in responses])
        )
        
    @overrides
    def _batch_verify(
        self,
        instances: Iterable[DecomposedLLMGenerationInstance]
    ) -> Iterable[VerifiedLLMGenerationInstance]:
        """ """
    
        iidx_cidx_to_id = {}
        retrieval_questions = []
        retrieval_topics = []
        
        for iidx, instance in enumerate(instances):
            assert 'topic' in instance.meta, "The instance must have a topic."
            for cidx, claim in enumerate(instance.claims):
                iidx_cidx_to_id[(iidx, cidx)] = len(retrieval_questions)
                retrieval_questions.append(claim.claim)
                retrieval_topics.append(instance.meta['topic'])
                
        passage_chunks = self._retriever.get_passages_batched(
            topics=retrieval_topics, questions=retrieval_questions, k=5
        )
        
        assert len(passage_chunks) == len(retrieval_questions), "The number of passage chunks must be equal to the number of claims."
        
        input_instances = [
            {
                "topic": retrieval_topics[iidx_cidx_to_id[(iidx, cidx)]],
                "parsed_passages": "\n\n".join(
                    [
                        f"Title: {passage['title']} Text: {passage['text']}"
                        for passage in passage_chunks[iidx_cidx_to_id[(iidx, cidx)]]
                    ]
                ),
                "input": claim.claim
            }
            for iidx, instance in enumerate(instances) for cidx, claim in enumerate(instance.claims)
        ]
        
        responses = self._agent.batch(input_instances, config=self._runnable_config)

        return [
            VerifiedLLMGenerationInstance(
                id_=instance.id_,
                generation=instance.generation,
                meta=instance.meta,
                claims=[
                    VerifiedAtomicClaim(
                        claim=claim.claim,
                        meta=claim.meta,
                        factual_score=responses[iidx_cidx_to_id[(iidx, cidx)]].evidential_support
                    )
                    for cidx, claim in enumerate(instance.claims)
                ],
                aggregated_score=self._aggregator(
                    [
                        responses[iidx_cidx_to_id[(iidx, cidx)]].evidential_support
                        for cidx, claim in enumerate(instance.claims)
                    ]
                )
            )
            for iidx, instance in enumerate(instances)
        ]