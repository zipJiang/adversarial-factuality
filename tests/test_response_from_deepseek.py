""" """

import unittest
from langchain_interface.models import ChatOpenAIWithBatchAPI
# from langchain_deepseek import ChatDeepSeek
from src.langchain_step.factscore_evidential_support_step import FActScoreEvidentialSupportStep
from langchain_interface.models.mixins import ReasoningContentMixin


class ChatOpenAIWithReasoning(ReasoningContentMixin, ChatOpenAIWithBatchAPI):
    """ """
    pass


class TestResponseFromDeepseek(unittest.TestCase):

    def setUp(self):
        
        self._base_url = "http://localhost:9871/v1"
        self._model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        self._api_key = "token-abc123"
        
        self.model = ChatOpenAIWithReasoning(
            model_name=self._model_name,
            base_url=self._base_url,
            api_key=self._api_key,
            # max_tokens=128,
            # top_p=0.98,
            temperature=0.0,
        )
        self.runnable = FActScoreEvidentialSupportStep().chain_llm(llm=self.model)
        
    def test_response_from_deepseek(self):
        """ """
        
        print(
            self.runnable.invoke({
                "topic": "Joe Biden",
                "parsed_passages": "As of 2025, Donald Trump is the President of the United States.",
                "input": "Joe Biden is the President of the United States.",
            }).reasoning_content
        )
        
    # def test_raw_openai_api(self):
    #     """ """

    #     from openai import OpenAI

    #     # Modify OpenAI's API key and API base to use vLLM's API server.
    #     openai_api_key = "token-abc123"
    #     openai_api_base = "http://localhost:9871/v1"

    #     client = OpenAI(
    #         api_key=openai_api_key,
    #         base_url=openai_api_base,
    #     )

    #     models = client.models.list()
    #     model = models.data[0].id

    #     # Round 1
    #     messages = [{"role": "user", "content": "9.11 and 9.8, which is greater?"}]
    #     response = client.chat.completions.create(model=model, messages=messages)

    #     print("=" * 80)
    #     reasoning_content = response.choices[0].message.reasoning_content
    #     print("=" * 80)
    #     content = response.choices[0].message.content

    #     print("reasoning_content:", reasoning_content)
    #     print("content:", content)