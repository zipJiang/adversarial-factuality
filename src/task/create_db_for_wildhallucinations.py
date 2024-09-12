"""
"""

from overrides import overrides
import spacy
import datasets
import logging
import ujson as json
from typing import Text
from tqdm import tqdm
from ..retriever import WikiDocRetriever
from ..retriever.wikidoc_retriever import DocDB
from .task import Task


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler("logs/create_db_for_wildhallucinations.log")
handler.setLevel(logging.INFO)
logger.addHandler(handler)


@Task.register("create-db-for-wildhallucinations")
class CreateDBForWildHallucinations(Task):
    """
    """
    __NAME__ = "create-db-for-wildhallucinations"
    def __init__(
        self,
        output_db_path: Text,
    ):
        """
        """
        super().__init__()
        self._retriever = WikiDocRetriever(
            db_path="db/enwiki-20230401.db",
            cache_path="db/.cache/enwiki-20230401.cache",
            embed_cache_path="db/.cache/enwiki-20230401.embed.cache",
            retrieval_type="bm25",
            batch_size=16,
            device="cpu"
        )
        self._output_db_path = output_db_path
        self._dataset = datasets.load_dataset("wentingzhao/WildHallucinations", split='train')
        self._dataset = self._dataset.filter(lambda x: x['category'] in {"culture & entertainment", "geographic"})
        self._spacy_nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "textcat"])
        self._spacy_nlp.add_pipe("sentencizer")
        self._spacy_nlp.max_length = 2000000

    @overrides
    def run(self):
        """
        """
        
        topic_dict = {}
        num_no_hit_counter = {}
        category_counter = {}
        
        index_mapping = {}
        docs = []
        
        for didx, datapoint in enumerate(tqdm(self._dataset)):
            for iidx, item in enumerate(datapoint['info']):
                if item['status_code'] == 200:
                    index_mapping[(didx, iidx)] = len(docs)
                    docs.append(item['text'][:self._spacy_nlp.max_length])
                    
        logger.info(f"Processing {len(docs)} datapoints sents.")
        sents = [[sent.text for sent in doc.sents] for doc in tqdm(self._spacy_nlp.pipe(docs, batch_size=16, n_process=8))]
        
        for didx, datapoint in enumerate(tqdm(self._dataset)):
            topic = datapoint['entity']
            category_counter[datapoint['category']] = category_counter.get(datapoint['category'], 0) + 1
            try:
                passages = self._retriever.db.get_text_from_title(title=topic)
                logger.info(f"Topic: {topic} found.")
            except Exception as e:
                logger.info(f"Topic: {topic} not found.")
                num_no_hit_counter[datapoint['category']] = num_no_hit_counter.get(datapoint['category'], 0) + 1
                passages = []
                
            need_processing = []
            for iidx, item in enumerate(datapoint['info']):
                if item['status_code'] == 200:
                    idx = index_mapping[(didx, iidx)]
                    need_processing.extend(sents[idx])

            topic_dict[topic] = {
                "need_processing": need_processing,
                "already_processed": [p['text'] for p in passages]
            }

        # create a new db
        db = DocDB(
            db_path=self._output_db_path,
            data_dict=topic_dict
        )
        
        try:
            db.close()
        except Exception as e:
            logger.error(f"Error closing the db: {e}")

        logger.info(f"DB created at: {self._output_db_path}")
        logger.info(f"Processed {len(topic_dict)} topics.")

        for category, count in category_counter.items():
            logger.info(f"Category: {category} count: {count} num_no_hit: {num_no_hit_counter.get(category, 0)}")