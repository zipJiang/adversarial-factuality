"""For the time being, we only use the retrieve
from the FActScore, and we will count on future
refactoring to make this more general.
"""

import json
import time
import os

import sqlite3
import numpy as np
import pickle as pkl

from rank_bm25 import BM25Okapi
from typing import List, Dict, Text, Any

SPECIAL_SEPARATOR = "####SPECIAL####SEPARATOR####"
MAX_LENGTH = 256


class DocDB(object):
    """Sqlite backed document storage.

    Implements get_doc_text(doc_id).
    """

    def __init__(self, db_path=None, data_path=None):
        self.db_path = db_path
        self.connection = sqlite3.connect(self.db_path, check_same_thread=False)

        cursor = self.connection.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")

        if len(cursor.fetchall()) == 0:
            assert (
                data_path is not None
            ), f"{self.db_path} is empty. Specify `data_path` in order to create a DB."
            print(f"{self.db_path} is empty. start building DB from {data_path}...")
            self.build_db(self.db_path, data_path)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def path(self):
        """Return the path to the file that backs this database."""
        return self.path

    def close(self):
        """Close the connection to the database."""
        self.connection.close()

    def build_db(self, db_path, data_path):
        from transformers import RobertaTokenizer

        tokenizer = RobertaTokenizer.from_pretrained("roberta-large")

        titles = set()
        output_lines = []
        tot = 0
        start_time = time.time()
        c = self.connection.cursor()
        c.execute("CREATE TABLE documents (title PRIMARY KEY, text);")

        with open(data_path, "r") as f:
            for line in f:
                dp = json.loads(line)
                title = dp["title"]
                text = dp["text"]
                if title in titles:
                    continue
                titles.add(title)
                if type(text) == str:
                    text = [text]
                passages = [[]]
                for sent_idx, sent in enumerate(text):
                    assert len(sent.strip()) > 0
                    tokens = tokenizer(sent)["input_ids"]
                    max_length = MAX_LENGTH - len(passages[-1])
                    if len(tokens) <= max_length:
                        passages[-1].extend(tokens)
                    else:
                        passages[-1].extend(tokens[:max_length])
                        offset = max_length
                        while offset < len(tokens):
                            passages.append(tokens[offset : offset + MAX_LENGTH])
                            offset += MAX_LENGTH

                psgs = [
                    tokenizer.decode(tokens)
                    for tokens in passages
                    if np.sum([t not in [0, 2] for t in tokens]) > 0
                ]
                text = SPECIAL_SEPARATOR.join(psgs)
                output_lines.append((title, text))
                tot += 1

                if len(output_lines) == 1000000:
                    c.executemany("INSERT INTO documents VALUES (?,?)", output_lines)
                    output_lines = []
                    print(
                        "Finish saving %dM documents (%dmin)"
                        % (tot / 1000000, (time.time() - start_time) / 60)
                    )

        if len(output_lines) > 0:
            c.executemany("INSERT INTO documents VALUES (?,?)", output_lines)
            print(
                "Finish saving %dM documents (%dmin)"
                % (tot / 1000000, (time.time() - start_time) / 60)
            )

        self.connection.commit()
        self.connection.close()

    def get_text_from_title(self, title):
        """Fetch the raw text of the doc for 'doc_id'."""
        cursor = self.connection.cursor()
        cursor.execute("SELECT text FROM documents WHERE title = ?", (title,))
        results = cursor.fetchall()
        results = [r for r in results]
        cursor.close()
        assert (
            results is not None and len(results) == 1
        ), f"`topic` in your data ({title}) is likely to be not a valid title in the DB."
        results = [
            {"title": title, "text": para}
            for para in results[0][0].split(SPECIAL_SEPARATOR)
        ]
        assert (
            len(results) > 0
        ), f"`topic` in your data ({title}) is likely to be not a valid title in the DB."
        return results


class Retriever(object):

    def __init__(
        self,
        db_path,
        cache_path,
        embed_cache_path,
        retrieval_type="gtr-t5-large",
        batch_size=None,
        device: Text = "cuda:0",
    ):
        # self.db = db
        self.db = DocDB(db_path=db_path)
        self.cache_path = cache_path
        self.embed_cache_path = embed_cache_path
        self.retrieval_type = retrieval_type
        self.batch_size = batch_size
        assert retrieval_type == "bm25" or retrieval_type.startswith("gtr-")

        self.encoder = None
        self.load_cache()
        self.add_n = 0
        self.add_n_embed = 0
        
        self._device = device

    def load_encoder(self):
        from sentence_transformers import SentenceTransformer

        encoder = SentenceTransformer("sentence-transformers/" + self.retrieval_type)
        encoder = encoder.to(self._device)
        encoder = encoder.eval()
        # encoder = encoder.cuda()
        # encoder = encoder.eval()
        self.encoder = encoder
        assert self.batch_size is not None

    def load_cache(self):
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "r") as f:
                self.cache = json.load(f)
        else:
            self.cache = {}
        if os.path.exists(self.embed_cache_path):
            with open(self.embed_cache_path, "rb") as f:
                self.embed_cache = pkl.load(f)
        else:
            self.embed_cache = {}

    def save_cache(self):
        if self.add_n > 0:
            if os.path.exists(self.cache_path):
                with open(self.cache_path, "r") as f:
                    new_cache = json.load(f)
                self.cache.update(new_cache)

            with open(self.cache_path, "w") as f:
                json.dump(self.cache, f)

        if self.add_n_embed > 0:
            if os.path.exists(self.embed_cache_path):
                with open(self.embed_cache_path, "rb") as f:
                    new_cache = pkl.load(f)
                self.embed_cache.update(new_cache)

            with open(self.embed_cache_path, "wb") as f:
                pkl.dump(self.embed_cache, f)

    def get_bm25_passages(self, topic, query, passages, k):
        if topic in self.embed_cache:
            bm25 = self.embed_cache[topic]
        else:
            bm25 = BM25Okapi(
                [
                    psg["text"].replace("<s>", "").replace("</s>", "").split()
                    for psg in passages
                ]
            )
            self.embed_cache[topic] = bm25
            self.add_n_embed += 1
        scores = bm25.get_scores(query.split())
        indices = np.argsort(-scores)[:k]
        return [passages[i] for i in indices]

    def get_gtr_passages(self, topic, retrieval_query, passages, k):
        if self.encoder is None:
            self.load_encoder()
        if topic in self.embed_cache:
            passage_vectors = self.embed_cache[topic]
        else:
            inputs = [
                psg["title"] + " " + psg["text"].replace("<s>", "").replace("</s>", "")
                for psg in passages
            ]
            passage_vectors = self.encoder.encode(
                inputs, batch_size=self.batch_size, device=self.encoder.device
            )
            self.embed_cache[topic] = passage_vectors
            self.add_n_embed += 1
        query_vectors = self.encoder.encode(
            [retrieval_query], batch_size=self.batch_size, device=self.encoder.device
        )[0]
        scores = np.inner(query_vectors, passage_vectors)
        indices = np.argsort(-scores)[:k]
        # print(retrieval_query, indices)
        return [passages[i] for i in indices]

    def get_gtr_passages_batched(
        self,
        topics: List[Text],
        retrieval_queries: List[Text],
        chunks_of_passages: List[List[Dict[Text, Any]]],
        k: int
    ) -> Dict[Text, List[Dict[Text, Text]]]:
        """ """

        if not topics:
           return [] 
        
        if self.encoder is None:
            self.load_encoder()
            
        # we don't have to cache here, as encoded cache
        # should already been cached with results (assuming k is fixed).
        
        # flattened_raw_psgs = [
        #     psg
        #     for passages in chunks_of_passages
        #     for psg in passages
        # ]
        # flattened_psgs = [
        #     psg["title"] + " " + psg["text"].replace("<s>", "").replace("</s>", "")
        #     for psg in flattened_raw_psgs
        # ]
        
        flattened_raw_psgs = []
        flattened_psgs = []

        # we select a subset of the passages to encode
        topics_already_seen = {}
        for t, chunk in zip(topics, chunks_of_passages):
            if t not in topics_already_seen:
                topics_already_seen[t] = (len(flattened_raw_psgs), len(chunk) + len(flattened_raw_psgs))
                flattened_raw_psgs.extend(chunk)
                flattened_psgs.extend([
                    psg["title"] + " " + psg["text"].replace("<s>", "").replace("</s>", "")
                    for psg in chunk
                ])
        
        topic_vectors = self.encoder.encode(
            sentences=retrieval_queries,
            batch_size=self.batch_size,
            device=self._device
        )
        
        passage_vectors = self.encoder.encode(
            sentences=flattened_psgs,
            batch_size=self.batch_size,
            device=self._device
        )

        relations = np.inner(topic_vectors, passage_vectors)
        
        return_vals = []
        for rel, topic in zip(relations, topics):
            pstarts, pends = topics_already_seen[topic]
            indices = np.argsort(-rel[pstarts:pends], axis=0)[:k]
            # print(retrieval_query, indices, "batched")
            cum_ind = [pstarts + i for i in indices]
            return_vals.append([
                flattened_raw_psgs[i]
                for i in cum_ind
            ])
            
        assert len(return_vals) == len(retrieval_queries)
            
        return return_vals

    def get_passages(self, topic, question, k):
        retrieval_query = topic + " " + question.strip()
        cache_key = topic + "#" + retrieval_query

        if cache_key not in self.cache:
            passages = self.db.get_text_from_title(topic)
            if self.retrieval_type == "bm25":
                self.cache[cache_key] = self.get_bm25_passages(
                    topic, retrieval_query, passages, k
                )
            else:
                self.cache[cache_key] = self.get_gtr_passages(
                    topic, retrieval_query, passages, k
                )
            assert len(self.cache[cache_key]) in [k, len(passages)]
            self.add_n += 1

        return self.cache[cache_key]

    def get_passages_batched(self, topics: List[Text], questions: List[Text], k: int) -> List[List[Dict[Text, Any]]]:
        """We optimize this by running the retrieval in parallel.
        """
        
        # generate cache key for all topics
        queries = [' '.join((topic, question.strip())) for topic, question in zip(topics, questions)]
        cache_keys = [topic + "#" + query for topic, query in zip(topics, queries)]
        
        result_dict = {}
        
        topics_need_to_check = []
        cache_keys_need_to_check = []
        queries_need_to_check = []
        passages_need_to_check = []
        
        cnc_set = set()

        for topic, query, key in zip(topics, queries, cache_keys):
            if key in self.cache:
                result_dict[key] = self.cache[key]
            elif key not in cnc_set:
                passages = self.db.get_text_from_title(topic)
                # if len(passages) <= k:
                #     result_dict[key] = passages
                # else:
                topics_need_to_check.append(topic)
                queries_need_to_check.append(query)
                cache_keys_need_to_check.append(key)
                passages_need_to_check.append(passages)
                cnc_set.add(key)
                
        if self.retrieval_type != 'bm25':
            filtered_passage_results = self.get_gtr_passages_batched(
                topics=topics_need_to_check,
                retrieval_queries=queries_need_to_check,
                chunks_of_passages=passages_need_to_check,
                k=k
            )
        else:
            filtered_passage_results = []
            raise NotImplementedError("Batched BM25 is not implemented yet.")
        
        for key, result in zip(cache_keys_need_to_check, filtered_passage_results):
            # assert topic not in result_dict, f"Topic {topic} is already in the result_dict."
            assert key not in result_dict, f"Key {key} is already in the result_dict."
            result_dict[key] = result
            self.cache[key] = result

        return [result_dict[key] for key in cache_keys]