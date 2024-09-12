"""
"""

import ujson as json
from tqdm import tqdm
from src.retriever import WikiDocRetriever


def main():
    """
    """
    
    with open("data/training_sft/wildhallucination-informative/test.jsonl", 'r', encoding='utf-8') as file_:
        topics = [json.loads(line)['topic'] for line in file_]

    retriever = WikiDocRetriever(
        db_path="db/enwiki-20230401.db",
        cache_path="db/.cache/enwiki-20230401.cache",
        embed_cache_path="db/.cache/enwiki-20230401.embed.cache",
        retrieval_type="bm25",
        batch_size=16,
        device="cpu"
    )

    not_found = 0
    for topic in tqdm(topics):
        try:
            passages = retriever.db.get_text_from_title(title=topic)
        except Exception as e:
            not_found += 1
            
    print(f"Number of topics not found: {not_found}")
    print(f"Total topics: {len(topics)}")

        
if __name__ == "__main__":
    main()