"""Download particular split of wikipedia entity
dump from the huggingface hub.
"""

import click
from huggingface_hub import hf_hub_download


@click.command()
def main():
    """
    """
    repo_id = "nkandpa2/pretraining_entities"
    
    hf_hub_download(repo_id, filename="wikipedia_entity_map.npz", cache_dir="data/", repo_type="dataset")

    
if __name__ == '__main__':
    main()