"""Train a REV model with the given input types.
"""
import click
from envyaml import EnvYAML
from registrable import Registrable
from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
from src.task import Task


@click.command()
@click.argument("config-path")
@click.option("--cache-path", type=click.Path(exists=False, file_okay=True, dir_okay=False, writable=True, readable=True), default=None)
def main(
    config_path,
    cache_path
):
    """
    """
    
    config = EnvYAML(config_path, flatten=False, env_file="ENV.env", include_environment=True).export()['task']
    if cache_path is not None:
        set_llm_cache(SQLiteCache(cache_path))
    Task.from_params(config).run()
    
    
if __name__ == '__main__':
    main()