"""Train a REV model with the given input types.
"""
import click
from envyaml import EnvYAML
from registrable import Registrable
from langchain.globals import set_llm_cache
from langchain.cache import SQLiteCache
from src.task import Task


@click.command()
@click.argument("config-path")
@click.option("--cache-path", default=".cache/.gpt-3.5-turbo-cache.db", type=click.Path(exists=False, file_okay=True, dir_okay=False, writable=True, readable=True))
def main(
    config_path,
    cache_path
):
    """
    """
    
    config = EnvYAML(config_path, flatten=False, env_file="ENV.env", include_environment=True).export()['task']
    set_llm_cache(SQLiteCache(cache_path))
    Task.from_params(config).run()
    
    
if __name__ == '__main__':
    main()