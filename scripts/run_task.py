""" """

import click
import logging
from tasker import BaseTask
from tasks import *


logging.basicConfig(level=logging.INFO)


@click.command()
@click.option("--config-path", type=click.Path(exists=True), help="Path to the config file.")
def main(
    config_path
):
    """ """
    BaseTask.construct_and_run(config_path)
    
    
if __name__ == "__main__":
    main()