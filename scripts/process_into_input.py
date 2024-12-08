""" Take some data and process it into the input format
for abstention_detection.
"""

import click
import ujson as json
from typing import Text


@click.command()
@click.option('--input-path', type=click.Path(exists=True), required=True)
@click.option('--output-path', type=click.Path(), required=True)
def main(
    input_path: Text,
    output_path: Text
):
    """ """

    with open(input_path, 'r') as f:
        data = json.load(f)
        
    with open(output_path, 'w') as f:
        for idx, item in enumerate(data):
            f.write(
                json.dumps({
                    "id_": idx,
                    "generation": item['output']['parsed'],
                    "meta": {
                        "topic": item['topic'],
                        "note": "dev data"
                    }
                }) + '\n'
            )
            
            
if __name__ == '__main__':
    main()