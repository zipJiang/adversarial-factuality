""" """

import os

try:
    import ujson as json
except ImportError:
    import json
import click


@click.command()
@click.option('--input', '-i', help='Path to the raw data file.', type=click.Path(exists=True))
@click.option('--output-dir', '-o', help='Path to the output dir.', type=click.Path(exists=True, dir_okay=True, file_okay=False))
def main(
    input: str,
    output_dir: str,
):
    """ """

    basename = os.path.basename(input)
    source = os.path.basename(input).split('.')[0].split('-')[1]
    output_file = open(os.path.join(output_dir, basename), 'w', encoding='utf-8')
    
    with open(input, 'r', encoding='utf-8') as file_:
        for ridx, row in enumerate(file_):
            data = json.loads(row)
            parsed = {
                "id_": data.get("_id", ridx),
                "generation": data["output"]['raw'],
                "meta": {
                    "topic": data["topic"],
                    "source": source
                }
            }
            
            output_file.write(json.dumps(parsed) + '\n')
            
    output_file.close()
    
    
if __name__ == '__main__':
    main()