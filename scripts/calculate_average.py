"""Get average score of a datafile.
"""

import click
import numpy as np
import ujson as json


@click.command()
@click.option('--filepath', '-f', help='Path to the datafile.')
def main(filepath):
    """
    """
    with open(filepath, 'r', encoding='utf-8') as file_:
        data = json.load(file_)
        
    print(np.mean([item['score'] for item in data]).item())
    
    
if __name__ == '__main__':
    main()