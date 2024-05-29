"""Take a document with a list of topics,
and a stats file that can be mapped back
to retrieve the frequency group.
"""

import os
import json
import sys
import click
from typing import List, Text, Dict


def get_topics(path: Text) -> List[Text]:
    """
    """
    with open(path, 'r', encoding='utf-8') as file_:
        topics = [line.strip() for line in file_.read().splitlines()]
        
    return topics


def build_stats_dict(path: Text) -> Dict[Text, Text]:
    """Take the stats path and try to build
    mapping from topic to frequency group.
    """
    
    import pickle
    with open(path, 'rb') as file_:
        data = pickle.load(file_)
        
    frequency_dict = {line['wikipedia_title']: line['adjusted_freq'] if line['adjusted_freq'] >= 0 else max(line['wikidata_freq'], line['popqa_freq']) for line in data}
    frequency_dict = {topic: frequency_dict[topic] for topic in frequency_dict if frequency_dict[topic] >= 0}

    # map frequency_dict to group_frequency_dict
    
    freq_groups = [
        (0, 100),
        (100, 1000),
        (1000, 5000),
        (5000, sys.maxsize),
    ]
    
    def _get_freq_group(freq):
        for gidx, group in enumerate(freq_groups):
            if freq >= group[0] and freq < group[1]:
                return gidx
            
    freq_group_indices = {topic: _get_freq_group(frequency_dict[topic]) for topic in frequency_dict}
    
    index_to_name = {
        0: "very rare",
        1: "rare",
        2: "medium",
        3: "frequent"
    }
    
    return {topic: index_to_name[freq_group_indices[topic]] for topic in freq_group_indices}


@click.command()
@click.option("--topic-list", type=click.Path(exists=True), help="Path to the topic list.", required=True)
@click.option("--stats-path", type=click.Path(exists=True), help="Path to the stats file.", required=True)
@click.option("--output-path", type=click.Path(exists=False), help="Path to the output file.", required=True)
def main(
    topic_list,
    stats_path,
    output_path
):
    """
    """
    topics = get_topics(topic_list)
    freq_group_map = build_stats_dict(stats_path)
    selected = {topic: freq_group_map[topic] for topic in topics}

    with open(output_path, 'w', encoding='utf-8') as file_:
        json.dump(selected, file_, indent=4)

        
if __name__ == '__main__':
    main()