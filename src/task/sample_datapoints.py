"""
"""

from .task import Task
import sys
from typing import Text, Optional
import random
import pickle
import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@Task.register("sample-datapoints")
class SampleDatapoints(Task):
    def __init__(
        self,
        dump_path: Optional[Text] = "data/entity_filtering/stats_checked_against_protected_entities.pkl",
        output_path: Optional[Text] = "data/entity_filtering/sampled_datapoints.txt"
    ):
        super().__init__()
        self._dump_path = dump_path
        self._random_seed = 42
        self._random_obj = random.Random(self._random_seed)
        self._continent_groups = [
            # This does not need to cover all the continents
            ("Insular Oceania", "Oceania", "Asia", "Indian subcontinent", "Australian continent"),
            ("North America",),
            ("Europe", "Eurasia"),
            ("Central America", "Afro-Eurasia", "South America", "Africa", "Caribbean", "Americas", None),
        ]
        self._frequency_ranges = [
            (0, 100),
            (100, 1000),
            (1000, 5000),
            (5000, sys.maxsize),
        ]
        
        self._each_split_size_train = 50
        self._each_split_size_dev = 7
        self._each_split_size_test = 7
        self._output_path = output_path

    def run(self):
        """
        """
        
        with open(self._dump_path, 'rb') as f:
            datapoints = pickle.load(f)
            
        results_train = []
        results_dev = []
        results_test = []

        for continent_group in self._continent_groups:
            for freq_range in self._frequency_ranges:
                filtered_datapoints = self._filter_datapoints(datapoints, continent_group, freq_range)
                
                samples = self._random_obj.sample(filtered_datapoints, self._each_split_size_train + self._each_split_size_test + self._each_split_size_dev)
                results_train.extend(samples[:self._each_split_size_train])
                results_dev.extend(samples[self._each_split_size_train:self._each_split_size_train + self._each_split_size_dev])
                results_test.extend(samples[self._each_split_size_train + self._each_split_size_dev:])
                
        with open(self._output_path + ".train", 'w', encoding='utf-8') as file_:
            for item in results_train:
                file_.write(item['wikipedia_title'] + '\n')
                
        with open(self._output_path + ".dev", 'w', encoding='utf-8') as file_:
            for item in results_dev:
                file_.write(item['wikipedia_title'] + '\n')
                
        with open(self._output_path + ".test", 'w', encoding='utf-8') as file_:
            for item in results_test:
                file_.write(item['wikipedia_title'] + '\n')
                
    def _filter_datapoints(self, datapoints, continent_group, freq_range):
        """
        """
        datapoints = [item for item in datapoints if item['is_in_dump'] and not item['already_selected'] and item['continent'] is not None]
        # print({item['continent'] for item in datapoints})
        
        def _validity_check(dp):
            return dp['is_in_dump'] and (not dp['already_selected']) and dp['popqa_freq'] > 0 and dp['wikidata_freq'] > 0 and dp['adjusted_freq'] < freq_range[1] and dp['adjusted_freq'] >= freq_range[0] and dp['continent'] in set(continent_group)
        
        def _relaxed_validity_check(dp):
            return dp['is_in_dump'] and dp['continent'] in set(continent_group) and max(dp['popqa_freq'], dp['wikidata_freq']) >= freq_range[0] and max(dp['popqa_freq'], dp['wikidata_freq']) < freq_range[1]
        
        filtered_datapoints = [dp for dp in datapoints if _validity_check(dp)]
        logger.info(f"Filtered {len(filtered_datapoints)} datapoints for continent group {continent_group} and frequency range {freq_range}")
        if len(filtered_datapoints) < self._each_split_size_train + self._each_split_size_dev + self._each_split_size_test:
            filtered_datapoints = [dp for dp in datapoints if _relaxed_validity_check(dp)]
            logger.info(f"(Re-)Filtered {len(filtered_datapoints)} datapoints for continent group {continent_group} and frequency range {freq_range}")
        
        return filtered_datapoints