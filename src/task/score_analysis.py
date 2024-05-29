"""Take multiple result files, and output relevant information,
and generate a directory contains all relevant analysis.
"""

import json
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from overrides import overrides
from typing import Dict, List, Text
from .task import Task


plt.style.use('ggplot')
plt.rcParams["font.family"] = "Nimbus Roman"
plt.rcParams["font.size"] = 16

# use the Set3 color scheme
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(
    color=[
        "#8dd3c7",
        "#ffffb3",
        "#bebada",
        "#fb8072",
        "#80b1d3",
        "#fdb462",
        "#b3de69",
        "#fccde5",
        "#d9d9d9",
        "#bc80bd",
        "#ccebc5",
        "#ffed6f",
        "#fcba03",
        "#d35ffa",
        "#fa9a5f"
    ]
)


@Task.register("score-analysis")
class ScoreAnalysisTask(Task):
    """
    """
    __NAME__ = "score-analysis"
    
    def __init__(
        self,
        result_paths: Dict[Text, Text],
        group_mapping_path: Text,
        output_dir: Text
    ):
        """
        """
        self._result_paths = result_paths
        self._output_dir = output_dir
        
        with open(group_mapping_path, 'r', encoding='utf-8') as file_:
            self._group_mapping = json.load(file_)
            
        self._breakdowns = [
            "very rare",
            "rare",
            "medium",
            "frequent"
        ]
            
    @overrides
    def run(self):
        """
        """
        
        all_result_dict = {}
        
        # create a figure for the analysis
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        def average(scores) -> float:
            return np.mean(scores).item()
        
        current_index = 0
        width = 1 / (len(self._result_paths) + 1)
        
        for key, path in self._result_paths.items():
            with open(path, 'r', encoding='utf-8') as file_:
                result = json.load(file_)
                
            result_dict = {"all": average([row['score'] for row in result])}

            for breakdown in self._breakdowns:
                result_dict[breakdown] = average([row['score'] for row in result if self._group_mapping[row['topic']] == breakdown])
                
            # create list of scores
            score_list = [result_dict[breakdown] * 100 for breakdown in self._breakdowns]
            offset = width * current_index
            rects = ax.bar(np.arange(len(self._breakdowns)) + offset, score_list, width, label=key)
            ax.bar_label(rects, padding=3, fmt='%.2f%%', fontsize=20)
            
            current_index += 1
            
            all_result_dict[key] = result_dict

        ax.set_ylabel('Scores', fontsize=20)
        ax.set_xlabel('Frequency', fontsize=20)
        # put legend at top right corner, fontsize 20
        ax.legend(loc='upper left', fontsize=20)
        ax.set_yticks(np.arange(0, 101, 25), [f"{level:.2f}%" for level in np.arange(0, 101, 25)], fontsize=20)
        ax.set_xticks(np.arange(len(self._breakdowns)), self._breakdowns, fontsize=20)
        
        fig.tight_layout()
        os.makedirs(self._output_dir, exist_ok=True)
        fig.savefig(os.path.join(self._output_dir, "score_analysis.svg"))
        
        with open(os.path.join(self._output_dir, "score_analysis.json"), 'w', encoding='utf-8') as file_:
            json.dump(all_result_dict, file_, indent=4)