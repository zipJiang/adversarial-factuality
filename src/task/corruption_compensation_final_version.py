"""We run final version plot for corruption_compensation
(This will be without argument so that the plots will be
there without any configuration).
"""

import json
import matplotlib.pyplot as plt
import itertools
import matplotlib as mpl
import numpy as np
import os
from typing import Text, List, Dict, Tuple
from dataclasses import dataclass
from overrides import overrides
from .task import Task


plt.style.use('ggplot')
plt.rcParams["font.family"] = "Nimbus Roman"
plt.rcParams["font.size"] = 16

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# use the Set3 color scheme
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(
    color=[
        "#fdb462",
        "#8dd3c7",
        "#ffffb3",
        "#bebada",
        "#fb8072",
        "#80b1d3",
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


@dataclass(frozen=True, eq=True)
class ParsedDP:
    splitname: Text
    score: float
    num: int
    method: Text


@Task.register("corruption-compensation-final-version")
class CorruptionCompensationFinalVersion(Task):
    """Run final version plot for corruption_compensation."""

    def __init__(
        self,
        splitname: Text,
        output_path: Text,
    ):
        super().__init__()

        self._corruption_dir = "data/scores/new_corrupted/entertainment"
        self._splitname = splitname
        self._output_path = output_path
        self._label_map = {
            "factscore": "FActScore",
            "dedup": "FActMore",
        }
        
        self._method_source_map = {
            "factscore": "data/scores/new_data/mistral-entertainment-normal-factscore.json",
            "dedup": "data/scores/new_data/mistral-entertainment-normal-core.json",
        }

    @overrides
    def run(self):
        """
        """
    
        def _load_data(path: Text) -> List[Dict]:
            """ """
            with open(path, "r", encoding='utf-8') as file_:
                data = json.load(file_)
                
            return data
        
        def _parse_filename(path: Text) -> Tuple[Text, int, Text]:
            """Return the number and method from the filename.
            The first field is splitname
            """
            assert path.endswith(".json"), f"Invalid path: {path}"
            filename = os.path.basename(path)[: -len(".json")]
            splitted = filename.split('-')
            
            if len(splitted) == 2:
                return None, 0, splitted[1]
            else:
                return splitted[-3], int(splitted[-2]), splitted[-1]
            
        
        selected = []
            
        for filename in os.listdir(self._corruption_dir):
            if not filename.endswith(".json"):
                continue
            
            splitname, num, method = _parse_filename(filename)
            data = _load_data(os.path.join(self._corruption_dir, filename))
            score = np.mean([item['score'] for item in data]).item()
            selected.append(ParsedDP(splitname=splitname, score=score, num=num, method=method))
            
        # our intention is to do a ggplot
        selected = sorted(
            [item for item in selected if item.splitname == self._splitname or item.splitname is None],
            key=lambda x: x.num if x.splitname is not None else 0,
            reverse=False
        )
        
        # create a figure for the analysis
        fig, ax = plt.subplots(1, 1, figsize=(16, 6))

        current_index = 0
        width = 1 / 3  # Since we only have two methods
        
        colors = [
            "#fdb462",
            "#8dd3c7",
        ]

        for method in ["factscore", "dedup"]:
            color = colors[current_index]
            group = [item for item in selected if item.method == method]
            print(group)
            x = np.arange(len(group) + 1)
            y = [item.score for item in group]
            
            base_data = _load_data(self._method_source_map[method])
            base_score = np.mean([item['score'] for item in base_data]).item()

            ax.bar(x + current_index * width, y + [base_score], width, label=method)
            current_index += 1
            
            # add a line from the base_score to the left
            ax.plot([-.5, len(group) + (current_index - .5) * width], [base_score, base_score], 'k--', linewidth=2, color=color)
            
        ax.set_yticks(np.arange(.4, .601, .04))
        ax.set_ylim([.4, .6])
        ax.set_yticklabels([f"{x * 100:.0f}%" for x in ax.get_yticks()], fontsize=30)
        # ax.set_xticks(sorted(set([s.num for s in selected]), reverse=False))
        ax.set_xticks(np.arange(len(group) + 1), [f"{item.num}" for item in group] + ['clean'], fontsize=30)
        ax.legend()
            
        fig.tight_layout()
        fig.savefig(self._output_path)