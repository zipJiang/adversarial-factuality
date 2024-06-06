"""Take the corrupted generation data, and check the compensation for the corruption.
"""

import json
import matplotlib.pyplot as plt
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


@Task.register("explore-corruption-compensation")
class ExploreCorruptionCompensation(Task):
    def __init__(
        self,
        corruption_dir: Text,
        method_map: Dict[Text, Text],
        orig_path: Text,
        output_path: Text
    ):
        """ """
        super().__init__()

        self._corruption_dir = corruption_dir
        self._orig_path = orig_path
        self._method_map = method_map
        self._output_path = output_path

    @overrides
    def run(self):
        """ """
        
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
            # if splitname != self._splitname:
            #     continue
                
            data = _load_data(os.path.join(self._corruption_dir, filename))
            score = np.mean([item['score'] for item in data]).item()
            selected.append(ParsedDP(splitname=splitname, score=score, num=num, method=method))
            
        # create canvas (landscape mode wider)
        fig, ax = plt.subplots(1, 1, figsize=(16, 6))
        
        selected = sorted(selected, key=lambda x: x.num, reverse=False)
        
        orig_data = _load_data(self._orig_path)
        orig_score = np.mean([item['score'] for item in orig_data]).item()
        
        # plot a orig_score_line for reference
        ax.axhline(orig_score, linestyle='--', label='Baseline', linewidth=5.0, color="#bebada")
            
        marker_list = ['o', 's', 'D', 'v', '^', '<', '>', 'p', 'P', '*', 'h', 'H', '+', 'x', 'X', 'D']
        colors = ["#fdb462", "#8dd3c7"] * 10
        linestyles = ['-', '-.'] * 10
        
        assert len(self._method_map) == 2, f"Invalid method_map: {self._method_map}"
        
        for color, marker, (tag, method) in zip(colors, marker_list, self._method_map.items()):
            for linestyle, split in zip(linestyles, set([s.splitname for s in selected if s.splitname is not None])):
                subselected = [item for item in selected if item.splitname == split or item.splitname is None]
                scores = [item.score for item in subselected if item.method == method]
                nums = [item.num for item in subselected if item.method == method]

                if linestyle == '-':
                    ax.plot(nums, scores, label=tag, marker=marker, linestyle=linestyle, linewidth=5.0, markersize=30.0, color=color)
                else:
                    # no label
                    ax.plot(nums, scores, marker=marker, linestyle=linestyle, linewidth=5.0, markersize=30.0, color=color)
            
        ax.set_yticks(np.arange(.4, .6, .04))
        ax.set_yticklabels([f"{x * 100:.0f}%" for x in ax.get_yticks()], fontsize=30)
        ax.set_xticks(sorted(set([s.num for s in selected]), reverse=False))
        ax.set_xticklabels([f"{x}" for x in ax.get_xticks()], fontsize=30)

        ax.set_xlabel("Number of Appended Sentences", fontsize=30)
        ax.set_ylabel("FP", fontsize=30)

        ax.legend(fontsize=30, facecolor='white', edgecolor='black', loc='right')
        fig.tight_layout()
        
        fig.savefig(self._output_path)