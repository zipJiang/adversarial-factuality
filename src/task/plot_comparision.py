"""Plot the evaluation result of three different prompts"""

import json
import matplotlib.pyplot as plt
import matplotlib as mpl
from overrides import overrides
from typing import Text, Dict
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



@Task.register("plot-comparision")
class PlotComparisionTask(Task):
    def __init__(
        self,
        targ_dict: Dict[Text, Text],
        output_path: Text
    ):
        """targ_dict: Dict[Text, Text] --- legend_name -> file_path
        """
        
        super().__init__()
        
        self._targ_dict = targ_dict
        self._output_path = output_path
        
    @overrides
    def run(self):
        """"""
        
        fig, ax = plt.subplots()

        width = 0.25
        multiplier = 0
        
        topics = None

        for legend_name, file_path in self._targ_dict.items():
            with open(file_path, "r", encoding='utf-8') as file_:
                data = json.load(file_)
                offset = width * multiplier
                
                # we assume that all data are generated around the same topics
                stats = {item['topic']: item['score'] for item in data}
                
            if topics is None:
                topics = list(stats.keys())
                    
            ys = [stats[topic] for topic in topics]
            xs = [i + offset for i in range(len(ys))]
            ax.bar(xs, ys, label=legend_name, width=width)
            multiplier += 1
            
        ax.set_xlabel("Topic")
        ax.set_xticks([i + width for i in range(len(topics))], topics, rotation=20, size=12)
        ax.set_ylabel("Score")
        ax.set_ylim(0.2, 1.02)
        
        ax.legend()
        
        plt.tight_layout()

        fig.savefig(self._output_path, bbox_inches='tight', dpi=300)