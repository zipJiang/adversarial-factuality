"""Calculate the correlation between the features of the dataset.
"""

import json
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy.stats import pearsonr, spearmanr
from overrides import overrides
from typing import Text, Dict, List, Tuple, Optional
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


@Task.register("calculate-correlation")
class CalculateCorrelationTask(Task):
    def __init__(
        self,
        data_dict: Dict[Text, Text],
        output_path: Text,
        comparisions: Optional[List[Tuple[Text, Text]]] = None,
    ):
        """
        """
        self._output_path = output_path
        self._data_dict = {}
        
        for key, value in data_dict.items():
            with open(value, 'r', encoding='utf-8') as f:
                split = json.load(f)
                self._data_dict[key] = [item['score'] for item in split]
                
        self._comparision = comparisions if comparisions is not None else [(key1, key2) for key1 in self._data_dict.keys() for key2 in self._data_dict.keys() if key1 != key2]
        
    @overrides
    def run(self):
        
        key_to_idx = {key: idx for idx, key in enumerate(self._data_dict.keys())}
        idx_to_key = [key for key in self._data_dict.keys()]
        for k, v in key_to_idx.items():
            idx_to_key[v] = k
        
        p_mat = np.zeros((len(key_to_idx), len(key_to_idx)))
        p_calculated = np.zeros((len(key_to_idx), len(key_to_idx)))
        s_mat = np.zeros((len(key_to_idx), len(key_to_idx)))
        s_calculated = np.zeros((len(key_to_idx), len(key_to_idx)))
        
        for comparision in self._comparision:
            x = self._data_dict[comparision[0]]
            y = self._data_dict[comparision[1]]
            
            pr = pearsonr(x, y)[0]
            sr = spearmanr(x, y)[0]
            
            p_mat[key_to_idx[comparision[0]], key_to_idx[comparision[1]]] = pr
            p_mat[key_to_idx[comparision[1]], key_to_idx[comparision[0]]] = pr
            p_calculated[key_to_idx[comparision[0]], key_to_idx[comparision[1]]] = 1
            p_calculated[key_to_idx[comparision[1]], key_to_idx[comparision[0]]] = 1
            s_mat[key_to_idx[comparision[0]], key_to_idx[comparision[1]]] = sr
            s_mat[key_to_idx[comparision[1]], key_to_idx[comparision[0]]] = sr
            s_calculated[key_to_idx[comparision[0]], key_to_idx[comparision[1]]] = 1
            s_calculated[key_to_idx[comparision[1]], key_to_idx[comparision[0]]] = 1
            
        # draw a markdown table for the pearson and spearman correlation (to .2f)
        
        with open(self._output_path, 'w', encoding='utf-8') as file_:
            file_.write('### Pearson\n\n')
            file_.write('| |')
            for key in idx_to_key:
                file_.write(f'{key}|')
            file_.write('\n')
            file_.write('|---|')
            for key in idx_to_key:
                file_.write('---|')
            file_.write('\n')
            for idx, key in enumerate(idx_to_key):
                file_.write(f'|{key}|')
                for idx2 in range(len(idx_to_key)):
                    if p_calculated[idx, idx2] == 1:
                        file_.write(f'{p_mat[idx, idx2]:.2f}|')
                    else:
                        file_.write(' |')
                file_.write('\n')
            file_.write('\n\n')
            
            file_.write('### Spearman\n\n')
            file_.write('| |')
            for key in idx_to_key:
                file_.write(f'{key}|')
            file_.write('\n')
            file_.write('|---|')
            for key in idx_to_key:
                file_.write('---|')
            file_.write('\n')
            for idx, key in enumerate(idx_to_key):
                file_.write(f'|{key}|')
                for idx2 in range(len(idx_to_key)):
                    if s_calculated[idx, idx2] == 1:
                        file_.write(f'{s_mat[idx, idx2]:.2f}|')
                    else:
                        file_.write(' |')
                file_.write('\n')
            file_.write('\n')