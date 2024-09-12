"""Get dataset statistics for the newly included dataset.
"""


import os
from io import StringIO
import ujson as json
from .task import Task
from typing import Text, Dict
from overrides import overrides


@Task.register("get-dataset-stats")
class GetDatasetStatsTask(Task):
    """
    """
    __NAME__ = "get-dataset-stats"

    def __init__(
        self,
        data_dir: Text,
        output_path: Text
    ):
        super().__init__()
        self._data_dir = data_dir
        self._output_path = output_path
        
    def _make_table(self, categories: Dict[Text, int]) -> Text:
        """
        """
        stream = StringIO()
        print("| Category | Count |", file=stream)
        print("|----------|-------|", file=stream)
        print("\n".join(f"| {k} | {v} |" for k, v in categories.items()), file=stream)
        
        return stream.getvalue()

    @overrides
    def run(self):
        
        with open(self._output_path, 'w', encoding='utf-8') as file_:
            for filename in os.listdir(self._data_dir):
                categories = {}
                if not filename.endswith(".jsonl"):
                    continue
                
                with open(os.path.join(self._data_dir, filename), "r") as f:
                    data = [json.loads(line) for line in f]

                for item in data:
                    item_categories = [c.strip() for c in item['category'].split('|')]
                    for category in item_categories:
                        if category not in categories:
                            categories[category] = 0
                        categories[category] += 1
                        
                stats = self._make_table(categories)
                file_.write(f"#### Statistics for {filename}:\n\n")
                file_.write(stats + "\n\n")