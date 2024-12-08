""" """

from functools import lru_cache
from abc import abstractmethod, ABC
from typing import Text, Dict, Any, Union, List, Generator
from langchain_interface.instances import Instance
from registrable import Registrable


class BaseDataReader(ABC, Registrable):
    def __init__(
        self
    ):
        """ """
        super().__init__()

    def __call__(
        self,
        file_paths: Union[List[Text], Text]
    ) -> Generator[Instance, None, None]:
        """ """
        
        if isinstance(file_paths, str):
            file_paths = [file_paths]

        for fp in file_paths:
            yield from self._read(fp)
            
    @lru_cache(maxsize=3)
    def _read(self, file_path: Text) -> Generator[Instance, None, None]:
        """ """
        with open(file_path, "r", encoding="utf-8") as file_:
            for line in file_:
                yield self._parse(line)
        
    @abstractmethod
    def _parse(self, line: Text) -> Instance:
        """ """
        
        raise NotImplementedError(
            "Override the line_parsing to get proper reader."
        )