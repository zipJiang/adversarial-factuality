"""
"""

from dataclasses import dataclass, field
from typing import Text, Union
from langchain_interface.instances import Instance


@dataclass(frozen=True, eq=True)
class ScorerInstance(Instance):
    text: Text
    topic: Union[None, Text]