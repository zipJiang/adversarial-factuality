"""
"""

from dataclasses import dataclass, field
from typing import Text, Optional
from langchain_interface.instances import Instance


@dataclass(frozen=True, eq=True)
class ScorerInstance(Instance):
    text: Text
    topic: Optional[Text]