"""
"""

from dataclasses import dataclass, field
from typing import Text, Union
from langchain_interface.instances import Instance


@dataclass(frozen=True, eq=True)
class ScorerInstance(Instance):
    text: Text
    topic: Union[None, Text]


@dataclass(frozen=True, eq=True)
class DedupScoreInstance(ScorerInstance):
    in_sent_claim_idx: int
    from_sent_idx: int
    sent: Text
    sent_checkworthy: float
    claim_checkworthy: float

