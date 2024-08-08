from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Self


class ConditionType(Enum):
    THRESHOLD = "THRESHOLD"
    IN_SET = "IN_SET"


@dataclass
class Condition:
    type: ConditionType
    information_gain: float
    feature: Optional[str] = None
    threshold: Optional[str] = None
    set: Optional[List[str]] = None


@dataclass
class Node:
    entropy: Optional[float] = None
    example_count: Optional[int] = None
    condition: Optional[Condition] = None
    label: Optional[str] = None
    left_child: Optional[Self] = None
    right_child: Optional[Self] = None
    predictions: Optional[dict] = None


class Tree:
    pass
