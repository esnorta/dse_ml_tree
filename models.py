from dataclasses import dataclass


@dataclass
class Node:
    entropy: float
    example_count: int
