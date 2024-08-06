from typing import List, Optional

from models import Node


class Splitter:
    @staticmethod
    def get_weighted_entropy_sum(children: List[Node]) -> Optional[float]:
        weighted_entropy_sum = 0
        total_example_count = sum(c.example_count for c in children)
        for child in children:
            weighted_entropy_sum += (
                child.entropy * child.example_count / total_example_count
            )

        return weighted_entropy_sum

    def get_information_gain(self, parent: Node, children: List[Node]) -> float:
        weighted_entropy_sum = self.get_weighted_entropy_sum(children)

        information_gain = parent.entropy - weighted_entropy_sum
        return round(information_gain, 2)
