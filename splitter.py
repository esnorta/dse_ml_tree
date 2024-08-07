from typing import List, Optional

import numpy as np
import numpy.typing as npt
import pandas as pd

from entropy import entropy_estimator
from models import Condition, Node


class Splitter:
    """
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
    """

    def __init__(self, tree):
        self.tree = tree

    @staticmethod
    def get_weighted_entropy_sum(arrays: List[npt.NDArray]) -> Optional[float]:
        weighted_entropy_sum = 0
        total_example_count = sum(len(a) for a in arrays)
        for array in arrays:
            entropy = entropy_estimator.get_shannon_entropy(array)
            example_count = len(array)
            weighted_entropy_sum += entropy * example_count / total_example_count

        return weighted_entropy_sum

    def get_information_gain(
        self, parent: Node, array_left: npt.NDArray, array_right: npt.NDArray
    ) -> float:
        weighted_entropy_sum = self.get_weighted_entropy_sum([array_left, array_right])

        information_gain = parent.entropy - weighted_entropy_sum
        return round(information_gain, 2)

    def perform_feature_splits(
        self, df_splits: pd.DataFrame, df: pd.DataFrame, feature: str, parent_node: Node
    ):
        target_feature = self.tree.target_feature
        values = df[feature].sort_values().unique()

        values = np.convolve(values, [0.5, 0.5], "valid")

        for value in values:
            df_left = df.loc[df[feature] < value]
            df_right = df.loc[df[feature] >= value]
            information_gain = self.get_information_gain(
                parent_node, df_left[target_feature], df_right[target_feature]
            )

            row = [feature, value, information_gain]
            df_splits.loc[len(df_splits)] = row

        return df_splits

    def find_best_split(
        self,
        parent_node: Node,
        df: pd.DataFrame,
        features: List[str],
    ) -> Condition:
        df_splits = pd.DataFrame(columns=["feature", "threshold", "information_gain"])
        for feature in features:
            df_splits = self.perform_feature_splits(df_splits, df, feature, parent_node)

        best_split = df_splits.loc[df_splits["information_gain"].idxmax()]
        condition = Condition(
            type="THRESHOLD",
            feature=best_split["feature"],
            threshold=best_split["threshold"],
        )

        return condition
