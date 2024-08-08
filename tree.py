from typing import Any, List

import numpy as np
import numpy.typing as npt
import pandas as pd

from entropy import entropy_estimator
from models import Node
from splitter import Splitter


class Tree:
    min_information_gain = 0.0004

    def __init__(self, df: pd.DataFrame, target_feature: str):
        self.root = Node(
            entropy_estimator.get_shannon_entropy(df[target_feature]), len(df)
        )
        self.target_feature = target_feature
        self.nodes = []

    def get_leaf_label_probabilities(self, array: npt.NDArray[Any]) -> dict:
        uniques, counts = np.unique(array, return_counts=True)
        fractions = dict(zip(uniques, counts / len(array)))
        return fractions

    def grow(self, node: Node, df: pd.DataFrame, features: List[str]) -> None:
        condition = Splitter(self).find_best_split(node, df, features)
        print(condition.information_gain)

        if condition.information_gain < self.min_information_gain:
            print("Growing stopped")
            node.predictions = self.get_leaf_label_probabilities(
                df[self.target_feature]
            )
            return
        node.condition = condition
        self.nodes.append(node)

        df_left = df.loc[df[condition.feature] < condition.threshold]
        df_right = df.loc[df[condition.feature] >= condition.threshold]

        left_child = Node(
            entropy=entropy_estimator.get_shannon_entropy(df_left[self.target_feature]),
            example_count=len(df_left),
        )
        right_child = Node(
            entropy=entropy_estimator.get_shannon_entropy(
                df_right[self.target_feature]
            ),
            example_count=len(df_right),
        )
        node.left_child = left_child
        node.right_child = right_child

        self.grow(left_child, df_left, features)
        self.grow(right_child, df_left, features)

    def predict_datapoint_label(self, datapoint: pd.core.series.Series) -> dict:
        node = self.root
        while node.condition:
            if datapoint[node.condition.feature] < node.condition.threshold:
                node = node.left_child
            elif datapoint[node.condition.feature] >= node.condition.threshold:
                node = node.right_child

        return node.predictions
