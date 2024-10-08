from typing import Any, List, get_args

import numpy as np
import numpy.typing as npt
import pandas as pd
from const import SPLIT_CRITERIA
from entropy import entropy_estimator
from models import Node
from splitter import Splitter


class Tree:
    min_split_size = 10

    def __init__(
        self,
        df: pd.DataFrame,
        target_feature: str,
        split_criterion: SPLIT_CRITERIA,
        min_gain: float = 0.1,
        max_depth: int = 40,
    ):
        self.root = Node(
            entropy_estimator.get_shannon_entropy(df[target_feature]), len(df)
        )
        self.target_feature = target_feature
        self.min_gain = min_gain
        self.split_criterion = split_criterion
        if split_criterion not in get_args(SPLIT_CRITERIA):
            raise Exception(f"Invalid split criterion: {split_criterion}")
        self.max_depth = max_depth
        self.depth = 0

    def get_leaf_label_probabilities(self, array: npt.NDArray[Any]) -> dict:
        uniques, counts = np.unique(array, return_counts=True)
        fractions = dict(zip(uniques, counts / len(array)))
        return fractions

    def make_leaf(self, node: Node, df: pd.DataFrame) -> Node:
        predictions = self.get_leaf_label_probabilities(df[self.target_feature])
        node.predictions = predictions
        return node

    def grow(
        self,
        node: Node,
        df: pd.DataFrame,
        features: List[str],
    ) -> None:
        if len(df) < self.min_split_size:
            self.make_leaf(node, df)
            return

        if self.depth + 1 > self.max_depth:
            self.make_leaf(node, df)
            return
        self.depth += 1

        condition = Splitter(self.target_feature).find_best_split(
            node, df, features, self.split_criterion
        )
        if condition.gain < self.min_gain:
            self.make_leaf(node, df)
            return
        node.condition = condition

        match condition.type:
            case "THRESHOLD":
                df_left = df.loc[df[condition.feature] < condition.threshold]
                df_right = df.loc[df[condition.feature] >= condition.threshold]
            case "IN_SET":
                df_left = df.loc[df[condition.feature].isin(condition.in_set)]
                df_right = df.loc[~df[condition.feature].isin(condition.in_set)]
            case _:
                raise Exception(f"Unknown condition type: {condition.type}")

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
        self.grow(right_child, df_right, features)

    def predict(self, datapoint: pd.core.series.Series) -> dict:
        node = self.root
        if not node:
            raise Exception("Root node has not been initialised")
        while node.condition:
            match node.condition.type:
                case "THRESHOLD":
                    if datapoint[node.condition.feature] < node.condition.threshold:
                        node = node.left_child
                    elif datapoint[node.condition.feature] >= node.condition.threshold:
                        node = node.right_child
                case "IN_SET":
                    if datapoint[node.condition.feature] in node.condition.in_set:
                        node = node.left_child
                    elif datapoint[node.condition.feature] not in node.condition.in_set:
                        node = node.right_child

        return node.predictions

    def predict_label(self, datapoint: pd.core.series.Series):
        predictions = self.predict(datapoint)
        max_prob_label = max(predictions, key=lambda x: predictions[x])
        return max_prob_label
