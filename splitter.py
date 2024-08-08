import itertools
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd

from entropy import entropy_estimator
from models import Condition, Node


class Splitter:
    def __init__(self, target_feature: Optional[str] = None):
        if target_feature:
            self.target_feature = target_feature

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

    def perform_numerical_feature_splits(
        self, df_splits: pd.DataFrame, df: pd.DataFrame, feature: str, parent_node: Node
    ):
        target_feature = self.target_feature
        values = df[feature].sort_values().unique()

        values = np.convolve(values, [0.5, 0.5], "valid")

        for value in values:
            df_left = df.loc[df[feature] < value]
            df_right = df.loc[df[feature] >= value]
            information_gain = self.get_information_gain(
                parent_node, df_left[target_feature], df_right[target_feature]
            )

            row = ["THRESHOLD", feature, value, None, information_gain]
            df_splits.loc[len(df_splits)] = row

        return df_splits

    def perform_categorical_feature_splits(
        self, df_splits: pd.DataFrame, df: pd.DataFrame, feature: str, parent_node: Node
    ):
        target_feature = self.target_feature
        sets = []
        categories = df[feature].unique()
        for i in range(len(categories) + 1):
            for subset in itertools.combinations(categories, i):
                subset = list(subset)
                sets.append(subset)

        for cat_set in sets:
            df_left = df.loc[df[feature].isin(cat_set)]
            df_right = df.loc[~df[feature].isin(cat_set)]
            information_gain = self.get_information_gain(
                parent_node, df_left[target_feature], df_right[target_feature]
            )

            row = ["IN_SET", feature, None, cat_set, information_gain]
            df_splits.loc[len(df_splits)] = row

        return df_splits

    def find_best_split(
        self,
        parent_node: Node,
        df: pd.DataFrame,
        features: List[str],
    ) -> Condition:
        df_splits = pd.DataFrame(
            columns=["type", "feature", "threshold", "in_set", "information_gain"]
        )
        for feature in features:
            is_numeric = pd.api.types.is_numeric_dtype(df[feature])
            if is_numeric:
                df_splits = self.perform_numerical_feature_splits(
                    df_splits, df, feature, parent_node
                )
            else:
                df_splits = self.perform_categorical_feature_splits(
                    df_splits, df, feature, parent_node
                )

        best_split = df_splits.loc[df_splits["information_gain"].idxmax()]
        if best_split["type"] == "THRESHOLD":
            condition = Condition(
                type="THRESHOLD",
                information_gain=best_split["information_gain"],
                feature=best_split["feature"],
                threshold=best_split["threshold"],
            )
        else:
            condition = Condition(
                type="IN_SET",
                information_gain=best_split["information_gain"],
                feature=best_split["feature"],
                in_set=best_split["in_set"],
            )

        return condition

    def train_test_split(
        self, df: pd.DataFrame, train_size: float = 0.7, random_state: int = 7
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Get number of samples
        n_samples = df.shape[0]

        # Set the seed for the random number generator
        np.random.seed(random_state)

        # Shuffle the indices
        shuffled_indices = np.random.permutation(np.arange(n_samples))

        # Determine the size of the test set
        test_size = int(n_samples * (1 - train_size))

        # Split the indices into test and train
        test_indices = shuffled_indices[:test_size]
        train_indices = shuffled_indices[test_size:]

        df_train = df.loc[train_indices]
        df_test = df.loc[test_indices]

        return df_train, df_test
