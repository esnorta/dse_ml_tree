import itertools
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
from const import SPLIT_CRITERIA
from entropy import entropy_estimator
from impurity import gini_impurity_estimator, scaled_impurity_estimator
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

    def _get_gain(
        self,
        parent: Node,
        array_left: npt.NDArray,
        array_right: npt.NDArray,
        split_criterion: SPLIT_CRITERIA,
    ) -> float:
        gain = 0

        match split_criterion:
            case "ENTROPY":
                weighted_entropy_sum = self.get_weighted_entropy_sum(
                    [array_left, array_right]
                )
                if not weighted_entropy_sum or not parent.entropy:
                    return 0
                gain = parent.entropy - weighted_entropy_sum
            case "GINI":
                all = np.append(array_right, array_left)
                parent.gini_impurity = gini_impurity_estimator.get_gini_impurity(all)
                weighted_impurity_sum = (
                    gini_impurity_estimator.get_weighted_impurity_sum(
                        [array_left, array_right]
                    )
                )
                if not weighted_impurity_sum or not parent.gini_impurity:
                    return 0
                gain = parent.gini_impurity - weighted_impurity_sum
            case "SCALED_IMPURITY":
                all = np.append(array_right, array_left)
                parent.scaled_impurity = scaled_impurity_estimator.get_scaled_impurity(
                    all
                )
                weighted_impurity_sum = (
                    scaled_impurity_estimator.get_weighted_impurity_sum(
                        [array_left, array_right]
                    )
                )
                if not weighted_impurity_sum or not parent.gini_impurity:
                    return 0
                gain = parent.scaled_impurity - weighted_impurity_sum
        return round(gain, 2)

    def _perform_numerical_feature_splits(
        self,
        df_splits: pd.DataFrame,
        df: pd.DataFrame,
        feature: str,
        parent_node: Node,
        split_criterion: SPLIT_CRITERIA,
    ):
        target_feature = self.target_feature
        values = df[feature].sort_values().unique()

        values = np.convolve(values, [0.5, 0.5], "valid")

        for value in values:
            df_left = df.loc[df[feature] < value]
            df_right = df.loc[df[feature] >= value]
            gain = self._get_gain(
                parent_node,
                df_left[target_feature],
                df_right[target_feature],
                split_criterion,
            )

            row = ["THRESHOLD", feature, value, np.nan, gain]
            df_splits.loc[len(df_splits)] = row

        return df_splits

    def _perform_categorical_feature_splits(
        self,
        df_splits: pd.DataFrame,
        df: pd.DataFrame,
        feature: str,
        parent_node: Node,
        split_criterion: SPLIT_CRITERIA,
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
            gain = self._get_gain(
                parent_node,
                df_left[target_feature],
                df_right[target_feature],
                split_criterion,
            )

            row = ["IN_SET", feature, np.nan, cat_set, gain]
            df_splits.loc[len(df_splits)] = row

        return df_splits

    def find_best_split(
        self,
        parent_node: Node,
        df: pd.DataFrame,
        features: List[str],
        split_criterion: SPLIT_CRITERIA,
    ) -> Condition:
        df_splits = pd.DataFrame(
            columns=["type", "feature", "threshold", "in_set", "gain"]
        )
        for feature in features:
            is_numeric = pd.api.types.is_numeric_dtype(df[feature])
            if is_numeric:
                df_splits = self._perform_numerical_feature_splits(
                    df_splits, df, feature, parent_node, split_criterion
                )
            else:
                df_splits = self._perform_categorical_feature_splits(
                    df_splits, df, feature, parent_node, split_criterion
                )
        best_split = df_splits.loc[df_splits["gain"].idxmax()]
        if best_split["type"] == "THRESHOLD":
            condition = Condition(
                type="THRESHOLD",
                gain=best_split["gain"],
                feature=best_split["feature"],
                threshold=best_split["threshold"],
            )
        else:
            condition = Condition(
                type="IN_SET",
                gain=best_split["gain"],
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
