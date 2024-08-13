import pandas as pd
import pytest
from splitter import Splitter
from tree import Tree


@pytest.fixture(scope="module")
def df():
    data = {
        "favourite_sport": [
            "cyber",
            "running",
            "football",
            "boxing",
            "backetball",
            "bowling",
            "snowboard",
            "running",
            "running",
            "chess",
        ],
        "age": [18, 56, 34, 39, 26, 44, 76, 99, 6, 77],
        "weight": [90.0, 74.0, 70.6, 30.0, 57.7, 70.0, 200.2, 65.5, 5.1, 66.3],
        "healthy": ["no", "yes", "yes", "no", "yes", "yes", "no", "yes", "yes", "no"],
    }

    return pd.DataFrame(data)


@pytest.fixture(scope="module")
def tree(df):
    tree = Tree(df, "healthy", "ENTROPY")
    return tree


class TestSplitter:
    def test_get_gain(self, df, tree):
        node = tree.root
        df_left = df.loc[df.weight < 82]
        df_right = df.loc[df.weight >= 82]

        information_gain = Splitter("healthy")._get_gain(
            node, df_left["healthy"], df_right["healthy"], tree.split_criterion
        )

        assert information_gain == 0.12

    def test_find_best_split_numerical(self, df, tree):
        node = tree.root

        condition = Splitter("healthy").find_best_split(
            node, df, ["age", "weight"], tree.split_criterion
        )

        assert condition.feature == "weight"
        assert condition.threshold == 82

    def test_find_best_split_categorical(self, df, tree):
        node = tree.root

        condition = Splitter("healthy").find_best_split(
            node, df, ["favourite_sport"], tree.split_criterion
        )

        assert condition.type == "IN_SET"
        assert condition.in_set == ["cyber", "boxing", "snowboard"]
        assert condition.gain == 0.26

    def test_find_best_split(self, df, tree):
        node = tree.root

        condition = Splitter("healthy").find_best_split(
            node, df, ["age", "weight", "favourite_sport"], tree.split_criterion
        )

        assert condition.in_set == ["cyber", "boxing", "snowboard"]
        assert condition.gain == 0.26

    def test_train_test_split(self, df):
        df_train, df_test = Splitter().train_test_split(df, 0.6, 1)

        assert len(df) == 10
        assert len(df_train) == 6
        assert len(df_test) == 4

    def test_find_best_split_numerical_gini(self, df):
        tree = Tree(df, "healthy", "GINI")
        node = tree.root

        condition = Splitter("healthy").find_best_split(
            node, df, ["age", "weight"], tree.split_criterion
        )

        assert condition.feature == "age"
        assert condition.threshold == 66
