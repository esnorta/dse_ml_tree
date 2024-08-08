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
        "weight": [90, 74, 70, 30, 57, 70, 200, 65, 5, 66],
        "healthy": ["no", "yes", "yes", "no", "yes", "yes", "no", "yes", "yes", "no"],
    }

    return pd.DataFrame(data)


class TestSplitter:
    def test_get_information_gain(self, df):
        tree = Tree(df, "healthy")
        node = tree.root
        df_left = df.loc[df.weight < 82]
        df_right = df.loc[df.weight >= 82]

        information_gain = Splitter(tree).get_information_gain(
            node, df_left["healthy"], df_right["healthy"]
        )

        assert information_gain == 0.12

    def test_find_best_split(self, df):
        tree = Tree(df, "healthy")
        node = tree.root

        condition = Splitter(tree).find_best_split(node, df, ["age", "weight"])

        assert condition.feature == "weight"
        assert condition.threshold == 82
