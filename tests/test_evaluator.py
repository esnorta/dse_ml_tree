import pandas as pd
import pytest

from evaluator import evaluator
from tree import Tree


@pytest.fixture(scope="module")
def tree(df):
    def predict_label(row):
        return tree.predict_label(row)

    df["healthy"] = (df.healthy == "yes").astype("int")

    tree = Tree(df, "healthy")
    node = tree.root

    tree.grow(node, df, ["age", "weight"])

    df["label_pred"] = df.apply(func=predict_label, axis=1)

    return tree


class TestEvaluator:
    def get_test_df(self) -> pd.DataFrame:
        data = {
            "true": [1, 0, 0, 1, 1, 0, 0, 1, 0, 0],
            "predicted": [1, 0, 0, 0, 1, 1, 1, 1, 1, 0],
        }

        return pd.DataFrame(data)

    def test_estimate_accuracy(self):
        df = self.get_test_df()
        accuracy = evaluator.estimate_accuracy(df["true"], df["predicted"])
        assert accuracy == 0.6

    def test_estimate_precision(self):
        df = self.get_test_df()
        precision = evaluator.estimate_precision(df["true"], df["predicted"])
        assert precision == 0.5

    def test_estimate_recall(self):
        df = self.get_test_df()
        recall = evaluator.estimate_recall(df["true"], df["predicted"])
        assert recall == 0.75
