import pytest
from tree import Tree


@pytest.fixture(scope="module")
def tree(df):
    tree = Tree(df, "healthy", "ENTROPY")
    return tree


class TestTree:
    def test_grow(self, df, tree):
        node = tree.root

        tree.grow(node, df, ["age", "weight"])

        assert tree.root.condition.type == "THRESHOLD"
        assert tree.root.condition.feature == "weight"

    def test_predict(self, df, tree):
        node = tree.root
        tree.grow(node, df, ["age", "weight"])

        label = tree.predict_label(df.loc[0])

        assert label == "no"

    def test_predict_gini(self, df):
        tree = Tree(df, "healthy", "GINI")
        node = tree.root
        tree.grow(node, df, ["age", "weight"])

        label = tree.predict_label(df.loc[0])

        assert label == "yes"

    def test_max_depth(self, df):
        tree = Tree(df, "healthy", "ENTROPY", 0.0001, 2)
        node = tree.root
        tree.grow(node, df, ["age", "weight"])

        tree_2 = Tree(df, "healthy", "ENTROPY", 0.0001, 1)
        node = tree.root
        tree.grow(node, df, ["age", "weight"])

        assert tree.depth > tree_2.depth
