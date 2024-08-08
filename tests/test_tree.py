from tree import Tree


class TestTree:
    def test_grow(self, df):
        tree = Tree(df, "healthy")
        node = tree.root

        tree.grow(node, df, ["age", "weight"])

        assert tree.root.condition.type == "THRESHOLD"
        assert tree.root.condition.feature == "weight"

    def test_predict(self, df):
        tree = Tree(df, "healthy")
        node = tree.root
        tree.grow(node, df, ["age", "weight"])

        label = tree.predict_label(df.loc[0])

        assert label == "yes"
