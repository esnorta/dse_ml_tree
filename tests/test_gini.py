import pandas as pd
from gini import gini_impurity_estimator


class TestGiniImpurityEstimator:
    def test_get_gini_impurity(self):
        ages = ["young"] * 100 + ["old"] * 300
        data = {
            "Age": ages,
        }

        df = pd.DataFrame(data)
        impurity = gini_impurity_estimator._get_gini_impurity(df["Age"])

        assert impurity == 0.375
