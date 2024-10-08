import pandas as pd
from impurity import gini_impurity_estimator, scaled_impurity_estimator


class TestGiniImpurityEstimator:
    def test_get_gini_impurity(self):
        ages = ["young"] * 100 + ["old"] * 300
        data = {
            "Age": ages,
        }

        df = pd.DataFrame(data)
        impurity = gini_impurity_estimator.get_gini_impurity(df["Age"])

        assert impurity == 0.375


class TestScaledGiniImpurityEstimator:
    def test_get_scaled_impurity(self):
        ages = ["young"] * 100 + ["old"] * 300
        data = {
            "Age": ages,
        }

        df = pd.DataFrame(data)
        impurity = scaled_impurity_estimator.get_scaled_impurity(df["Age"])

        assert impurity == 0.433
