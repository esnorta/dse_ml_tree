import pandas as pd
import pytest

from entropy import EntropyEstimator


@pytest.fixture(scope="module")
def df():
    data = {"Name": ["Tom", "nick", "krish", "jack"], "Age": [18, 19, 19, 18]}

    return pd.DataFrame(data)


class TestEntropy:
    def test_get_fractions(self, df):
        fractions = EntropyEstimator().get_fractions(df["Age"])

        assert fractions[18] == 0.5
        assert fractions[19] == 0.5

    def test_get_fractions_string_values(self):
        names = ["Tom"] * 10
        data = {
            "Name": names,
            "Age": [
                "young",
                "old",
                "young",
                "old",
                "old",
                "old",
                "old",
                "old",
                "old",
                "old",
            ],
        }

        df = pd.DataFrame(data)

        fractions = EntropyEstimator().get_fractions(df["Age"])

        assert fractions["young"] == 0.2
        assert fractions["old"] == 0.8

    def test_estimate_shannon_entropy_1(self, df):
        entropy = EntropyEstimator().estimate_shannon_entropy(df["Age"])

        assert entropy == 1

    def test_estimate_shannon_entropy_075(self, df):
        names = ["Tom"] * 100
        ages = ["young"] * 25 + ["old"] * 75
        data = {
            "Name": names,
            "Age": ages,
        }

        df = pd.DataFrame(data)
        entropy = EntropyEstimator().estimate_shannon_entropy(df["Age"])

        assert entropy == 0.81
