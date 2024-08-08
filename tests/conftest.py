import pandas as pd
import pytest


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
