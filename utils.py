from typing import Tuple

import numpy as np
import pandas as pd


def get_fractions(array: pd.DataFrame) -> Tuple[float, float]:
    uniques, counts = np.unique(array, return_counts=True)
    fractions = dict(zip(uniques, counts / len(array)))

    if len(fractions.keys()) > 2:
        raise Exception("More that 2 values found in numpy array")
    values = list(fractions.values())

    if len(values) <= 1:
        return 0, 0
    p = values[0]
    q = values[1]

    return p, q
