import math
from typing import Any

import numpy as np
import numpy.typing as npt


class EntropyEstimator:
    @staticmethod
    def get_fractions(array: npt.NDArray[Any]) -> dict:
        uniques, counts = np.unique(array, return_counts=True)
        fractions = dict(zip(uniques, counts / len(array)))
        return fractions

    def get_shannon_entropy(self, dataset: npt.NDArray[Any]) -> float:
        fractions = self.get_fractions(dataset)

        if len(fractions.keys()) > 2:
            raise Exception("More thatn 2 values found in numpy array")
        values = list(fractions.values())

        if len(values) == 1:
            return 1

        p = values[0]
        q = values[1]

        entropy = -p * math.log2(p) - q * math.log2(q)

        return round(entropy, 2)


entropy_estimator = EntropyEstimator()
