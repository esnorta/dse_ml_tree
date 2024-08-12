import math
from typing import Any, List, Optional

import numpy.typing as npt
import pandas as pd
from utils import get_fractions


class GiniImpurityEstimator:
    @staticmethod
    def get_gini_impurity(dataset: pd.DataFrame) -> float:
        p, q = get_fractions(dataset)
        impurity = 1 - (p**2 + q**2)

        return round(impurity, 3)

    def get_weighted_impurity_sum(self, arrays: List[pd.DataFrame]) -> Optional[float]:
        weighted_impurity_sum = 0
        total_example_count = sum(len(a) for a in arrays)
        for array in arrays:
            impurity = self.get_gini_impurity(array)
            example_count = len(array)
            weighted_impurity_sum += impurity * example_count / total_example_count

        return weighted_impurity_sum


class ScaledImpurityEstimator:
    @staticmethod
    def get_scaled_impurity(dataset: pd.DataFrame) -> float:
        p, q = get_fractions(dataset)
        impurity = math.sqrt(p * q)

        return round(impurity, 3)

    def get_weighted_impurity_sum(
        self, arrays: List[pd.DataFrame] | List[npt.NDArray[Any]]
    ) -> Optional[float]:
        weighted_impurity_sum = 0
        total_example_count = sum(len(a) for a in arrays)
        for array in arrays:
            impurity = self.get_scaled_impurity(array)
            example_count = len(array)
            weighted_impurity_sum += impurity * example_count / total_example_count

        return weighted_impurity_sum


gini_impurity_estimator = GiniImpurityEstimator()
scaled_impurity_estimator = ScaledImpurityEstimator()
