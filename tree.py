import pandas as pd

from entropy import entropy_estimator
from models import Node


class Tree:
    def __init__(self, df: pd.DataFrame, target_feature: str):
        self.root = Node(
            entropy_estimator.get_shannon_entropy(df[target_feature]), len(df)
        )
        self.target_feature = target_feature
