import pandas as pd


class Evaluator:
    @staticmethod
    def estimate_accuracy(
        true_labels: pd.core.series.Series, pred_labels: pd.core.series.Series
    ) -> float:
        n_examples = len(true_labels)
        n_correct_predictions = sum(true_labels == pred_labels)
        accuracy = n_correct_predictions / n_examples

        return accuracy

    @staticmethod
    def estimate_precision(
        true_labels: pd.core.series.Series, pred_labels: pd.core.series.Series
    ) -> float:
        precision = 0
        true_positives = ((true_labels == 1) & (pred_labels == 1)).sum()
        false_positives = ((true_labels == 0) & (pred_labels == 1)).sum()

        if (true_positives + false_positives) != 0:
            precision = true_positives / (true_positives + false_positives)

        return precision

    @staticmethod
    def estimate_recall(
        true_labels: pd.core.series.Series, pred_labels: pd.core.series.Series
    ) -> float:
        recall = 0
        true_positives = ((true_labels == 1) & (pred_labels == 1)).sum()
        false_positives = ((true_labels == 1) & (pred_labels == 0)).sum()

        if (true_positives + false_positives) != 0:
            recall = true_positives / (true_positives + false_positives)

        return recall


evaluator = Evaluator()
