import pandas as pd
from geneeval import AutoClassifier
from geneeval.classifiers import BinaryClassifierLR, BinaryClassifierMLP
from geneeval.common.utils import TASK_NAMES


class TestAutoClassifier:
    def test_autoclassifier(self, preprocessed_data: pd.DataFrame) -> None:
        for task in TASK_NAMES:
            classifier = AutoClassifier(task=task, data=preprocessed_data)
            if task.endswith("binary_classification"):
                assert isinstance(classifier[0], BinaryClassifierLR)
                assert isinstance(classifier[1], BinaryClassifierMLP)
