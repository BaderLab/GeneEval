import pandas as pd
from geneeval import AutoClassifier
from geneeval.classifiers import LRClassifier, MLPClassifier
from geneeval.common.utils import TASK_NAMES


class TestAutoClassifier:
    def test_autoclassifier(self, preprocessed_data: pd.DataFrame) -> None:
        for task in TASK_NAMES:
            classifier = AutoClassifier(task=task, data=preprocessed_data)
            if task.endswith("classification"):
                assert isinstance(classifier[0], LRClassifier)
                assert isinstance(classifier[1], MLPClassifier)
