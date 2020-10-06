import pandas as pd
from geneeval import AutoClassifier
from geneeval.classifiers import LRClassifier, MLPClassifier
from geneeval.common.utils import TASKS, CLASSIFICATION


class TestAutoClassifier:
    def test_autoclassifier(self, preprocessed_data: pd.DataFrame) -> None:
        for task in TASKS:
            classifier = AutoClassifier(task=task, data=preprocessed_data)
            if task in CLASSIFICATION:
                assert isinstance(classifier[0], LRClassifier)
                assert isinstance(classifier[1], MLPClassifier)
