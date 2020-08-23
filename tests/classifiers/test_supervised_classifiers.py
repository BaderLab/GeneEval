import pandas as pd
from geneeval.classifiers import BinaryClassifierLR, BinaryClassifierMLP


class TestBinaryClassifierLR:
    def test_can_fit_and_score(self, preprocessed_data: pd.DataFrame) -> None:
        classifier = BinaryClassifierLR(data=preprocessed_data)
        classifier.fit()
        results = classifier.score()
        isinstance(results["valid"], float)
        isinstance(results["test"], float)


class TestBinaryClassifierMLP:
    def test_can_fit_and_score(self, preprocessed_data: pd.DataFrame) -> None:
        classifier = BinaryClassifierMLP(data=preprocessed_data)
        classifier.fit()
        results = classifier.score()
        isinstance(results["valid"], float)
        isinstance(results["test"], float)
