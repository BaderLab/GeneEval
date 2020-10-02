import pandas as pd
from geneeval.classifiers import LRClassifier, MLPClassifier


class TestLRClassifier:
    def test_can_fit_and_score(self, preprocessed_data: pd.DataFrame) -> None:
        classifier = LRClassifier(data=preprocessed_data)
        classifier.fit()
        results = classifier.score()
        isinstance(results["valid"], float)
        isinstance(results["test"], float)


class TestMLPClassifier:
    def test_can_fit_and_score(self, preprocessed_data: pd.DataFrame) -> None:
        classifier = MLPClassifier(data=preprocessed_data)
        classifier.fit()
        results = classifier.score()
        isinstance(results["valid"], float)
        isinstance(results["test"], float)
