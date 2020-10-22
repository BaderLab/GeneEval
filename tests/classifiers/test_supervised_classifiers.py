import pandas as pd
from geneeval.classifiers import MLPClassifier


class TestMLPClassifier:
    def test_can_fit_and_score(self, preprocessed_data: pd.DataFrame) -> None:
        classifier = MLPClassifier(data=preprocessed_data)
        classifier.fit()
        results = classifier.score()
        isinstance(results["valid"], float)
        isinstance(results["test"], float)
