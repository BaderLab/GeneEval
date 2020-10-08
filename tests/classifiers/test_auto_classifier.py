import pandas as pd
import pytest
from geneeval import AutoClassifier
from geneeval.classifiers import LRClassifier, MLPClassifier
from geneeval.common.utils import CLASSIFICATION, TASKS
from hypothesis import given
from hypothesis.strategies import text


class TestAutoClassifier:
    def test_autoclassifier(self, preprocessed_data: pd.DataFrame) -> None:
        for task in TASKS:
            classifier = AutoClassifier(task=task, data=preprocessed_data)
            if task in CLASSIFICATION:
                assert isinstance(classifier[0], LRClassifier)
                assert isinstance(classifier[1], MLPClassifier)

    # Hypothesis throws a deprecation warning because the preprocessed_data fixture is reused.
    # There is no risk of test contamination here as preprocessed_data is a dataclass
    # with froze=True. I cannot scope the fixture as module or session, as pandas throws an error.
    @given(task=text())
    @pytest.mark.filterwarnings("ignore::FutureWarning")
    def test_autoclassifier_invalid_task_value_error(
        self, task: str, preprocessed_data: pd.DataFrame
    ) -> None:
        with pytest.raises(ValueError):
            _ = AutoClassifier(task=task, data=preprocessed_data)
