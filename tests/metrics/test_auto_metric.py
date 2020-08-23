import pytest
from hypothesis.strategies import text
from hypothesis import given
from geneeval.common.utils import TASK_NAMES
from geneeval.metrics.auto_metric import AutoMetric
from sklearn import metrics


class TestAutoMetric:
    @given(task=text())
    def test_value_error_invalid_task_name(self, task: str) -> None:
        with pytest.raises(ValueError):
            AutoMetric(task)

    def test_autometric(self) -> None:
        for task in TASK_NAMES:
            metric = AutoMetric(task=task)
            if task.endswith("binary_classification"):
                assert metric == metrics.accuracy_score
