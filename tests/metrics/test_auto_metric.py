import pytest
from geneeval.common.utils import TASKS
from geneeval.metrics.auto_metric import _METRICS, AutoMetric
from hypothesis import given
from hypothesis.strategies import text


class TestAutoMetric:
    @given(task=text())
    def test_value_error_invalid_task_name(self, task: str) -> None:
        with pytest.raises(ValueError):
            AutoMetric(task)

    def test_autometric(self) -> None:
        for task in TASKS:
            metric = AutoMetric(task=task)
            assert metric == _METRICS[task]
