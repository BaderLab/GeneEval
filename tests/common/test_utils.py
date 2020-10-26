from typing import List

import pytest
from geneeval.common.utils import TASKS, resolve_tasks
from hypothesis import given
from hypothesis.strategies import lists, text


class TestUtils:
    def test_value_error_include_and_exclude(self) -> None:
        with pytest.raises(ValueError):
            _ = resolve_tasks(include_tasks=["arbitrary"], exclude_tasks=["arbitrary"])

    @given(tasks=lists(text(), min_size=1))
    def test_value_error_invalid_task_names(self, tasks: List[str]) -> None:
        with pytest.raises(ValueError):
            _ = resolve_tasks(include_tasks=tasks)
        with pytest.raises(ValueError):
            _ = resolve_tasks(exclude_tasks=tasks)

    def test_include_tasks_exclude_task_none(self) -> None:
        # We treat falsy values as unspecified, so the full list of task_names should be returned.
        tasks = resolve_tasks()
        assert tasks == TASKS
        tasks = resolve_tasks(include_tasks="")
        assert tasks == TASKS
        tasks = resolve_tasks(include_tasks=[])
        assert tasks == TASKS

    def test_include_tasks(self) -> None:
        task_name = list(TASKS)[0]
        expected = {task_name}
        actual = resolve_tasks(include_tasks=task_name)

        assert expected == actual

    def test_exclude_tasks(self) -> None:
        task_name = list(TASKS)[0]
        expected = TASKS - {task_name}
        actual = resolve_tasks(exclude_tasks=task_name)

        assert expected == actual
