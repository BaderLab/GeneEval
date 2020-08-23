import pytest
from typing import List
from hypothesis.strategies import text, lists
from hypothesis import given
from geneeval.common.utils import resolve_tasks, TASK_NAMES


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
        assert tasks == TASK_NAMES
        tasks = resolve_tasks(include_tasks="")
        assert tasks == TASK_NAMES
        tasks = resolve_tasks(include_tasks=[])
        assert tasks == TASK_NAMES

        tasks = resolve_tasks()
        assert tasks == TASK_NAMES
        tasks = resolve_tasks(exclude_tasks="")
        assert tasks == TASK_NAMES
        tasks = resolve_tasks(exclude_tasks=[])
        assert tasks == TASK_NAMES

    def test_include_tasks(self) -> None:
        task_name = list(TASK_NAMES)[0]
        partially_specified = resolve_tasks(include_tasks=task_name.split(".")[0])
        fully_specified = resolve_tasks(include_tasks=task_name)

        assert partially_specified == ["subcellular_localization.binary_classification"]
        assert fully_specified == ["subcellular_localization.binary_classification"]

    def test_exclude_tasks(self) -> None:
        task_name = list(TASK_NAMES)[0]
        partially_specified = resolve_tasks(exclude_tasks=task_name.split(".")[0])
        fully_specified = resolve_tasks(exclude_tasks=task_name)
        expected = list(TASK_NAMES - {task_name})

        assert partially_specified == expected
        assert fully_specified == expected
