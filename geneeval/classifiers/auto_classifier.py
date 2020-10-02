from typing import Tuple

from geneeval.classifiers.supervised_classifiers import (
    LRClassifier,
    MLPClassifier,
    SupervisedClassifier,
)
from geneeval.common.utils import TASK_NAMES
from geneeval.data import PreprocessedData


class AutoClassifier:
    """A factory function, which returns the correct classifiers for a given `task`.
    """

    def __new__(
        self, task: str, data: PreprocessedData
    ) -> Tuple[SupervisedClassifier, SupervisedClassifier]:
        if task not in TASK_NAMES:
            raise ValueError(f"task must be one of: {TASK_NAMES}. Got: {task}")

        if task.endswith("classification"):
            return LRClassifier(data), MLPClassifier(data)
