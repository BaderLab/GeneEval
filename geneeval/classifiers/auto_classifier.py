from geneeval.classifiers.supervised_classifiers import (
    BinaryClassifierLR,
    BinaryClassifierMLP,
    SupervisedClassifier,
)
from typing import Tuple
from geneeval.data import PreprocessedData
from geneeval.common.utils import TASK_NAMES


class AutoClassifier:
    """A factory function, which returns the correct classifiers for a given `task`."""

    def __new__(
        self, task: str, data: PreprocessedData
    ) -> Tuple[SupervisedClassifier, SupervisedClassifier]:
        if task not in TASK_NAMES:
            raise ValueError(f"task must be one of: {TASK_NAMES}. Got: {task}")

        if task.endswith("binary_classification"):
            return BinaryClassifierLR(data), BinaryClassifierMLP(data)
