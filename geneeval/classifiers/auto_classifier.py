from typing import Tuple

from geneeval.classifiers.supervised_classifiers import (
    LRClassifier,
    MLPClassifier,
    SupervisedClassifier,
)
from geneeval.common.utils import TASKS, CLASSIFICATION, REGRESSION
from geneeval.data import PreprocessedData


class AutoClassifier:
    """A factory function, which returns the correct classifiers for a given `task`."""

    def __new__(
        self, task: str, data: PreprocessedData
    ) -> Tuple[SupervisedClassifier, SupervisedClassifier]:

        if task in CLASSIFICATION:
            # return LRClassifier(data, metric=metric), MLPClassifier(data, metric=metric)
            return LRClassifier(data), MLPClassifier(data)
        elif task in REGRESSION:
            pass
        else:
            raise ValueError(f"task must be one of: {', '.join(TASKS)}. Got: {task}")
