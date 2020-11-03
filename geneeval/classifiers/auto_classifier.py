from geneeval.classifiers.supervised_classifiers import MLPClassifier, SupervisedClassifier
from geneeval.common.utils import CLASSIFICATION, TASKS
from geneeval.data import PreprocessedData


class AutoClassifier:
    """A factory function, which returns the correct classifiers for a given `task`."""

    def __new__(self, task: str, data: PreprocessedData) -> SupervisedClassifier:

        if task not in TASKS:
            raise ValueError(f"task must be one of: {', '.join(TASKS)}. Got: {task}")

        if task in CLASSIFICATION:
            return MLPClassifier(data)
        else:
            raise NotImplementedError("Regression is not currently supported.")
