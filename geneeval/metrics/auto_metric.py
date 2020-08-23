from sklearn import metrics
from sklearn.metrics import accuracy_score
from geneeval.common.utils import TASK_NAMES


class AutoMetric:
    """A factory function, which returns the correct metric for a given `task`.
    """

    def __new__(self, task: str) -> metrics:
        if task not in TASK_NAMES:
            raise ValueError(f'task must be one of: {TASK_NAMES}. Got: "{task}".')

        if task.endswith("binary_classification"):
            return accuracy_score
