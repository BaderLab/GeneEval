from geneeval.common.utils import TASKS, METRICS
from sklearn import metrics


class AutoMetric:
    """A factory function, which returns the correct metric for a given `task`."""

    def __new__(self, task: str) -> metrics:
        if task not in TASKS:
            raise ValueError(f"task must be one of: {', '.join(TASKS)}. Got: {task}")

        return METRICS[task]
