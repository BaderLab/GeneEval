from geneeval.common.utils import TASKS
from geneeval.metrics import f1_micro_score
from sklearn import metrics

_METRICS = {"subcellular_localization": f1_micro_score}


class AutoMetric:
    """A factory function, which returns the correct metric for a given `task`."""

    def __new__(self, task: str) -> metrics:
        if task not in TASKS:
            raise ValueError(f"task must be one of: {', '.join(TASKS)}. Got: {task}")

        return _METRICS[task]
