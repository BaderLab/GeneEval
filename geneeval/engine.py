import pandas as pd
from typing import Optional, List, Union
from geneeval.common.utils import resolve_tasks
from geneeval.data import DatasetReader
from geneeval.classifiers.auto_classifier import AutoClassifier


class Engine:
    """Coordinates an evaluation of the given gene `features` on the benchmark, optionally
    including only or excluding only the `include_tasks` and `exclude_tasks` respectively
    """

    def __init__(
        self,
        embeddings: pd.DataFrame,
        include_tasks: Optional[Union[str, List[str]]] = None,
        exclude_tasks: Optional[Union[str, List[str]]] = None,
    ) -> None:
        self._embeddings = embeddings
        self._tasks = resolve_tasks(include_tasks, exclude_tasks)
        self.results = {}

    def run(self) -> None:
        for task in self._tasks:
            data = DatasetReader(self._embeddings, task)

            classifier = AutoClassifier(task, data)
            if isinstance(classifier, tuple):
                results = {"logreg": None, "mlp": None}
                for estimator, model in zip(classifier, results):
                    estimator.fit()
                    results[model] = estimator.score()
            else:
                estimator = classifier
                estimator.fit()
                results = estimator.score()
            self.results[task] = results
