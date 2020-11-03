from typing import Dict, List, Optional, Union

import pandas as pd

from geneeval.classifiers import SupervisedClassifier
from geneeval.classifiers.auto_classifier import AutoClassifier
from geneeval.common.utils import resolve_tasks
from geneeval.data import DatasetReader, PreprocessedData


class Engine:
    """Coordinates an evaluation of the given gene `features` on the benchmark, optionally
    including only or excluding only the `include_tasks` and `exclude_tasks` respectively
    """

    def __init__(
        self,
        features: pd.DataFrame,
        include_tasks: Optional[Union[str, List[str]]] = None,
        exclude_tasks: Optional[Union[str, List[str]]] = None,
    ) -> None:
        self._features = features
        self._tasks = resolve_tasks(include_tasks, exclude_tasks)
        self.results: Dict[str, Dict[str, Dict[str, float]]] = {}

    def run(self) -> None:
        for task in self._tasks:
            data: PreprocessedData = DatasetReader(self._features, task)

            classifier: SupervisedClassifier = AutoClassifier(task, data)
            estimator = classifier
            estimator.fit()
            results = estimator.score()
            self.results[task] = results
