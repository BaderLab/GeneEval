from functools import partial
from typing import Dict, List, Optional, Union

import numpy as np
from geneeval.data import PreprocessedData
from sklearn import metrics
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from skorch import NeuralNetClassifier
from torch import nn


class SupervisedClassifier:
    """All classifiers inherit from this class."""

    def __init__(
        self,
        estimator,
        data: PreprocessedData,
        metric: metrics,
        param_grid: Optional[Union[Dict, List[Dict]]] = None,
    ) -> None:
        self.estimator = estimator
        self.data = data
        self.param_grid = param_grid
        self.metric = metric

        if self.param_grid is not None:
            self.estimator = GridSearchCV(
                estimator=self.estimator,
                param_grid=self.param_grid,
                scoring=make_scorer(self.metric),
                cv=self.data.splits,
                refit=True,
            )

    def fit(self) -> None:
        """Wrapper around `self.estimator.fit`."""
        self.estimator.fit(self.data.X_train, self.data.y_train)

    def score(self) -> Dict:
        """Wrapper around `self.estimator.score`."""
        X_valid = self.data.X_train[self.data.splits.test_fold == 0]
        y_valid = self.data.y_train[self.data.splits.test_fold == 0]
        return {
            "valid": {self.metric.__name__: self.estimator.score(X_valid, y_valid)},
            "test": {
                self.metric.__name__: self.estimator.score(self.data.X_test, self.data.y_test),
            },
        }


class LRClassifier(SupervisedClassifier):
    """A logistic regression classifier for classification tasks."""

    def __init__(self, data: PreprocessedData) -> None:
        multi_class: bool = data.y_train.shape[-1] > 1
        multi_label: bool = np.sum(data.y_train, axis=-1).max() > 1

        f1_micro_score = partial(metrics.f1_score, average="micro")
        f1_micro_score.__name__ = "f1_micro_score"

        metric = f1_micro_score if multi_class or multi_label else metrics.accuracy_score
        estimator = LogisticRegressionCV(cv=data.splits, refit=True)
        if multi_label:
            estimator = OneVsRestClassifier(estimator)
        super().__init__(estimator=estimator, data=data, metric=metric)


class MLPClassifier(SupervisedClassifier):
    """A multi-layer perceptron classifier for classification tasks."""

    param_grid = {
        "lr": [1e-4, 5e-5],
        "module__hidden_dim": [50, 100, 200],
        "module__dropout": [0.0, 0.1, 0.25, 0.5],
    }

    def __init__(self, data: PreprocessedData) -> None:
        embedding_dim = data.X_train[0].shape[-1]
        num_classes = data.y_train.shape[-1]
        multi_class = num_classes > 1
        multi_label = np.sum(data.y_train, axis=-1).max() > 1

        f1_micro_score = partial(metrics.f1_score, average="micro")
        f1_micro_score.__name__ = "f1_micro_score"

        metric = f1_micro_score if multi_class or multi_label else metrics.accuracy_score
        criterion = nn.BCEWithLogitsLoss if multi_label else nn.CrossEntropyLoss

        estimator = NeuralNetClassifier(
            module=MLP,
            criterion=criterion,
            train_split=None,
            module__embedding_dim=embedding_dim,
            module__num_classes=num_classes,
        )

        super().__init__(
            estimator=estimator,
            data=data,
            metric=metric,
            param_grid=MLPClassifier.param_grid,
        )


class MLP(nn.Module):
    """A simple feed-forward neural network."""

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        hidden_dim: int = 100,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, X, **kwargs):
        return self.model(X)
