from typing import Dict, Union, List, Optional
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GridSearchCV
from geneeval.data import PreprocessedData
from torch import nn
from sklearn import metrics
import numpy as np
from skorch import NeuralNetClassifier
from sklearn.metrics import make_scorer


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

    def fit(self):
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


class BinaryClassifierLR(SupervisedClassifier):
    """A logistic regression classifier for binary classification tasks.
    """

    metric = metrics.accuracy_score

    def __init__(self, data: PreprocessedData) -> None:
        estimator = LogisticRegressionCV(cv=data.splits, refit=True)
        super().__init__(estimator=estimator, data=data, metric=BinaryClassifierLR.metric)


class BinaryClassifierMLP(SupervisedClassifier):
    metric = metrics.accuracy_score
    param_grid = {
        "lr": [1e-4, 5e-5],
        "module__hidden_dim": [50, 100, 200],
        "module__dropout": [0.0, 0.1, 0.25, 0.5],
    }

    def __init__(self, data: PreprocessedData) -> None:
        embedding_dim = data.X_train[0].shape[-1]
        multi_label = np.sum(data.y_train, axis=-1).max() > 1

        estimator = NeuralNetClassifier(
            module=MLP,
            train_split=None,
            module__embedding_dim=embedding_dim,
            module__num_classes=2,
            module__multi_label=multi_label,
        )

        super().__init__(
            estimator=estimator,
            data=data,
            metric=BinaryClassifierMLP.metric,
            param_grid=BinaryClassifierMLP.param_grid,
        )


class MLP(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        hidden_dim: int = 100,
        dropout: float = 0.1,
        multi_label: bool = False,
    ):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
            nn.Softmax() if num_classes > 2 and not multi_label else nn.Sigmoid(),
        )

    def forward(self, X, **kwargs):
        return self.model(X)
