from typing import Dict, List, Optional, Union

import numpy as np
import torch
from geneeval.data import PreprocessedData
from geneeval.metrics.auto_metric import f1_micro_score
from sklearn import metrics
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from skorch import NeuralNet
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

    def score(self) -> Dict[str, float]:
        """Wrapper around `self.estimator.score`."""
        X_valid = self.data.X_train[self.data.splits.test_fold == 0]
        y_valid = self.data.y_train[self.data.splits.test_fold == 0]

        y_valid_pred = self.estimator.best_estimator_.predict(X_valid)
        y_test_pred = self.estimator.best_estimator_.predict(self.data.X_test)

        return {
            "valid": {self.metric.__name__: self.metric(y_valid, y_valid_pred)},
            "test": {self.metric.__name__: self.metric(self.data.y_test, y_test_pred)},
        }


class MLPClassifier(SupervisedClassifier):
    """A multi-layer perceptron classifier for classification tasks."""

    param_grid = {
        "lr": [1e-1, 5e-2, 1e-2],
        "batch_size": [64, 128],
        "module__num_hidden_layers": [0, 1],
        "module__hidden_dim": [50, 100, 200],
        "module__dropout": [0.0, 0.1, 0.2],
    }

    __name__ = "mlp"

    def __init__(self, data: PreprocessedData) -> None:
        embedding_dim = data.X_train[0].shape[-1]
        num_classes = data.y_train.shape[-1]

        multi_class = num_classes > 1
        multi_label = np.sum(data.y_train, axis=-1).max() > 1

        metric = f1_micro_score if multi_class or multi_label else metrics.accuracy_score
        criterion = nn.BCEWithLogitsLoss if multi_label else nn.CrossEntropyLoss
        predict_nonlinearity = (
            lambda x: (torch.sigmoid(x) >= 0.5).float() if multi_label else "auto"
        )  # noqa: E731
        estimator = NeuralNet(
            module=MLP,
            criterion=criterion,
            optimizer=torch.optim.Adam,
            max_epochs=10,
            train_split=None,
            predict_nonlinearity=predict_nonlinearity,
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
        num_hidden_layers: int = 1,
        hidden_dim: int = 100,
        dropout: float = 0.1,
    ):
        super().__init__()

        if num_hidden_layers == 0:
            self.model = nn.Sequential(
                nn.Linear(embedding_dim, num_classes),
            )
        else:
            self.model = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.Dropout(dropout),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_classes),
            )

    def forward(self, X, **kwargs):
        return self.model(X)
