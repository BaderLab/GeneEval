from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
from typing import Dict
import numpy as np
from sklearn.model_selection import PredefinedSplit
from dataclasses import dataclass
import pandas as pd
from geneeval.common.data_utils import load_benchmark
from geneeval.common.utils import CLASSIFICATION


@dataclass(frozen=True)
class PreprocessedData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    splits: PredefinedSplit


class DatasetReader:
    """Given a dataframe of gene features, returns a `PreprocessedData` containing everything we
    need to train and evaluate with Sklearn."""

    def __new__(self, features: pd.DataFrame, task: str) -> Dict[str, PreprocessedData]:

        benchmark = load_benchmark()
        partitions = benchmark[task]

        if task in CLASSIFICATION:
            # If the labels are a list with length > 1 it is multilabel
            data = list(partitions.values())
            multilabel: bool = isinstance(data, list) and len(max(data, key=len)) > 1

            lb = MultiLabelBinarizer() if multilabel else LabelBinarizer()

            X_train = features.loc[
                list(partitions["train"].keys()) + list(partitions["valid"].keys())
            ].values
            X_test = features.loc[list(partitions["test"].keys())].values

            # fit_transform partitions together so binarization is the same across partitions.
            binarized_labels = lb.fit_transform(
                [label for partition in partitions.values() for label in partition.values()]
            ).astype(np.float)

            y_train = binarized_labels[: X_train.shape[0]]
            y_test = binarized_labels[X_train.shape[0] :]

            if y_train.shape[-1] == 1 and y_test.shape[-1] == 1:
                y_train, y_test = y_train.squeeze(-1), y_test.squeeze(-1)

            test_fold = np.asarray(len(partitions["train"]) * [-1] + len(partitions["valid"]) * [0])
            splits = PredefinedSplit(test_fold)

        else:
            pass

        return PreprocessedData(X_train, y_train, X_test, y_test, splits)
