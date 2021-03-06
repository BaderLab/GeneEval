from pathlib import Path
from typing import Dict

import numpy as np
import orjson
import pandas as pd
import pytest
from geneeval.common import utils
from geneeval.data import PreprocessedData
from sklearn.model_selection import PredefinedSplit
from sklearn.preprocessing import MultiLabelBinarizer


@pytest.fixture
def features_json_filepath() -> Path:
    return Path("tests/data/features.json")


@pytest.fixture
def features_tsv_filepath() -> Path:
    return Path("tests/data/features.tsv")


@pytest.fixture
def features_csv_filepath() -> Path:
    return Path("tests/data/features.csv")


@pytest.fixture
def features_txt_filepath() -> Path:
    return Path("tests/data/features.txt")


@pytest.fixture
def predictions_filepath() -> Path:
    return Path("tests/data/predictions.json")


@pytest.fixture
def benchmark_filepath() -> Path:
    return Path("tests/data/benchmark.json")


@pytest.fixture
def benchmark_filepath_manager(benchmark_filepath: str) -> None:
    """Temporarily changes the benchmark filepath to the dummy benchmark for testing."""
    prev = utils.BENCHMARK_FILEPATH
    utils.BENCHMARK_FILEPATH = benchmark_filepath
    yield
    utils.BENCHMARK_FILEPATH = prev


@pytest.fixture
def features_dataframe() -> pd.DataFrame:
    return (
        pd.DataFrame(
            {
                "P32916": [0.2343, -0.1242, 0.5431, -0.3475, 0.9373],
                "Q8L5Y0": [-0.9323, 0.2212, -0.4331, -0.8634, 0.8373],
                "Q9VP48": [0.5633, -0.6242, 0.3723, -0.2375, -0.1673],
                "Q8W5R2": [0.1433, -0.3242, 0.5323, -0.9975, -0.4573],
                "Q9BVK8": [0.5621, -0.4272, 0.9743, -0.1373, -0.2173],
            }
        )
        .astype("float32", copy=False)
        .T
    )


@pytest.fixture
def preprocessed_data(features_dataframe: pd.DataFrame) -> PreprocessedData:
    kwargs = {
        "X_train": features_dataframe[:4].to_numpy(),
        "y_train": np.array(
            [[0, 0, 0, 1], [1, 0, 0, 1], [1, 0, 1, 0], [1, 1, 0, 0]], dtype=np.float32
        ),
        "X_test": features_dataframe[4:5].to_numpy(),
        "y_test": np.array([[0, 0, 0, 1]], dtype=np.float32),
        "splits": PredefinedSplit([-1, -1, -1, 0]),
        "lb": MultiLabelBinarizer(),
    }
    return PreprocessedData(**kwargs)


@pytest.fixture
def benchmark(benchmark_filepath: str) -> Dict:
    with open(benchmark_filepath, "r") as f:
        benchmark = orjson.loads(f.read())
    return benchmark
