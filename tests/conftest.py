import pytest
import pandas as pd
from pathlib import Path
import orjson
from typing import Dict
import numpy as np
from sklearn.model_selection import PredefinedSplit
from geneeval.data import PreprocessedData
from geneeval.common import data_utils


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
def benchmark_filepath() -> Path:
    return Path("tests/data/benchmark.json")


@pytest.fixture
def benchmark_filepath_manager(benchmark_filepath: str) -> None:
    """Temporarily changes the benchmark filepath to the dummy benchmark for testing."""
    prev = data_utils.BENCHMARK_FILEPATH
    data_utils.BENCHMARK_FILEPATH = benchmark_filepath
    yield
    data_utils.BENCHMARK_FILEPATH = prev


@pytest.fixture
def features_dataframe() -> pd.DataFrame:
    return (
        pd.DataFrame(
            {
                "Q8W5R2": [0.2343, -0.1242, 0.5431, -0.3475, 0.9373],
                "Q99732": [-0.9323, 0.2212, -0.4331, -0.8634, 0.8373],
                "P83774": [0.5633, -0.6242, 0.3723, -0.2375, -0.1673],
                "Q1ENB6": [0.1433, -0.3242, 0.5323, -0.9975, -0.4573],
                "Q9XF19": [0.5621, -0.4272, 0.9743, -0.1373, -0.2173],
            }
        )
        .astype("float32", copy=False)
        .T
    )


@pytest.fixture
def preprocessed_data(features_dataframe: pd.DataFrame) -> PreprocessedData:
    kwargs = {
        "X_train": features_dataframe[:4].to_numpy(),
        "y_train": np.array([0, 0, 1, 1]),
        "X_test": features_dataframe[4:5].to_numpy(),
        "y_test": np.array([1]),
        "splits": PredefinedSplit([-1, -1, -1, 0]),
    }
    return PreprocessedData(**kwargs)


@pytest.fixture
def benchmark(benchmark_filepath: str) -> Dict:
    with open(benchmark_filepath, "r") as f:
        benchmark = orjson.loads(f.read())
    return benchmark
