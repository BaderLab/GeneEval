import pytest
import pandas as pd
from pathlib import Path
import orjson
from typing import Dict
import numpy as np


@pytest.fixture
def embeddings_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gene_id_1": [0.2343, -0.1242, 0.5431, -0.3475, 0.9373],
            "gene_id_2": [-0.9323, 0.2212, -0.4331, -0.8634, 0.8373],
            "gene_id_3": [0.5633, -0.6242, 0.3723, -0.2375, -0.1673],
        }
    ).T


@pytest.fixture
def embeddings_json_filepath() -> Path:
    return Path("tests/data/embeddings.json")


@pytest.fixture
def embeddings_tsv_filepath() -> Path:
    return Path("tests/data/embeddings.tsv")


@pytest.fixture
def embeddings_csv_filepath() -> Path:
    return Path("tests/data/embeddings.csv")


@pytest.fixture
def embeddings_txt_filepath() -> Path:
    return Path("tests/data/embeddings.txt")


@pytest.fixture
def benchmark_filepath() -> Path:
    return Path("tests/data/benchmark.json")


@pytest.fixture
def benchmark(benchmark_filepath: Path) -> Dict:
    with open(benchmark_filepath, "r") as f:
        benchmark = orjson.loads(f.read())
    return benchmark


@pytest.fixture
def benchmark_embeddings_dataframe(benchmark: Dict) -> pd.DataFrame:
    ids = benchmark["inputs"].keys()
    embeddings = np.random.randn(len(ids), 5)
    return pd.DataFrame({id_: embedding for id_, embedding in zip(ids, embeddings)}).T
