import pytest
import pandas as pd


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
def embeddings_json_filepath() -> str:
    return "tests/data/embeddings.json"


@pytest.fixture
def embeddings_tsv_filepath() -> str:
    return "tests/data/embeddings.tsv"


@pytest.fixture
def embeddings_csv_filepath() -> str:
    return "tests/data/embeddings.csv"


@pytest.fixture
def embeddings_txt_filepath() -> str:
    return "tests/data/embeddings.txt"
