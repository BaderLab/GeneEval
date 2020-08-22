import pandas as pd
from typing import Union
from pathlib import Path
import orjson

BENCHMARK_FILEPATH = "benchmark.json"


def load_embeddings(filepath: Union[str, Path], **kwargs) -> pd.DataFrame:
    """Load delimited data from `filepath`. `**kwargs` will be passed to `pd.load_json` or
    `pd.load_csv`.
    """
    filepath = Path(filepath) if isinstance(filepath, str) else filepath
    delimiters = {".tsv": "\t", ".csv": ",", ".txt": r"\s+"}  # noqa

    if filepath.suffix == ".json":
        embeddings = pd.read_json(filepath, **kwargs).T
    elif filepath.suffix in delimiters:
        sep = delimiters[filepath.suffix]
        embeddings = pd.read_csv(filepath, sep=sep, header=None, index_col=0, **kwargs)
        embeddings.index.name = None
        embeddings.columns = list(range(embeddings.shape[1]))
    else:
        raise ValueError(
            "Expected file extension to be one of: .json, .tsv, .csv or .txt."
            f" Got {filepath.suffix}."
        )

    return embeddings.astype("float32", copy=False)


def load_benchmark():
    with open(BENCHMARK_FILEPATH, "r") as f:
        return orjson.loads(f.read())
