from pathlib import Path
from typing import Optional, Union

import orjson
import pandas as pd

from .utils import BENCHMARK_FILEPATH


def load_features(filepath: Union[str, Path], **kwargs) -> pd.DataFrame:
    """Load delimited data from `filepath`. `**kwargs` will be passed to `pd.load_json` or
    `pd.load_csv`.
    """
    filepath = Path(filepath) if isinstance(filepath, str) else filepath
    delimiters = {".tsv": "\t", ".csv": ",", ".txt": r"\s+"}  # noqa

    if filepath.suffix == ".json":
        features = pd.read_json(filepath, **kwargs).T
    elif filepath.suffix in delimiters:
        sep = delimiters[filepath.suffix]
        features = pd.read_csv(filepath, sep=sep, header=None, index_col=0, **kwargs)
        features.index.name = None
        features.columns = list(range(features.shape[1]))
    else:
        raise ValueError(
            "Expected file extension to be one of: .json, .tsv, .csv or .txt."
            f" Got {filepath.suffix}."
        )

    return features.astype("float32", copy=False)


def load_benchmark(filepath: Optional[str] = None):
    with open(filepath or BENCHMARK_FILEPATH, "r") as f:
        return orjson.loads(f.read())
