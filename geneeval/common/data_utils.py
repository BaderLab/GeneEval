import pandas as pd
from typing import Union
from pathlib import Path


def load_embeddings(filepath: Union[str, Path]) -> pd.DataFrame:
    filepath = Path(filepath) if isinstance(filepath, str) else filepath
    delimiters = {".tsv": "\t", ".csv": ",", ".txt": r"\s+"}  # noqa

    if filepath.suffix == ".json":
        embeddings = pd.read_json(filepath).T
    elif filepath.suffix in delimiters:
        sep = delimiters[filepath.suffix]
        embeddings = pd.read_csv(filepath, sep=sep, header=None, index_col=0)
        embeddings.index.name = None
        embeddings.columns = list(range(embeddings.shape[1]))
    else:
        raise ValueError(
            "Expected file extension to be one of: .json, .tsv, .csv or .txt."
            f" Got {filepath.suffix}."
        )

    return embeddings
