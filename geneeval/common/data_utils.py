import pandas as pd
from pathlib import Path


def load_embeddings(filepath: Path) -> pd.DataFrame:
    delimiters = {".tsv": "\t", ".csv": ", ", ".txt": "\s+"}  # noqa

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
