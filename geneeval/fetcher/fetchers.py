import io
import re
import json
from pathlib import Path
from collections import Counter
from tempfile import TemporaryFile
from abc import ABCMeta, abstractmethod
from typing import List, Tuple, Callable, Union, Dict, Any, Optional

import requests
from requests.adapters import HTTPAdapter
from requests.exceptions import ConnectionError

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

from abc import ABCMeta, abstractmethod
from typing import List, Callable, Dict, Any
from geneeval.common.utils import multi_label_split, BENCHMARK_FILEPATH


class Fetcher:
    """All data fetchers inherit from this class."""

    def __init__(self):
        self.adapter = HTTPAdapter(max_retries=5)
        self.fetchers = []

    def fetch(self) -> dict:
        fetcher_results = [fetcher.fetch() for fetcher in self.fetchers]
        results = {key: val for result_dct in fetcher_results for key, val in result_dct.items()}
        return results

    def register(self, fetcher_instance):
        self.fetchers.append(fetcher_instance)


class UniprotFetcher(Fetcher):
    """Fetches relevant data from UniProt.

    Column names are here: https://www.uniprot.org/help/uniprotkb_column_names"""

    URL = "https://www.uniprot.org/uploadlists/"

    def __init__(self):
        self.fetch_callbacks: List[Callable] = []  # Fetches data from endpoint
        self.parse_callbacks: List[Callable] = []  # Parses fetched data
        self.process_callbacks: List[Callable] = []  # Post-processes parsed data
        super().__init__()

    def fetch(self):

        columns = self._build_columns()
        protein_id_file = self._create_file()
        protein_id_file.seek(0)

        with requests.Session() as session:
            session.mount(UniprotFetcher.URL, self.adapter)
            try:
                response = session.post(
                    UniprotFetcher.URL,
                    data={
                        "format": "tab",
                        "from": "ACC+ID",
                        "to": "ACC",
                        "columns": columns,
                    },
                    headers={"From": "duncan.forster@mail.utoronto.ca"},  # remove this later
                    files={"file": protein_id_file.read()},
                    timeout=600,
                )

                protein_id_file.close()  # `TempFile` is destroyed on close
                response = pd.read_csv(io.StringIO(response.text), sep="\t", index_col=0)
                print(response.head())

                return self._process(self._parse(response))

            except ConnectionError as connection_error:
                # TODO: Probably should handle this
                protein_id_file.close()  # `TempFile` is destroyed on close
                print(connection_error)

    def register(self, fetcher_class):
        self.fetch_callbacks.append(fetcher_class.fetch_callback)
        self.parse_callbacks.append(fetcher_class.parse_callback)
        self.process_callbacks.append(fetcher_class.process_callback)

    def _build_columns(self) -> str:
        return ",".join(["id"] + [callback() for callback in self.fetch_callbacks])

    def _create_file(self) -> TemporaryFile:
        benchmark = json.load(BENCHMARK_FILEPATH.open())
        protein_ids = benchmark["inputs"].keys()
        protein_id_file = TemporaryFile()
        for protein_id in protein_ids:
            protein_id_file.write(str.encode(f"{protein_id}\n"))
        # for i, protein_id in enumerate(protein_ids):  # Debug
        #     if i > 400:
        #         break
        #     protein_id_file.write(str.encode(f"{protein_id}\n"))
        return protein_id_file

    def _parse(self, response: pd.DataFrame) -> Dict[str, Any]:
        callback_results: List[dict] = [callback(response) for callback in self.parse_callbacks]
        parsed_response = self._merge_dicts(callback_results)
        return parsed_response

    def _process(self, parsed_response: Dict[str, Any]) -> Dict[str, Any]:
        callback_results: List[dict] = [
            callback(parsed_response) for callback in self.process_callbacks
        ]
        processed_response = self._merge_dicts(callback_results)
        return processed_response

    def _merge_dicts(self, dcts: List[dict]) -> dict:
        return {key: val for dct in dcts for key, val in dct.items()}


class TaskFetcherInterface(metaclass=ABCMeta):
    """Any subclass of this class should implement `fetch_callback` and `parse_callback`."""

    @staticmethod
    @abstractmethod
    def fetch_callback() -> Any:
        pass

    @staticmethod
    @abstractmethod
    def parse_callback() -> Any:
        pass

    @staticmethod
    @abstractmethod
    def process_callback() -> Any:
        pass


class SequenceFetcher(TaskFetcherInterface):
    """Fetches protein sequences."""

    @staticmethod
    def fetch_callback():
        return "sequence"

    @staticmethod
    def parse_callback(df: pd.DataFrame) -> Dict[str, Dict[str, List[str]]]:
        column = df["Sequence"]
        genes = df.index
        parsed = {gene: sequence for gene, sequence in zip(genes, column)}
        return {"sequence": parsed}

    @staticmethod
    def process_callback():
        pass


class LocalizationFetcher(TaskFetcherInterface):
    """Fetches the subcellular localization standard."""

    @staticmethod
    def fetch_callback() -> str:
        return "comment(SUBCELLULAR LOCATION)"

    @staticmethod
    def parse_callback(df: pd.DataFrame) -> Dict[str, Dict[str, List[str]]]:
        column = df["Subcellular location [CC]"]
        genes = df.index
        parsed = {}
        for value, gene in zip(column, genes):
            if not isinstance(value, str) or value == "":
                continue  # Skips empty or NaN values
            value = re.sub(r"^[^:]*:", "", value)  # Remove column identifier
            value = re.sub(r"\{[^}]*\}", "", value)  # Remove content in braces
            value = re.sub(r"Note=(.*)", "", value)  # Remove any freeform "Note" text
            value = re.sub(r"[.;]", ",", value)  # Replace ";" and "." characters with ","
            categories = [
                category.strip().lower()
                for category in value.split(",")
                if not category.isspace() and category != ""
            ]  # Remove whitespace around categories, lowercase, and drop whitespace and empty strings
            categories = list(set(categories))  # Drop duplicates
            parsed[gene] = categories
        return {"subcellular_localization": parsed}

    @staticmethod
    def process_callback(parsed_dct: Dict[str, Any]) -> Dict[str, Dict[str, List[str]]]:

        LOCALIZATION_COUNT_UPPER_LIMIT = 1000
        LOCALIZATION_COUNT_LOWER_LIMIT = 50

        localization_dct = parsed_dct["subcellular_localization"]
        counter = Counter()
        for localization_list in localization_dct.values():
            counter.update(localization_list)

        retained_localizations = set(
            [
                key
                for key, val in counter.items()
                if LOCALIZATION_COUNT_LOWER_LIMIT <= val <= LOCALIZATION_COUNT_UPPER_LIMIT
            ]
        )

        processed_localization_dct = {}
        for gene, localization_list in localization_dct.items():
            filtered_localizations = set(localization_list).intersection(retained_localizations)
            if len(filtered_localizations) > 0:
                processed_localization_dct[gene] = list(filtered_localizations)

        # Split into train, validation and test sets
        X = np.array(list(processed_localization_dct.keys())).reshape((-1, 1))
        y, classes = LocalizationFetcher.binarize_labels(list(processed_localization_dct.values()))
        X_train, y_train, X_valid, y_valid, X_test, y_test = multi_label_split(X, y)

        train_split: Dict[str, List[str]] = LocalizationFetcher.type_cast_split(
            X_train, y_train, classes
        )
        valid_split: Dict[str, List[str]] = LocalizationFetcher.type_cast_split(
            X_valid, y_valid, classes
        )
        test_split: Dict[str, List[str]] = LocalizationFetcher.type_cast_split(
            X_test, y_test, classes
        )

        return {
            "subcellular_localization": {
                "train": train_split,
                "valid": valid_split,
                "test": test_split,
            }
        }

    @staticmethod
    def binarize_labels(y: List[List[str]]) -> Tuple[np.array, np.array]:
        mlb = MultiLabelBinarizer()
        y = mlb.fit_transform(y)
        classes = mlb.classes_
        return y, classes

    @staticmethod
    def type_cast_split(X: np.array, y: np.array, classes: np.array) -> Dict[str, List[str]]:
        """Ensures the data split (given by `X` and `y`) are type cast to
        dictionary format compatible with final benchmark.
        """
        X = list(X.flatten())
        y = y.astype(bool)
        y = [list(classes[y[i, :]]) for i in range(len(y))]
        split_dict = {str(x): y_ for x, y_ in zip(X, y)}
        return split_dict
