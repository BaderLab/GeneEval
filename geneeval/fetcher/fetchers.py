import io
import re
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from requests.exceptions import ConnectionError

import pandas as pd

from abc import ABCMeta, abstractmethod
from typing import List, Tuple, Callable, Union, Dict, Any


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
        self.fetch_callbacks: List[Callable] = []
        self.parse_callbacks: List[Callable] = []
        super().__init__()

    def fetch(self):

        columns = self._build_columns()

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
                    files={"file": Path("uniprot_test.txt").read_bytes()},  # TODO: figure this out}
                    timeout=60,
                )

                response = pd.read_csv(io.StringIO(response.text), sep="\t", index_col=0)

                return self._parse(response)

            except ConnectionError as connection_error:
                # TODO: Probably should handle this
                print(connection_error)

    def register(self, fetcher_class):
        self.fetch_callbacks.append(fetcher_class.fetch_callback)
        self.parse_callbacks.append(fetcher_class.parse_callback)

    def _build_columns(self) -> str:
        return ",".join(["id"] + [callback() for callback in self.fetch_callbacks])

    def _parse(self, response: pd.DataFrame) -> Dict[str, Any]:
        callback_results: List[dict] = [callback(response) for callback in self.parse_callbacks]
        response = {key: val for result_dct in callback_results for key, val in result_dct.items()}
        return response


class TaskFetcherInterface(metaclass=ABCMeta):
    """Any subclass of this class should implement `fetch_callback` and `parse_callback`.
    """

    @staticmethod
    @abstractmethod
    def fetch_callback() -> Any:
        pass

    @staticmethod
    @abstractmethod
    def parse_callback() -> Any:
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
        return {"sequence" : parsed}


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
            value = re.sub(r"^[^:]*:", "", value)      # Remove column identifier
            value = re.sub(r"\{[^}]*\}", "", value)    # Remove content in braces
            value = re.sub(r"Note=(.*)", "", value)    # Remove any freeform "Note" text
            value = re.sub(r"[.;]", ",", value)        # Replace ";" and "." characters with ","
            categories = [
                category.strip() for category in value.split(",") if not category.isspace() and category != ""
            ]  # Remove whitespace around categories and drop whitespace and empty strings
            parsed[gene] = categories
        return {"subcellular_localization": parsed}


class OtherDataFetcher:
    """A dummy data fetcher class that can be replaced."""

    def register(self, fetcher_instance):
        pass
