from typing import List, Union

from geneeval.common.utils import TASK_NAMES
from .fetchers import Fetcher, UniprotFetcher, OtherDataFetcher, LocalizationFetcher, SequenceFetcher


class AutoFetcher:
    """A factory function which returns the correct data fetcher for the given `tasks`.

    A `Fetcher` is returned which requests all the data relevant to `tasks` in a single call to its
    `fetch` method. This ensures the API endpoints are only queried once, rather than for every
    task individually.
    """

    def __new__(cls, tasks: List[str]):

        fetcher = Fetcher()

        uniprot_fetcher = UniprotFetcher()
        other_data_fetcher = None

        for task in tasks:
            if task not in TASK_NAMES:
                raise ValueError(f"task must be one of: {TASK_NAMES}. Got: {task}")

        for task in tasks:

            if task.startswith("sequence"):
                uniprot_fetcher.register(SequenceFetcher)

            if task.startswith("subcellular_localization"):
                uniprot_fetcher.register(LocalizationFetcher)

        fetcher.register(uniprot_fetcher)
        return fetcher
