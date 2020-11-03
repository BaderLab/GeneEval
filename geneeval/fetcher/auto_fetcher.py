from typing import Iterable

from geneeval.fetcher.fetchers import Fetcher, LocalizationFetcher, SequenceFetcher, UniprotFetcher


class AutoFetcher:
    """A factory function which returns the correct data fetcher for the given `tasks`.

    A `Fetcher` is returned which requests all the data relevant to `tasks` in a single call to its
    `fetch` method. This ensures the API endpoints are only queried once, rather than for every
    task individually.

    NOTE: It is assumed that a `benchmark.json` file already exists, with at least the gene IDs
    present. This file can be created by running the `get_protein_ids.py` file in `scripts`.
    """

    def __new__(cls, tasks: Iterable[str]) -> Fetcher:

        fetcher = Fetcher()

        uniprot_fetcher = UniprotFetcher()

        for task in tasks:

            if task.startswith("sequence"):
                uniprot_fetcher.register(SequenceFetcher)

            if task.startswith("subcellular_localization"):
                uniprot_fetcher.register(LocalizationFetcher)

        fetcher.register(uniprot_fetcher)
        return fetcher
