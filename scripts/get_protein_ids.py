import json
from pathlib import Path

import typer
import fastaparser

import requests
from requests.adapters import HTTPAdapter
from requests.exceptions import ConnectionError

from geneeval.common.utils import BENCHMARK_FILEPATH

# URL template to fetch protein ids
URL_TEMPLATE = "https://www.uniprot.org/uniprot/?query={0}+AND+{1}+AND+{2}+AND+{3}&format=fasta"
EVIDENCE_PARAM = 'annotation%3A(evidence%3A"Inferred+from+experiment+[ECO%3A0000269]")'  # Experimentally verified
TAXONOMY_PARAM = "taxonomy%3A2759"  # All eukaryotes
LENGTH_PARAM = "length%3A[50+TO+2000]"  # Proteins with sequence length between 50 and 2000
FRAGMENT_PARAM = "fragment%3Ano"  # No fragments
URL = URL_TEMPLATE.format(EVIDENCE_PARAM, TAXONOMY_PARAM, LENGTH_PARAM, FRAGMENT_PARAM)

app = typer.Typer()


@app.command()
def fetch_protein_ids() -> None:

    adapter = HTTPAdapter(max_retries=5)

    with requests.Session() as session:
        session.mount(URL, adapter)
        try:
            response = session.get(
                URL,
                timeout=600,
            )

            # Create a fasta file with filtered protein ids. This file is then uploaded
            # to http://weizhong-lab.ucsd.edu/cdhit_suite/cgi-bin/index.cgi?cmd=cd-hit to
            # perform homology mapping with default settings and 30% sequence identity cutoff.
            typer.echo(
                "Opening CD-Hit: perform homology mapping with default settings and 30%% sequence identity cutoff."
            )
            typer.launch("http://weizhong-lab.ucsd.edu/cdhit_suite/cgi-bin/index.cgi?cmd=cd-hit")
            out_path = Path("filtered_protein_ids.fasta")
            out_path.write_text(response.text)

        except ConnectionError as connection_error:
            print(connection_error)


@app.command()
def fasta_to_json(fasta_path: Path) -> None:
    benchmark_dct = {}
    with fasta_path.open() as fasta_file:
        parser = fastaparser.Reader(fasta_file, parse_method="quick")
        for sequence in parser:
            protein_id = sequence.header.split("|")[1]
            sequence_str = sequence.sequence
            benchmark_dct[protein_id] = sequence_str

    benchmark_dct = {"inputs": benchmark_dct}

    with BENCHMARK_FILEPATH.open("w") as fp:
        json.dump(benchmark_dct, fp)


if __name__ == "__main__":
    app()
