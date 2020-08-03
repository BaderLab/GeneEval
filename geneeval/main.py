import typer
from typing import List, Union
from pathlib import Path
from geneeval import Engine
from geneeval.common.data_utils import load_embeddings

app = typer.Typer()


@app.command()
def prepare(
    filepath: Path = typer.Argument(..., writable=True, help="Filepath to save the task data."),
    include_tasks: Union[str, List[str]] = typer.Option(
        None, help="A task name (or list of task names) to include in the prepared data."
    ),
    exclude_tasks: Union[str, List[str]] = typer.Option(
        None, help="A task name (or list of task names) to exclude in the prepared data."
    ),
) -> None:
    typer.echo("Downloading data for task...")


@app.command()
def evaluate(
    filepath: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="Filepath to the gene embeddings."
    ),
    include_tasks: Union[str, List[str]] = typer.Option(
        None, help="A task name (or list of task names) to include in the evaluation."
    ),
    exclude_tasks: Union[str, List[str]] = typer.Option(
        None, help="A task name (or list of task names) to exclude in the evaluation."
    ),
) -> None:

    embeddings = load_embeddings(filepath)
    engine = Engine(embeddings, include_tasks, exclude_tasks)
    engine.run()

    # Nicely display results in the console.
    # Save the results to disk (should be a format that is easy to load)
    results = engine.results
    print(results)

    typer.echo("Evaluating data for task...")
