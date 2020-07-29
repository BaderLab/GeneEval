import typer
from typing import Optional, List
from pathlib import Path

app = typer.Typer()


@app.command()
def prepare(
    filepath: Path = typer.Argument(..., writable=True, help="Filepath to save the task data."),
    include_tasks: Optional[List[str]] = typer.Option(
        None, help="A list of task names to include in the evaluation."
    ),
    exclude_tasks: Optional[List[str]] = typer.Option(
        None, help="A list of task names to exclude in the evaluation."
    ),
) -> None:
    typer.echo("Downloading data for task...")


@app.command()
def evaluate(
    filepath: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="Filepath to the gene embeddings."
    ),
    include_tasks: Optional[List[str]] = typer.Option(
        None, help="A list of task names to include in the evaluation."
    ),
    exclude_tasks: Optional[List[str]] = typer.Option(
        None, help="A list of task names to exclude in the evaluation."
    ),
) -> None:

    typer.echo("Evaluating data for task...")
