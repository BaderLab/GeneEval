import typer

app = typer.Typer()


def run() -> None:
    app()


@app.command()
def download() -> None:
    typer.echo("Downloading data for task...")


@app.command()
def evaluate() -> None:
    typer.echo("Evaluating data for task...")


if __name__ == "__main__":
    app()
