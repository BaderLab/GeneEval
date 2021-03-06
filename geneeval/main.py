from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import orjson
import typer
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer

from geneeval.common.data_utils import load_benchmark, load_features
from geneeval.common.utils import BENCHMARK_FILEPATH, TASKS, resolve_tasks
from geneeval.engine import Engine
from geneeval.fetcher.auto_fetcher import AutoFetcher
from geneeval.metrics.auto_metric import AutoMetric

app = typer.Typer()
evaluate_app = typer.Typer()
app.add_typer(evaluate_app, name="evaluate")


@app.command()
def build_benchmark() -> None:
    """Downloads the benchmark data. Include or exclude specific tasks with `include_tasks` and
    `exclude_tasks` respectively.
    """

    fetcher = AutoFetcher(list(TASKS.keys()))
    benchmark = fetcher.fetch()
    benchmark = {**benchmark, **orjson.loads(BENCHMARK_FILEPATH.read_bytes())}
    benchmark = orjson.dumps(benchmark, option=orjson.OPT_INDENT_2)
    BENCHMARK_FILEPATH.write_bytes(benchmark)


@app.command()
def prepare(
    filepath: Path = typer.Argument(
        ..., writable=True, help="Filepath to save prepared benchmark file."
    ),
    include_tasks: List[str] = typer.Option(
        None, help="A task name (or list of task names) to include in the prepared data."
    ),
    exclude_tasks: List[str] = typer.Option(
        None, help="A task name (or list of task names) to exclude in the prepared data."
    ),
):
    tasks = resolve_tasks(include_tasks, exclude_tasks)
    benchmark = load_benchmark()
    prepared_benchmark = {task: benchmark[task] for task in tasks}
    task_specific_genes = list(
        set(
            [
                gene
                for partition in prepared_benchmark.values()
                for genes in partition.values()
                for gene in genes.keys()
            ]
        )
    )  # Find the subset of genes specific to the `tasks` specified
    prepared_benchmark["inputs"] = {gene: benchmark["inputs"][gene] for gene in task_specific_genes}
    prepared_benchmark = orjson.dumps(prepared_benchmark, option=orjson.OPT_INDENT_2)
    filepath.write_bytes(prepared_benchmark)


@evaluate_app.command("features")
def evaluate_features(
    filepath: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="Filepath to the gene features."
    ),
    include_tasks: List[str] = typer.Option(
        None, help="A task name (or list of task names) to include in the evaluation."
    ),
    exclude_tasks: List[str] = typer.Option(
        None, help="A task name (or list of task names) to exclude in the evaluation."
    ),
) -> Dict:
    """Evaluate fixed-length feature vectors on the benchmark. Include or exclude specific tasks
    with `include_tasks` and `exclude_tasks` respectively.
    """

    features = load_features(filepath)
    engine = Engine(features, include_tasks, exclude_tasks)
    engine.run()

    # Nicely display results in the console.
    # Save the results to disk (should be a format that is easy to load)
    results = engine.results
    print(
        orjson.dumps(results, option=orjson.OPT_INDENT_2 | orjson.OPT_SERIALIZE_NUMPY).decode("utf")
    )
    return results


@evaluate_app.command("predictions")
def evaluate_predictions(
    filepath: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="Filepath to the gene label predictions."
    ),
) -> Dict:
    """Evaluate predictions on the benchmark."""

    with open(filepath, "r") as f:
        predictions = orjson.loads(f.read())
    benchmark = load_benchmark()

    # Create a recursive dictionary so that we can add keys willy-nilly.
    # https://stackoverflow.com/a/19189356/6046281
    def recursive_defaultdict():
        return defaultdict(recursive_defaultdict)

    results = recursive_defaultdict()

    for task, partitions in predictions.items():
        metric = AutoMetric(task)
        # Fit the label binarizer on the train set of the benchmark.
        multilabel = (
            isinstance(list(benchmark[task]["train"].values())[0], list)
            and len(max(list(benchmark[task]["train"].values()), key=len)) > 1
        )
        lb = MultiLabelBinarizer() if multilabel else LabelBinarizer()
        lb.fit(list(benchmark[task]["train"].values()))

        for partition, data in partitions.items():
            y_true = lb.transform(list(benchmark[task][partition].values())).astype(np.float32)
            y_pred = lb.transform(list(data.values())).astype(np.float32)
            results[task][partition][metric.__name__] = round(metric(y_true, y_pred), 2)

    print(
        orjson.dumps(results, option=orjson.OPT_INDENT_2 | orjson.OPT_SERIALIZE_NUMPY).decode("utf")
    )
    return results
