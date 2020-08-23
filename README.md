![build](https://github.com/BaderLab/GeneEval/workflows/build/badge.svg)
[![codecov](https://codecov.io/gh/BaderLab/GeneEval/branch/master/graph/badge.svg)](https://codecov.io/gh/BaderLab/GeneEval)
![GitHub](https://img.shields.io/github/license/JohnGiorgi/allennlp-multi-label-classification?color=blue)

# GeneEval

A Python library for benchmarking gene function prediction.

## Installation

Latest PyPI release

```bash
pip install geneeval
```

From source

```bash
# Install poetry for your system: https://python-poetry.org/docs/#installation
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python

# Clone and move into the repo
git clone https://github.com/BaderLab/GeneEval.git
cd GeneEval

# Install the package with poetry
poetry install
```

If you plan on evaluating fixed-length feature vectors (see [Usage](#usage)), please install with `pip install "geneeval[features]"` (or `poetry install -E "features"` if installing from source).

## Usage

First, download the benchmark with the `prepare` command

```bash
geneeval prepare "./benchmark.json"
```

There are two ways to run the evaluation, depending on your method.

### Methods that produce fixed-length feature vectors

If your method produces a fixed-length feature vector for each gene ID in the benchmark, collect these in a comma-separated file, e.g.

```
Q8W5R2, 0.2343, -0.1242, 0.5431, -0.3475, 0.9373
Q99732, -0.9323, 0.2212, -0.4331, -0.8634, 0.8373
P83774, 0.5633, -0.6242, 0.3723, -0.2375, -0.1673
Q1ENB6, 0.1433, -0.3242, 0.5323, -0.9975, -0.4573
Q9XF19, 0.5621, -0.4272, 0.9743, -0.1373, -0.2173
```

> You can prepare a `.csv`, `.tsv`, `.txt` (separated by spaces) or a `.json` file (where the vectors are keyed by gene IDs). We will correctly parse the file parsed on its file extension.

and then call the `evaluate features` command

```bash
geneeval evaluate features "./features.csv"
```

These features will be used as input to simple classifiers, which will be evaluated with a grid search over the benchmark tasks.

### Methods that do not produce fixed-length feature vectors

For all other methods, you simply need to produce predictions for each task in the benchmark that you wish to evaluate on. Predictions should be collected in a `.json` file keyed by task name, data partition, and gene ID, e.g.

```json
{
    "subcellular_localization": {
      "binary_classification": {
        "train": {
          "Q8W5R2": "M",
          "Q99732": "M",
          "P83774": "S"
        },
        "valid": {
          "Q1ENB6": "S"
        },
        "test": {
          "Q9XF19": "S"
        }
      }
    }
}
```

> We make no assumptions about how these predictions are obtained.

then, the `evaluate predictions` command can be used to obtain a score on the tasks

```bash
geneeval predictions "./predictions.json"
```