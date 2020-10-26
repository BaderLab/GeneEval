from pathlib import Path
from typing import List, Optional, Set, Tuple, Union

import numpy as np
from skmultilearn.model_selection import iterative_train_test_split

CLASSIFICATION = {
    "subcellular_localization",
}
REGRESSION = set()
TASKS = CLASSIFICATION | REGRESSION

TRAIN_SIZE = 0.7
VALID_SIZE = 0.1
TEST_SIZE = 0.2

BENCHMARK_FILEPATH = Path.home() / ".geneeval" / "benchmark.json"
BENCHMARK_FILEPATH.parents[0].mkdir(parents=True, exist_ok=True)


def resolve_tasks(
    include_tasks: Optional[Union[str, List[str]]] = None,
    exclude_tasks: Optional[Union[str, List[str]]] = None,
) -> Set[str]:
    if include_tasks and exclude_tasks:
        raise ValueError("Only one of include_tasks or exclude_tasks can be provided, not both.")
    else:
        if include_tasks:
            user_tasks = include_tasks
        elif exclude_tasks:
            user_tasks = exclude_tasks
        else:
            return TASKS

        # Allow user to provide string or list of strings
        user_tasks = user_tasks if isinstance(user_tasks, List) else [user_tasks]

        # For each of the requested inclusion/exclusions, we (possibly) need to resolve task names.
        resolved_tasks = set()
        for user_task in user_tasks:
            if user_task in TASKS:
                resolved_tasks.add(user_task)
            else:
                raise ValueError(
                    f'Provided task name "{user_task}" is not a valid task name or task group.'
                )

        return resolved_tasks & TASKS if include_tasks else resolved_tasks - TASKS


def multi_label_split(
    X: np.array, y: np.array
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train, y_train, X_test, y_test = iterative_train_test_split(X, y, test_size=1 - TRAIN_SIZE)
    X_valid, y_valid, X_test, y_test = iterative_train_test_split(
        X_test, y_test, test_size=TEST_SIZE / (TEST_SIZE + VALID_SIZE)
    )
    return X_train, y_train, X_valid, y_valid, X_test, y_test
