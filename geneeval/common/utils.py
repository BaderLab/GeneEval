from typing import Optional, List, Union
from sklearn.metrics import f1_score
import numpy as np
from pathlib import Path
from skmultilearn.model_selection import iterative_train_test_split

CLASSIFICATION = {
    "subcellular_localization",
}
REGRESSION = set()
TASKS = CLASSIFICATION | REGRESSION
METRICS = {"subcellular_localization": f1_score}

TRAIN_SIZE = 0.7
VALID_SIZE = 0.1
TEST_SIZE = 0.2

BENCHMARK_FILEPATH = Path.home() / ".geneeval" / "benchmark.json"
BENCHMARK_FILEPATH.parents[0].mkdir(parents=True, exist_ok=True)


def resolve_tasks(
    include_tasks: Optional[Union[str, List[str]]] = None,
    exclude_tasks: Optional[Union[str, List[str]]] = None,
) -> List[str]:
    if include_tasks and exclude_tasks:
        raise ValueError("Only one of include_tasks or exclude_tasks can be provided, not both.")
    else:
        if not include_tasks and not exclude_tasks:
            return list(TASKS)
        else:
            if include_tasks:
                user_tasks = include_tasks
            elif exclude_tasks:
                user_tasks = exclude_tasks
            else:
                raise ValueError("include_tasks or exclude_tasks ")

            # Allow user to provide string or list of strings
            user_tasks = user_tasks if isinstance(user_tasks, List) else [user_tasks]

            # For each of the requested inclusion/exclusions, we (possibly) need to resolve task names.
            resolved_tasks = set()
            for user_task in user_tasks:
                if user_task in TASKS:
                    resolved_tasks.add(user_task)
                else:
                    resolved_task = {
                        resolved_task
                        for resolved_task in TASKS
                        if resolved_task.startswith(f"{user_task}.")
                    }

                    if not resolved_task:
                        raise ValueError(
                            f'Provided task name "{user_task}" is not a valid task name or task group.'
                        )

                    resolved_tasks.update(resolved_task)

            return list(resolved_tasks & TASKS if include_tasks else resolved_tasks - TASKS)


def multi_label_split(X: np.array, y: np.array):
    X_train, y_train, X_test, y_test = iterative_train_test_split(X, y, test_size=1 - TRAIN_SIZE)
    X_valid, y_valid, X_test, y_test = iterative_train_test_split(
        X_test, y_test, test_size=TEST_SIZE / (TEST_SIZE + VALID_SIZE)
    )
    return X_train, y_train, X_valid, y_valid, X_test, y_test
