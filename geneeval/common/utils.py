from typing import Optional, List, Union
from sklearn.metrics import f1_score

CLASSIFICATION = {
    "subcellular_localization",
}
REGRESSION = set()
TASKS = CLASSIFICATION | REGRESSION
METRICS = {"subcellular_localization": f1_score}


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
