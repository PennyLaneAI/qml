from __future__ import annotations
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from ..job_distributor import SortedWorkerHandler, QMLDemo, ReturnTypes


def build_strategy_matrix_offsets(
    num_workers: int,
    sphinx_examples_dir: Path,
    sphinx_examples_execution_times_file_loc: str = None,
    glob_pattern: str = "*.py",
) -> ReturnTypes.DictSortedWorkerHandler:
    """
    Generates a JSON Dict of the following schema:

    {
        "num_workers": <int: Total number of workers jobs were distributed across>
        "workers": [
            {
                "load": <int: Total load on this worker (sum of load on all assigned tasks)>
                "tasks": [
                    {
                        "name": <str: The name of the demo (example.py)>
                        "load": <int: The millisecond representation of how long it took to execute this demo>
                    }
                ]
            }
        ]
    }

    The jobs are distributed across the workers as evenly as possible. To see the methodology of the distribution,
    please see ../../job_distributor.py. Details are there.

    This function also adds 1 to the load of all demos. This is done to handle the case where you may not know the
    load of the demos or unable to fetch that information. This would make the SortedWorkerHandler job to distribute
    the jobs evenly across all the workers, if all the demos had a load of 0, then they would all go into 1 worker
    as the SortedWorkerHandler would see them all not requiring any power to build.


    Args:
        num_workers: The total number of nodes that needs to be spawned
        sphinx_examples_dir: The directory where all the sphinx demonstrations reside
        sphinx_examples_execution_times_file_loc: The path to the JSON file
                                                  containing the name of demos to execution time
        glob_pattern: The pattern use to glob all demonstration files inside sphinx_examples_dir. Defaults to "*.py"

    Returns:
        ReturnTypes.DictSortedWorkerHandler
    """
    if sphinx_examples_execution_times_file_loc is not None:
        with open(sphinx_examples_execution_times_file_loc) as fh:
            execution_times = json.load(fh)
    else:
        execution_times = {}
    job_distribution_handler = SortedWorkerHandler(num_workers=num_workers)
    for sphinx_examples_file_name in sphinx_examples_dir.glob(glob_pattern):
        # Adding +1 to load of each demo as we want the load on all demos to be >1 in order for distribution
        # To work well
        job = QMLDemo(
            name=sphinx_examples_file_name.name,
            load=execution_times.get(sphinx_examples_file_name.name, 0) + 1,
        )
        job_distribution_handler.add_task(job)
    job_distribution_handler.assign_tasks_to_workers()

    # Drop all workers with a load of 0
    job_distribution_worker_list = [ worker for worker in job_distribution_handler.asdict()["workers"] if worker["load"] ]
    return {
        "num_workers": len(job_distribution_worker_list),
        "workers": job_distribution_worker_list
    }
