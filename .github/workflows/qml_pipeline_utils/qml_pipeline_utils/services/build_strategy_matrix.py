from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from ..common import calculate_files_per_worker, WorkerFileCount


def build_strategy_matrix_offsets(
    num_workers: int, sphinx_examples_dir: "Path", glob_pattern: str = "*.py"
) -> List[str]:
    """
    Generates a JSON list of "offsets" that indicate where each node should start building tutorials from.
    This function calculates how many files should be allocated per worker, then generates a list indicating the index
    each worker should start executing tutorials at.

    Example:
        If you have 15 tutorials (files are globbed using pathlib):
        [
          "0.py",  # File names in this case are their respective indexes in this list (illustrative purpose)
          "1.py",
          "2.py",
          "3.py",
          "4.py",
          "5.py",
          "6.py",
          "7.py",
          "8.py",
          "9.py",
          "10.py",
          "11.py",
          "12.py",
          "13.py",
          "14.py"
        ]
        And you have 3 workers, then the files to be executed per worker is:
        math.ceil(15 / 3) = 5

        The above list is then broken into chunks of 5
        [
          "0.py",  <-- Worker 1 start point
          "1.py",
          "2.py",
          "3.py",
          "4.py",
          "5.py",  <-- Worker 2 start point
          "6.py",
          "7.py",
          "8.py",
          "9.py",
          "10.py", <-- Worker 3 start point
          "11.py",
          "12.py",
          "13.py",
          "14.py"
        ]

        Keeping the above in mind, the output of the function for the above case would be:

        -> [0, 5, 10]
        This when fed into GitHub's strategy.matrix will tell the workflow to spawn 3 nodes.

        Each node will have access to its own offset from this list. It will recalculate the total files to execute
        and determine the files relevant to it.
    Args:
        num_workers: The total number of nodes that needs to be spawned
        sphinx_examples_dir: The directory where all the sphinx demonstrations reside
        glob_pattern: The pattern use to glob all demonstration files inside sphinx_examples_dir. Defaults to "*.py"

    Returns:
        List[str]. JSON list of integers indicating the offset of each node to execute tutorials.
    """
    file_info: WorkerFileCount = calculate_files_per_worker(num_workers, sphinx_examples_dir, glob_pattern)
    file_count = file_info.total_files
    files_per_worker = file_info.files_per_worker

    return list(range(0, file_count, files_per_worker))
