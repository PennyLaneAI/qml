from typing import List, TYPE_CHECKING

from ..common import calculate_files_to_retain

if TYPE_CHECKING:
    from pathlib import Path


def show_worker_files(
    num_workers: int, offset: int, sphinx_examples_dir: "Path", glob_pattern: str = "*.py"
) -> List[str]:
    """
    Return a List of string, containing filenames of demos that the current worker will execute.

    This is a helper function that echoes back the value from common.calculate_files_to_retain;
    but since only modules in services.* are exposed to the cli, this function was created.

    Args:
        num_workers:  The total number of workers that have been spawned
        offset: The current strategy matrix offset in the GitHub workflow
        sphinx_examples_dir: The directory where all the sphinx demonstrations reside
        glob_pattern: The pattern use to glob all demonstration files inside build_directory. Defaults to "*.py"

    Returns:
        List[str]. List where each element is the name of a file the current worker has to execute.
    """

    return calculate_files_to_retain(
        num_workers=num_workers,
        offset=offset,
        sphinx_examples_dir=sphinx_examples_dir,
        glob_pattern=glob_pattern,
    )
