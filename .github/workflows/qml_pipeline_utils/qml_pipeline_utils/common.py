import re
import math
from dataclasses import dataclass
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class WorkerFileCount:
    total_files: int
    files_per_worker: int


def calculate_files_per_worker(
    num_workers: int, sphinx_examples_dir: "Path", glob_pattern: str = "*.py"
) -> WorkerFileCount:
    """
    Calculates how many files should be allocated per worker based on the total number of files found.

    Example:
        If you have 15 tutorials (files are globed using pathlib):
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

        So in this scenario this function would return the WorkerFileCount(total_files=15, files_per_worker=5)

    Args:
        num_workers:  The total number of workers that will be spawned.
        sphinx_examples_dir: The directory where all the sphinx demonstrations reside
        glob_pattern: The pattern use to glob all demonstration files inside sphinx_examples_dir. Defaults to "*.py"

    Returns:
        WorkerFileCount. An object containing the total number of files globed `total_examples` and allocation
        of files to workers `files_per_worker`.
    """
    files = list(sphinx_examples_dir.glob(glob_pattern))
    files_count = len(files)
    return WorkerFileCount(
        total_files=files_count, files_per_worker=math.ceil(files_count / num_workers)
    )


def calculate_files_to_retain(
    num_workers: int, offset: int, sphinx_examples_dir: "Path", glob_pattern: str = "*.py"
) -> List[str]:
    """
    Determines the exact file names that a worker needs to execute.

    Args:
        num_workers:  The total number of workers that have been spawned
        offset: The current strategy matrix offset in the GitHub workflow
        sphinx_examples_dir: The directory where all the sphinx demonstrations reside
        glob_pattern: The pattern use to glob all demonstration files inside build_directory. Defaults to "*.py"

    Returns:
        List[str]. List where each element is the name of a file the current worker has to execute.
    """
    file_info: WorkerFileCount = calculate_files_per_worker(
        num_workers=num_workers,
        sphinx_examples_dir=sphinx_examples_dir,
        glob_pattern=glob_pattern
    )

    files = sorted(sphinx_examples_dir.glob(glob_pattern))
    files_to_retain: List[Path] = files[offset:offset + file_info.files_per_worker]

    return list(map(lambda x: x.name, files_to_retain))


def get_sphinx_role_targets(
    sphinx_file_location: "Path",
    sphinx_role_name: str,
    sphinx_file_encoding: str = "utf-8",
    re_flags: int = 0,
) -> List[str]:
    """
    Given the path to a sphinx python file, this function finds the usage of a given sphinx role in the file and returns
    the targets for those roles.

    To read more on sphinx roles, refer to: https://www.sphinx-doc.org/en/master/usage/restructuredtext/roles.html

    The general syntax for a sphinx role consists of  ":role:`target`"

    This function uses regex to parse out the given role from the sphinx file. It returns a list that are targets
    in a given sphinx file for a given sphinx role.

    It behaves as the following:

    :role: `target`
               ^
            The value of this target is returned

    :role: `title <target>`
                    ^
                The value of this target is returned

    Args:
        sphinx_file_location: The path to the sphinx file that has to be read for roles. Should be a PosixPath object.
                              PosixPath("path/to/file.py")
        sphinx_role_name: A string to indicate the sphinx role to parse for. Ex: 'doc', 'download'.
        sphinx_file_encoding: The encoding to use while reading the sphinx file.
        re_flags: Additional flags to passed to re.compile. Pass flags the same way you would to re.
                  Ex: flag1 | flag2.
    """
    with sphinx_file_location.open("r", encoding=sphinx_file_encoding) as fh:
        sphinx_file_content = fh.read()

    # Matches instances of the following two patterns in a file:
    #  :{given sphinx role name}: `direct link to another asset`
    #            OR
    #  :{given sphinx role name}: `link text <link to another asset>`
    # The regex itself is quite simple, the role name is added as an f-string format.
    # The parsing of the text portion is done as:
    #  `(.+ ?<.+>|.+)`
    # So this matches `text <text>` OR `text`
    # The ?P<label> notation is a feature of regex called "named groups".
    # They make the captured data easier to use.
    # Read more on Named groups: https://docs.python.org/3/howto/regex.html#non-capturing-and-named-groups
    sphinx_role_pattern = re.compile(
        fr":{sphinx_role_name}: ?`(.+ ?<(?P<hyperlinked_target>.+)>|(?P<direct_target>.+))`",
        flags=re_flags | re.MULTILINE,
    )

    role_targets = []
    for match in sphinx_role_pattern.finditer(sphinx_file_content):
        groups = match.groupdict()

        match_target = groups.get("hyperlinked_target") or groups.get("direct_target")
        if match_target and match_target not in role_targets:
            role_targets.append(match_target)

    return role_targets
