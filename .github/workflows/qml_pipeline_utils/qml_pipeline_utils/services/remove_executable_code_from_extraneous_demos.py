from __future__ import annotations
import json
from enum import Enum, IntEnum
from typing import Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

# Used to handle an edge case where you may have a multi-line marked comment start and end on the same line
# """ hello """ <-- Example
MULTI_LINE_MIN_LEN = 6


class CommentType(Enum):
    SINGLE_LINE = "#"
    MULTI_LINE = ("'''", "r'''", '"""', 'r"""')


class FileReadState(IntEnum):
    NORMAL = 1
    MULTI_LINE_COMMENT = 2


def remove_executable_code_from_extraneous_demos(
    worker_tasks_file_loc: Path,
    sphinx_examples_dir: Path,
    dry_run: bool = False,
    verbose: bool = False,
    glob_pattern: str = "*.py",
) -> Optional[List[str]]:
    """
    Deletes executable code from all tutorials that are not relevant to the current node calling this function.

    The files that are relevant is determined as such:
      -> Glob all tutorial files (sorted order)
      -> Re-calculate the number of files this worker needs to execute `math.ceil(TOTAL_FILES / TOTAL_WORKERS)`
      -> Start the current matrix offset and add the total files from previous step
      -> All other files will have executable code removed

    Args:
        worker_tasks_file_loc: Path to JSON file that contains the tasks relevant to the current worker.
                  Expected synatx of file:
                  ```
                  [
                    {"name": "demo_name.py", "load": 123123},
                    ...
                  ]
                  ```
        sphinx_examples_dir: The directory where all the sphinx demonstrations reside
        dry_run: Indicate if the current call is a dry run, files that will be updated will be returned
        verbose: Output additional logging data to output
        glob_pattern: Pattern to glob all files in the sphinx_examples_dir

    Returns:
        Optional[List[str]]. By default, return `None`. If dry_run flag is set from cli, then returns a list of file names that will
        have executable code retained.
    """

    with worker_tasks_file_loc.open() as fh:
        worker_tasks_all = json.load(fh)

    files_to_retain = [task["name"] for task in worker_tasks_all]
    sphinx_examples_files = sphinx_examples_dir.glob(glob_pattern)

    if dry_run:
        return files_to_retain

    for file in sphinx_examples_files:
        if file.name in files_to_retain:
            continue
        remove_executable_from_doc(file, file)
        if verbose:
            print("Removing executable code from", file.name)
    return None


def remove_executable_from_doc(
    input_file_path: Path, output_file_path: Path, encoding: str = "utf-8"
) -> None:
    """
    Given a python file to read, this function will remove all executable code from the file but retain all comments.

    Sample python file:
    ```
    #!/usr/bin/env python3

    # This is a python script

    import requests

    print("Hello World")

    # Send HTTP Request
    resp = requests.get("https://example.com")
    print(resp.status_code)
    # Thanks
    ```

    This function would update that file to following:
    ```
    #!/usr/bin/env python3

    # This is a python script





    # Send HTTP Request


    # Thanks
    ```

    Args:
        input_file_path: PosixPath of python file to read in
        output_file_path: PosixPath of where to save python file with all executable code removed
        encoding: The encoding type used to read and write to the file, defaults to UTF-8

    Returns:
        None
    """
    output_lines = []
    current_read_state = FileReadState.NORMAL
    with input_file_path.open(encoding=encoding) as fh:
        for line in fh:
            original_line = line
            line = line.rstrip()
            if current_read_state == FileReadState.NORMAL:
                if line.startswith(CommentType.SINGLE_LINE.value):
                    output_lines.append(original_line)
                elif line.startswith(CommentType.MULTI_LINE.value):
                    output_lines.append(original_line)
                    if (
                        not line.endswith(CommentType.MULTI_LINE.value)
                        or len(line) < MULTI_LINE_MIN_LEN
                    ):
                        current_read_state = FileReadState.MULTI_LINE_COMMENT
                else:
                    output_lines.append("\n")
            elif current_read_state == FileReadState.MULTI_LINE_COMMENT:
                output_lines.append(original_line)
                if line.startswith(CommentType.MULTI_LINE.value) or line.endswith(
                    CommentType.MULTI_LINE.value
                ):
                    current_read_state = FileReadState.NORMAL
    with output_file_path.open("w", encoding=encoding) as fh:
        # More memory efficient than using str.join as the entire document isn't held in memory as one string
        for line in output_lines:
            fh.write(line)
    return None
