#!/usr/bin/env python3

"""
This script can be invoked by GitHub Actions workflow to generate the strategy.matrix that the workflow will use
in order to spawn worker nodes to build the QML docs.

This script determines the subset of tutorial a worker will build during the sphinx-build process.
Please refer to the doc strings of each function to see how this is done. Summary is provided below.

Summary of workflow:

1) Call `build-matrix` function
```
python3 github_job_scheduler.py build-matrix {PATH_TO_REPO} --num-workers={TOTAL WORKERS TO SPAWN}
```
This simply calculates how many tutorials are there in total, then divides that by the total number of workers
to get  the 'jobs per worker' count. It then outputs a JSON list of 'offset' which basically means where in the
tutorial list should the worker start executing jobs.

So if there are four tutorials
[
  "a.py",
  "b.py",
  "c.py",
  "d.py"
]
And two workers, then the output would be:
[0, 2]

This information is then used by the workers to determine as such:
[
  "a.py", <-- Worker 0 starts at index '0' (stops at 1)
  "b.py",
  "c.py", <-- Worker 1 starts at index '2' (stops at 3)
  "d.py"
]


2) Call 'execute-matrix' function
```
python3 github_job_scheduler.py build-matrix {PATH_TO_REPO} --num-workers={TOTAL WORKERS TO SPAWN} \
 --offset={CURRENT WORKER MATRIX OFFSET FROM 'build-matrix'}
```

There is no option to actually build 'partial gallery' in sphinx. It is also not possible to simply delete tutorials
that are not relevant as that would break table of contents, thumbnails and other index pages.

To achieve parallelisation, the workers remove executable code from tutorials that are not relevant to them.

Example:
If worker 1 from above example called 'execute-matrix', then the following would happen to "a.py" tutorial.

Contents of a.py (before execute-matrix):
```
# This is a tutorial file

print("Starting a loop")
for i in range(10):
  print(i)

# Conclusion
```

Contents of a.py (after execute-matrix):
```
# This is a tutorial file


# Conclusion
# %%%%%%RUNNABLE_CODE_REMOVED%%%%%%
```

This allows all workers to build all tutorials, but only the tutorials relevant to each worker will have appropriate
executable code preserved. This also preserves proper indexing, and table of contents.


2) Call 'clean-html' function
```
python3 github_job_scheduler.py build-matrix {PATH_TO_REPO} --num-workers={TOTAL WORKERS TO SPAWN} \
 --offset={CURRENT WORKER MATRIX OFFSET FROM 'build-matrix'}
```
Once the sphinx-build process has completed, the html files that were generated which are not relevant to the current
worker are deleted. This ensures that when all the html files are aggregated, the proper executed html files is present
from each worker.
"""

import re
import json
import math
import argparse
from enum import IntEnum
from itertools import chain
from pathlib import PosixPath
from typing import List, Optional


class State(IntEnum):
    NORMAL = 0
    BLOCK_SINGLE_QUOTE = 1
    BLOCK_DOUBLE_QUOTE = 2


def _remove_executable_from_doc(input_file_path: PosixPath, output_file_path: PosixPath) -> None:
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
    ```

    This function would update that file to following:
    ```
    #!/usr/bin/env python3

    # This is a python script





    # Send HTTP Request


    # %%%%%%RUNNABLE_CODE_REMOVED%%%%%%
    ```

    Args:
        input_file_path: PosixPath of python file to read in
        output_file_path: PosixPath of where to save python file with all executable code removed

    Returns:
        None
    """
    lines = []
    current_state = State.NORMAL
    with input_file_path.open() as fh:
        for line in fh:
            if current_state == State.NORMAL:
                if line.startswith("#"):
                    lines.append(line)
                    current_state = State.NORMAL
                elif line.startswith(('"""', 'r"""')) and line.endswith(('"""', '"""\n')) and len(line) > 6:
                    lines.append(line)
                    current_state = State.NORMAL
                elif line.startswith(('"""', '"""\n', 'r"""', 'r"""\n')):
                    lines.append(line)
                    current_state = State.BLOCK_DOUBLE_QUOTE
                elif line.startswith(("'''", "r'''")) and line.endswith(("'''", "'''\n")) and len(line) > 6:
                    lines.append(line)
                    current_state = State.NORMAL
                elif line.startswith(("'''", "'''\n", "r'''", "r'''\n")):
                    lines.append(line)
                    current_state = State.BLOCK_SINGLE_QUOTE
                else:
                    lines.append("\n")
                    current_state = State.NORMAL
            elif current_state == State.BLOCK_DOUBLE_QUOTE:
                lines.append(line)
                current_state = State.NORMAL if line.startswith(('"""', '"""\n')) or line.endswith(('"""', '"""\n')) \
                    else State.BLOCK_DOUBLE_QUOTE
            elif current_state == State.BLOCK_SINGLE_QUOTE:
                lines.append(line)
                current_state = State.NORMAL if line.startswith(("'''", "'''\n")) or line.endswith(("'''", "'''\n")) \
                    else State.BLOCK_SINGLE_QUOTE
    lines.append("\n# %%%%%%RUNNABLE_CODE_REMOVED%%%%%%")

    with output_file_path.open("w", encoding="utf-8") as fh:
        for line in lines:  # More space efficient in this case than using str.join
            fh.write(line)

    return None


def build_matrix(num_workers: int,
                 build_directory: PosixPath,
                 parser_namespace: argparse.Namespace) -> List[str]:
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
        And you have 3 workers, then the files to be excuted per worker is:
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

        Each node will have access to it's own offset from this list. It will recalculate the total files to execute
        and determine the files relevant to it.
    Args:
        num_workers: The total number of nodes that needs to be spawned
        build_directory: The directory where all the demonstrations reside
        parser_namespace: The argparse object containing input from cli

    Returns:
        List[str]. JSON list of integers indicating the offset of each node to execute tutorials.
    """
    glob_pattern = parser_namespace.glob_pattern
    file_count = len(list(build_directory.glob(glob_pattern)))
    files_per_worker = math.ceil(file_count / num_workers)

    return list(range(0, file_count, files_per_worker))


def execute_matrix(num_workers: int,
                   build_directory: PosixPath,
                   parser_namespace: argparse.Namespace) -> Optional[List[str]]:
    """
    Deletes executable to code from all tutorials that are not relevant to the current node calling this function.

    The files that are relevant is determined as such:
      -> Glob all tutorial files (sorted order)
      -> Re-calculate the number of files this worker needs to execute `math.ceil(TOTAL_FILES / TOTAL_WORKERS)`
      -> Start the the current matrix offset and add the total files from previous step
      -> All other files will have executable code removed

    Args:
        num_workers: The total number of workers that are in the workflow
        build_directory: The directory where all the demonstrations reside
        parser_namespace: The argparse object containing input from cli

    Returns:
        By default, return `None`. If dry_run flag is set from cli, then returns a list of file names that will
        have executable code retained.
    """
    glob_pattern = parser_namespace.glob_pattern
    offset = parser_namespace.offset
    dry_run = parser_namespace.dry_run
    verbose = parser_namespace.verbose

    assert offset >= 0, f"Invalid value for offset. Expected positive int; Got: {offset}"

    files = sorted(build_directory.glob(glob_pattern))
    files_per_worker = math.ceil(len(files) / num_workers)

    files_to_retain = files[offset:offset + files_per_worker]
    if dry_run:
        return list(map(lambda x: x.name, files_to_retain))

    files_to_delete_1 = files[:offset]
    files_to_delete_2 = files[offset + files_per_worker:]

    for file in chain(files_to_delete_1, files_to_delete_2):
        _remove_executable_from_doc(file, file)
        if verbose:
            print("Removing executable code from", file.name)
    return None


def clean_html(num_workers: int,
               root_directory: PosixPath,
               build_directory: PosixPath,
               parser_namespace: argparse.Namespace) -> Optional[List[str]]:
    """
    Deletes all html files after sphinx-build that are not relevant to the current node.

    The `execute_matrix` function is called to get the list of relevant files.
    And since the html and py file names are the same, it is possible to determine which html files will be deleted.

    The same goes for images. Sphinx generated images (graphs etc) have the prefix of 'sphx_glr_{file name}.png'.
    That can be used to determine which images are relevant to this node and the remaining images are deleted.

    Args:
        num_workers: The total number of workers that are in the workflow
        root_directory: The directory where the QML repo resides.
        build_directory: The directory where all the demonstrations reside
        parser_namespace: The argparse object containing input from cli

    Returns:
        By default, return `None`. If dry_run flag is set from cli, then returns a list of file names that will
        be deleted.
    """
    current_dry_run = parser_namespace.dry_run
    verbose = parser_namespace.verbose
    parser_namespace.dry_run = True
    files_to_retain = execute_matrix(num_workers, build_directory, parser_namespace)
    files_to_retain = list(map(lambda f: "".join(f.split(".")[:-1]), files_to_retain))

    image_files = (root_directory / "_build" / "html" / "_images").glob("*")
    html_files = (root_directory / "_build" / "html" / "demos").glob("*.html")

    dry_run = []
    for file in html_files:
        file_stem = file.stem

        if file_stem == "index":
            continue
        file_name = f"html/{file.name}"
        if file_stem not in files_to_retain:
            if current_dry_run:
                dry_run.append(file_name)
            else:
                file.unlink()
            if verbose:
                print("Deleted", file_name)

    image_file_possible_prefixes = [
        f"sphx_glr_{file_name}"
        for file_name in files_to_retain
    ]
    image_file_regex_pattern = f"^({'|'.join(image_file_possible_prefixes)}).*"
    image_file_regex = re.compile(image_file_regex_pattern, re.IGNORECASE)
    for file in image_files:
        file_stem = file.stem
        file_name = f"_images/{file.name}"

        if parser_namespace.preserve_non_sphinx_images and not file_stem.startswith("sphx_glr"):
            continue

        if not image_file_regex.match(file_stem):
            if current_dry_run:
                dry_run.append(file_name)
            else:
                file.unlink()
            if verbose:
                print("Deleted", file_name)

    if current_dry_run:
        return dry_run
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="QML Build Scheduler",
        description="This Python script aids in splitting the build process of the QML docs across multiple nodes"
    )
    parser.add_argument("action",
                        help="build-matrix -> Build strategy.matrix that will be used to schedule jobs dynamically. "
                             "execute-matrix -> Remove executable code from non-relevant files in current offset. "
                             "clean-html -> Delete html files built that are not relevant for current matrix offset.",
                        choices=["build-matrix", "execute-matrix", "clean-html"],
                        type=str)
    parser.add_argument("directory",
                        help="The path to the qml directory",
                        type=str)
    parser.add_argument("--examples-dir",
                        help="The directory where sphinx documents exists. Similar to Sphinx examples-dir."
                             "Default: 'demonstrations'",
                        default="demonstrations")
    parser.add_argument("--num-workers",
                        help="The total number of jobs the build should be split across",
                        type=int,
                        required=True)
    parser.add_argument("--glob-pattern",
                        help="The pattern to use to search for files in the examples-dir."
                             "Default: '*.py'",
                        type=str,
                        default="*.py")
    parser_build = parser.add_argument_group("Build Matrix")

    parser_execute = parser.add_argument_group("Execute Matrix / Clean")
    parser_execute.add_argument("--offset",
                                help="The current matrix output to retain files from",
                                type=int)
    parser_execute.add_argument("--dry-run",
                                help="Will not delete files but list what will be deleted",
                                action="store_true")
    parser_execute.add_argument("--preserve-non-sphinx-images",
                                help="Do not delete static images in the gallery _images directory",
                                action="store_true")
    parser_execute.add_argument("--verbose",
                                help="Print name of files that are being cleaned up for current offset",
                                action="store_true")

    parser_results = parser.parse_args()

    directory_qml = PosixPath(parser_results.directory)
    directory_examples = directory_qml / parser_results.examples_dir
    assert directory_examples.exists(), f"Could not find {parser_results.examples_dir!r} folder " \
                                        f"under {parser_results.directory!r}"

    worker_count = parser_results.num_workers
    assert worker_count > 0, "Total number of workers has to be greater than 1"

    if parser_results.action == "clean-html":
        output = clean_html(worker_count, directory_qml, directory_examples, parser_results)
    else:
        action_dict = {
            "build-matrix": build_matrix,
            "execute-matrix": execute_matrix
        }
        output = action_dict[parser_results.action](worker_count, directory_examples, parser_results)

    print(json.dumps(output) if output else "")
