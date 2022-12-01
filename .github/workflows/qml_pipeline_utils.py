#!/usr/bin/env python3

"""
This script can be invoked by GitHub Actions workflow to generate the strategy.matrix that the workflow will use
in order to spawn worker nodes to build the QML docs.

This script determines the subset of tutorial a worker will build during the sphinx-build process.
Please refer to the doc strings of each function to see how this is done. Summary is provided below.

Summary of workflow:

1) Call `build-strategy-matrix` function
```
python3 github_job_scheduler.py build-strategy-matrix {PATH_TO_REPO} --num-workers={TOTAL WORKERS TO SPAWN}
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


2) Call 'remove-executable-code' function
```
python3 github_job_scheduler.py remove-executable-code {PATH_TO_REPO} --num-workers={TOTAL WORKERS TO SPAWN} \
 --offset={CURRENT WORKER MATRIX OFFSET FROM 'build-strategy-matrix'}
```

There is no option to actually build 'partial gallery' in sphinx. It is also not possible to simply delete tutorials
that are not relevant as that would break table of contents, thumbnails and other index pages.

To achieve parallelisation, the workers remove executable code from tutorials that are not relevant to them.

Example:
If worker 1 from above example called 'remove-executable-code', then the following would happen to "a.py" tutorial.

Contents of a.py (before remove-executable-code):
```
# This is a tutorial file

print("Starting a loop")
for i in range(10):
  print(i)

# Conclusion
```

Contents of a.py (after remove-executable-code):
```
# This is a tutorial file


# Conclusion
```

This allows all workers to build all tutorials, but only the tutorials relevant to each worker will have appropriate
executable code preserved. This also preserves proper indexing, and table of contents.


3) Call 'remove-html' function
```
python3 github_job_scheduler.py remove-html {PATH_TO_REPO} --num-workers={TOTAL WORKERS TO SPAWN} \
 --offset={CURRENT WORKER MATRIX OFFSET FROM 'build-strategy-matrix'}
```
Once the sphinx-build process has completed, the html files that were generated which are not relevant to the current
worker are deleted. This ensures that when all the html files are aggregated, the proper executed html files is present
from each worker.
"""

import re
import json
import math
import argparse
from shutil import rmtree
from itertools import chain
from enum import Enum, IntEnum
from pathlib import PosixPath
from dataclasses import dataclass
from typing import List, Optional
from xml.etree import ElementTree

# Used to handle an edge case where you may have a multi-line marked comment start and end on the same line
# """ hello """ <-- Example
MULTI_LINE_MIN_LEN = 6


@dataclass
class WorkerFileCount:
    total_files: int
    files_per_worker: int


class CommentType(Enum):
    SINGLE_LINE = "#"
    MULTI_LINE = ("'''", "r'''", '"""', 'r"""')


class FileReadState(IntEnum):
    NORMAL = 1
    MULTI_LINE_COMMENT = 2


def _remove_executable_from_doc(
    input_file_path: PosixPath, output_file_path: PosixPath, encoding: str = "utf-8"
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


def _calculate_files_per_worker(
    num_workers: int, build_directory: PosixPath, glob_pattern: str = "*.py"
) -> WorkerFileCount:
    """
    Calculates how many files should be allocated per worker based on the total number of files found.

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

        So in this scenario this function would return the WorkerFileCount(total_files=15, files_per_worker=5)

    Args:
        num_workers:  The total number of workers that will be spawned.
        build_directory: The directory where all the demonstrations reside
        glob_pattern: The pattern use to glob all demonstration files inside build_directory. Defaults to "*.py"

    Returns:
        WorkerFileCount. An object containing the total number of files globbed `total_examples` and allocation
        of files to workers `files_per_worker`.
    """
    files = list(build_directory.glob(glob_pattern))
    files_count = len(files)
    return WorkerFileCount(
        total_files=files_count, files_per_worker=math.ceil(files_count / num_workers)
    )


def _calculate_files_to_retain(
    num_workers: int, offset: int, build_directory: PosixPath, glob_pattern: str = "*.py"
) -> List[str]:
    """
    Determines the exact file names that a worker needs to execute.

    Args:
        num_workers:  The total number of workers that have been spawned
        offset: The current strategy.matrix offset in the GitHub workflow
        build_directory: The directory where all the demonstrations reside
        glob_pattern: The pattern use to glob all demonstration files inside build_directory. Defaults to "*.py"

    Returns:
        List[str]. List where each element is the name of a file the current worker has to execute.
    """
    file_info = _calculate_files_per_worker(num_workers, build_directory, glob_pattern)

    files = sorted(build_directory.glob(glob_pattern))
    files_to_retain = files[offset : offset + file_info.files_per_worker]

    return list(map(lambda x: x.name, files_to_retain))


def _get_sphinx_role_targets(
    sphinx_file_location: PosixPath,
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


def build_strategy_matrix_offsets(
    num_workers: int, build_directory: PosixPath, parser_namespace: argparse.Namespace
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
        build_directory: The directory where all the demonstrations reside
        parser_namespace: The argparse object containing input from cli

    Returns:
        List[str]. JSON list of integers indicating the offset of each node to execute tutorials.
    """
    glob_pattern = parser_namespace.glob_pattern
    file_info = _calculate_files_per_worker(num_workers, build_directory, glob_pattern)
    file_count = file_info.total_files
    files_per_worker = file_info.files_per_worker

    return list(range(0, file_count, files_per_worker))


def remove_extraneous_executable_code(
    num_workers: int, build_directory: PosixPath, parser_namespace: argparse.Namespace
) -> Optional[List[str]]:
    """
    Deletes executable code from all tutorials that are not relevant to the current node calling this function.

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
        Optional[List[str]]. By default, return `None`. If dry_run flag is set from cli, then returns a list of file names that will
        have executable code retained.
    """
    glob_pattern = parser_namespace.glob_pattern
    offset = parser_namespace.offset
    dry_run = parser_namespace.dry_run
    verbose = parser_namespace.verbose

    assert offset >= 0, f"Invalid value for offset. Expected positive int; Got: {offset}"

    files_to_retain = _calculate_files_to_retain(num_workers, offset, build_directory, glob_pattern)

    if dry_run:
        return files_to_retain

    for file in build_directory.glob(glob_pattern):
        if file.name in files_to_retain:
            continue
        _remove_executable_from_doc(file, file)
        if verbose:
            print("Removing executable code from", file.name)
    return None


def remove_extraneous_html(
    num_workers: int,
    root_directory: PosixPath,
    build_directory: PosixPath,
    parser_namespace: argparse.Namespace,
) -> Optional[List[str]]:
    """
    Deletes all html files after sphinx-build that are not relevant to the current node.

    The `execute_matrix` function is called to get the list of relevant files.
    And since the html and py file names are the same, it is possible to determine which html files will be deleted.

    The same goes for images. Sphinx generated images (graphs etc) have the prefix of 'sphx_glr_{file name}.png'.
    That can be used to determine which images are relevant to this node and the remaining images are deleted.

    Args:
        num_workers: The total number of workers that are in the workflow
        root_directory: The directory where the QML repo resides
        build_directory: The directory where all the demonstrations reside
        parser_namespace: The argparse object containing input from cli

    Returns:
        By default, return `None`. If dry_run flag is set from cli, then returns a list of file names that will
        be deleted.
    """
    current_dry_run = parser_namespace.dry_run
    verbose = parser_namespace.verbose
    offset = parser_namespace.offset
    glob_pattern = parser_namespace.glob_pattern
    files_to_retain_with_suffix = _calculate_files_to_retain(
        num_workers, offset, build_directory, glob_pattern
    )
    files_to_retain = list(map(lambda f: "".join(f.split(".")[:-1]), files_to_retain_with_suffix))

    image_files = (root_directory / "_build" / "html" / "_images").glob("*")

    # Path.rglob returns a generator that scans the globbed directory as you iterate the generator.
    # This causes a scanner error as the directories are deleted as the loop iterates through the generator.
    # Converting the generator to a list ensures the all the files are scanned first prior to the loop.
    downloadable_python_files = list(
        (root_directory / "_build" / "html" / "_downloads").rglob("*.py")
    )
    downloadable_notebook_files = list(
        (root_directory / "_build" / "html" / "_downloads").rglob("*.ipynb")
    )

    html_files = (root_directory / "_build" / "html" / "demos").glob("*.html")

    dry_run = []
    for file in html_files:
        file_stem = file.stem

        if file_stem == "index":
            continue
        file_name = f"demos/{file.name}"
        if file_stem not in files_to_retain:
            if current_dry_run:
                dry_run.append(file_name)
            else:
                file.unlink()
            if verbose:
                print("Deleted", file_name)

    image_file_possible_prefixes = [f"sphx_glr_{file_name}" for file_name in files_to_retain]
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

    files_downloadable_artifact_targets = [
        # Get the last part (the actual file name, instead of full path to file)
        target.split("/")[-1]
        for file_to_retain in files_to_retain_with_suffix
        for target in _get_sphinx_role_targets(
            PosixPath(f"{build_directory}/{file_to_retain}"), "download"
        )
    ]
    files_downloadable_artifact_stem = list(
        map(lambda f: "".join(f.split(".")[:-1]), files_downloadable_artifact_targets)
    )
    for file in chain(downloadable_python_files, downloadable_notebook_files):
        file_stem = file.stem
        file_parent = file.parent
        file_name = "/".join(["_downloads", file_parent.name, file.name])

        if file_stem not in files_to_retain and file_stem not in files_downloadable_artifact_stem:
            if current_dry_run:
                dry_run.append(file_name)
            else:
                rmtree(file_parent)
            if verbose:
                print("Deleted", file_name)

    if current_dry_run:
        return dry_run
    return None


def remove_html_from_sitemap(
    root_directory: PosixPath, html_files_to_remove: List[str], parser_namespace: argparse.Namespace
):
    """
    Removes html files from the _build/html folder similar to `remove_extraneous_html` function.
    However, this function is used to remove files that are not relevant to *any node*.
    It not only removes the given html files but further updates the sitemap of the built demos
    so that SEO coverage is not affected.

    To call this function, pass the directory where the qml repo resides, and a list of files to delete.
    The path to the list of files need to be the path *after* the base url e.g: pennylane.ai/qml.

    Example:
        If the path to a file is: https://pennylane.ai/qml/demos/my_old_demo.html
        And the base url to qml is: https://pennylane.ai/qml
        Then to delete that html file, pass:
          remove_html_from_sitemap("/path/to/qml/repository",
                                   ["demos/my_old_demo.html"])
        Then the set HTML would get removed from the gallery directory AND also the sitemap.xml file.

    Args:
        root_directory: The directory where the QML repo resides
        html_files_to_remove: List of string. Each string being a file name relative to the base url.
        parser_namespace: The argparse object containing input from cli

    Returns:
        None
    """
    verbose = parser_namespace.verbose
    dry_run = parser_namespace.dry_run

    sitemap_location = root_directory / "_build" / "html" / "sitemap.xml"
    sitemap_tree = ElementTree.parse(sitemap_location)
    sitemap_root = sitemap_tree.getroot()

    default_ns_match = re.match(r"{(.*)}", sitemap_root.tag)
    default_ns = "" if not default_ns_match else default_ns_match.group(1)

    # Prefixing the default namespace here to the xml tag as it seems to work more steadily
    # across both Python 3.7 and above. Once QML starts using a higher version of Python then this
    # method can be converted to passing the namespace to findall using the `namespaces` parameter.
    xml_search_paths = {
        "url": "url" if not default_ns else f"{{{default_ns}}}url",
        "loc": "loc" if not default_ns else f"{{{default_ns}}}loc",
    }
    if default_ns and verbose:
        print(f"Detected default namespace for sitemap: '{default_ns}'")
    sitemap_urls = sitemap_root.findall(xml_search_paths["url"])
    for file_to_remove in html_files_to_remove:
        file = root_directory / "_build" / "html" / file_to_remove
        if file.exists():
            if verbose:
                print(f"Deleting file from _build/html: '{file_to_remove}'")
            if not dry_run:
                file.unlink()
        for url in sitemap_urls:
            url_location = url.find(xml_search_paths["loc"])
            loc = url_location.text
            if loc.endswith(file_to_remove):
                if verbose:
                    print(f"Deleting following url from sitemap.xml: '{loc}'")
                if not dry_run:
                    sitemap_root.remove(url)
    if default_ns:
        ElementTree.register_namespace("", default_ns)
    sitemap_tree.write(str(sitemap_location))
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="QML Build Scheduler",
        description="This Python script aids in splitting the build process of the QML docs across multiple nodes",
    )
    parser.add_argument(
        "action",
        help="build-matrix -> Build strategy.matrix that will be used to schedule jobs dynamically. "
        "execute-matrix -> Remove executable code from non-relevant files in current offset. "
        "remove-html -> Delete html files built that are not relevant for current matrix offset."
        "clean-sitemap -> Delete html files and remove them from sitemap.xml",
        choices=["build-strategy-matrix", "remove-executable-code", "remove-html", "clean-sitemap"],
        type=str,
    )
    parser.add_argument("directory", help="The path to the qml directory", type=str)
    parser.add_argument(
        "--examples-dir",
        help="The directory where sphinx documents exists. Similar to Sphinx examples-dir."
        "Default: 'demonstrations'",
        default="demonstrations",
    )
    parser.add_argument(
        "--num-workers",
        help="The total number of jobs the build should be split across",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--glob-pattern",
        help="The pattern to use to search for files in the examples-dir." "Default: '*.py'",
        type=str,
        default="*.py",
    )

    parser_execute = parser.add_argument_group("Execute Matrix / Clean")
    parser_execute.add_argument(
        "--offset", help="The current matrix output to retain files from", type=int
    )
    parser_execute.add_argument(
        "--dry-run", help="Will not delete files but list what will be deleted", action="store_true"
    )
    parser_execute.add_argument(
        "--preserve-non-sphinx-images",
        help="Do not delete static images in the gallery _images directory",
        action="store_true",
    )
    parser_execute.add_argument(
        "--verbose",
        help="Print name of files that are being cleaned up for current offset",
        action="store_true",
    )

    parser_sitemap = parser.add_argument_group("Sitemap cleanup")
    parser_sitemap.add_argument(
        "--html-files",
        help="A comma separated list of html files that needs to be deleted from build directory and sitemap.xml",
        default="",
    )

    parser_results = parser.parse_args()

    directory_qml = PosixPath(parser_results.directory)
    directory_examples = directory_qml / parser_results.examples_dir
    assert directory_examples.exists(), (
        f"Could not find {parser_results.examples_dir!r} folder "
        f"under {parser_results.directory!r}"
    )

    worker_count = parser_results.num_workers
    assert worker_count > 0, "Total number of workers has to be greater than 1"

    if parser_results.action == "remove-html":
        output = remove_extraneous_html(
            worker_count, directory_qml, directory_examples, parser_results
        )
    elif parser_results.action == "clean-sitemap":
        html_files_to_remove = list(
            map(str.strip, filter(None, parser_results.html_files.split(",")))
        )
        assert len(html_files_to_remove), (
            "Action `clean-sitemap` requires flag `--html-files` "
            "and a comma separated list of files to remove"
        )
        remove_html_from_sitemap(directory_qml, html_files_to_remove, parser_results)
        output = None
    else:
        action_dict = {
            "build-strategy-matrix": build_strategy_matrix_offsets,
            "remove-executable-code": remove_extraneous_executable_code,
        }
        output = action_dict[parser_results.action](
            worker_count, directory_examples, parser_results
        )

    if output:
        print(json.dumps(output))
