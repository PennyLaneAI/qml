from __future__ import annotations
import re
import json
from shutil import rmtree
from pathlib import Path
from itertools import chain
from typing import Optional, List, TYPE_CHECKING

from ..common import get_sphinx_role_targets

if TYPE_CHECKING:
    from pathlib import Path


def remove_extraneous_built_html_files(
    worker_tasks_file_loc: Path,
    sphinx_build_directory: Path,
    sphinx_examples_dir: Path,
    sphinx_gallery_dir_name: str,
    preserve_non_sphinx_images: bool,
    sphinx_build_type: str = "html",
    dry_run: bool = False,
    verbose: bool = False,
) -> Optional[List[str]]:
    """
    Deletes all html files after sphinx-build that are not relevant to the current node.

    The `execute_matrix` function is called to get the list of relevant files.
    And since the html and py file names are the same, it is possible to determine which html files will be deleted.

    The same goes for images. Sphinx generated images (graphs etc.) have the prefix of 'sphx_glr_{file name}.png'.
    That can be used to determine which images are relevant to this node and the remaining images are deleted.

    Args:
        worker_tasks_file_loc: Path to JSON file that contains the tasks relevant to the current worker.
                  Expected synatx of file:
                  ```
                  [
                    {"name": "demo_name.py", "load": 123123},
                    ...
                  ]
                  ```
        sphinx_build_directory: The directory where sphinx outputs the built demo html files reside
        sphinx_examples_dir: The directory where all the sphinx demonstrations reside
        sphinx_gallery_dir_name: The same of the directory in sphinx_build_directory
        preserve_non_sphinx_images: Indicate if images that are static across all workers should be preserved
        dry_run: Return files that will be updated instead of updating them
        verbose: Additional logging output

    Returns:
        By default, return `None`. If dry_run flag is set from cli, then returns a list of file names that will
        be deleted.
    """
    with worker_tasks_file_loc.open() as fh:
        worker_tasks_all = json.load(fh)

    if sphinx_build_type == "json":
        sphinx_build_type = "fjson"

    files_to_retain_with_suffix = [task["name"] for task in worker_tasks_all]
    files_to_retain = list(map(lambda f: "".join(f.split(".")[:-1]), files_to_retain_with_suffix))

    image_files = (sphinx_build_directory / "_images").glob("*")

    # Path.rglob returns a generator that scans the globed directory as you iterate the generator.
    # This causes a scanner error as the directories are deleted as the loop iterates through the generator.
    # Converting the generator to a list ensures the all the files are scanned first prior to the loop.
    downloadable_python_files = list((sphinx_build_directory / "_downloads").rglob("*.py"))
    downloadable_notebook_files = list((sphinx_build_directory / "_downloads").rglob("*.ipynb"))

    html_files = (sphinx_build_directory / sphinx_gallery_dir_name).glob(f"*.{sphinx_build_type}")

    dry_run_files = []
    for file in html_files:
        file_stem = file.stem

        if file_stem == "index":
            continue
        file_name = f"demos/{file.name}"
        if file_stem not in files_to_retain:
            if dry_run:
                dry_run_files.append(file_name)
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

        if preserve_non_sphinx_images and not file_stem.startswith("sphx_glr"):
            continue

        if not image_file_regex.match(file_stem):
            if dry_run:
                dry_run_files.append(file_name)
            else:
                file.unlink()
            if verbose:
                print("Deleted", file_name)

    files_downloadable_artifact_targets = [
        # Get the last part (the actual file name, instead of full path to file)
        target.split("/")[-1]
        for file_to_retain in files_to_retain_with_suffix
        for target in get_sphinx_role_targets(
            Path(f"{sphinx_examples_dir}/{file_to_retain}"), "download"
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
            if dry_run:
                dry_run_files.append(file_name)
            else:
                rmtree(file_parent)
            if verbose:
                print("Deleted", file_name)

    if dry_run:
        return dry_run_files
    return None
