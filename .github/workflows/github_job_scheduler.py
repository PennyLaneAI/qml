#!/usr/bin/env python3

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
    glob_pattern = parser_namespace.glob_pattern
    file_count = len(list(build_directory.glob(glob_pattern)))
    files_per_worker = math.ceil(file_count / num_workers)

    return list(range(1, file_count, files_per_worker))


def execute_matrix(num_workers: int,
                   build_directory: PosixPath,
                   parser_namespace: argparse.Namespace) -> Optional[List[str]]:
    glob_pattern = parser_namespace.glob_pattern
    offset = parser_namespace.offset
    dry_run = parser_namespace.dry_run
    verbose = parser_namespace.verbose

    assert offset and offset > 0, f"Invalid value for offset. Expected int greater than 0; Got: {offset}"

    files = sorted(build_directory.glob(glob_pattern))
    files_per_worker = math.ceil(len(files) / num_workers)

    files_to_retain = files[offset - 1:offset + files_per_worker - 1]
    if dry_run:
        return list(map(lambda x: x.name, files_to_retain))

    files_to_delete_1 = files[:offset - 1]
    files_to_delete_2 = files[offset + files_per_worker - 1:]

    for file in chain(files_to_delete_1, files_to_delete_2):
        _remove_executable_from_doc(file, file)
        if verbose:
            print("Removing executable code from", file.name)
    return None


def clean_html(num_workers: int,
               root_directory: PosixPath,
               build_directory: PosixPath,
               parser_namespace: argparse.Namespace) -> Optional[List[str]]:
    current_dry_run = parser_namespace.dry_run
    verbose = parser_namespace.verbose
    parser_namespace.dry_run = True
    files_to_retain = execute_matrix(num_workers, build_directory, parser_namespace)
    files_to_retain = list(map(lambda f: "".join(f.split(".")[:-1]), files_to_retain))

    html_files = (root_directory / "_build" / "html" / "demos").glob("*.html")

    dry_run = []
    for file in html_files:
        file_stem = file.stem

        if file_stem == "index":
            continue

        if file_stem not in files_to_retain:
            if current_dry_run:
                dry_run.append(file.name)
            else:
                file.unlink()
            if verbose:
                print("Deleted", file.name)

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
                             "execute-matrix -> Remove executable code from non-relevant files in current matrix offset. "
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
        output = {
            "build-matrix": build_matrix,
            "execute-matrix": execute_matrix
        }[parser_results.action](worker_count, directory_examples, parser_results)

    print(json.dumps(output) if output else "")
