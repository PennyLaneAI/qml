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
