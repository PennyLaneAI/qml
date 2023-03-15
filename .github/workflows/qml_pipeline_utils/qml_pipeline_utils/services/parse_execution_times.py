from __future__ import annotations
import re
from typing import Dict, TYPE_CHECKING

from ..common import calculate_files_to_retain

if TYPE_CHECKING:
    from pathlib import Path

PATTERN_FLAGS = re.MULTILINE
PATTERN_TUTORIAL_NAME = re.compile(
    r"<span\sclass=\\?\"pre\\?\">([a-zA-Z0-9_\-]+\.py)</span>", flags=PATTERN_FLAGS
)
PATTERN_TUTORIAL_TIME = re.compile(r"<td><p>(\d{2}:\d{2}.\d{3})</p></td>", flags=PATTERN_FLAGS)

MS_IN_MIN = 60000  # Number of milliseconds in a minute
MS_IN_SEC = 1000  # Number of milliseconds in a second
MS_IN_DSC = 100  # Number of milliseconds in a decisecond (tenth of a second)


def convert_execution_time_to_ms(execution_time: str) -> int:
    """
    This function takes the execution time output from the sphinx_execution_times.html file and
    converts the time from:
      MM:SS.SSS -> <int: in milliseconds>
    """
    execution_time_parts = re.split(r"[:.]", execution_time)

    execution_time_min = int(execution_time_parts[0])
    execution_time_sec = int(execution_time_parts[1])
    execution_time_dsc = int(execution_time_parts[2])

    return (
        (execution_time_min * MS_IN_MIN)
        + (execution_time_sec * MS_IN_SEC)
        + (execution_time_dsc * MS_IN_DSC)
    )


def parse_execution_times(
    num_workers: int,
    offset: int,
    sphinx_examples_dir: Path,
    sphinx_build_directory: Path,
    sphinx_gallery_dir_name: str,
    sphinx_build_type: str = "html",
    glob_pattern: str = "*.py",
) -> Dict[str, int]:
    """
    This function parses the `sg_execution_times.html` file generated by sphinx and returns the time it took
    to execute each demo in milliseconds.

    It parses this information using regex. The basic structure of the execution times html is as follows:

    <html>
        <head>
          {Omitted}
        </head>
        <body>
            {Omitted}

            <tr>
              <td>
                <p>
                  <code>
                    <span class="pre">demo_file_name.py</span>
                  </code>
                </p>
              </td>
              <td>
                <p>MM:SS.SSS</p>
              </td>
            </tr>

            {Omitted}
        </body>
    </html>

    For each row, it captures the file name and the execution duration and converts the result back to a
    dictionary:

    {
      "demo_file_name.py": {int} (MM:SS.SSS converted to millisecond)
    }

    Args:
        num_workers: The total number of workers that have been spawned
        offset: The current strategy matrix offset in the GitHub workflow
        sphinx_build_directory: The directory where sphinx outputs the built demo html files
        sphinx_gallery_dir_name: The gallery directory name inside sphinx_build_directory
                                 where sphinx puts all gallery demo html files
        sphinx_build_type: The output format of sphinx-build, Valid values are "html" and "json"
        glob_pattern: The pattern use to glob all demonstration files inside build_directory. Defaults to "*.py"
    """

    assert sphinx_build_type in {"html", "json"}, "Invalid sphinx build type"
    if sphinx_build_type == "json":
        sphinx_build_type = "fjson"

    # Hard coding the filename here as it is not something the user controls.
    # The sg_execution_times exists inside the directory sphinx puts all the built "galleries"
    sg_execution_file_location = (
        sphinx_build_directory
        / sphinx_gallery_dir_name
        / f"sg_execution_times.{sphinx_build_type}"
    )

    with sg_execution_file_location.open("r") as fh:
        sg_execution_file_content = fh.read()

    tutorial_name_matches = PATTERN_TUTORIAL_NAME.findall(sg_execution_file_content)
    tutorial_time_matches = PATTERN_TUTORIAL_TIME.findall(sg_execution_file_content)

    relevant_demos = calculate_files_to_retain(
        num_workers=num_workers,
        offset=offset,
        sphinx_examples_dir=sphinx_examples_dir,
        glob_pattern=glob_pattern,
    )

    assert len(tutorial_name_matches) == len(tutorial_time_matches), (
        f"Unable to properly parse {str(sg_execution_file_location)}. "
        f"Got {len(tutorial_name_matches)} tutorial names, "
        f"but {len(tutorial_time_matches)} exection times. "
        f"execution time content: \n\n{sg_execution_file_content}"
    )

    return {
        tutorial_name: convert_execution_time_to_ms(tutorial_time)
        for tutorial_name, tutorial_time in zip(tutorial_name_matches, tutorial_time_matches)
        if tutorial_name in relevant_demos
    }
