import re
from typing import Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

PATTERN_FLAGS = re.MULTILINE
PATTERN_TUTORIAL_NAME = re.compile(r"<span\sclass=\"pre\">(.*\.py)</span>", flags=PATTERN_FLAGS)
PATTERN_TUTORIAL_TIME = re.compile(r"<td><p>(\d{2}:\d{2}.\d{3})</p></td>", flags=PATTERN_FLAGS)

MS_IN_MIN = 60000  # Number of milliseconds in a minute
MS_IN_SEC = 1000   # Number of milliseconds in a second
MS_IN_DSC = 100    # Number of milliseconds in a decisecond (tenth of a second)


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

    return (execution_time_min * MS_IN_MIN) + (execution_time_sec * MS_IN_SEC) + (execution_time_dsc * MS_IN_DSC)


def parse_execution_times(sphinx_build_directory: "Path", sphinx_gallery_dir_name: str) -> Dict[str, int]:
    # Hard coding the filename here as it is not something the user controls.
    # The sg_execution_times exists inside the directory sphinx puts all the built "galleries"
    sg_execution_file_location = sphinx_build_directory / sphinx_gallery_dir_name / "sg_execution_times.html"

    with sg_execution_file_location.open("r") as fh:
        sg_execution_file_content = fh.read()

    tutorial_name_matches = PATTERN_TUTORIAL_NAME.findall(sg_execution_file_content)
    tutorial_time_matches = PATTERN_TUTORIAL_TIME.findall(sg_execution_file_content)

    assert len(tutorial_name_matches) == len(tutorial_time_matches), (
        f"Unable to properly parse "
        f"{str(sg_execution_file_location)}."
        f"Got {len(tutorial_name_matches)} tutorial "
        f"names, but {len(tutorial_time_matches)} "
        f"execution time matches"
    )

    return {
        tutorial_name: convert_execution_time_to_ms(tutorial_time)
        for tutorial_name, tutorial_time in zip(tutorial_name_matches, tutorial_time_matches)
    }
