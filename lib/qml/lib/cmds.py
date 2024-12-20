import subprocess
from pathlib import Path
from collections.abc import Iterable
from typing import Literal


def poetry_export(
    python: str | Path,
    output: Path,
    *,
    format: Literal["requirements.txt", "constraints.txt"] = "requirements.txt",
    groups: Iterable[str] = (),
) -> None:
    """Executes `poetry export` with the given Python interpreter.

    Args:
        output: Path to output file
        format: Format, either 'constraints.txt' or 'requirements.txt'
        groups: If provided, only include dependencies from the groups
            listed. Otherwise, include only dependencies from the main
            group.

    Raies:
        CalledProcessError: The command does not complete successfully
    """
    cmd = [
        str(python),
        "-m",
        "poetry",
        "export",
        "--without-hashes",
        "--all-extras",
        "--format",
        format,
        "--output",
        str(output),
    ]
    for group in groups:
        cmd.extend(("--only", group))

    subprocess.run(
        cmd,
    ).check_returncode()


def pip_install(
    python: str | Path,
    *args: str | Path,
    requirements: Path | None = None,
    constraints: Path | None = None,
    quiet: bool = True,
):
    """Executes `pip install` with the given python
    interpreter and args."""
    cmd = [str(python), "-m", "pip", "install"]
    if requirements:
        cmd.extend(("--requirement", str(requirements)))
    if constraints:
        cmd.extend(("--constraint", str(constraints)))
    if quiet:
        cmd.append("--quiet")

    cmd.extend(str(arg) for arg in args)
    subprocess.run(cmd).check_returncode()
