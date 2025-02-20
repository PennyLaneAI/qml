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
        python: Path to Python executable
        output: Path to output file
        format: Format, either 'constraints.txt' or 'requirements.txt'
        groups: If provided, only include dependencies from the groups
            listed. Otherwise, include only dependencies from the main
            group.

    Raises:
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
    use_uv: bool = True,
) -> None:
    """Executes `pip install` with the given python
    interpreter and args.

    Args:
        python: Path to python executable
        args: Command line args passed to pip install
        requirements: Path to a requirements file
        constraints: Path to a constriants file
        quiet: Whether to suppress output to stdout

    Raises:
        CalledProcessError: The command does not complete successfully
    """
    cmd = [str(python), "-m"]
    if use_uv:
        cmd.extend(
            [
                "uv",
                "pip",
                "install",
                "--index-strategy",
                "unsafe-best-match",
            ]
        )
    else:
        cmd.extend(["pip", "install"])

    if requirements:
        cmd.extend(("--requirement", str(requirements)))
    if constraints:
        cmd.extend(("--constraint", str(constraints)))
    if quiet:
        cmd.append("--quiet")

    cmd.extend(str(arg) for arg in args)
    subprocess.run(cmd).check_returncode()


def pip_sync(
    python: str | Path,
    requirements: Path,
    *args: Path | str,
    target_python: Path | None = None,
    quiet: bool = True,
) -> None:
    """Run `uv sync` with the given arguments.

    Args:
        python: Path to python executable
        args: Command line args passed to pip install
        requirements: Path to a requirements file
        args: Extra requirements files or args to pass to `uv`.
        quiet: Whether to suppress output to stdout
    """

    cmd = [
        str(python),
        "-m",
        "uv",
        "pip",
        "sync",
        "--index-strategy",
        "unsafe-best-match",
        str(requirements),
        *(str(arg) for arg in args),
    ]
    if target_python:
        cmd.extend(["--python", str(python)])

    if quiet:
        cmd.append("--quiet")

    subprocess.run(cmd).check_returncode()
