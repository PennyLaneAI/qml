from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence, Iterator
import shutil
from qml.lib import fs, cmds
from qml.lib.virtual_env import Virtualenv
import os
import sys
from logging import getLogger
import subprocess
from enum import Enum
import re

logger = getLogger("qml")


class BuildTarget(Enum):
    """Sphinx-build targets."""

    HTML = "html"
    JSON = "json"


@dataclass
class Demo:
    """Represents a demo and its metadata."""

    name: str
    path: Path

    @property
    def py_file(self) -> Path:
        """The python file containing this demo's code and
        markup."""
        return self.path / "demo.py"

    @property
    def metadata_file(self) -> Path:
        """Metadata for this demo."""
        return self.path / "metadata.json"

    @property
    def requirements_file(self) -> Path:
        """Path to a requirements file containing
        unversioned dependencies for this demo."""
        return self.path / "requirements.in"

    @property
    def resources(self) -> Sequence[Path]:
        """Other files in the demo's directory."""
        return tuple(
            p
            for p in self.path.iterdir()
            if p not in {self.py_file, self.metadata_file, self.requirements_file}
        )

    @property
    def executable(self) -> bool:
        """Whether this demo can be exeucted."""
        return self.name.startswith("tutorial_")

    def requirements(self):
        """Return a list of this demo's unversioned
        requirements."""
        with open(self.requirements_file, "r") as f:
            return f.read().splitlines()


def find(search_dir: Path, *names: str) -> Iterator[Demo]:
    """Find demos with given names in `search_dir`."""
    if not names:
        yield from (
            Demo(name=demo_dir.name, path=demo_dir.resolve())
            for demo_dir in search_dir.iterdir()
            if demo_dir.is_dir()
        )

        return

    for name in set(names):
        demo_dir = search_dir / name
        if not (demo_dir / "demo.py").exists():
            raise ValueError(f"No demo exists with name '{name}")

        yield Demo(name=name, path=demo_dir.resolve())


def build(
    sphinx_dir: Path,
    build_dir: Path,
    venv_path: Path,
    demos: Sequence[Demo],
    target: BuildTarget,
    execute: bool,
):
    """Build the provided demos using 'sphinx-build', optionally
    executing them to generate plots and cell outputs.

    Args:
        sphinx_dir: The directory containing the sphinx conf.py file
        build_dir: Build directory
        venv_path: Path to virtual environment into which build and
            dependencies will be installed
        demos: List of demos to build
        target: The target build format
        execute: Whether to execute demos
    """
    logger.info("Building %d demos", len(demos))

    build_venv = Virtualenv(venv_path)
    stage_dir = build_dir / "demonstrations"
    if stage_dir.exists():
        shutil.rmtree(stage_dir)

    stage_dir.mkdir(parents=True)
    # Need a 'GALLERY_HEADER' file for sphinx-gallery
    with open(stage_dir / "GALLERY_HEADER.rst", "w"):
        pass

    for demo in demos:
        # Use copy2 to perserve file modification time
        shutil.copy2(demo.py_file, (stage_dir / demo.name).with_suffix(".py"))

        for resource in demo.resources:
            fs.copy_any(resource, (stage_dir / resource.name))

    _install_build_dependencies(build_venv, build_dir)
    if execute:
        _install_execution_dependencies(build_venv, build_dir, demos)

    cmd = [
        str(build_venv.path / "bin" / "sphinx-build"),
        "-b",
        target.value,
    ]
    if not execute:
        cmd.extend(("-D", "plot_gallery=0"))

    cmd.extend((str(sphinx_dir), str(build_dir / target.value)))
    sphinx_env = os.environ | {"DEMO_STAGING_DIR": str(stage_dir.resolve())}
    subprocess.run(cmd, env=sphinx_env).check_returncode()


def _install_build_dependencies(venv: Virtualenv, build_dir: Path):
    """Install dependencies for running sphinx-build into `venv`."""
    logger.info("Installing sphinx-build dependencies")

    build_requirements_file = build_dir / "requirements-build.txt"
    cmds.poetry_export(
        sys.executable,
        build_requirements_file,
        groups=("base",),
        format="requirements.txt",
    )
    cmds.pip_install(venv.python, "-r", build_requirements_file)


def _install_execution_dependencies(
    venv: Virtualenv, build_dir: Path, demos: Sequence[Demo]
):
    """Install dependencies for executing provided demos into
    `venv`."""
    constraints_file = (build_dir / "constraints.txt").resolve()
    cmds.poetry_export(
        sys.executable,
        constraints_file,
        format="constraints.txt",
        groups=("executable-dependencies",),
    )
    if sys.platform == "darwin":
        _fix_pytorch_constraint_macos(constraints_file)

    requirements: set[str] = set()
    for demo in demos:
        if demo.executable:
            requirements.update(demo.requirements())

    logger.info("Installing execution dependencies")
    cmds.pip_install(
        venv.python,
        *requirements,
        constraints=constraints_file,
    )


def _fix_pytorch_constraint_macos(constraints_file: Path):
    """`poetry export` keeps the '+cpu' extra on torch and torchvision when exporting
    on MacOS, which uses the PyPi wheels. Because the wheels have no '+cpu' extra on
    PyPi, package resolution will fail.

    This function strips the +cpu extra from the constraint.
    """
    with open(constraints_file, "r") as f:
        constraints = f.read()

    for pattern in (r"(torch==[0-9\.]+)\+cpu", r"(torchvision==[0-9\.]+)\+cpu"):
        constraints = re.sub(
            pattern, lambda m: m.group(1), constraints, flags=re.MULTILINE
        )

    with open(constraints_file, "w") as f:
        f.write(constraints)
