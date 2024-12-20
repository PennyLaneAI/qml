import typer
from pathlib import Path
from qml.lib.demo import Demo, find_demos
from qml.context import Context
from qml.lib.virtual_env import Virtualenv
from qml.lib.fs import copy_any
import shutil
import os
import subprocess
import sys

app = typer.Typer(name="qml", no_args_is_help=True)


def install_build_dependencies(build_dir: Path, venv: Virtualenv):
    build_requirements_file = build_dir / "requirements-build.txt"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "poetry",
            "export",
            "--without-hashes",
            "--format",
            "requirements.txt",
            "--only",
            "base",
            "--output",
            str(build_requirements_file),
        ]
    ).check_returncode()

    print("Installing build dependencies...")
    subprocess.run(
        [venv.python, "-m", "pip", "install", "-r", str(build_requirements_file)]
    ).check_returncode()


def install_dependencies(build_dir: Path, venv: Virtualenv, demos: list[Demo]):
    constraints_file = (build_dir / "constraints.txt").resolve()

    print("Generating constraints...")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "poetry",
            "export",
            "--without-hashes",
            "--format",
            "constraints.txt",
            "--only",
            "executable-dependencies",
            "--output",
            str(constraints_file),
        ]
    ).check_returncode()

    print("Generating requirements...")
    requirements: set[str] = set()
    for demo in demos:
        if demo.executable:
            requirements.update(demo.requirements())

    print("Installing runtime dependencies...")
    subprocess.run(
        [
            venv.python,
            "-m",
            "pip",
            "install",
            "--constraint",
            str(constraints_file),
            *requirements,
        ]
    ).check_returncode()


@app.command()
def help():
    print("Help!")


@app.command()
def build(demo_names: list[str], target: str = "html", execute: bool = False):
    ctx = Context()

    demos = list(find_demos(ctx.demos_dir, *demo_names))
    ctx.build_dir.mkdir(exist_ok=True)
    shutil.copytree(
        ctx.repo_root / "_static", ctx.build_dir / "_static", dirs_exist_ok=True
    )
    stage_dir = ctx.build_dir / "demonstrations"
    if stage_dir.exists():
        shutil.rmtree(stage_dir)

    stage_dir.mkdir(parents=True)
    # Need a 'GALLERY_HEADER' file for sphinx-gallery
    with open(stage_dir / "GALLERY_HEADER.rst", "w"):
        pass

    print(f"Building {len(demos)} demos")
    for demo in demos:
        # Use copy2 to perserve file modification time
        shutil.copy2(demo.py_file, (stage_dir / demo.name).with_suffix(".py"))

        for resource in demo.resources:
            copy_any(resource, (stage_dir / resource.name))

    env = os.environ | {"DEMO_STAGING_DIR": str(stage_dir.resolve())}
    build_venv = ctx.build_venv()
    install_build_dependencies(ctx.build_dir, build_venv)

    cmd = [
        str(build_venv.path / "bin" / "sphinx-build"),
        "-b",
        target,
    ]
    if execute:
        install_dependencies(ctx.build_dir, build_venv, demos)
    else:
        cmd.extend(("-D", "plot_gallery=0"))

    cmd.append(str(ctx.repo_root))
    cmd.append(str((ctx.build_dir / target).resolve()))

    subprocess.run(cmd, env=env).check_returncode()
