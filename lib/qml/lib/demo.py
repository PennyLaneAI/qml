from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from collections.abc import Sequence, Iterator
import shutil
from qml.lib import fs, cmds
from qml.lib.virtual_env import Virtualenv
from qml.lib.pip_tools import RequirementsGenerator
import os
import sys
from logging import getLogger
import subprocess
from enum import Enum
import functools
import requirements
import json
import lxml.html
from qml.context import Context


logger = getLogger("qml")


class BuildTarget(Enum):
    """Sphinx-build targets."""

    HTML = "html"
    JSON = "json"


@dataclass
class Demo:
    """Represents a demo and its metadata."""

    CORE_DEPENDENCIES = frozenset(
        (
            "matplotlib",
            "numpy",
            "pennylane",
            "pennylane-lightning",
            "pennylane-catalyst",
            "jax",
            "jaxlib",
        )
    )
    """Dependencies installed for every demo that do not
    need to be specified by `requirements.in`."""

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
    def requirements_file(self) -> Path | None:
        """Path to a requirements file containing
        unversioned dependencies for this demo."""
        if (path := self.path / "requirements.in").exists():
            return path

        return None

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

    @functools.cached_property
    def requirements(self) -> frozenset[str]:
        if not (path := self.requirements_file):
            return self.CORE_DEPENDENCIES

        reqs = set(self.CORE_DEPENDENCIES)
        with open(path, "r") as f:
            for req in requirements.parse(f):
                reqs.discard(req.name)
                reqs.add(req.line)

        return frozenset(reqs)


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
            raise ValueError(f"No demo exists with name '{name}'")

        yield Demo(name=name, path=demo_dir.resolve())


def search(search_dir: Path, pattern: str) -> Iterator[str]:
    """Yield demo names in `search_dir` matching `pattern`."""
    for path in search_dir.glob(pattern=pattern):
        if (path / "demo.py").exists():
            yield path.name


def build(
    ctx: Context,
    demos: Sequence[Demo],
    target: BuildTarget,
    execute: bool,
    quiet: bool = False,
    keep_going: bool = False,
) -> None:
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
    failed: list[str] = []
    logger.info("Building %d demos", len(demos))

    build_venv = Virtualenv(ctx.build_venv_path)
    _install_build_dependencies(build_venv, ctx.build_dir)

    requirements_generator = RequirementsGenerator(
        Path(sys.executable),
        global_constraints_file=ctx.constraints_file,
        extra_index_urls=("https://download.pytorch.org/whl/cpu",),
    )

    for demo in demos:
        try:
            _build_demo(
                sphinx_dir=ctx.repo_root,
                build_dir=ctx.build_dir,
                build_venv=build_venv,
                requirements_generator=requirements_generator,
                target=target,
                execute=execute,
                demo=demo,
                package=target is BuildTarget.JSON,
                quiet=quiet,
            )
        except subprocess.CalledProcessError as exc:
            if not keep_going:
                raise exc

            failed.append(demo.name)
            logger.error("Build failed for demo '%s'", demo.name)
            if quiet:
                if (
                    error_summary := _find_sphinx_gallery_execution_error(exc.stdout)
                    is None
                ):
                    error_summary = exc.stdout

                logger.error("%s", error_summary)

    if failed:
        raise RuntimeError("Failed to build demos: " + ", ".join(failed))


def _build_demo(
    sphinx_dir: Path,
    build_dir: Path,
    build_venv: Virtualenv,
    demo: Demo,
    target: BuildTarget,
    requirements_generator: "RequirementsGenerator",
    execute: bool,
    package: bool,
    quiet: bool,
):
    logger.info("Building '%s', execute=%s", demo.name, execute)

    out_dir = sphinx_dir / "demos" / demo.name
    fs.clean_dir(out_dir)
    execute = execute and demo.executable

    with open(out_dir / "requirements.txt", "w") as f:
        f.write(requirements_generator.generate_requirements(demo.requirements))

    if execute:
        cmds.pip_install(
            build_venv.python, requirements=out_dir / "requirements.txt", quiet=True
        )

    stage_dir = build_dir / "demonstrations"
    fs.clean_dir(stage_dir)
    # Need a 'GALLERY_HEADER' file for sphinx-gallery
    with open(stage_dir / "GALLERY_HEADER.rst", "w"):
        pass

    shutil.copy2(demo.py_file, (stage_dir / demo.name).with_suffix(".py"))
    for resource in demo.resources:
        fs.copy_any(resource, (stage_dir / resource.name))

    cmd = [
        str(build_venv.path / "bin" / "sphinx-build"),
        "-b",
        target.value,
    ]
    if not execute:
        cmd.extend(("-D", "plot_gallery=0"))

    cmd.extend((str(sphinx_dir), str(build_dir / target.value)))
    sphinx_env = os.environ | {
        "DEMO_STAGING_DIR": str(stage_dir.resolve()),
        "GALLERY_OUTPUT_DIR": str(out_dir.resolve().relative_to(sphinx_dir)),
    }
    if quiet:
        stdout, stderr, text = subprocess.PIPE, subprocess.STDOUT, True
    else:
        stdout, stderr, text = None, None, None

    subprocess.run(
        cmd, env=sphinx_env, stdout=stdout, stderr=stderr, text=text
    ).check_returncode()

    if package:
        _package_demo(
            demo,
            build_dir / "pack",
            sphinx_dir / "_static",
            build_dir / target.value,
            out_dir,
        )


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
    cmds.pip_install(venv.python, "-r", build_requirements_file, use_uv=False)


def _package_demo(
    demo: Demo,
    pack_dir: Path,
    static_dir: Path,
    sphinx_output: Path,
    sphinx_gallery_output: Path,
):
    """Package a demo into a .zip file for distribution.

    Args:
        demo: The demo to package
        pack_dir: The directory in which to place the packaged demo
        static_dir: The /static directory in the repo root
        sphinx_output: The directory containing the sphinx output
        sphinx_gallery_output: The directory containing files genreated by
            sphinx-gallery
    """
    dest = pack_dir / demo.name
    fs.clean_dir(dest)

    with open(
        (sphinx_output / "demos" / demo.name / demo.name).with_suffix(".fjson"), "r"
    ) as f:
        html_body = json.load(f)["body"]

    asset_paths: set[tuple[Path, str]] = set()
    html_body: str = lxml.html.rewrite_links(
        html_body,
        functools.partial(
            _link_rewriter, static_dir, sphinx_output / "_images", asset_paths
        ),
    )
    with open(dest / "body.html", "w") as f:
        f.write(html_body)

    for asset, asset_dest in asset_paths:
        try:
            fs.copy_parents(asset, Path(dest, "_assets", asset_dest))
        except FileNotFoundError:
            logger.warning("Could not find asset '%s' for demo '%s'", asset, demo.name)
            continue

    shutil.copy(
        (sphinx_gallery_output / demo.name).with_suffix(".ipynb"), dest / "demo.ipynb"
    )
    shutil.copy(demo.metadata_file, dest / "metadata.json")
    shutil.copy(demo.py_file, dest / "demo.py")
    shutil.copy(sphinx_gallery_output / "requirements.txt", dest / "requirements.txt")
    for resource in demo.resources:
        fs.copy_any(resource, dest / resource.relative_to(demo.path))

    with open(demo.metadata_file, "r") as f:
        metadata = json.load(f)

    for preview_image in metadata["previewImages"]:
        if (uri := preview_image["uri"]).startswith("/_static/"):
            src = static_dir / uri.removeprefix("/_static/")
            path = PurePosixPath(
                "_assets", "thumbnails", preview_image["type"]
            ).with_suffix(src.suffix)
            preview_image["uri"] = path.as_posix()
            fs.copy_parents(src, dest / path)

    for hardware in metadata.get("hardware", []):
        if (uri := hardware["logo"]).startswith("/_static/"):
            src = static_dir / uri.removeprefix("/_static/")
            path = PurePosixPath("_assets", "logos", Path(src).name)
            hardware["logo"] = path.as_posix()
            fs.copy_parents(src, dest / path)

    with open(dest / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    shutil.make_archive(
        base_name=str(pack_dir / dest.name), format="zip", base_dir=dest, root_dir=dest
    )


def _link_rewriter(
    static_dir: Path, image_dir: Path, asset_paths: set[tuple[Path, str]], link: str
):
    if "_images/" in link:
        _, path = link.split("_images/", maxsplit=2)
        asset_paths.add((image_dir / path, f"images/{path}"))
        return f"_assets/images/{path}"
    elif "_static/" in link:
        _, path = link.split("_static/", maxsplit=2)
        asset_paths.add((static_dir / path, f"static/{path}"))
        return f"_assets/static/{path}"

    return link


def _find_sphinx_gallery_execution_error(stdout: str) -> str | None:
    i = stdout.find(
        "Here is a summary of the problems encountered when running the examples:"
    )
    if i != -1:
        return stdout[i:]

    return None
