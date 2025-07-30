from lib.qml.app.utils import slugify
import typer
from qml.context import Context
from qml.lib import demo, repo, cli, fs, template
import shutil
import logging
from typing import Annotated, Optional
import typing
import inflection
import re
import json
import rich

logging.basicConfig(level=logging.INFO)

app = typer.Typer(name="qml")


@app.command()
def help():
    print("QML Demo build tool.")


@app.command()
def build(
    demo_names: Annotated[
        Optional[list[str]],
        typer.Argument(
            help="Names of demos to build. If not provided, build all demos."
        ),
    ] = None,
    format: Annotated[
        demo.BuildTarget, typer.Option(help="Format to build demos")
    ] = typing.cast(demo.BuildTarget, "html"),
    execute: Annotated[
        bool, typer.Option(help="Whether to execute demos and generate output cells")
    ] = False,
    quiet: Annotated[bool, typer.Option(help="Suppress sphinx output")] = False,
    keep_going: Annotated[
        bool, typer.Option(help="Continue if sphinx-build fails for a demo")
    ] = False,
    dev: Annotated[bool, typer.Option(help="Whether to use dev dependencies")] = False,
) -> None:
    """Build the named demos."""
    ctx = Context()
    demo_names = demo_names or []
    demos = list(demo.find(ctx.demos_dir, *demo_names))

    ctx.build_dir.mkdir(exist_ok=True)
    shutil.copytree(
        ctx.repo_root / "_static", ctx.build_dir / "_static", dirs_exist_ok=True
    )
    demo.build(
        ctx,
        demos=demos,
        target=format,
        execute=execute,
        quiet=quiet,
        keep_going=keep_going,
        dev=dev,
    )


@app.command()
def new():
    """Create a new demo."""
    ctx = Context()
    title: str = typer.prompt("Title")
    name_default = slugify(title)

    while True:
        name: str = typer.prompt("Custom directory name", name_default)
        if demo.get(ctx.demos_dir, name):
            print(f"A demo with the directory name '{name}' already exists, please choose a different name.")
        else:
            break

    description = typer.prompt("Description", "")

    author_prompt = "Author's pennylane.ai username"
    authors: list[str] = []
    authors.append(typer.prompt(author_prompt))

    while True:
        if not typer.confirm("Would you like to add another author?"):
            break

        authors.append(typer.prompt(author_prompt))

    small_thumbnail = cli.prompt_path("Thumbnail image")
    large_thumbnail = cli.prompt_path("Large thumbnail image")

    demo_dir = ctx.demos_dir / name
    demo_dir.mkdir(exist_ok=True)

    if small_thumbnail:
        dest = (demo_dir / "thumbnail").with_suffix(small_thumbnail.suffix)
        fs.copy_parents(small_thumbnail, dest)
        small_thumbnail = str(dest.relative_to(demo_dir))

    if large_thumbnail:
        dest = (demo_dir / "large_thumbnail").with_suffix(large_thumbnail.suffix)
        fs.copy_parents(large_thumbnail, dest)
        large_thumbnail = str(dest.relative_to(demo_dir))

    with open(demo_dir / "demo.py", "w") as f:
        f.write(template.demo(title))

    with open(demo_dir / "metadata.json", "w") as f:
        json.dump(
            template.metadata(
                title=title,
                description=description,
                authors=authors,
                thumbnail=small_thumbnail,
                large_thumbnail=large_thumbnail,
                categories=[],
            ),
            f,
            indent=2,
        )

    rich.print(
        f"Created new demo in [bold]{demo_dir.relative_to(ctx.repo_root)}[/bold]"
    )


@app.command()
def sync_v2():
    """Copy new and changed demos from /demonstrations to /demonstrations_v2."""
    ctx = Context()
    for v1_demo in (ctx.repo_root / "demonstrations").glob("*.py"):
        demo_name = v1_demo.stem
        v1_metadata = v1_demo.with_suffix(".metadata.json")

        v2_demo_dir = ctx.demos_dir / demo_name
        v2_demo = v2_demo_dir / "demo.py"
        v2_metadata = v2_demo_dir / "metadata.json"

        if not v2_demo_dir.exists():
            v2_demo_dir.mkdir()
            shutil.copy2(v1_demo, v2_demo)
            shutil.copy2(v1_metadata, v2_metadata)

            print(
                f"Copied new demo {v1_demo} to {v2_demo_dir}. Please update the requirements.in file."
            )
        else:
            for src, dest in [(v1_demo, v2_demo), (v1_metadata, v2_metadata)]:
                if repo.file_should_update(ctx.repo, src, dest):
                    shutil.copy2(src, dest)
                    print(f"Updated {dest} from {src}")
