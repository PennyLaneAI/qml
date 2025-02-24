import typer
from qml.context import Context
from qml.lib import demo, repo
import shutil
import logging
from typing import Annotated, Optional
import typing


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
                f"Copied new demo {v1_demo} to {v2_demo_dir}. Please updated the requirements.in file."
            )
        else:
            for src, dest in [(v1_demo, v2_demo), (v1_metadata, v2_metadata)]:
                if repo.file_should_update(ctx.repo, src, dest):
                    shutil.copy2(src, dest)
                    print(f"Updated {dest} from {src}")
