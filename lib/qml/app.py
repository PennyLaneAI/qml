import typer
from qml.context import Context
from qml.lib import demo
import shutil
import logging
from typing import Annotated


logging.basicConfig(level=logging.INFO)

app = typer.Typer(name="qml")


@app.command()
def help():
    print("QML Demo build tool.")


@app.command()
def build(
    demo_names: Annotated[
        list[str],
        typer.Argument(
            help="Names of demos to build. If not provided, build all demos."
        ),
    ] = None,
    target: Annotated[
        demo.BuildTarget, typer.Option(help="Format to build demos")
    ] = "html",
    execute: Annotated[
        bool, typer.Option(help="Whether to execute demos and generate output cells")
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
        sphinx_dir=ctx.repo_root,
        build_dir=ctx.build_dir,
        venv_path=ctx.build_venv_path,
        demos=demos,
        target=target,
        execute=execute,
    )
