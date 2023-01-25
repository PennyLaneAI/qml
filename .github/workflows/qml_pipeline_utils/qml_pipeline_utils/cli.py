import json
import argparse
from pathlib import Path

import qml_pipeline_utils.services


COMMON_CLI_FLAGS = {
    "num-workers": {
        "type": int,
        "help": "The total number of workers to be spawned",
        "required": True,
    },
    "examples-dir": {
        "type": str,
        "help": "The directory where the sphinx demonstrations reside",
        "required": True,
    },
    "build-dir": {
        "type": str,
        "help": "The directory where sphinx outputs the built demo html files",
        "required": True,
    },
    "glob-pattern": {
        "type": str,
        "help": "The glob wilcard pattern to use to get all the demo files in examples-dir",
        "required": False,
        "default": "*.py",
    },
    "offset": {
        "type": int,
        "help": "The offset of the current worker from the GitHub strategy matrix",
        "required": True,
    },
    "dry-run": {
        "action": "store_true",
        "help": "Show files will be affected without updating anything",
    },
    "verbose": {"action": "store_true", "help": "Additional logging output"},
}


def add_flags_to_subparser(subparser: argparse.ArgumentParser, *flags: str) -> None:
    assert all([flag in COMMON_CLI_FLAGS for flag in flags])
    for flag in flags:
        flag_kwargs = COMMON_CLI_FLAGS[flag]
        subparser.add_argument(f"--{flag}", **flag_kwargs)


def cli_parser():
    parser = argparse.ArgumentParser(
        prog="qml_pipeline_utils",
        description="This package contains utilities used during the QML CI/CD pipeline",
    )

    subparsers = parser.add_subparsers(dest="action")

    subparsers_build_strategy_matrix = subparsers.add_parser(
        "build-strategy-matrix",
        description="Builds the strategy matrix used by GitHub to spawn additional workers",
    )
    add_flags_to_subparser(
        subparsers_build_strategy_matrix, "num-workers", "examples-dir", "glob-pattern"
    )

    subparsers_remove_executable_code = subparsers.add_parser(
        "remove-executable-code-from-extraneous-demos",
        description="Remove executable code from sphinx examples-dir demos that are not relevant to the current worker",
    )
    add_flags_to_subparser(
        subparsers_remove_executable_code,
        "num-workers",
        "examples-dir",
        "offset",
        "dry-run",
        "verbose",
        "glob-pattern",
    )

    subparsers_remove_html = subparsers.add_parser(
        "remove-extraneous-built-html-files",
        description="Remove the HTML files and accompanying images/assets for demos "
        "that are not relevant to the current worker",
    )
    add_flags_to_subparser(
        subparsers_remove_html,
        "num-workers",
        "build-dir",
        "examples-dir",
        "offset",
        "dry-run",
        "verbose",
        "glob-pattern",
    )
    subparsers_remove_html.add_argument(
        "--preserve-non-sphinx-images",
        action="store_true",
        help="Indicate if static images in the build-dir/_images directory should be deleted or not",
    )
    subparsers_remove_html.add_argument(
        "--gallery-dir-name",
        type=str,
        default="demos",
        help="The gallery directory name inside build-dir where sphinx puts all gallery demo html files",
    )

    subparsers_clean_sitemap = subparsers.add_parser(
        "clean-sitemap", description="Delete html files and remove them from sitemap.xml"
    )
    add_flags_to_subparser(subparsers_clean_sitemap, "build-dir", "verbose", "dry-run")
    subparsers_clean_sitemap.add_argument(
        "--html-files",
        type=str,
        help="A comma separated list of html files that needs to be deleted from build directory and sitemap.xml",
    )

    subparsers_show_worker_files = subparsers.add_parser(
        "show-worker-files",
        description="Returns a list of files that the current worker will need to execute once sphinx built commences",
    )
    add_flags_to_subparser(
        subparsers_show_worker_files, "num-workers", "offset", "examples-dir", "glob-pattern"
    )

    parser_results = parser.parse_args()

    cli_actions = {
        "build-strategy-matrix": {
            "func": qml_pipeline_utils.services.build_strategy_matrix_offsets,
            "kwargs": {
                "num_workers": getattr(parser_results, "num_workers", None),
                "sphinx_examples_dir": Path(getattr(parser_results, "examples_dir", "")),
                "glob_pattern": getattr(parser_results, "glob_pattern", None),
            },
        },
        "remove-executable-code-from-extraneous-demos": {
            "func": qml_pipeline_utils.services.remove_executable_code_from_extraneous_demos,
            "kwargs": {
                "num_workers": getattr(parser_results, "num_workers", None),
                "sphinx_examples_dir": Path(getattr(parser_results, "examples_dir", "")),
                "offset": getattr(parser_results, "offset", None),
                "dry_run": getattr(parser_results, "dry_run", None),
                "verbose": getattr(parser_results, "verbose", None),
                "glob_pattern": getattr(parser_results, "glob_pattern", None),
            },
        },
        "remove-extraneous-built-html-files": {
            "func": qml_pipeline_utils.services.remove_extraneous_built_html_files,
            "kwargs": {
                "num_workers": getattr(parser_results, "num_workers", None),
                "sphinx_build_directory": Path(getattr(parser_results, "build_dir", "")),
                "sphinx_examples_dir": Path(getattr(parser_results, "examples_dir", "")),
                "sphinx_gallery_dir_name": getattr(parser_results, "gallery_dir_name", None),
                "preserve_non_sphinx_images": getattr(
                    parser_results, "preserve_non_sphinx_images", None
                ),
                "offset": getattr(parser_results, "offset", None),
                "dry_run": getattr(parser_results, "dry_run", None),
                "verbose": getattr(parser_results, "verbose", None),
                "glob_pattern": getattr(parser_results, "glob_pattern", None),
            },
        },
        "clean-sitemap": {
            "func": qml_pipeline_utils.services.clean_sitemap,
            "kwargs": {
                "sphinx_build_directory": Path(getattr(parser_results, "build_dir", "")),
                "html_files_to_remove": list(
                    filter(
                        None, map(str.strip, getattr(parser_results, "html_files", "").split(","))
                    )
                ),
                "verbose": getattr(parser_results, "verbose", None),
                "dry_run": getattr(parser_results, "dry_run", None),
            },
        },
        "show-worker-files": {
            "func": qml_pipeline_utils.services.show_worker_files,
            "kwargs": {
                "num_workers": getattr(parser_results, "num_workers", None),
                "offset": getattr(parser_results, "offset", None),
                "sphinx_examples_dir": Path(getattr(parser_results, "examples_dir", "")),
                "glob_pattern": getattr(parser_results, "glob_pattern", None),
            },
        },
    }

    if parser_results.action not in cli_actions:
        raise ValueError(
            f"Invalid action '{parser_results.action}'. "
            f"Expected one of: {json.dumps(list(cli_actions.keys()))}"
        )

    action = parser_results.action
    func = cli_actions[action]["func"]
    kwargs = cli_actions[action]["kwargs"]

    result = func(**kwargs)

    if result:
        print(json.dumps(result))
