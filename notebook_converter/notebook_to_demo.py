#!/usr/bin/env python3

import json
import os
import re
from base64 import b64decode
from pathlib import Path
from typing import Dict, List, Optional, Union

import pypandoc

CWD = Path(os.path.dirname(os.path.realpath(__file__)))
REPO_ROOT = CWD.parent


DIRS = {
    "demo_images": REPO_ROOT / "_static" / "demonstration_assets",
    "demo": REPO_ROOT / "demonstrations_v2"
}

def str_to_bool(s: str) -> Optional[bool]:
    if isinstance(s, bool):
        return s
    elif s is None:
        return None
    elif not isinstance(s, str):
        raise TypeError(f"Unexpected type for casting to bool. Expected str, got {type(s)}")
    s = s.lower()
    if s in {"true", "t", "yes", "y", "1"}:
        return True
    elif s in {"false", "f", "no", "n", "0"}:
        return False
    raise ValueError(f"Unable to cast {s} to boolean")


def update_sphinx_tags(rst: str) -> str:
    """
    Update `.. container :: {tag}` to `.. {tag}::`

    :param rst: The converted rst source
    :return:
    """
    return re.sub(r"\.\. container:: (\w+)", r".. \1::", rst)


def add_property_newline(rst: str) -> str:
    """
    If a line ends as following:

    foobar :property=lorem

    Then it is updated to:

    foobar
    :property=lorem
    """
    return re.sub(r"(\w+) (:property=)", r"\1\n   \2", rst)


def generate_code_output_block(output_source: Optional[List[str]] = None, only_header: bool = False) -> str:
    output_header = "\n".join(
        [
            f"# {line}"
            for line in [
                ".. rst-class:: sphx-glr-script-out",
                "",
                ".. code-block:: none",
                "",
            ]
        ]
    )
    if only_header:
        return output_header
    output_text = (
        "\n" + "\n".join([f"#    {line.rstrip()}" for line in output_source])
        if output_source
        else ""
    )
    return f"\n\n{'#' * 70}\n{output_header}{output_text}"


def generate_sphinx_role_comment(role_name: str, target: str, **attrs: Union[str, int]) -> str:
    return "\n".join(
        [
            f"# .. {role_name}:: {target}",
            *[f"#    :{attr_name}: {attr_value}" for attr_name, attr_value in attrs.items()],
        ]
    )


def fix_image_alt_tag_as_text(rst: str) -> str:
    return re.sub(r" {3}:alt: (.+)\n\n {3}(.+)", r"   :alt: \1", rst)


def convert_notebook_to_python(
    notebook: Dict,
    notebook_name: str,
    assets_folder_name: str,
    is_executable: bool
) -> str:
    # Initial validations
    assert "cells" in notebook
    assert isinstance(notebook["cells"], list)
    assert len(notebook["cells"])
    assert notebook["cells"][0]["cell_type"] == "markdown"

    ret_python_str = ""

    for i, cell in enumerate(notebook["cells"]):
        cell_type = cell["cell_type"]
        cell_source = "".join(cell.get("source", []))

        if cell_type == "markdown" and cell_source:
            cell_rst_source = pypandoc.convert_text(
                cell_source, format="md", to="rst", extra_args=["--wrap=auto", "--columns=100"]
            )
            cell_rst_source_formatted = fix_image_alt_tag_as_text(
                add_property_newline(update_sphinx_tags(cell_rst_source))
            )

            # First cell (Header)
            if i == 0:
                ret_python_str = f'r"""{cell_rst_source_formatted}"""'
            else:  # Subsequent text sections
                commented_source = "\n".join(
                    [f"# {line}" for line in cell_rst_source_formatted.split("\n")]
                )

                ret_python_str += f"\n\n{'#' * 70}\n{commented_source}"
        elif cell_type == "code" and cell_source:
            ret_python_str += f"\n\n{cell_source}"
            if not is_executable:
                # The output needs to be put into the demo file
                code_outputs = cell.get("outputs", [])
                num_images = 0
                for j, output in enumerate(code_outputs):
                    output_data = output.get("data")
                    if output["output_type"] == "execute_result" and "text/plain" in output_data:
                        ret_python_str += generate_code_output_block(
                            output_data["text/plain"], only_header=j != 0
                        )
                    elif output["output_type"] == "display_data":
                        cell_id = cell.get("id", f"c_{i}")
                        if "text/plain" in output_data and "image/png" not in output_data:
                            if j == 0:
                                ret_python_str += generate_code_output_block()
                            ret_python_str += "\n" + "\n".join(
                                [f"#    {line.strip()}" for line in output_data["text/plain"]]
                            )

                        if "image/png" in output_data:
                            if j == 0:
                                ret_python_str += f"\n\n{'#' * 70}"
                            num_images += 1
                            image_filename = f"{notebook_name}_{cell_id}_{num_images}.png"
                            image_file_dir = DIRS["demo_images"] / assets_folder_name
                            image_file_path = (
                                image_file_dir / image_filename
                            )
                            image_file_link_path = Path("../_static/demonstration_assets") / assets_folder_name / image_filename
                            role_text = generate_sphinx_role_comment(
                                "figure", image_file_link_path.as_posix(), align="center", width="80%"
                            )
                            image_data = b64decode(output_data["image/png"])

                            if not image_file_dir.exists():
                                image_file_dir.mkdir(parents=True)
                            with open(image_file_path, "wb") as ifh:
                                ifh.write(image_data)
                            ret_python_str += f"\n#\n{role_text}"
                    elif output["output_type"] == "stream":
                        text = output.get("text")
                        if text:
                            ret_python_str += generate_code_output_block(text)
    else:
        ret_python_str = ret_python_str.replace("\n%", "\n# %")

    return ret_python_str


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert Jupyter Notebook to QML Demo")

    parser.add_argument("notebook", help="Path to file that needs to be converted")

    parser.add_argument(
        "--is-executable",
        help="Indicate if the notebook is executable. "
        "If not passed, this information is inferred "
        "from the notebook file name. "
        "If the notebook name startswith tutotial_, "
        "then it is treated as executable.",
        type=str_to_bool,
        default=None
    )

    results = parser.parse_args()

    notebook_file = Path(results.notebook)
    notebook_file_name = notebook_file.stem
    notebook_is_executable = notebook_file_name.startswith("tutorial_") if results.is_executable is None else results.is_executable
    notebook_assets_folder_name = (
        notebook_file_name[len("tutorial_") :]
        if notebook_file_name.startswith("tutorial_")
        else notebook_file_name
    )

    with notebook_file.open() as fh:
        nb = json.load(fh)

    nb_py = convert_notebook_to_python(
        nb,
        notebook_file_name,
        notebook_assets_folder_name,
        notebook_is_executable
    )
    
    demo_dir = DIRS["demo"] / notebook_file_name
    if not demo_dir.exists():
        demo_dir.mkdir(parents=True)

    with (demo_dir / "demo.py").open("w") as fh:
        fh.write(nb_py)

