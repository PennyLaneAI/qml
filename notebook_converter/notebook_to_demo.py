#!/usr/bin/env python3

import os
import re
import json
import shutil
from itertools import chain
from pathlib import Path, PurePosixPath
from base64 import b64decode
from typing import Dict, List, Union, Optional

import pypandoc

CWD = Path(os.path.dirname(os.path.realpath(__file__)))
REPO_ROOT = CWD.parent

MATCH_AUTHOR_FILE = re.compile(r"\.{2} +bio:{2} +(?P<name>[\w '\-]+)\n+ *(:photo:)? *(?P<profile_picture>.*\.[a-zA-Z0-9]+)?\n+ *(?P<bio>.*)",
                               flags=re.M | re.S)

AUTHORS = {
    "link-dir": PurePosixPath("../_static/authors"),
    "save-dir": REPO_ROOT / "_static" / "authors"
}
DEMO = {
    "link-dir": PurePosixPath("../demonstrations"),
    "save-dir": REPO_ROOT / "demonstrations"
}


def format_author_name(name: str) -> str:
    return re.sub(r"[ \-'\u0080-\uFFFF]+", "_", name).lower()

def parse_author_file(author_file_path: Union[Path, str]) -> Optional[Dict]:
    author_file_loc = Path(author_file_path)
    author_file_name = author_file_loc.stem
    with author_file_loc.open() as fh:
        content = fh.read()

    m = MATCH_AUTHOR_FILE.match(content)
    if not m:
        return None

    profile_picture_path = m.group("profile_picture")
    if profile_picture_path:
        profile_picture_path = Path(profile_picture_path)
        if not profile_picture_path.is_absolute():
            if profile_picture_path.is_relative_to(AUTHORS["link-dir"]):
                profile_picture_path = (REPO_ROOT / "_static" / profile_picture_path).resolve()
            else:
                profile_picture_path = (author_file_loc.parent / profile_picture_path).resolve()
    else:
        profile_picture_path = None
    return {
        "name": m.group("name"),
        "bio": m.group("bio"),
        "profile_picture": profile_picture_path,
        "formatted_name": format_author_name(author_file_name)
    }


def set_author_info(author: Dict) -> Path:
    """

    :param author:  A dictionary of author info with the following syntax
    {
        "name": Author Name,
        "bio": Author Info,
        "profile_picture": /path/to/profile_picture.png,
        "formatted_name": "<Optional> Author name"
    }
    :return: Path object of the file that was saved and author info txt
    """
    name = author["name"]
    bio = author.get("bio", "").strip()
    profile_picture_loc = author.get("profile_picture")
    if profile_picture_loc:
        profile_picture_loc = Path(profile_picture_loc)
    name_formatted = author.get("name_formatted", profile_picture_loc.stem if profile_picture_loc else format_author_name(name)).lower()

    if profile_picture_loc:
        new_profile_picture_save_loc = AUTHORS["save-dir"] / profile_picture_loc.name
    else:
        new_profile_picture_save_loc = None

    info_file_name = f"{name_formatted}.txt"
    info_file_save_loc = AUTHORS["save-dir"] / info_file_name

    if profile_picture_loc:
        try:
            shutil.copy(profile_picture_loc, new_profile_picture_save_loc)
        except shutil.SameFileError:
            pass

    if profile_picture_loc:
        author_link = (AUTHORS["link-dir"] / profile_picture_loc.name).as_posix()
        photo_text = f":photo: {author_link}"
    else:
        photo_text = ""
    author_txts = [f".. bio:: {name}"]
    if photo_text:
        author_txts.append(f"   {photo_text}")
    if bio:
        if photo_text:
            author_txts.append("")
        author_txts.append(f"   {bio}")
    author_txt = "\n".join(author_txts)

    with info_file_save_loc.open("w") as fh:
        fh.write(author_txt)

    return info_file_save_loc

def set_authors(*authors: Dict) -> str:
    """
    :param authors: An unpacked list of dictionaries, each one having the following schema:
    {
        "name": Author Name,
        "bio": Author Info,
        "profile_picture": /path/to/profile_picture.png,
        "formatted_name": "<Optional> Author name"
    }
    :return:
    """
    author_files = [
        AUTHORS['link-dir'] / set_author_info(author).name
        for author in authors
    ]

    header = [
        "About the author",
        "----------------"
    ]
    author_txts = [
        f"# .. include:: {author_txt_link}\n"
        for author_txt_link in author_files
    ]

    author_sphinx_txt = "\n".join([f"# {line}" for line in chain(header, author_txts)])
    return f"\n\n{'#' * 70}\n{author_sphinx_txt}"

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


def generate_code_output_block(output_source: List[str] = None, only_header: bool = False) -> str:
    output_header = "\n".join(
        [
            f"# {line}"
            for line in [
                ".. rst-class :: sphx-glr-script-out",
                "",
                "  .. code-block: none",
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
                        cell_id = cell["id"]
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
                            image_file_dir = DEMO["save-dir"] / notebook_assets_folder_name
                            image_file_path = (
                                image_file_dir / image_filename
                            )
                            image_file_link_path = DEMO["link-dir"] / notebook_assets_folder_name / image_filename
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

    parser.add_argument("--author",
                        help="Information about Demo Author, must be in format \"Name\" \"Bio\" \"/path/to/profile_picture.png\"",
                        action="append",
                        nargs=3)
    parser.add_argument("--author-file",
                        help="Path to an existing author file that is formatted with sphinx roles",
                        action="append")

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

    authors = []
    cli_authors = results.author or []
    cli_authors_files = results.author_file or []
    for author_file in cli_authors_files:
        author_info = parse_author_file(author_file)
        if author_info is None:
            raise ValueError(f"Unable to parse author file {author_file}")
        authors.append(author_info)
    for author_info in cli_authors:
        name = author_info[0]
        bio = author_info[1]
        profile_picture = author_info[2]
        formatted_name = format_author_name(name)
        authors.append({
            "name": name,
            "bio": bio,
            "profile_picture": profile_picture,
            "formatted_name": formatted_name
        })

    nb_py = convert_notebook_to_python(
        nb,
        notebook_file_name,
        notebook_is_executable
    )

    if authors:
        author_sphinx = set_authors(*authors)
        nb_py += author_sphinx

    with (DEMO["save-dir"] / f"{notebook_file_name}.py").open("w") as fh:
        fh.write(nb_py)

