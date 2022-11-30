import re
from xml.etree import ElementTree
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def clean_sitemap(
    sphinx_build_directory: "Path",
    html_files_to_remove: List[str],
    verbose: bool = False,
    dry_run: bool = False,
):
    """
    Removes html files from the _build/html folder similar to `remove_extraneous_html` function.
    However, this function is used to remove files that are not relevant to *any node*.
    It not only removes the given html files but further updates the sitemap of the built demos
    so that SEO coverage is not affected.

    To call this function, pass the directory where the qml repo resides, and a list of files to delete.
    The path to the list of files need to be the path *after* the base url e.g: pennylane.ai/qml.

    Example:
        If the path to a file is: https://pennylane.ai/qml/demos/my_old_demo.html
        And the base url to qml is: https://pennylane.ai/qml
        Then to delete that html file, pass:
          remove_html_from_sitemap("/path/to/qml/repository/_build/html",
                                   ["demos/my_old_demo.html"])
        Then the set HTML would get removed from the gallery directory AND also the sitemap.xml file.

    Args:
        sphinx_build_directory: The directory where sphinx outputs the built demo html files
        html_files_to_remove: List of string. Each string being a file name relative to the base url.
        verbose: Additional logging output
        dry_run: If True, does nothing. If doing a dry_run, set verbose to True.

    Returns:
        None
    """
    sitemap_location = sphinx_build_directory / "sitemap.xml"
    sitemap_tree = ElementTree.parse(sitemap_location)
    sitemap_root = sitemap_tree.getroot()

    default_ns_match = re.match(r"{(.*)}", sitemap_root.tag)
    default_ns = "" if not default_ns_match else default_ns_match.group(1)

    # Prefixing the default namespace here to the xml tag as it seems to work more steadily
    # across both Python 3.7 and above. Once QML starts using a higher version of Python then this
    # method can be converted to passing the namespace to findall using the `namespaces` parameter.
    xml_search_paths = {
        "url": "url" if not default_ns else f"{{{default_ns}}}url",
        "loc": "loc" if not default_ns else f"{{{default_ns}}}loc",
    }
    if default_ns and verbose:
        print(f"Detected default namespace for sitemap: '{default_ns}'")
    sitemap_urls = sitemap_root.findall(xml_search_paths["url"])
    for file_to_remove in html_files_to_remove:
        file = sphinx_build_directory / file_to_remove
        if file.exists():
            if verbose:
                print(f"Deleting file from {str(sphinx_build_directory)}: '{file_to_remove}'")
            if not dry_run:
                file.unlink()
        for url in sitemap_urls:
            url_location = url.find(xml_search_paths["loc"])
            loc = url_location.text
            if loc.endswith(file_to_remove):
                if verbose:
                    print(f"Deleting following url from sitemap.xml: '{loc}'")
                if not dry_run:
                    sitemap_root.remove(url)
    if default_ns:
        ElementTree.register_namespace("", default_ns)
    sitemap_tree.write(str(sitemap_location))
    return None
