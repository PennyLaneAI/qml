#!/usr/bin/env python

"""
Pandoc filter to process intersphinx links.
"""
import tempfile
import time
from pathlib import Path
from typing import Any, Dict

import sphobjinv as soi
from pandocfilters import toJSONFilter, Link, RawInline
import requests
from requests import exceptions as requests_exceptions

DEMOS_URL = "https://pennylane.ai/qml/demos/"
PL_OBJ_INV_URL = "https://docs.pennylane.ai/en/stable/"
CAT_OBJ_INV_URL = "https://docs.pennylane.ai/projects/catalyst/en/stable/"

# Mirror Sphinx's sphinx.util.requests: use GET with User-Agent and timeout.
# Sphinx uses requests.get()/head() via sphinx.util.requests.get(), which sets
# User-Agent (Firefox-like + Sphinx version) and intersphinx_timeout.
# See: sphinx/util/requests.py (get/head), intersphinx_timeout in conf.py.
REQUEST_TIMEOUT = 5  # seconds, analogous to intersphinx_timeout
USER_AGENT = "Mozilla/5.0 (compatible; QML-Pandoc-Filter; +https://github.com/PennyLaneAI/qml)"

def load_inventory_with_retry(url: str, retries: int = 3, delay: float = 1.0) -> soi.Inventory:
    """Download objects.inv from URL and load inventory with retry logic for transient HTTP errors."""
    headers = {"User-Agent": USER_AGENT}
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            break
        except requests_exceptions.HTTPError:
            if attempt == retries:
                raise
            time.sleep(delay * attempt)
    with tempfile.NamedTemporaryFile(suffix=".inv", delete=False) as f:
        f.write(response.content)
        path = Path(f.name)
    try:
        return soi.Inventory(path)
    finally:
        path.unlink(missing_ok=True)

def make_named_inventory(inv: soi.Inventory) -> Dict[str, Any]:
    """Make a dictionary of objects from an inventory."""
    return {item.name: item for item in inv.objects}

def process_link(text: str, key: str) -> tuple[str, str]:
    """Process a link to a PennyLane or Catalyst object."""
    # Check if it's a PennyLane object. These are more popular, so check first.
    if key in pl_obj_inv.keys():
        return text if text else pl_obj_inv[key].dispname, PL_OBJ_INV_URL + pl_obj_inv[key].uri
    if key in cat_obj_inv.keys():
        return text if text else cat_obj_inv[key].dispname, CAT_OBJ_INV_URL + cat_obj_inv[key].uri
    return text, key

def process_doc_link(text: str, key: str) -> tuple[str, str]:
    """Process a doc/demo link."""
    # Check if it's a demo link. If so, we need to add the demos URL.
    if "/" in key:
        type, slug = key.split("/",1)
        if type == "demos":
            text = text if text else slug
            return text, DEMOS_URL + slug
        
    # If we haven't returned, this is a pennylane or catalyst doc.
    return process_link(text, key)

def parse_body(link: str) -> tuple[str, str]:
    """Parse the body of an intersphinx link."""
    # Intersphinx links have three formats:
    # 1. Some text followed by <sphinx/link>
    # 2. ~.pennylane.some.function.name
    # 3. sphinx/link
    if "<" in link:
        text, key = link.split("<")
        return text.strip(), key.strip(">").lstrip(".")
    elif link.startswith("~"):
        return link.split(".")[-1], link.lstrip("~.")
    else:
        return "", link

def pandocify_string(string: str) -> list:
    """Helper function to convert a string to a list of Pandoc Str nodes."""
    pandoc_string = []
    for substring in string.split(" "):
        pandoc_string.append({"t": "Str", "c": substring},)
        pandoc_string.append({"t": "Space"},)
    pandoc_string.pop() # Remove the last space
    return pandoc_string

def filter_links(key, value, format, _):
    """Pandoc filter to process intersphinx links."""
    # Pandoc thinks intersphinx links are Code blocks.
    # The class will be interpreted-text, with the role defining the type of link.
    if key == 'Code':
        [[_, classes, role], body] = value

        if "interpreted-text" in classes:
            if "html" in role[0]:
                return RawInline("markdown", body)
            else:
                text, key = parse_body(body)
                # The doc type could be a demo, so needs special treatment
                if "doc" in role[0]:
                    name, link = process_doc_link(text, key)
                else:
                    name, link = process_link(text, key)

                # Some links end with a dollar sign for some reason, but this is interpreted as math. 
                # Remove it to avoid this.
                return Link(["",[],[]], pandocify_string(name), [link.removesuffix("$"),""])

if __name__ == '__main__':
    pl_obj_inv = make_named_inventory(load_inventory_with_retry(PL_OBJ_INV_URL+"objects.inv"))
    cat_obj_inv = make_named_inventory(load_inventory_with_retry(CAT_OBJ_INV_URL+"objects.inv"))
    toJSONFilter(filter_links)
