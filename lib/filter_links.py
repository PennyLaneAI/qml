#!/usr/bin/env python

"""
Pandoc filter to process intersphinx links.
"""
import json
import sphobjinv as soi
from pandocfilters import toJSONFilter, Link

DEMOS_URL = "https://pennylane.ai/qml/demos/"
PL_OBJ_INV_URL = "https://docs.pennylane.ai/en/stable/"
CAT_OBJ_INV_URL = "https://docs.pennylane.ai/projects/catalyst/en/stable/"

def process_link(text: str, key: str) -> tuple[str, str]:
    """Process a link to a PennyLane or Catalyst object."""
    # Check if it's a PennyLane object. These are more popular, so check first.
    for item in pl_obj_inv.objects:
        if item.name == key:
            return text if text else item.dispname, PL_OBJ_INV_URL + item.uri

    for item in cat_obj_inv.objects:
        if item.name == key:
            return text if text else item.dispname, CAT_OBJ_INV_URL + item.uri
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
    elif "~" == link[0]:
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
            text, key = parse_body(body)
            
            # The doc type could be a demo, so needs special treatment
            if "doc" in role[0]:
                name, link = process_doc_link(text, key)
            else:
                name, link = process_link(text, key)

            return Link(["",[],[]], pandocify_string(name), [link,""])

if __name__ == '__main__':
    pl_obj_inv = soi.Inventory(url=PL_OBJ_INV_URL+"objects.inv")
    cat_obj_inv = soi.Inventory(url=CAT_OBJ_INV_URL+"objects.inv")
    toJSONFilter(filter_links)
