#!/usr/bin/env python

"""
Pandoc filter to convert all unrecognized directives to epigraphs (BlockQuotes).
Also convert reference links to links to the online demo's references section.
Code, link URLs, etc. are not affected.
"""
import os
from pandocfilters import toJSONFilter, BlockQuote, Link, Table, CodeBlock

DEMOS_URL = "https://pennylane.ai/qml/demos/"

def parse_rst_class(body):
    """Parse the body of an rst-class directive. This could either be a table or a sphinx-glr-script-out block."""
    [metadata, content] = body
    class_type = metadata.get("c")[0].get("c")
    if class_type == "sphx-glr-script-out":
        return CodeBlock(content)
    elif class_type == "docstable":
        return Table(content)
    else:
        return

def filter_directives(key, value, format, _):
    if key == 'Div':
        [[_, classes, _], body] = value
        if "related" in classes or "meta" in classes:
            return []
        elif "rst-class" in classes:
            return parse_rst_class(body)
        else:
            return BlockQuote(body)

    if key == 'Link':
        [roles, _, [link, _]] = value
        if len(link) > 8:
            if link[:8] == "##NOTE##":
                # This is a reference link.
                # Replace it with a link to the online demo.
                link = DEMOS_URL + os.getenv("CURRENT_DEMO") + "#references"
                return Link(roles, [{"t": "Str", "c": " [Refs]"}], [link, ""])

    # Remove the references section from the demo.
    if key == 'Header':
        [level, keys, _] = value
        if "references" in keys:
            return []

if __name__ == '__main__':
    toJSONFilter(filter_directives)
