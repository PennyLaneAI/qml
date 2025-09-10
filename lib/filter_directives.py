#!/usr/bin/env python

"""
Pandoc filter to convert all unrecognized directives to epigraphs (BlockQuotes).
Code, link URLs, etc. are not affected.
"""

from pandocfilters import toJSONFilter, BlockQuote

def filter_directives(key, value, format, _):
    if key == 'Div':
        [[_, classes, _], body] = value
        if "related" in classes or "meta" in classes:
            return []
        else:
            return BlockQuote(body)

if __name__ == '__main__':
    toJSONFilter(filter_directives)
