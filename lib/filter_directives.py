#!/usr/bin/env python

"""
Pandoc filter to convert all unrecognized directives to epigraphs (BlockQuotes).
Code, link URLs, etc. are not affected.
"""

import pandocfilters as pf

def filter_directives(key, value, format, _):
    if key == 'Div':
        [[ident, classes, keyvals], code] = value
        if "related" or "meta" in classes:
            return []

if __name__ == '__main__':
    pf.toJSONFilter(filter_directives)
