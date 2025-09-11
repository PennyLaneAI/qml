#!/usr/bin/env python

"""
Pandoc filter to convert figure directives to images.
"""

from pandocfilters import toJSONFilter, Image

def filter_images(key, value, format, _):
    if key == 'Image':
        specs, [alt], src = value

        try:
            alt["c"] = ""
        except KeyError:
            alt = { "t": "Str", "c": "" }
        
        return Image(specs, [alt], src)

if __name__ == '__main__':
    toJSONFilter(filter_images)
