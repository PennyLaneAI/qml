#!/usr/bin/env python

"""
Pandoc filter to convert figure directives to images.
"""

from pandocfilters import toJSONFilter, Image
from filter_helpers import parse_img_source


def filter_images(key, value, format, _):
    if key == 'Image':
        _, _, [url, _] = value
        cleaned_url = parse_img_source(url)

        alt = { "t": "Str", "c": cleaned_url }
        
        return Image(["", [], []], [alt], [cleaned_url, ""])

if __name__ == '__main__':
    toJSONFilter(filter_images)
