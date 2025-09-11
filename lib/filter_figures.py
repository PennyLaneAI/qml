#!/usr/bin/env python

"""
Pandoc filter to convert figure directives to images.
"""

import os
from pandocfilters import toJSONFilter, Image

ASSETS_DIR = "https://blog-assets.cloud.pennylane.ai/demos/"

def parse_img_source(src: str) -> str:
    if "../_static/" in src:
        return src.replace("../_", ASSETS_DIR + os.getenv("CURRENT_DEMO") + "/main/_assets/")
    else:
        return src

def filter_images(key, value, format, _):
    if key == 'Image':
        _, _, [url, _] = value
        cleaned_url = parse_img_source(url)

        alt = { "t": "Str", "c": cleaned_url }
        
        return Image(["", [], []], [alt], [cleaned_url, ""])

if __name__ == '__main__':
    toJSONFilter(filter_images)
