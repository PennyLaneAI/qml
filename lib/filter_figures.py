#!/usr/bin/env python

"""
Pandoc filter to convert figure directives to images.
"""
import os
from pandocfilters import toJSONFilter, Image

ASSETS_DIR = "https://blog-assets.cloud.pennylane.ai/demos/"

def parse_img_source(src: str) -> str:
    if "../_static/demonstration_assets/" in src:
        return src.replace("../_", ASSETS_DIR + os.getenv("CURRENT_DEMO") + "/main/_assets/")
    else:
        return src

def filter_images(key, value, format, _):
    if key == 'Image':
        specs, [alt], [url, _] = value
        cleaned_url = parse_img_source(url)

        try:
            alt["c"] = cleaned_url
        except KeyError:
            alt = { "t": "Str", "c": cleaned_url }
        
        return Image(specs, [alt], [cleaned_url, ""])

if __name__ == '__main__':
    toJSONFilter(filter_images)
