#!/usr/bin/env python

"""
Pandoc filter to convert figure directives to images.
"""

from pandocfilters import toJSONFilter, Image

def filter_images(key, value, format, _):
    if key == 'Image':
        specs, url, alt = value
        
        return Image([specs, url, alt])

if __name__ == '__main__':
    toJSONFilter(filter_images)
