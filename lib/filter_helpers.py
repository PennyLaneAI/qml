import os
from pandocfilters import stringify, walk

ASSETS_DIR = "https://blog-assets.cloud.pennylane.ai/demos/"

def parse_img_source(src: str) -> str:
    if "../_static/" in src:
        return src.replace("../_", ASSETS_DIR + os.getenv("CURRENT_DEMO") + "/main/_assets/")
    else:
        return src

def make_markdown_figure(figure: list) -> str:
    """Make a markdown figure from a figure block.
    Returns a markdown text string for the figure.
    The format for the figure block is a list containing three elements:
    [metadata, [url_dict], [url, alt_text]]
    """
    [_, [url_dict], _] = figure
    return f"![]({url_dict.get('c', '')})"

def make_markdown_link(link: list) -> str:
    """Make a markdown link from a link block.
    Returns a markdown text string for the link.
    The format for the link block is a list containing three elements:
    [metadata, link_text, [link_url, caption]]
    """
    [_, link_text, link_url] = link
    text = stringify(link_text)
    url = link_url[0]
    return f"[{text}]({url})"

def process_text(text):
    """Process a block of 'text'.
    Return a markdown text string representing the content.
    """
    result = []
    def parse_text(key, val, format, meta):
        if key in ['Str', 'MetaString']:
            result.append(val)
        elif key == 'Code':
            result.append(val[1])
        elif key == 'Math':
            result.append(f"$${val[1]}$$")
        elif key == 'LineBreak':
            result.append(" ")
        elif key == 'SoftBreak':
            result.append(" ")
        elif key == 'Space':
            result.append(" ")
        elif key == 'Image':
            result.append(make_markdown_figure(val))
            # Don't process the rest of the block
            return []
        elif key == 'Link':
            result.append(make_markdown_link(val))
            # Don't process the rest of the block
            return []
    walk(text, parse_text, "", {})
    return ''.join(result)
