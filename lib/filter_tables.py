#! /usr/bin/env python

"""
Pandoc filter to convert grid tables to simple markdowntables.
"""

from pandocfilters import toJSONFilter, LineBlock

def process_header(header_cells):
    header_text = []
    for cell in header_cells:
        [_, _, _, _, blocks] = cell
        text = blocks[0].get("c")[1].get("c")[0].get("c")
        header_text.append(text)
    return header_text

def make_markdown_table(content):
    # Parse the table into headers and cells.
    # Table structure in Pandoc AST:
    # [attr, caption, colspecs, head, bodies, foot]
    [_, _, _, header_rows, bodies, _] = content
    header_row = header_rows[0]
    header_cells = header_row[1]
    header_text = process_header(header_cells)
    with open("header_text.txt", "w") as f:
        f.write(str(header_text))
    #return header_text
    return []

def filter_tables(key, value, format, _):
    if key == 'Div':
        [[_, classes, _], body] = value
        if  "rst-class" in classes:
            [metadata, content] = body
            rst_class_type = metadata.get("c")[0].get("c")
            if rst_class_type == "docstable":
                return make_markdown_table(content)

if __name__ == '__main__':
    toJSONFilter(filter_tables)
