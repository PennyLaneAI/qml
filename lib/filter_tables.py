#! /usr/bin/env python

"""
Pandoc filter to convert grid tables to simple markdowntables.
"""
import json
from pandocfilters import toJSONFilter, RawInline

def process_header(header_cells):
    header_text = []
    for cell in header_cells:
        [_, _, _, _, blocks] = cell
        text = blocks[0].get("c")[0].get("c")[0].get("c")
        header_text.append(text)
    return header_text

def process_rows(rows):
    row_content_buffer = []
    cell_content_buffer = []
    for row in rows:
        row_content = row[1]
        for cell in row_content:
            [_, _, _, _, block] = cell
            cell_content_buffer.append(block)
        row_content_buffer.append(cell_content_buffer)
        cell_content_buffer = []
    return row_content_buffer

def make_markdown_table(table_content):
    # Parse the table into headers and cells.
    # Table structure in Pandoc AST:
    # [attr, caption, colspecs, head, bodies, foot]
    [_, _, _, header_rows, bodies, _] = table_content
    
    header_row = header_rows[1][0]
    header_cells = header_row[1]
    header_text = process_header(header_cells)
    
    body = bodies[0]
    [_, _, _, rows] = body
    row_content = process_rows(rows)

    # Build the header row
    lineblock_c = []
    header_list = []
    header_div = []
    space = {"t": "Space"}
    bar = {"t": "Str", "c": "|"}
    div = {"t": "Str", "c": "---"}
    for header in header_text:
        header_list.append({
            "t": "Str",
            "c": header
        })
        header_list.append(space)
        header_list.append(bar)
        header_list.append(space)

        header_div.append(div)
        header_div.append(space)
        header_div.append(bar)
        header_div.append(space)

    header_list.pop(-1)
    header_div.pop(-1)
    lineblock_c.append(header_list)
    lineblock_c.append(header_div)
    return [{"t": "LineBlock", "c": lineblock_c}]

def filter_tables(key, value, format, _):
    if key == 'Div':
        [[_, classes, _], body] = value
        if  "rst-class" in classes:
            [metadata, content] = body
            rst_class_type = metadata.get("c")[0].get("c")
            if rst_class_type == "docstable":
                return make_markdown_table(content.get("c"))

if __name__ == '__main__':
    toJSONFilter(filter_tables)
