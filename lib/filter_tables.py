#! /usr/bin/env python

"""
Pandoc filter to convert grid tables to simple markdowntables.
"""
from pandocfilters import toJSONFilter, walk
from filter_helpers import make_markdown_figure, make_markdown_link


def process_header(header_cells):
    """Process the header cells of a table.
    Return a list of text strings for each header cell.
    """
    header_text = []
    for cell in header_cells:
        [_, _, _, _, blocks] = cell
        text = blocks[0].get("c")[0].get("c")[0].get("c")
        header_text.append(text)
    return header_text

def process_rows(rows):
    """Process the rows of a table.
    Return a list of lists of text strings for each cell in each row.
    """
    row_content_buffer = []
    cell_content_buffer = []
    for row in rows:
        row_content = row[1]
        for cell in row_content:
            [_, _, _, _, block] = cell
            cell_content_buffer.append(process_block(block))
        row_content_buffer.append(cell_content_buffer)
        cell_content_buffer = []
    return row_content_buffer

def process_block(block):
    """Process a block of a table.
    Return a markdown text string for the content in the block.
    """
    result = []
    def parse_block(key, val, format, meta):
        if key in ['Str', 'MetaString']:
            result.append(val)
        elif key == 'Code':
            result.append(val[1])
        elif key == 'Math':
            result.append(val[1])
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
    walk(block, parse_block, "", {})
    return ''.join(result)

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

    table_string = ""
    for header in header_text:
        table_string += f"| {header} "
    table_string += "|\n"
    table_string += "|---" * len(header_text) + "|\n"
    for row in row_content:
        for cell in row:
            table_string += f"| {cell}"
        table_string += "|\n"

    return [{"t": "RawBlock", "c": ["markdown", table_string]}]


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
