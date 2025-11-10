#! /usr/bin/env python

"""
Pandoc filter to convert grid tables to simple markdowntables.
"""
from pandocfilters import stringify, toJSONFilter
from filter_helpers import process_text


def process_header(header_cells):
    """Process the header cells of a table.
    Return a list of text strings for each header cell.
    """
    header_text = []
    for cell in header_cells:
        [_, _, _, _, blocks] = cell
        header_text.append(process_text(blocks))
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
            cell_content_buffer.append(process_text(block))
        row_content_buffer.append(cell_content_buffer)
        cell_content_buffer = []
    return row_content_buffer

def make_markdown_table(table_content, caption):
    """Make a markdown table from a table block.
    Returns a markdown text string for the table.
    The format for the table block is a list containing six elements:
    [attr, caption, colspecs, head, bodies, foot]
    """
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
            table_string += f"| {cell} "
        table_string += "|\n"

    if caption:
        caption_string = f"Table: {process_text(caption)}"
        table_string = f"{table_string}\n{caption_string}"

    return [{"t": "RawBlock", "c": ["markdown", table_string]}]


def filter_tables(key, value, format, _):
    if key == 'Div':
        [[_, classes, _], body] = value
        if  "rst-class" in classes:
            metadata = body[0]
            content = body[1]
            rst_class_type = metadata.get("c")[0].get("c")
            if rst_class_type == "docstable":
                caption = []
                if len(body) > 2:
                    caption.append(body[2])
                return make_markdown_table(content.get("c"), caption)

if __name__ == '__main__':
    toJSONFilter(filter_tables)
