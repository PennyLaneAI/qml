import os
import sys
import difflib

import pytz
from datetime import datetime
from html_parser import DemoOutputParser

TIMEZONE = pytz.timezone("America/Toronto")

def parse_demo_outputs(filename):
    """Parse the outputs produced of a QML repo demonstration from file.

    Args:
        filename (str): The name of the demonstration file. The file is
            expected to be in HTML format.

    Returns:
        list: the list of demonstration outputs
    """
    f = open(filename, "r")
    html_file = f.read()

    parser = DemoOutputParser()
    parser.feed(html_file)

    outputs = []
    for d in parser.data:

        # We expect to find 'Out' sections as per how Sphinx produces outputs
        if d != 'Out:':

            if '\n' in d:
                # If there are newlines in the string, then extract each line
                # by splitting and only keep the non-empty ones
                lines = [line for line in d.split("\n") if line != '']
                outputs.extend(lines)
            else:
                outputs.append(d)
    return outputs

def write_file_diff(file_obj, qml_version, file_url, outputs, diff_indices):
    """Write the outputs produced by a demonstration from the QML repository to
    a file.

    Args:
        file_obj (object): the file object used to write the diffs found when
            checking a specific version of the demonstration
        qml_version (str): the version of the QML repository for which results
            are written; e.g., "Master" or "Dev"
        file_url (str): the URL for the demo; either a page on the PennyLane
            website or a page on the hosted dev version of the PennyLane website
        outputs (list): the list of demo outputs
        diff_indices (set): the set of indices where a difference was found
    """
    # Write the version name with the associated URL pointing to the demo
    file_obj.write(f'[{qml_version}]({file_url}):\n\n')

    if len(diff_indices) > 20:

        # Insert a dropdown option if too many outputs
        # Note: html tags are being used that are compatible with GitHub
        # markdown

        # End html details and code sections, provide a quick summary
        summary_section = "<summary>\n More \n </summary>\n"
        file_obj.write(f'<details> \n {summary_section} <pre>\n <code>\n')

        # Dump the outputs
        for idx in diff_indices:
            file_obj.write(f'{outputs[idx]}\n')

        # End html code and details sections
        file_obj.write(f' </code>\n </pre>\n </details>\n\n')

    else:

        # Start a markdown code block
        file_obj.write(f'```\n')

        # Dump the outputs
        for idx in diff_indices:
            file_obj.write(f'{outputs[idx]}\n')

        # End the markdown code block
        file_obj.write(f'```\n\n')


def main():
    """Parses two versions of the automatically run demonstrations from the QML
    repository, compares the output of each demo and writes to a file based on the
    differences found.
    """
    master_path = "/tmp/master/home/runner/work/qml/qml/_build/html/demos/"
    dev_path = "/tmp/dev/home/runner/work/qml/qml/_build/html/demos/"

    master_url = 'https://pennylane.ai/qml/demos/'
    dev_url = 'http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/'

    # Get all the filenames
    master_files = os.listdir(master_path)
    dev_files = os.listdir(dev_path)

    master_automatically_run = set([f for f in master_files if f.startswith("tutorial_")])
    dev_automatically_run = set([f for f in dev_files if f.startswith("tutorial_")])

    automatically_run = master_automatically_run.union(dev_automatically_run)

    output_file = open('demo_diffs.md','w')

    # Write a time update
    update_time = pytz.utc.localize(datetime.utcnow())
    update_time = update_time.astimezone(TIMEZONE)
    update_time_str = update_time.strftime("%Y-%m-%d  %H:%M:%S")
    output_file.write(f"Last update: {update_time_str} (All times shown in Eastern time)\n")

    demos_with_diffs = []
    database_of_differences = {}
    for filename in automatically_run:
        master_file = os.path.join(master_path, filename)
        master_outputs = parse_demo_outputs(master_file)

        dev_file = os.path.join(dev_path, filename)
        dev_outputs = parse_demo_outputs(dev_file)

        outputs_with_diffs = set()
        for out_idx, (a,b) in enumerate(zip(master_outputs, dev_outputs)):

            for i,s in enumerate(difflib.ndiff(a, b)):

                # The output of difflib.ndiff can be one of three cases:
                # 1. Whitespace (no difference found)
                # 2-3. The '-' or '+' characters: for character differences
                if s[0]==' ':
                    continue

                # If any diff found: keep track of the index
                elif s[0]=='-':
                    outputs_with_diffs.add(out_idx)

                elif s[0]=='+':
                    outputs_with_diffs.add(out_idx)

        if outputs_with_diffs:

            demos_with_diffs.append(filename)
            database_of_differences[filename] = (master_outputs, dev_outputs, outputs_with_diffs)

    if not demos_with_diffs:
        output_file.write(f'### No differences found between the tutorial outputs. 🎉\n')
    else:
        output_file.write("# List of differences in demonstration outputs\n\n")

        # 1. Create a list of all the demos that are different in the table of
        # contents
        output_file.write("# Table of contents\n\n")
        for i, demo_name in enumerate(demos_with_diffs):
            output_file.write(f'{i + 1}. [{demo_name}](#demo{i})\n')

        # 2. Note the number of all demos
        output_file.write(f"\n\nNumber of demos different/all demos: {len(demos_with_diffs)}/{len(automatically_run)}\n\n")

        # 3. Bump the differences
        for i, demo_name in enumerate(demos_with_diffs):
            master_outputs, dev_outputs, outputs_with_diffs = database_of_differences[demo_name]

            file_html = demo_name.replace('.py', '.html')
            output_file.write(f'## {i + 1}. {demo_name} <a name="demo{i}"></a>\n\n')
            output_file.write('---\n\n')

            # Write the Master version difference to file
            master_file_url = master_url + file_html
            write_file_diff(output_file, "Master", master_file_url, master_outputs, outputs_with_diffs)

            # Write the Dev version difference to file
            dev_file_url = dev_url + file_html
            write_file_diff(output_file, "Dev", dev_file_url, dev_outputs, outputs_with_diffs)

            output_file.write('---\n\n')

    return 0

if __name__ == '__main__':
    sys.exit(main())
