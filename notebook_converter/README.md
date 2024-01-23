# QML Notebook to Demo Converter
This converter aids the Jupyter Notebook to QML Demo conversion. Though this converter will not completely convert
a notebook to a demo, it will get it most of the way with minor tweaks that may still require manual updates.

## Setup

To use the notebook converter, you will first need to install several dependencies.
This can be done via `pip`, by using the `requirements.txt` file in this
directory:

```bash
pip install -r notebook_converter/requirements.txt
```

In addition, you will need [Pandoc](https://pandoc.org) to be available.
You may follow the official [installation instructions](https://pandoc.org/installing.html) or use [conda](https://docs.conda.io/):

```bash
conda install -c conda-forge pandoc
```

## Running the Converter
```bash
python3 notebook_converter/notebook_to_demo.py \
   path/to/my/notebook.ipynb \
   --author "John Doe" "Some info about author" "some/path/to/profile/picture/john_doe.png"
```

### If the notebook is not going to be executable by sphinx-build
For non-executable demos, first ensure that all the executable code has been run locally and all the output is 
available in the notebook.

After that, you can indicate if the notebook is not executable by setting `--is-executable` for False

```bash
python3 notebook_converter/notebook_to_demo.py \
   path/to/my/notebook.ipynb \
   --author "John Doe" "Some info about author" "some/path/to/profile/picture/john_doe.png"
   --is-executable=False
```

The `is-executable` can be explicitly passed to indicate if a notebook should be an executable demo or not.
If it is omitted, then this information is determined based on the notebook name. If the notebook name starts with
`tutorial_` then it is treated as executable, otherwise it is treated as not executable.

### CLI Flags

#### `/path/to/notebook` (required)
The path to the notebook, if absolute path is passed, it will be used verbatim. If relative path is passed,
it must be relative to the script's location.

#### `--author` (optional)
Information about the author, in the syntax of `--author "Full Name" "Bio" "path/to/profile_picture.png"`

For demos with multiple authors, then this flag can be passed multiple times:
```bash
python3 notebook_converter/notebook_to_demo.py \
   path/to/my/notebook.ipynb \
   --author "John Doe" "Some info about author" "some/path/to/profile/picture/john_doe.png" \
   --author "Jane Doe" "Some info about author" "some/path/to/profile/picture/jane_doe.png" \
   --is-executable=False
```

#### `--author-file` (optional)
If the notebook being converted is being used by an existing author, this option can be used to pass the location of the existing author-file.

```bash
--author-file "qml/_static/authors/john_doe.txt"
```

Similar to `--author`, this option can be passed multiple times. It can also be passed alongside `--author`

```bash
python3 notebook_converter/notebook_to_demo.py \
   path/to/my/notebook.ipynb \
   --author "John Doe" "Some info about author" "some/path/to/profile/picture/john_doe.png" \
   --author "Jane Doe" "Some info about author" "some/path/to/profile/picture/jane_doe.png" \
   --author-file "path/to/bob_doe.txt" \
   --author-file "path/to/rob_doe.txt" \
   --is-executable=False
```

#### `--is-executable` (True|False) (optional)
Indicate if the notebook is intended to be an executable demo or non-executable. If this is not passed,
the information is inferred from the notebook name. If the notebook name startswith `tutorial_` then it is
considered executable.

