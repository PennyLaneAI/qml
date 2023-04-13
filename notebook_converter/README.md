# QML Notebook to Demo Converter
This converter aids the Jupyter Notebook to QML Demo conversion. Though this converter will not completely convert
a notebook to a demo, it will get it most of the way with minor tweaks that may still require manual updates.

## Setup

To use the notebook converter, you will first need to install several dependencies.
This can be done via `pip`, by using the `requirements.txt` file in this
directory:

```console
$ pip install -r notebook_converter/requirements.txt
```

In addition, you will need [Pandoc](https://pandoc.org) to be available.
You may follow the official [installation instructions](https://pandoc.org/installing.html) or use [conda](https://docs.conda.io/):

```console
$ conda install -c conda-forge pandoc
```

## Running the Converter
```bash
python3 notebook_converter/notebook_to_demo.py \
   path/to/my/notebook.ipynb \
   --author-file=path/to/author_file.yaml
```

Details on author file shown below

### Author File
The converter also generates an author block at the footer of the demo file. The information is fetched from the
author file the user provides.

```yaml
name: Author Full Name
profile_picture: path/to/author_picture.png
bio: |
  Information about the Author to put in their bio section
```

#### Options available in author file

##### `name` (required)
The display name for the Author

#### `profile_picture` (required)
The path to the display picture for the author.

#### `bio` (optional)
A small description about the author that will be displayed alongside the name and picture

#### `formatted_name` (optional)
The name that will be used to save the picture and text file for that author. Use this option if you want those
files named a certain way.

The default behavior is to use the `name` field but make it lower case, and replace `<space>, -, ', "` and any
non-ascii character with '_'. 

Example:
```yaml
name: John Doe
```
Author asset files will be named `john_doe`

However,
```yaml
name: John Doé
```
In this case,author asset files will be named `john_do_`

So the option here is to do:
```yaml
name: John Doé
formatted_name: john_doe
```

Another formatted name example:
```yaml
name: Rashid N H M  # This would by default format to `rashid_n_h_m`
formatted_name: rashid_nhm
```

### CLI Flags

#### `/path/to/notebook` (required)
The path to the notebook, if absolute path is passed, it will be used verbatim. If relative path is passed,
it must be relative to the script's location.

#### `--author-file` (required)
Path to a YAML file with information about the author. See information above to what this file has to contain.
If a relative path is used, it must be relative to the script's location, absolute path is used verbatim.

#### `--is-executable` (optional)
Indicate if the notebook is intended to be an executable demo or non-executable. If this is not passed,
the information is inferred from the notebook name. If the notebook name startswith `tutorial_` then it is
considered executable.

#### `--sphinx-gallery-dir` (optional)
The path to the sphinx-gallery-dir where the generated QML Python Demo and assets will be saved. 
This flag defaults to `../demonstrations` and in most cases do not need to be passed from cli.
If a relative path is used (like the default), it must be relative to the script's location, absolute
path is used verbatim.

#### `--authors-directory` (optional)
The path to the directory where author information is saved.
This flag defaults to `../_static/authors` and in most cases do not need to be passed from cli.
If a relative path is used (like the default), it must be relative to the script's location, absolute
path is used verbatim.
