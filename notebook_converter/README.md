# QML Notebook to Demo Converter
This converter aids the Jupyter Notebook to QML Demo conversion. Though this converter will not completely convert
a notebook to a demo, it will get it most of the way with minor tweaks that may still require manual updates.

## Setup
`pip install -r notebook_converter/requirements.txt`

Also, ensure [pandoc](https://pandoc.org/installing.html) is installed.

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
