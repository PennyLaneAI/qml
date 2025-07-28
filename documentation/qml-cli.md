# QML CLI Tool

## Overview

The **QML CLI tool** streamlines the creation, testing, and building of QML demonstrations. It provides an intuitive command-line interface to simplify the authoring process and maintain consistency across demonstration projects.

## Installation
To install the QML CLI you will need to have python version `3.10`. We reccoment using a virtual environment to install the CLI tool.

```bash
# Step 1. Create a virtual environment (optional but recommended)
python3.10 -m venv .venv

# Step 2. Activate the virtual environment
source .venv/bin/activate

# Step 3. Install the QML CLI tool
pip install .

# Step 4. Verify the installation
qml help

# Step 5. If successful, you should see the following output:
## > QML Demo build tool
```

## Commands

### `help`

Displays help information for the QML CLI tool.

```bash
qml help
```

---

### `new`

Creates a new QML demonstration in the `demonstrations-v2` directory. To make a demonstration executable, prefix its name with `tutorial__`. For example, `tutorial__my_demo`.

```bash
qml new
```

Upon execution, you will be prompted to provide the following details:

*   **Title**: A human-readable title for the demonstration.
*   **Name**: A unique identifier for the demo, using `snake_case` formatting.
*   **Description**: A brief explanation of the demonstration's purpose or functionality.
*   **Author(s)**: The PennyLane handle(s) of the author(s). If there are multiple authors, you will be prompted to enter them one at a time. (\*Note\*: If you don't have a PennyLane handle, you will need to [create one](https://auth.cloud.pennylane.ai/u/signup?state=hKFo2SBqQzM4RlJmNDJZdzNjX0UwbHpYYVU2a012eUlTWDZBd6Fur3VuaXZlcnNhbC1sb2dpbqN0aWTZIHZaVUM2TGhRNjVtM3YtWjhjWGZiaTc0T1ZqTW16ZWVGo2NpZNkgU1hka2hOc2lMVDBHZHJPVEZBUjJnSjV0cThvR1ZjZzM) before authoring a demonstration.)
*   **Thumbnail**: (Optional) Path to a thumbnail image. The file must be located in `/_static/demo_thumbnails/`. You may leave this blank and add the image later.
*   **Large Thumbnail**: (Optional) Path to a larger thumbnail image, also to be placed in `/_static/demo_thumbnails/`. This can be left blank and added later.

Once all inputs are provided, a new subdirectory will be created under `demonstrations-v2`, named according to the provided demo name. The directory will include the following files:

*   `demo.py`: The main script for the demonstration.
*   `metadata.json`: A metadata file describing the demonstration.
*   `requirements.in`: A list of Python dependencies specific to the demo.

---

### `sync-v2`

Synchronizes the demonstration project with the latest version specifications and dependencies.

```bash
qml sync-v2
```

---

### `build`

Compiles the demonstration(s), preparing them for execution or packaging.

```bash
# Demo names/directories should be separated by spaces.
# If no demo names are provided, all demos in `demonstrations-v2` will be built.

# Example of building all demonstrations:
qml build

# Example of building a specific demonstration:
qml build my_demo

# Example of building multiple demonstrations:
qml build demo_one demo_two
```

#### Build Flags:

##### `--execute`

Executes demos using Sphinx Gallery. This option only applies to demos flagged as executable via the `tutorial__` prefix in their directory name.

```bash
qml build --execute
```

##### `--format <html|json>`

Specifies the output format for the built demonstration: either `html` or `json`.

```bash
# Output as HTML
qml build --format html demo_name

# Output as JSON
qml build --format json demo_name
```

##### `--no-dev`

Uses the stable versions of PennyLane, Catalyst, Plugins, and various other dependencies.

```bash
qml build --no-dev demo_name
```

##### `--dev`

Uses the development versions of PennyLane, Catalyst, Plugins, and various other dependencies.

```bash
qml build --dev demo_name
```

---

## Viewing Build Outputs

### Viewing Built Demonstrations (HTML Output)

If building the HTML output, you can view the demonstration in a web browser. The built files will be located in the `_build/html/demos/<demo_name>/` directory once the build has successfully completed. Every change will require a rebuild to reflect updates.

Example:

```bash
# Open the built demonstration in a web browser (replace `my_demo` with your demo name)
open _build/html/demos/my_demo/my_demo.html
```

### Viewing Built Demonstrations (JSON Output)

If building the JSON output, you can view the demonstration in a JSON viewer or editor. The built files will be located in the `_build/json/demos/<demo_name>/` directory once the build has successfully completed. Every change will require a rebuild to reflect updates.

Example:

```bash
# Open the directory containing all the JSON files for the demonstration.
open _build/json/demos/my_demo/
```