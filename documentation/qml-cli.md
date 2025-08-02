# QML CLI Tool

## Overview

The **QML CLI tool** streamlines the creation, testing, and building of QML demonstrations. It provides an intuitive command-line interface to simplify the authoring process and maintain consistency across demonstration projects.

## Table of Contents

*   [Overview](#overview)
*   [Installation](#installation)
*   [Commands](#commands)
    *   [`help`](#help)
    *   [`new`](#new)
    *   [`sync-v2`](#sync-v2)
    *   [`build`](#build)
        *   [Build Flags](#build-flags)
*   [Viewing Build Outputs](#viewing-build-outputs)
*   [Support](#support)
*   [License](#license)

---

## Installation

To install the QML CLI, you will need Python version `3.10`. We highly recommend using a virtual environment to install the CLI tool to manage dependencies effectively.

```bash
# Step 1. Create a virtual environment (optional but highly recommended)
python3.10 -m venv .venv

# Step 2. Activate the virtual environment
source .venv/bin/activate

# Step 3. Install the QML CLI tool
# This command assumes you are in the root directory of the QML repository.
pip install .

# Step 4. Verify the installation
qml help

# Step 5. If successful, you should see output similar to this:
## > QML Demo build tool
```

---

## Commands

### `help`

Displays comprehensive help information for the QML CLI tool, including a list of available commands and their usage.

```bash
qml help
```

---

### `new`

Creates a new QML demonstration project within the `demonstrations-v2` directory. To ensure a demonstration is executable via Sphinx Gallery, its name must be prefixed with `tutorial__` (e.g., `tutorial__my_demo`).

```bash
qml new
```

Upon executing this command, you will be prompted to provide the following details interactively:

*   **Title**: A human-readable title for the demonstration.
*   **Custom directory name**: A unique directory name (used as demonstration slug on the [PL.ai](https://pennylane.ai/) website.
*   **Description**: A concise explanation of the demonstration's purpose or functionality.
*   **Author(s)**: The PennyLane handle(s) of the author(s). If there are multiple authors, you will be prompted to enter each one individually. (\*Note\*: If you do not have a PennyLane handle, you will need to [create one](https://auth.cloud.pennylane.ai/u/signup?state=hKFo2SBqQzM4RlJmNDJZdzNjX0UwbHpYYVU2a012eUlTWDZBd6Fur3VuaXZlcnNhbC1sb2dpbqN0aWTZIHZaVUM2TGhRNjVtM3YtWjhjWGZiaTc0T1ZqTW16ZWVGo2NpZNkgU1hka2hOc2lMVDBHZHJPVEZBUjJnSjV0cThvR1ZjZzM) before authoring a demonstration.)
*   **Thumbnail**: (Optional) The path to a thumbnail image. This file must be located in `/_static/demo_thumbnails/regular_demo_thumbnails/`. You may leave this field blank and add the image later.
*   **Large Thumbnail**: (Optional) The path to a larger thumbnail image, also to be placed in `/_static/demo_thumbnails/large_demo_thumbnails/`. This can be left blank and added later.

Once all required inputs are provided, a new subdirectory will be created under `demonstrations-v2`, named according to the specified demo name. This new directory will include the following essential files:

*   `demo.py`: The main Python script containing the demonstration's code.
*   `metadata.json`: A JSON file storing descriptive metadata about the demonstration.
*   `requirements.in`: A file listing Python dependencies specifically required by this demonstration.

---

### `sync-v2`

Synchronizes the demonstration project with the latest version specifications and dependencies, ensuring compatibility and up-to-date requirements.

```bash
qml sync-v2
```

---

### `build`

Compiles one or more demonstrations, preparing them for local execution or packaging.

```bash
# Provide demo names or directories separated by spaces.
# If no demo names are provided, all demonstrations within `demonstrations-v2` will be built.

# Example: Building all demonstrations
qml build

# Example: Building a specific demonstration
qml build my_demo

# Example: Building multiple demonstrations
qml build demo_one demo_two
```

#### Build Flags:

##### `--execute`

Executes the specified demos using Sphinx Gallery. This flag is effective only for demos whose directory names are prefixed with `tutorial__`, marking them as executable tutorials.

```bash
qml build --execute
```

##### `--format <html|json>`

Specifies the desired output format for the built demonstration. Options include `html` for web output or `json` for structured data.

```bash
# Output as HTML
qml build --format html demo_name

# Output as JSON
qml build --format json demo_name
```

##### `--no-dev`

Instructs the build process to use stable (release) versions of PennyLane, Catalyst, official plugins, and other core dependencies.

```bash
qml build --no-dev demo_name
```

##### `--dev`

Instructs the build process to use the latest development (unreleased) versions of PennyLane, Catalyst, official plugins, and other core dependencies.

```bash
qml build --dev demo_name
```

---

## Viewing Build Outputs

### Viewing Built Demonstrations (HTML Output)

If you have built the HTML output, you can view the demonstration in any web browser. The generated files will be located in the `_build/html/demos/<demo_name>/` directory upon successful completion of the build. Note that any changes made to the source files will require a rebuild to be reflected in the HTML output.

Example:

```bash
# Open the built demonstration in your default web browser (replace `my_demo` with your demo name)
open _build/html/demos/my_demo/my_demo.html
```

### Viewing Built Demonstrations (JSON Output)

If you have built the JSON output, you can inspect the demonstration's structured data using a JSON viewer or editor. The JSON files will be found in the `_build/json/demos/<demo_name>/` directory after the build is complete. Similar to HTML, any source changes necessitate a rebuild.

Example:

```bash
# Open the directory containing all JSON files for the demonstration.
open _build/json/demos/my_demo/
```

---

## Support

*   **Source Code:** https://github.com/PennyLaneAI/QML
*   **Issue Tracker:** https://github.com/PennyLaneAI/QML/issues

For any issues or questions, please utilize our GitHub issue tracker.

---

## License

The materials and demonstrations provided in this repository are **free** and **open source**, distributed under the Apache License, Version 2.0.

The file `custom_directives.py` is available under the BSD 3-Clause License, with Copyright (c) 2017, Pytorch contributors.