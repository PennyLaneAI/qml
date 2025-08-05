# Contributing to PennyLane Demos

This document provides comprehensive guidelines for contributing to the PennyLane demonstrations repository. It covers the entire process, from setting up your environment and creating new demos to managing dependencies and building your contributions for review.


## Table of Contents

- [Getting Started](#getting-started)
  - [Cloning/Forking the Repository](#cloningforking-the-repository)
  - [Creating a New Demo](#creating-a-new-demo)
    - [Using the QML CLI Tool](#using-the-qml-cli-tool)
    - [Manually Creating a Demo](#manually-creating-a-demo)
- [General Guidelines](#general-guidelines)
- [Image Guidelines](#image-guidelines)
  - [File Size and Format](#file-size-and-format)
  - [Dimensions](#dimensions)
- [Metadata Guidelines](#metadata-guidelines)
  - [Example `metadata.json` Structure](#example-metadatajson-structure)
  - [Validate your metadata file](#validate-your-metadata-file)
- [Dependency Management](#dependency-management)
- [Building and Testing Locally](#building-and-testing-locally)
- [Support](#support)
- [License](#license)


## Getting Started

### Cloning/Forking the Repository

To contribute to the PennyLane demonstrations, begin by forking and cloning the QML repository. All contributions should be made by opening a pull request against the `master` branch to use stable versions or `dev` branch to use the latest features from PennyLane, Catalyst, and other plugins.

### Creating a New Demo

There are multiple ways to create a new demo. The recommended method is to use the QML CLI tool. This tool provides a structured way to set up your demo environment and ensures that all necessary files are created.

#### Using the QML CLI Tool

To create a new demonstration, use the QML CLI tool's [`new` command](/documentation/qml-cli.md#new). This command will guide you through the initial setup. You will be prompted to provide a title, a custom directory name, a description, and the author's username. Optionally, you can add thumbnail images for the demo.

To make a demo executable, ensure that the directory name starts with `tutorial_` (legacy), or set the `executable_stable` or `executable_latest` flag to `true` in the metadata.json file.

```
## Example interaction with the `qml new` command:
❯ qml new
Title: Your demo title
Custom directory name [your_demo_title]: your_demo_directory_name
Description []: A description of the demo you are creating
Author's pennylane.ai username: author_username
Would you like to add another author? [y/N]: n
Thumbnail image [_static/demo_thumbnails/regular_demo_thumbnails/thumbnail_placeholder.png]: 
Large thumbnail image []: 
```

#### Manually Creating a Demo
If you prefer to set up your demo manually, follow these steps:

1. **Create a New Directory:** Create a new directory within the `demonstrations_v2` folder. The directory name must start with `tutorial_` unless the metadata includes either the `executable_stable` or `executable_latest` flag set to `true`, as required. All demos are saved in the `demonstrations_v2` directory. For example: `demonstrations_v2/tutorial_my_demo`.

2. **Add Required Files:** At a minimum, your demo directory should contain:
   - `demo.py`: The main executable Python script for your demo.
   - `metadata.json`: A JSON file containing metadata about your demo (see [Metadata Guidelines](#metadata-guidelines) for details).
   - `requirements.in` (optional): A file listing any additional dependencies required by your demo (see [Dependency Management](#dependency-management) for details).

### General Guidelines

- **Format:** Demos are written as executable Python scripts.
- **Jupyter Notebook Conversion:** If you prefer writing your demo in a Jupyter notebook, you can convert it to the required executable Python format by following the [QML Notebook to Demo Converter](https://github.com/PennyLaneAI/qml/tree/master/notebook_converter).
- **Naming Convention:** All demo directories must begin with `tutorial_`. They are saved in the `demonstrations_v2` directory. For example: `demonstrations_v2/tutorial_my_demo`.
- **Frameworks:** New demos should avoid using `autograd` or `TensorFlow`. `JAX` and `PyTorch` are recommended instead. Whenever possible, the use of `lightning.qubit` is also encouraged.
- **Restructured Text (ReST):** Restructured Text sections can be included anywhere within the script by beginning the comment with 79 hash characters (`#`). These are useful for breaking up large code blocks and providing extensive explanations.
- **LaTeX Macros:** Avoid using LaTeX macros within your comments. Even if they appear to work in development, they will not be displayed correctly once the demo is published.
- **Author Information:** Include the author's information in the metadata.json file. See the metadata guidelines below.
- **Code Formatting:** Before submitting, run your script through the [Black Python formatter](https://github.com/psf/black):

    ```bash
    pip install black
    
    black -l 100 /demonstrations_v2/tutorial_your_demo/demo.py
    ```
### Image Guidelines

#### File Size and Format

- Always optimize images for web use. Aim for a file in kilobytes (KB) rather than megabytes (MB).
- Use `.png` for all static images (decorative, descriptive, logos, etc.).
- Use `.gif` for animated images.

#### Dimensions

- To maintain quality and performance, every image should be twice (2X) its visible dimension size on the web page, and at a minimum of `150 ppi/dpi` (preferably `300 ppi/dpi`).

### Metadata Guidelines

Every demo requires an accompanying `metadata.json` file located in its directory. This JSON file stores crucial information about the demo. Refer to [this example](https://github.com/PennyLaneAI/qml/blob/master/demonstrations_v2/tutorial_here_comes_the_sun/metadata.json) for the required fields:

- **Filename:** Ensure the file is named `metadata.json`.
- **Authors:** Ensure that the `"authors"` field is populated with the `"username"` of the author(s) as registered on pennylane.ai. This should match the username provided when creating the bio.
- **`executable_stable` and `executable_latest`:** Set either of these fields to `true` if your demo is executable with the stable or latest versions of PennyLane, respectively. If neither field is set to `true`, the demo directory name must start with `tutorial_` to be executable.
- **Dates:** Leave publication and modification dates empty if you do not know them or use UTC format (e.g., `"2023-10-01T12:00:00Z"`).
- **Categories:** Choose relevant categories for your demo, such as: `"Getting Started"`,
- **`previewImages`:** Simply modify the final part of the image file names to match your demo's name. These two images will be provided to you once the review process begins. After receiving them, you must upload them to the address indicated in the metadata.
- **`relatedContent`:** This field refers to demos related to yours. You will need to include the corresponding `id` and set the `weight` to `1.0`.
- **Questions:** If you have any doubts about a specific field, do not hesitate to post a comment for your demo's reviewer.

##### Example `metadata.json` Structure

```json
{
    "title": "Your Demo Title",
    "description": "A brief description of your demo.",
    "authors": [
        {
            "username": "author_username"
        }
    ],
    "executable_stable": true,
    "executable_latest": true,
    "dateOfPublication": "",
    "dateOfLastModification": "",
    "categories": [
        "Quantum Computing"
    ],
    "tags": [],
    "previewImages": [
        "_static/demonstration_assets/your_demo_name/your_demo_name_thumbnail.png",
        "_static/demonstration_assets/your_demo_name/your_demo_name_large_thumbnail.png"
    ],
    "relatedContent": [
        {
            "id": "related_demo_id",
            "weight": 1.0
        }
    ],
    "basedOnPapers": [
        "10.48550/arXiv.XXXX.XXXXX"
    ],
     "basedOnPapersDetailed": [
        {
            "doi": "10.48550/arXiv.XXXX.XXXXX",
            "title": "Title of the paper",
            "authors": "Author One, Author Two, Author Three",
            "year": "2023",
            "url": "https://arxiv.org/pdf/XXXX.XXXXX"
        }
    ],
    "referencedByPapers": [],
    "referencesAndFurtherReading": []
}
```

#### Validate your metadata file:

```bash
pip install check-jsonschema 'jsonschema[format]'
check-jsonschema \
  --schemafile metadata_schemas/demo.metadata.schema.<largest_number>.json \
  demonstrations_v2/<your_demo_name>/metadata.json
```

Once your script and metadata are ready, you can submit a pull request!


## Dependency Management

Demo dependencies are automatically installed by the `qml` tool during demo execution. For detailed information on dependency specifications, refer to [dependencies/README.md](./dependencies/README.md). A `requirements.txt` file will be created in the demo directory after a successful build.

## Building and Testing Locally

To build and execute demos locally, use the following `qml` command:

```bash
qml build --execute <demo_name(s)>
```

### To build demos using development dependencies:

```bash
qml build --execute --dev <demo_name>
```

For more details on building options, run `qml build --help`.

You can view the built demo in your browser by navigating to the URL: `_build/html/demos/<name of your tutorial>.html`. This requires you to first build the HTML output.

## Support

- **Source Code:** https://github.com/PennyLaneAI/QML
- **Issue Tracker:** https://github.com/PennyLaneAI/QML/issues

If you encounter any issues, please report them on our GitHub issue tracker.

We are committed to providing a friendly, safe, and welcoming environment for all contributors. Please read and respect the [Code of Conduct](/.github/CODE_OF_CONDUCT.md).

## License

The materials and demos in this repository are **free** and **open source**, released under the Apache License, Version 2.0.

Please note, the file `custom_directives.py` is available under the BSD 3-Clause License, with copyright © 2017, PyTorch contributors.

