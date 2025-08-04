# Contributing to Demos in QML

This document provides comprehensive guidelines for contributing to the QML demonstrations repository. It covers the entire process, from setting up your environment and creating new demos to managing dependencies and building your contributions for review.

## Table of Contents

*   [Getting Started](#getting-started)
*   [Creating and Adding a New Demo](#creating-and-adding-a-new-demo)
    *   [General Guidelines](#general-guidelines)
    *   [Image Guidelines](#image-guidelines)
    *   [Metadata Guidelines](#metadata-guidelines)
    *   [Tutorial Content Guidelines](#tutorial-content-guidelines)
*   [Dependency Management](#dependency-management)
*   [Building and Testing Locally](#building-and-testing-locally)
*   [Support](#support)
*   [License](#license)

---

## Getting Started

To contribute to the QML demonstrations, begin by forking and cloning the QML repository. All contributions should be made by opening a pull request against the `master` branch.

Ensure you have the necessary tools installed, including Python and the QML CLI tool. Refer to the [QML CLI README for installation instructions](/documentation/qml-cli.md#installation). Once installed, you can create a new demo using the [`qml new` command](#creating-and-adding-a-new-demo).

## Creating and Adding a New Demo

To create a new demonstration, use the QML CLI tool's [`new` command](/documentation/qml-cli.md#new). This command will guide you through the initial setup.

### General Guidelines

*   **Format:** Demos are written as executable Python scripts.
    *   Packages listed in `dependencies/requirements-core.in` are available to all demos by default. Additional dependencies can be specified in a `requirements.in` file within the demo's directory. See [Dependency Management](#dependency-management) for more details.
    *   Matplotlib plots will be automatically rendered and displayed on the QML website.
    *   **Execution Time:** Aim to keep the execution time of your script under 10 minutes.
*   **Jupyter Notebook Conversion:** If you prefer writing your demo in a Jupyter notebook, you can convert it to the required executable Python format by following the [QML Notebook to Demo Converter](https://github.com/PennyLaneAI/qml/tree/master/notebook_converter).
*   **Naming Convention:** All demo directories must begin with `tutorial_`. They are saved in the `demonstrations_v2` directory. For example: `demonstrations_v2/tutorial_my_demo`.
*   **Frameworks:** New demos should avoid using `autograd` or `TensorFlow`. `Jax` and `torch` are recommended instead. Whenever possible, the use of `lightning.qubit` is also encouraged.
*   **Restructured Text (ReST):** Restructured Text sections can be included anywhere within the script by beginning the comment with 79 hash characters (`#`). These are useful for breaking up large code blocks and providing extensive explanations.
*   **LaTeX Macros:** Avoid using LaTeX macros within your comments. Even if they appear to work in development, they will not be displayed correctly once the demo is published.
*   **Author Information:** Include the author's information in the `.metadata.json` file. This can be either the author's PennyLane profile `username` or their `name`. Providing the PennyLane profile username allows author details to be sourced directly from that profile, and the demo will then appear as a contribution on the author's profile.
*   **End Marker:** Always conclude your demo script with the following line:

    ```python
    ##############################################################################
    # About the author
    # ----------------
    #
    ```

*   **Code Formatting:** Before submitting, run your script through the [Black Python formatter](https://github.com/psf/black):

    ```bash
    pip install black
    black -l 100 demo_new.py
    ```

### Image Guidelines

You may add figures within ReST comments using the following syntax:

```python
##############################################################################
#.. figure:: ../_static/demonstration_assets/<demo name>/image.png
#    :align: center
#    :width: 90%
```

Here, `<demo name>` refers to a subdirectory within `_static/demonstration_assets/` named after your demo.

<details>
  <summary><b>Follow these standards when adding images:</b></summary>

  ### File Size

  *   Always aim to keep image file sizes in kilobytes (KB).
  *   Always compress images to the best possible size where quality remains acceptable.

  ### Formats

  *   Use `.png` for all static images (decorative, descriptive, logos, etc.).
  *   Use `.gif` for animated images.

  ### Dimensions

  *   To maintain quality and performance, every image should be twice (2X) its visible dimension size on the web page, and at a minimum of `150 ppi/dpi` (preferably `300 ppi/dpi`).
</details>
<br>

### Metadata Guidelines

Every demo requires an accompanying `metadata.json` file located in its directory. This JSON file stores crucial information about the demo. Refer to [this example](https://github.com/PennyLaneAI/qml/blob/master/demonstrations/tutorial_here_comes_the_sun.metadata.json) for the required fields:

*   **Filename:** Ensure the file is named `<name_of_your_tutorial>.metadata.json`.
*   **Author ID:** The "id" of the author should match the one chosen when creating the bio.
*   **Dates:** Leave publication and modification dates empty if you do not know them.
*   **Categories:** Choose relevant categories for your demo, such as: `"Getting Started"`, `"Optimization"`, `"Quantum Machine Learning"`, `"Quantum Chemistry"`, `"Devices and Performance"`, `"Quantum Computing"`, `"Quantum Hardware"`, `"Algorithms"`, or `"How-to"`. You may select multiple categories.
*   **`previewImages`:** Simply modify the final part of the image file names to match your demo's name. These two images will be provided to you once the review process begins. After receiving them, you must upload them to the address indicated in the metadata.
*   **`relatedContent`:** This field refers to demos related to yours. You will need to include the corresponding `id` and set the `weight` to `1.0`.
*   **Questions:** If you have any doubts about a specific field, do not hesitate to post a comment for your demo's reviewer.

**Validate your metadata file:**

```bash
pip install check-jsonschema 'jsonschema[format]'
check-jsonschema \
  --schemafile metadata_schemas/demo.metadata.schema.<largest_number>.json \
  demonstrations/<your_demo_name>.metadata.json
```

Once your script and metadata are ready, you can submit a pull request!

### Tutorial Content Guidelines

While you are encouraged to be creative with your demo, please keep the following guidelines in mind:

*   **License:** All contributions must be made under the Apache 2.0 license.
*   **Title:** The title should be clear and concise. If based on a research paper, the title should be similar to the paper being implemented.
*   **Summary:** All demos should include a 1-3 sentence summary below the title. This summary should clearly state the goal and outcome of the demo and link to any relevant papers or resources used.
*   **Code Clarity:** Code should be clearly commented and explained, either through ReST-formatted comments or standard Python comments.
*   **Reproducibility:** If your content involves random variables or outputs, a fixed seed should be set for reproducibility.
*   **Originality:** All content must be original or free to reuse subject to license compatibility. For example, if you are implementing someone else's research, obtain permission before reproducing exact figures. Otherwise, avoid direct screenshots from papers and instead refer to figures in the paper within the text.
*   **Code Review:** All submissions must pass code review before being merged into the repository.
*   **Branching:** If your demo uses the latest stable release of PennyLane, submit your PR against the `master` branch. If it requires cutting-edge development versions of PennyLane or relevant plugins, submit your PR against the `dev` branch instead.
*   **Privacy:** By submitting your demo, you consent to our [Privacy Policy](https://pennylane.ai/privacy/).

## Dependency Management

Demo dependencies are automatically installed by the `qml` tool during demo execution. For detailed information on dependency specifications, refer to [dependencies/README.md](./dependencies/README.md). A `requirements.txt` file will be created in the demo directory after a successful build.

## Building and Testing Locally

To build and execute demos locally, use the following `qml` command:

```bash
qml build --execute <demo_name(s)>
```

To build demos using development dependencies:

```bash
qml build --execute --dev <demo_name>
```

For more details on building options, run `qml build --help`.

You can view the built demo in your browser by navigating to the URL: `_build/html/demos/<name of your tutorial>.html`. This requires you to first build the HTML output.

## Support

*   **Source Code:** https://github.com/PennyLaneAI/QML
*   **Issue Tracker:** https://github.com/PennyLaneAI/QML/issues

If you encounter any issues, please report them on our GitHub issue tracker.

We are committed to providing a friendly, safe, and welcoming environment for all contributors. Please read and respect the [Code of Conduct](.github/CODE_OF_CONDUCT.md).

## License

The materials and demos in this repository are **free** and **open source**, released under the Apache License, Version 2.0.

Please note, the file `custom_directives.py` is available under the BSD 3-Clause License, with copyright © 2017, PyTorch contributors.