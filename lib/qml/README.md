# EXPERIMENTAL QML V2 README

<p align="center">
  <a href="https://pennylane.ai/qml">
    <img width=60% src="https://raw.githubusercontent.com/PennyLaneAI/qml/master/_static/readme_header.png">
  </a>
</p>

<p align="center">
  <a href="https://github.com/PennyLaneAI/qml/actions?query=workflow%3Abuild-master">
    <img src="https://img.shields.io/github/actions/workflow/status/PennyLaneAI/qml/build-branch-master.yml?label=master&logo=github&style=flat-square" />
  </a>
  <a href="https://github.com/PennyLaneAI/qml/actions?query=workflow%3Abuild-dev">
    <img src="https://img.shields.io/github/actions/workflow/status/PennyLaneAI/qml/build-branch-dev.yml?label=dev&logo=github&style=flat-square" />
  </a>
  <img src="https://img.shields.io/badge/contributions-welcome-orange?style=flat-square"/>
</p>

This repository contains materials on Quantum Machine Learning and other quantum computing topics, as well as Python code
demos using [PennyLane](https://pennylane.ai), a cross-platform Python library for [differentiable
programming](https://en.wikipedia.org/wiki/Differentiable_programming) of quantum computers.

<a href="https://pennylane.ai/qml">
  <img src="https://raw.githubusercontent.com/PennyLaneAI/qml/master/_static/readme_panel_new.png" width="900px">
</a>

The content here will be presented in the form of [tutorial, demos and how-to's](https://pennylane.ai/qml/demonstrations/). Take a dive into quantum
  computing with fully-coded implementations of major works.

Explore these materials on our website: https://pennylane.ai. All tutorials are fully executable,
and can be downloaded as Jupyter notebooks and Python scripts.

## Contributing

You can contribute by submitting a demo via a pull request implementing a recent
quantum computing paper/result.


### Installing the QML tool.

The `qml` command line tool can be installed by running `pip install .` in the repo root.

### Adding demos

- Run `qml new` to create a new demo interactively.

- Demos are written in the form of an executable Python script.
  - Packages listed in `dependencies/requirements-core.in` will be available to all demos by default.
    Extra dependencies can be named using a `requirements.in` file in the demo directory.
    See section below on `Dependency Management` for more details.
  - Matplotlib plots will be automatically rendered and displayed on the QML website.

  _Note: try and keep execution time of your script to within 10 minutes_.

- If you would like to write the demo using a Jupyter notebook, you can convert
  the notebook to the required executable Python format by following
  [these steps](https://github.com/PennyLaneAI/qml/tree/master/notebook_converter).

- All demos should have a file name beginning with `tutorial_`.
  The python files are saved in the `demonstrations_v2/{demo_name}` directory.

- The new demos will avoid using `autograd` or `TensorFlow`, `Jax` and `torch` are recommended instead. Also, if possible, the use of `lightning.qubit` is recommended. 
- [Restructured Text](http://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html)
  sections may be anywhere within the script by beginning the comment with
  79 hashes (`#`). These are useful for breaking up large code-blocks.

- Avoid the use of LaTeX macros. Even if they work in the deployment, they will not be displayed once the demo is published.
- You may add figures within ReST comments by using the following syntax:

  ```python
  ##############################################################################
  #.. figure:: ../_static/demonstration_assets/<demo name>/image.png
  #    :align: center
  #    :width: 90%
  ```

  where `<demo name>` is a sub-directory with the name of
  your demo.
  <details>
    <summary><b>Follow these standards when adding images</b></summary>

    ### File Size
    * Always aim to keep the image file size in kilobytes (KB’s) 
    * Always compress the image to the best possible size where quality is acceptable.
    
    ### Formats
    * Use `.png` for everything (decorative images, descriptive images, logos etc)
    * Use `.gif` for animated images

    ### Dimensions
    * To maintain quality and performance, every image should be twice (2X) its visible dimension size on the web page, and at a minimum of `150 ppi/dpi` (preferably `300 ppi/dpi`).
  </details><br>

- Lastly, your demo will need an accompanying _metadata_ file in the demo directory.

- Include the author's information in the `.metadata.json` file. This can be either the author's PennyLane profile `username` or their `name`. If you provide the PennyLane profile username, the author details will be sourced directly from that profile, and the demo will then appear as a contribution on the author's profile.
  
- Don't forget to end with the following line

  ```python
  ##############################################################################
  # About the author
  # ----------------
  # 
  ```

- At this point, run your script through the [Black Python formatter](https://github.com/psf/black),

  ```bash
  pip install black
  black -l 100 demo_new.py
  ```

- Finally, add the metadata. The metadata is a `json` file in which we will store information about the demo.
  In [this example](https://github.com/PennyLaneAI/qml/blob/master/demonstrations/tutorial_here_comes_the_sun.metadata.json) you will see the fields you need to fill in.
  - Make sure the file name is `<name of your tutorial>.metadata.json`.
  - The "id" of the author will be the same as the one you chose when creating the bio. 
  - The date of publication and modification. Leave them empty in case you don't know them.
  - Choose the categories your demo fits into: `"Getting Started"`, `"Optimization"`, `"Quantum Machine Learning"`, `"Quantum Chemistry"`, `"Devices and Performance"`, `"Quantum Computing"`, `"Quantum Hardware"`, `"Algorithms"` or `"How-to"`. Feel free to add more than one.
  - In `previewImages` you should simply modify the final part of the file's name to fit the name of your demo. These two images will be sent to you once the review process begins. Once sent, you must upload them to the address indicated in the metadata.
  - `relatedContent` refers to the demos related to yours. You will have to put the corresponding id and set the `weight` to `1.0`. 
  - If there is any doubt with any field, do not hesitate to post a comment to the reviewer of your demo. 

  Don't forget to validate your metadata file as well.

  ```bash
  pip install check-jsonschema 'jsonschema[format]'
  check-jsonschema \
    --schemafile metadata_schemas/demo.metadata.schema.<largest_number>.json \
    demonstrations/<your_demo_name>.metadata.json
  ```

  and you are ready to submit a pull request!


In order to see the demo on the deployment, you can access through the url. For this, once deployed, you should change `index.html` to `demos/<name of your tutorial>.html` in the url. 
If your demo uses the latest release of PennyLane, simply make your PR against the
`master` branch. If you instead require the cutting-edge development versions of
PennyLane or any relevant plugins, make your PR against the `dev` branch instead.

By submitting your demo, you consent to our [Privacy Policy](https://pennylane.ai/privacy/).

#### Tutorial guidelines

While you are free to be as creative as you like with your demo,
there are a couple of guidelines to keep in mind.

- All contributions must be made under the Apache 2.0 license.

- The title should be clear and concise, and if based on a paper it should be similar to the paper
  that is being implemented.

- All demos should include a summary below the title.
  The summary should be 1-3 sentences that makes clear the
  goal and outcome of the demo, and links to any papers/resources used.

- Code should be clearly commented and explained, either
  as a ReST-formatted comment or a standard Python comment.

- If your content contains random variables/outputs, a fixed seed should
  be set for reproducibility.

- All content must be original or free to reuse subject to license compatibility.
  For example, if you are implementing someone else's research, reach out first to
  recieve permission to reproduce exact figures. Otherwise, avoid direct screenshots
  from papers, and instead refer to figures in the paper within the text.

- All submissions must pass code review before being merged into the repository.

## Dependency Management

Demo dependencies are automatically installed by the `qml` tool during demo
execution. See [dependencies/README.md](./dependencies/README.md) for details
on dependency specifications.    

### Installing dependencies

Dependencies are automatically installed when executing demos. A `requirements.txt` file
will be created in the `demo` directory after the build.

## Building

To build and execute demos locally, use `qml build --execute {demo_name} ...`.

To build demos using dev dependencies, use `qml build --execute --dev {demo_name}`.

Run `qml build --help` for more details.

## Support

- **Source Code:** https://github.com/PennyLaneAI/QML
- **Issue Tracker:** https://github.com/PennyLaneAI/QML/issues

If you are having issues, please let us know by posting the issue on our GitHub issue tracker.

We are committed to providing a friendly, safe, and welcoming environment for all.
Please read and respect the [Code of Conduct](/.github/CODE_OF_CONDUCT.md).

## License

The materials and demos in this repository are **free** and
**open source**, released under the Apache License, Version 2.0.

The file `custom_directives.py` is available under the BSD 3-Clause License with
Copyright (c) 2017, Pytorch contributors.