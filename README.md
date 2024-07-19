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

### Adding demos

- Demos are written in the form of an executable Python script.
  - Packages listed in `pyproject.toml` will be available for import during execution.
    See section below on `Dependency Management` for more details.
  - Matplotlib plots will be automatically rendered and displayed on the QML website.

  _Note: try and keep execution time of your script to within 10 minutes_.

- If you would like to write the demo using a Jupyter notebook, you can convert
  the notebook to the required executable Python format by following
  [these steps](https://github.com/PennyLaneAI/qml/tree/master/notebook_converter).

- All demos should have a file name beginning with `tutorial_`.
  The python files are saved in the `demonstrations` directory.

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

- Add and select an author photo from the `_static/authors` folder. The image name should be as `<author name>_<author surname>.<format>`. If this is a new author and their image is not a headshot, store the original image as `<author name>_<author surname>_original.<format>` and create a cropped headshot with the aforementioned name.
- In the same folder create a `<author name>.txt` file where to include the bio following this structure:

  ```txt
  .. bio:: <author name> <author surname>
   :photo: ../_static/authors/<author name>_<author surname>.<format>

   <author's bio>
  ```
  Note that if you want to include a middle name, it must be included in both the first and second line and in the file name.
  
- Your bio will be added at the end of the demo automatically. Don't forget to end with the following line

  ```python
  ##############################################################################
  # About the author
  # ----------------
  # 
  ```

- Lastly, your demo will need an accompanying _metadata_ file. This file should be named
  the same as your python file, but with the `.py` extension replaced with
  `.metadata.json`. Check out the `demonstrations_metadata.md` file in this repo for
  details on how to format that file and what to include.

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
  - Choose the categories your demo fits into: `"Getting Started"`, `"Optimization"`, `"Quantum Machine Learning"`, `"Quantum Chemistry"`, `"Devices and Performance"`, `"Quantum Computing"`, `"Quantum Hardware"`, `"Algorithms"` or `"How-To"`. Feel free to add more than one.
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
Due to the large scope of requirements in this repository, the traditional `requirements.txt` file is being phased out 
and `pyproject.toml` is being introduced instead, the goal being easier management in regard to adding/updating packages.

To install all the dependencies locally, [poetry](https://python-poetry.org/) needs to be installed. Please follow the
[official installation documentation](https://python-poetry.org/docs/#installation).

### Installing dependencies

Once poetry has been installed, the dependencies can be installed as follows:
```bash
make environment
```
Note: This makefile target calls `poetry install` under the hood, you can pass any poetry arguments to this by passing
the `POETRYOPTS` variable.
```bash
make environment POETRYOPTS='--sync --dry-run --verbose'
```

The `master` branch of QML uses the latest stable release of PennyLane, whereas the `dev` branch uses the most 
up-to-date version from the GitHub repository. If your demo relies on that, install the `dev` dependencies instead
by upgrading all PennyLane and its various plugins to the latest commit from GitHub.
```bash
# Run this instead of running the command above
make environment UPGRADE_PL=true
```

#### Installing only the dependencies to build the website without executing demos
It is possible to build the website without executing any of the demo code using `make html-norun` (More details below).

To install only the base dependencies without the executable dependencies, use:
```bash
make environment BASE_ONLY=true
```
(This is the equivalent to the previous method of `pip install -r requirements_norun.txt`).

### Adding / Editing dependencies

All dependencies need to be added to the pyproject.toml. It is recommended that unless necessary, 
all dependencies be pinned to as tight of a version as possible.

Add the new dependency in the `[tool.poetry.group.executable-dependencies.dependencies]` section of the toml file.

Once pyproject.toml files have been updated, the poetry.lock file needs to be refreshed:
```bash
poetry lock --no-update
```
This command will ensure that there are no dependency conflicts with any other package, and everything works.

The `--no-update` ensures existing package versions are not bumped as part of the locking process.

If the dependency change is required in prod, open the PR against `master`, or if it's only required in dev, then open
the PR against the `dev` branch, which will be synced to master on the next release of PennyLane.

#### Adding / Editing PennyLane (or plugin) versions
This process is slightly different from other packages. It is due to the fact that the `master` builds use the stable
releases of PennyLane as stated in the pyproject.toml file. However, for dev builds, we use the latest commit from 
GitHub.

##### Adding a new PennyLane package (plugin)
- Add the package to `pyproject.toml` file with the other pennylane packages and pin it to the latest stable release.
- Add the GitHub installation link to the Makefile, so it is upgraded for dev builds with the other PennyLane packages.
    - This should be under the format `$$PYTHON_VENV_PATH/bin/python -m pip install --upgrade git+https://github.com/PennyLaneAI/<repo>.git#egg=<repo>;\`
- Refresh the poetry lock file by running `poetry lock`

## Building

To build the website locally, simply run `make html`. The rendered HTML files
will now be available in `_build/html`. Open `_build/html/index.html` to browse
the built site locally.

Note that the above command may take some time, as all demos
will be executed and built! Once built, only _modified_ demos will
be re-executed/re-built.

Alternatively, you may run `make html-norun` to build the website _without_ executing
demos, or build only a single demo using the following command:

```console
sphinx-build -D sphinx_gallery_conf.filename_pattern=tutorial_QGAN\.py -b html . _build
```

where `tutorial_QGAN` should be replaced with the name of the demo to build.

## Building and running locally on Mac (M1)

To install dependencies on an M1 Mac and build the QML website, the following instructions may be useful.

- If python3 is not currently installed, we recommend you install via [Homebrew](https://github.com/conda-forge/miniforge):

  ```bash
  brew install python
  ```

- Follow the steps from `Dependency Management` to setup poetry.

- Install the base packages by running

  ```bash
  make environment BASE_ONLY=true
  ```

  Alternatively, you can do this in a new virtual environment using

  ```bash
  python -m venv [venv_name]
  cd [venv_name] && source bin/activate
  make environment BASE_ONLY=true
  ```

Once this is complete, you should be able to build the website using `make html-norun`. If this succeeds, the `build` folder should be populated with files. Open `index.html` in your browser to view the built site.

If you are running into the error message

```
command not found: sphinx-build
```

you may need to make the following change:

- In the `Makefile` change `SPHINXBUILD = sphinx-build` to `SPHINXBUILD = python3 -m sphinx.cmd.build`.

If you are running into the error message

```
ModuleNotFoundError: No module named 'the-module-name'
```

you may need to install the module manually:

```
pip3 install the-module-name
```

## Support

- **Source Code:** https://github.com/PennyLaneAI/QML
- **Issue Tracker:** https://github.com/PennyLaneAI/QML/issues

If you are having issues, please let us know by posting the issue on our GitHub issue tracker.

We are committed to providing a friendly, safe, and welcoming environment for all.
Please read and respect the [Code of Conduct](.github/CODE_OF_CONDUCT.md).

## License

The materials and demos in this repository are **free** and
**open source**, released under the Apache License, Version 2.0.

The file `custom_directives.py` is available under the BSD 3-Clause License with
Copyright (c) 2017, Pytorch contributors.
