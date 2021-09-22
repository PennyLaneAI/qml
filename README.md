<p align="center">
  <a href="https://pennylane.ai/qml">
    <img width=60% src="https://raw.githubusercontent.com/PennyLaneAI/qml/master/_static/readme_header.png">
  </a>
</p>

<p align="center">
  <a href="https://github.com/PennyLaneAI/qml/actions?query=workflow%3Abuild-master">
    <img src="https://img.shields.io/github/workflow/status/PennyLaneAI/qml/build-master?label=master&logo=github&style=flat-square" />
  </a>
  <a href="https://github.com/PennyLaneAI/qml/actions?query=workflow%3Abuild-dev">
    <img src="https://img.shields.io/github/workflow/status/PennyLaneAI/qml/build-dev?label=dev&logo=github&style=flat-square" />
  </a>
  <img src="https://img.shields.io/badge/contributions-welcome-orange?style=flat-square"/>
</p>

This repository contains introductory materials on Quantum Machine Learning, as well as Python code
demos using [PennyLane](https://pennylane.ai), a cross-platform Python library for [differentiable
programming](https://en.wikipedia.org/wiki/Differentiable_programming) of quantum computers.

<a href="https://pennylane.ai/qml">
  <img src="https://raw.githubusercontent.com/PennyLaneAI/qml/master/_static/readme_panel1.png" width="900px">
</a>

The content consists of four main areas:

* [What is quantum machine learning?](https://pennylane.ai/qml/whatisqml.html) Understand what
  quantum computing means for machine learning.

* [QML tutorials and demos](https://pennylane.ai/qml/demonstrations.html). Take a dive into quantum
  machine learning with fully-coded implementations of major works.

* [Key concepts of QML](https://pennylane.ai/qml/glossary.html). A glossary of key ideas for
  quantum machine learning and optimization.

* [QML videos](https://pennylane.ai/qml/videos.html). A selection of curated expert videos
  discussing various aspects of quantum machine learning.

<a href="https://pennylane.ai/qml/demonstations.html">
<img src="https://raw.githubusercontent.com/PennyLaneAI/qml/master/_static/readme_panel3.png" width="900px">
</a>

Explore these materials on our website: https://pennylane.ai/qml. All tutorials are fully executable,
and can be downloaded as Jupyter notebooks and Python scripts.

## Contributing

You can contribute by submitting a demo via pull request implementing a recent
quantum machine learning paper/result.

### Adding demos

* Demos are written in the form of an executable Python script.
  Any package listed in `requirements.txt` you can assume is available to be imported.
  Matplotlib plots will be automatically rendered and displayed on the QML website.

  _Note: try and keep execution time of your script to within 10 minutes_.

* If you would like to write the demo using a Jupyter notebook, you can convert
  the notebook to the required executable Python format by using
  [this script](https://gist.github.com/chsasank/7218ca16f8d022e02a9c0deb94a310fe).

* All demos should have a file name beginning with `tutorial_`.
  The python files are saved in the `demonstrations` directory.

* [Restructured Text](http://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html)
  sections may be anywhere within the script by beginning the comment with
  79 hashes (`#`). These are useful for breaking up large code-blocks.

* You may add figures within ReST comments by using the following syntax:

  ```python
  ##############################################################################
  #.. figure:: ../demonstrations/<demo name>/image.png
  #    :align: center
  #    :width: 90%
  ```

  where `<demo name>` is a sub-directory with the name of
  your demo.

* When complete, create a gallery link to your demo. This can be done by adding the
  snippet below to `demos_getting-started.rst` for introductory demos.

  ```rest
  .. customgalleryitem::
      :tooltip: An extended description of the demo
      :figure: demonstrations/<demo name>/thumbnail.png
      :description: :doc:`demos/pytorch_noise`
  ```

  You should also add a link to your demo to the table of contents, by adding to the
  end of the `.. toctree::` in the appropriate file.

  If you're unsure which file to put your demo in, choose the one you think is  best,
  and we will work together to sort it during the review process.

* Finally, run your script through the [Black Python formatter](https://github.com/psf/black),

  ```bash
  pip install black
  black -l 100 demo_new.py
  ```

  and you are ready to submit a pull request!

If your demo uses the latest release of PennyLane, simply make your PR against the
`master` branch. If you instead require the cutting-edge development versions of
PennyLane or any relevant plugins, make your PR against the `dev` branch instead.


#### Tutorial guidelines

While you are free to be as creative as you like with your demo,
there are a couple of guidelines to keep in mind.

* Submissions should include your name (and optionally email) at the top
  under the title.

* All contributions must be made under the Apache 2.0 license.

* The title should be clear and concise, and if based on a paper it should be similar to the paper
  that is being implemented.

* All demos should include a summary below the title.
  The summary should be 1-3 sentences that makes clear the
  goal and outcome of the demo, and links to any papers/resources used.

* Code should be clearly commented and explained, either
  as a ReST-formatted comment or a standard Python comment.

* Thumbnails should be legible, interesting, and unique --- but not too busy!
  Any included text should be minimal and legible.

* If your content contains random variables/outputs, a fixed seed should
  be set for reproducibility.

* All content must be original or free to reuse subject to license compatibility.
  For example, if you are implementing someone else's research, reach out first to
  recieve permission to reproduce exact figures. Otherwise, avoid direct screenshots
  from papers, and instead refer to figures in the paper within the text.

* All submissions must pass code review before being merged into the repository.

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
