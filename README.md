# Quantum machine learning tutorials and implementations

This repository contains the Python tutorials and implementations available
at https://pennylane.ai/qml. Content includes:

* [Key concepts of QML](https://pennylane.ai/qml/concepts.html). Explore and
  understand the key concepts underpinning variational quantum circuits and
  quantum machine learning.

* [Beginner tutorials](https://pennylane.ai/qml/beginner.html). Tutorials
  introduce core QML concepts, including quantum nodes, optimization, and devices,
  via easy-to-follow examples.

* [Implementations of cutting-edge QML research](https://pennylane.ai/qml/implementations.html).
  Take a deeper dive into quantum machine learning by exploring cutting-edge algorithms
  on near-term quantum hardware.

## Contributing

There are three ways you can contribute: add or extend to the QML key concepts section,
write a beginners QML tutorial, or by submitting an implementation of a recent
quantum machine learning paper/result.

### Adding new concepts

* All sections in the Key Concepts section are written using
  [Restructured Text](http://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html)
  markup, and are placed in the `concepts` folder. Have a look at some of the
  existing files, such as `concept_embeddings.rst`, for examples and to see
  how math is scripted.

* Once written, a small summary and a link should be added in the top-level `concepts.rst`
  file.

* You are now ready to submit a pull request!

### Adding tutorials and implementations

* Tutorials and implementations are written in the form of an executable Python script.
  Any package listed in `requirements.txt` you can assume is available to be imported.
  Matplotlib plots will be automatically rendered and displayed on the QML website.

  _Note: try and keep execution time of your script to within 10 minutes_.

* All tutorials/implementations should have a file name beginning with `tutorial_`.
  Beginner tutorials go in the `beginner` directory, while implementations go in
  the `implementations` directory.

* [Restructured Text](http://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html)
  sections may be anywhere within the script by beginning the comment with
  79 hashes (`#`). These are useful for breaking up large code-blocks.

* You may add figures within ReST comments by using the following syntax:

  ```python
  ##############################################################################
  #.. figure:: ../<tutorial type>/<tutorial name>/image.png
  #    :align: center
  #    :width: 90%
  ```

  where `<tutorial type>` is one of `beginner` or `implementation` (depending on what
  content you submitting), and `<tutorial name>` is a sub-directory with the name of
  your tutorial.

* When complete, create a gallery link to your tutorial/implementation, by adding the
  following to either `beginner.rst` or `implementations.rst`:

  ```rest
  .. customgalleryitem::
      :tooltip: An extended description of the tutorial/implementation
      :figure: <tutorial type>/<tutorial name>/thumbnail.png
      :description: :doc:`<build location>/pytorch_noise`
  ```

  Here, `<build_location>` is either `tutorial` (for beginner tutorials), or `app` (for
  implementations).

  You should also add a link to your tutorial to the table of contents, by adding to the
  end of the `.. toctree::`.

* Finally, run your script through the [Black Python formatter](https://github.com/psf/black),

  ```bash
  pip install black
  black -l 100 tutorial_new.py
  ```

  and you are ready to submit a pull request!


#### Tutorial guidelines

While you are free to be as creative as you like with your tutorial or implementation,
there are a couple of guidelines to keep in mind.

* Submissions should include your name (and optionally email) at the top
  under the title.

* All contributions must be made under the Apache 2.0 license.

* The title should be clear and concise, and if an implementation, be similar to the paper
  that is being implemented.

* All tutorials/implementations should include a summary below the title.
  The summary should be 1-3 sentences that makes clear the
  goal and outcome of the tutorial, and links to any papers/resources used.

* Code should be clearly commented and explained, either
  as a ReST-formatted comment or a standard Python comment.

* Thumbnails should be legible, interesting, and unique --- but not too busy!
  Any included text should be minimal and legible

* All content must be original or free to reuse subject to license compatibility.
  For example, if you are implementing someone elses research, reach out first to
  recieve permission to reproduce exact figures. Otherwise, avoid direct screenshots
  from papers, and instead refer to figures in the paper within the text.

## Building

To build the website locally, simply run `make html`. The rendered HTML files
will now be available in `_build/html`. Open `_build/html/index.html` to browse
the built site locally.

Note that the above command may take some time, as all tutorials and implementations
will be executed and built! Once built, only _modified_ tutorials will
be re-executed/re-built.

Alternatively, you may run `make html-norun` to build the website _without_ executing
tutorials/implementations.

## Support

- **Source Code:** https://github.com/XanaduAI/QML
- **Issue Tracker:** https://github.com/XanaduAI/QML/issues

If you are having issues, please let us know by posting the issue on our GitHub issue tracker.

We are committed to providing a friendly, safe, and welcoming environment for all.
Please read and respect the [Code of Conduct](.github/CODE_OF_CONDUCT.md).

## License

The materials, tutorials, and implementations in this repository are **free** and
**open source**, released under the Apache License, Version 2.0.

The file `custom_directives.py` is available under the BSD 3-Clause License with
Copyright (c) 2017, Pytorch contributors.
