 .. role:: html(raw)
   :format: html

How to submit a demo
====================

.. meta::
   :property="og:description": Take a deeper dive into quantum machine learning by exploring cutting-edge algorithms using PennyLane and near-term quantum hardware.
   :property="og:image": https://pennylane.ai/qml/_static/demos_card.png

To submit your own demo you can follow the steps below, beginning with creating
a demo, followed by filling in a demo-submission template and posting it as an
issue to the `PennyLane QML GitHub repository <https://github.com/PennyLaneAI/qml>`_

Create a demo
-------------
Create a demonstration/tutorial that is using PennyLane and upload it to e.g.
GitHub, Bitbucket or GitLab. Preferably, the demo is simply written as a Jupyter
notebook or a Python document (using either Python comments or Restructured Text
sections starting with 79 hashes ``#``), but you may write the demo in any way
you like.

Guidelines
^^^^^^^^^^

While you are free to be as creative as you like with your demo, there are a
couple of guidelines to keep in mind.

* Submissions should include your name (and optionally email) at the top under
  the title.

* The title should be clear and concise, and if based on a paper it should be
  similar to the paper that is being implemented.

* All demos should include a summary below the title. The summary should be 1-3
  sentences that makes clear the goal and outcome of the demo, and links to any
  papers/resources used.

* Code should be clearly commented and explained.

* If your content contains random variables/outputs, a fixed seed should be set
  for reproducibility.

* All content should be original or free to reuse subject to license
  compatibility. For example, if you are implementing someone else's research,
  reach out first to recieve permission to reproduce exact figures. Otherwise,
  avoid direct screenshots from papers, and instead refer to figures in the
  paper within the text.


Fill in template
----------------

Fill in the following template with the relevant information. Please include at
least your name, a title, an abstract and a link to your uploaded
demonstration/tutorial. An example follows (you can simply overwrite the
relevant parts with your own information).

.. code-block:: none

    #### General information

    **name**
    John Doe

    **affiliation**
    Quantum University

    --------------------------------------------------------------------------------

    #### Demo information

    **title**
    Frugal shot optimization with Rosalin

    **abstract**
    In this tutorial we investigate and implement the Rosalin (Random Operator
    Sampling for Adaptive Learning with Individual Number of shots) from
    Arrasmith et al. [#arrasmith2020]_. In this paper, a strategy is introduced
    for reducing the number of shots required when optimizing variational
    quantum algorithms, by both:

    * Frugally adapting the number of shots used per parameter update, and
    * Performing a weighted sampling of operators from the cost Hamiltonian.

    **relevant links**
    https://pennylane.ai/qml/demos/tutorial_rosalin.html
    https://arxiv.org/abs/2004.06252


Open an issue
-------------

Open a `new issue on the PennyLane QML GitHub repository
<https://github.com/PennyLaneAI/qml/issues/new>`_ and paste the filled in
template from above in the description box. Remove and replace all the text that
was already there.

Write "[DEMO] your-demo-title" (e.g. "[DEMO] Frugal shot optimization with
Rosalin", without the quotes) as the title of the issue, and add the "demos"
label to the issue by clicking on the cogwheel on the right-hand side of the
**Labels** tag to the right of the text-box and then clicking on "demos".

Don't forget to push the "Submit new issue" button!


.. raw:: html

    <link href="https://cdnjs.cloudflare.com/ajax/libs/mdbootstrap/4.8.10/css/mdb.min.css" rel="stylesheet">


.. toctree::
    :maxdepth: 2
    :hidden:
