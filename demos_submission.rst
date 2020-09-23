 .. role:: html(raw)
   :format: html

How to submit a demo
====================

.. meta::
   :property="og:description": Take a deeper dive into quantum machine learning by exploring cutting-edge algorithms using PennyLane and near-term quantum hardware.
   :property="og:image": https://pennylane.ai/qml/_static/demos_card.png

To submit your own demo you can follow the steps below, beginning with creating
a demo, followed by filling in a demo-submission template and posting it as an
`issue to the PennyLane QML GitHub repository <https://github.com/PennyLaneAI/qml/issues/new?assignees=&labels=demos&template=community-demo.md&title=%5BDEMO%5D>`_

Create a demo
-------------
Create a demonstration/tutorial that is using PennyLane and upload it to e.g.
GitHub, Bitbucket or GitLab. Preferably, the demo is simply written as a Jupyter
notebook a Python document or simply as a repository (e.g. `like this <https://github.com/PatrickHuembeli/vqc_loss_landscapes>`_), but you may
write the demo in any way you like.

Alternatively, you can also submit a link to a rendered Jupyter notebook using e.g. `Colab
<https://colab.research.google.com/notebooks/intro.ipynb>`_, `nbviewer
<https://nbviewer.jupyter.org/>`_, `Sagemaker <https://aws.amazon.com/sagemaker/>`_ or `Binder <https://mybinder.org/>`_.

Guidelines
^^^^^^^^^^

While you are free to be as creative as you like with your demo, there are a
couple of guidelines that might be good keep in mind when creating a demo.
Specifically when it's a Jupyter notebook or a single Python document. The
following concerns the content of the demo itself, and is in addition to the
submission details explained in the next section.

* The demo should include your name (and optionally email) at the top under the
  title.

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
  reach out first to receive permission to reproduce exact figures. Otherwise,
  avoid direct screenshots from papers, and instead refer to figures in the
  paper within the text.

* Include the demos dependencies (e.g. PennyLane version along with any relevant
  PennyLane plugin version). If possible, include a ``requirements.txt`` file
  along with your local output after running ``pip freeze``.


Open an issue
-------------

Open a `new issue on the PennyLane QML GitHub repository
<https://github.com/PennyLaneAI/qml/issues/new?assignees=&labels=demos&template=community-demo.md&title=%5BDEMO%5D>`_
using the "Community demo" template. Please include at least your name, a title, an abstract and a
link to your uploaded demonstration/tutorial. An example follows (you can simply
replace relevant parts with your own information).

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

The title of your issue should be "[DEMO] your-demo-title" (e.g. "[DEMO] Frugal shot optimization with
Rosalin", without the quotes), and there should be a "demos" label on the
right-hand side.

*Don't forget to push the "Submit new issue" button!*


.. raw:: html

    <link href="https://cdnjs.cloudflare.com/ajax/libs/mdbootstrap/4.8.10/css/mdb.min.css" rel="stylesheet">


.. toctree::
    :maxdepth: 2
    :hidden:
