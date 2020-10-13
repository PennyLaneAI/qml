 .. role:: html(raw)
   :format: html

Community
=========

.. meta::
   :property="og:description": PennyLane demonstrations created by the community showcasing quantum machine learning and other topics of interest.
   :property="og:image": https://pennylane.ai/qml/_static/demos_card.png

.. raw:: html

    <style>
    #right-column {
        max-width: 1200px;
    }
    .up-button {
        left: calc(50% - 650px);
    }
    .jumbotron {
        box-shadow: none!important;
    }
    </style>


    <div class="jumbotron p-0 other">
        <div class="view">
            <img src="_static/demo-quilt-wide.png" class="img-fluid" alt="Sample image">
            <a href="#">
            <div class="mask rgba-stylish-slight"></div>
            </a>
        </div>

        <div class="card-body text-center mb-3">
            <p class="card-text py-2 lead">
                Have an existing GitHub repository or Jupyter notebook showing
                off quantum machine learning with PennyLane? Read the guidelines
                and submission instructions <a href="demos_submission.html">here</a>,
                and have your demonstration and research featured on our community page.
            </p>
        </div>
    </div>

    <link href="https://cdnjs.cloudflare.com/ajax/libs/mdbootstrap/4.8.10/css/mdb.min.css" rel="stylesheet">

    <hr>
    <br>


.. For two cards per row, create a new card deck every two cards.

:html:`<div class="card-deck">`

.. Community card required options:
     - title
     - author
     - date
   All remaining options (code, paper, color) are *optional*.
   All fields, including the description can contain arbitrary HTML.

.. community-card::
    :title: Characterizing the loss landscape of variational quantum circuits
    :author: Patrick Huembeli and Alexandre Dauphin
    :date: 30/09/2020
    :code: https://github.com/PatrickHuembeli/vqc_loss_landscapes
    :paper: https://arxiv.org/abs/2008.02785
    :color: heavy-rain-gradient

    Using PennyLane and complex PyTorch, we compute the Hessian of the loss function of VQCs and
    show how to characterize the loss landscape with it. We show how the Hessian can be used to
    escape flat regions of the loss landscape.

.. If the final card deck only has a single card, we insert a 'hidden card'
   so that the card does not become full-width.

:html:`<div class="card hidden-card"></div></div>`

.. toctree::
    :maxdepth: 2
    :hidden:

