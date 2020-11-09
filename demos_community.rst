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

.. community-card::
    :title: Iris Classification using Angle Embedding and qml.qnn.KerasLayer
    :author: Hemant Gahankari
    :date: 09/10/2020
    :code: https://colab.research.google.com/drive/13PvS2D8mxBvlNw6_5EapUU2ePKdf_K53#scrollTo=1fJWDX5LxfvB
    :color: heavy-rain-gradient

    This example is created to explain how to pass classical data into the quantum
    function and convert it to quantum data. Later it shows how to create
    qml.qnn.KerasLayer from qnode and train it and also check model performance.

.. community-card::
    :title: Iris Classification using Amplitude Embedding and qml.qnn.KerasLayer
    :author: Hemant Gahankari
    :date: 09/11/2020
    :code: https://colab.research.google.com/drive/12ls_GkSD2t0hr3Mx9-qzVvSWxR3-N0WI#scrollTo=4PQTkXpv52vZ
    :color: heavy-rain-gradient

    This example is created to explain how to pass classical data into the
    quantum function and convert it to quantum data. Later it shows how to
    create qml.qnn.KerasLayer from qnode and train it and also check model
    performance.

.. community-card::
    :title: Linear Regression using Angle Embedding and Single Qubit
    :author: Hemant Gahankari
    :date: 09/11/2020
    :code: https://colab.research.google.com/drive/1ABVtBjwcGNNIfmiwEXRdFdZ47K1vZ978?usp=sharing
    :color: heavy-rain-gradient

    This example is created to explain how to create a Hybrid Neural Network
    (Mix of Classical and Quantum Layer) and train it and get predictions from
    it. The data set consists of a temperature readings in Degree Centigrade
    and corresponding Fahrenheit. Objective is to train a neural network to
    predict Fahrenheit value given Deg Centigrade value.

.. community-card::
    :title: Iris Classification using Qnode and Keras Optimizer and Loss function
    :author: Hemant Gahankari
    :date: 09/11/2020
    :code: https://colab.research.google.com/drive/17Qri3jUBpjjkhmO6ZZZNXwm511svSVPw?usp=sharing
    :color: heavy-rain-gradient

    This example is created to explain how to create a quantum function and
    train a quantum function using keras optimizer directly. i.e. not using
    Keras Layer. Objective is to train a quantum function to predict classes of
    Iris dataset.

.. If the final card deck only has a single card, we insert a 'hidden card'
   so that the card does not become full-width.

:html:`<div class="card hidden-card"></div></div>`

.. toctree::
    :maxdepth: 2
    :hidden:

