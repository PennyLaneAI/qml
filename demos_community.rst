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
    </style>


    <div class="jumbotron p-0 other">
        <div class="view rounded-top">
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


    <div class="card-deck mt-5">
        <div class="card plugin-card">
            <h4 class="card-header teal lighten-4">Characterizing the loss landscape of variational quantum circuits</h4>
            <div class="card-body">
                <h6>Patrick Huembeli and Alexandre Dauphin</h6>
                <p class="card-text">
                    Using PennyLane and complex PyTorch, we compute the Hessian of the loss function of VQCs and show how to characterize the loss landscape with it. We show how the Hessian can be used to escape flat regions of the loss landscape.
                </p>
            </div>
            <!-- Card footer -->
            <div class="rounded-bottom mdb-color lighten-5 text-right pt-3 pr-1">
                <ul class="list-unstyled list-inline font-small">
                    <li class="list-inline-item pr-2">
                        <a href="https://arxiv.org/abs/2008.02785" class="black-text"><i class="fas fa-book"></i> Paper</a>
                    </li>
                    <li class="list-inline-item pr-2"><a href="https://github.com/PatrickHuembeli/vqc_loss_landscapes" class="black-text">
                        <i class="fas fa-code-branch"></i></i> Code</a>
                    </li>
                    <li class="list-inline-item pr-2 black-text">
                        <i class="far fa-clock pr-1"></i>30/09/2020
                    </li>
                </ul>
            </div>
        </div>
        <div class="card hidden-card"></div>
    </div>


.. toctree::
    :maxdepth: 2
    :hidden:


.. <h3 class="card-title h3 my-4"><strong>Community demos</strong></h3>
