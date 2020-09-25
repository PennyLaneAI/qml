.. raw:: html

    <style>
        h1 {
            text-align: center;
        }
    </style>

QML Demos
=========

.. meta::
   :property="og:description": Take a deeper dive into quantum machine learning by exploring cutting-edge algorithms using PennyLane and near-term quantum hardware.
   :property="og:image": https://pennylane.ai/qml/_static/demos_card.png

.. raw:: html

    <link href="https://cdnjs.cloudflare.com/ajax/libs/mdbootstrap/4.8.10/css/mdb.min.css" rel="stylesheet">

    <div class="container">
        <!-- Section: Features v.1 -->
            <p class="lead grey-text text-center mx-auto mb-5">
            Take a deeper dive into quantum machine learning by exploring cutting-edge algorithms using PennyLane and near-term quantum hardware.</p>

        <section class="my-5">
            <div class="row justify-content-center" id="main-cards">
                <div class="col-lg-3 mb-5">
                    <a href="demos_basics.html">
                    <div class="card rounded-lg">
                        <div class="text-center d-flex align-items-center pb-2">
                            <div>
                                <h3 class="card-title">
                                <img src="_static/board.png" class="img-fluid" style="max-width: 88px;"></img>
                                <br>
                                <strong>Basics</strong>
                                </h3>
                                <p class="mb-1 grey-text px-3">Begin your journey into quantum machine learning using PennyLane.</p>
                                <div class="white-text d-flex justify-content-center"><h5>Read more <i class="fas fa-angle-double-right"></i></h5></div>
                            </div>
                        </div>
                    </div>
                </a>
                </div>
                <div class="col-lg-3 mb-5">
                    <a href="demos_research.html">
                    <div class="card rounded-lg">
                        <div class="text-center d-flex align-items-center pb-2">
                            <div>
                                <h3 class="card-title">
                                <img src="_static/research.png" class="img-fluid" style="max-width: 88px;"></img>
                                <br>
                                <strong>Research</strong>
                                </h3>
                                <p class="mb-1 grey-text px-3">Explore cutting-edge research in quantum machine learning using PennyLane.</p>
                                <div class="white-text d-flex justify-content-center"><h5>Read more <i class="fas fa-angle-double-right"></i></h5></div>
                            </div>
                        </div>
                    </div>
                </a>
                </div>
                <div class="col-lg-3 mb-5">
                    <a href="demos_community.html">
                    <div class="card rounded-lg">
                        <div class="text-center d-flex align-items-center pb-2">
                            <div>
                                <h3 class="card-title">
                                <img src="_static/community.png" class="img-fluid" style="max-width: 105px;"></img>
                                <br>
                                <strong>Community</strong>
                                </h3>
                                <p class="mb-1 grey-text px-3">Discover PennyLane demonstrations created by other users, or submit one yourself.</p>
                                <div class="white-text d-flex justify-content-center"><h5>Read more <i class="fas fa-angle-double-right"></i></h5></div>
                            </div>
                        </div>
                    </div>
                </a>
                </div>
            </div>

            <blockquote class="blockquote border rounded ">
                <p class="mb-0 text-muted text-center">
                    Do you want to make your own demo using PennyLane? Read
                    the guidelines and submission instructions <a href="demos_submission.html">here</a>, and have
                    your demo featured on our community page.
                </p>
            </blockquote>

            <h2 class="text-center mx-auto my-0">Featured</h2>

            <ul class="light-slider" id="featured-demos">
                <li>
                    <a href="demos/tutorial_quantum_transfer_learning.html">
                        <div class="card">
                          <img class="card-img-top img-fluid" src="_static/thumbs/transfer_images.png" alt="Card image cap">
                          <div class="card-body">
                            <h4 class="card-title">Quantum transfer learning</h4>
                            <p class="card-text">Learn how to apply a machine learning method, known as transfer learning, to a hybrid classical-quantum image classifier.</p>
                          </div>
                        </div>
                    </a>
                </li>
                <li>
                    <a href="demos/qgrnn.html">
                        <div class="card">
                          <img class="card-img-top img-fluid" src="_static/thumbs/qgrnn_thumbnail.png" alt="Card image cap">
                          <div class="card-body">
                            <h4 class="card-title">The quantum graph recurrent neural network</h4>
                            <p class="card-text">Use a quantum graph recurrent neural network to learn quantum dynamics.</p>
                          </div>
                        </div>
                    </a>
                </li>
                <li>
                    <a href="demos/tutorial_data_reuploading_classifier.html">
                        <div class="card">
                          <img class="card-img-top img-fluid" src="_static/thumbs/universal_dnn.png" alt="Card image cap">
                          <div class="card-body">
                            <h4 class="card-title">Data re-uploading classifier</h4>
                            <p class="card-text">A universal single-qubit quantum classifier using the idea of 'data re-uploading' by Pérez-Salinas et al. (2019), akin to a single hidden-layered neural network.</p>
                          </div>
                        </div>
                    </a>
                </li>
                <li>
                    <a href="demos/tutorial_vqe.html">
                        <div class="card">
                          <img class="card-img-top img-fluid" src="_static/thumbs/pes_h2.png" alt="Card image cap">
                          <div class="card-body">
                            <h4 class="card-title">Doubly stochastic gradient descent</h4>
                            <p class="card-text">Calculate the ground-state energy of the hydrogen molecule by sampling from terms in the VQE Hamiltonian, getting the doubly stochastic gradient descent.</p>
                          </div>
                        </div>
                    </a>
                </li>
                <li>
                    <a href="demos/tutorial_quantum_natural_gradient.html">
                        <div class="card">
                          <img class="card-img-top" src="_static/thumbs/qng_optimization.png" alt="Card image cap">
                          <div class="card-body">
                            <h4 class="card-title">Quantum natural gradient</h4>
                            <p class="card-text">Achieve faster optimization convergence using the quantum natural gradient.</p>
                          </div>
                        </div>
                    </a>
                </li>
                <li>
                    <a href="demos/tutorial_qaoa_maxcut.html">
                        <div class="card">
                          <img class="card-img-top" src="_static/thumbs/qaoa_maxcut_partition.png" alt="Card image cap">
                          <div class="card-body">
                            <h4 class="card-title">QAOA for MaxCut</h4>
                            <p class="card-text">Implement the QAOA algorithm using PennyLane to solve the MaxCut problem.</p>
                          </div>
                        </div>
                    </a>
                </li>
                <li>
                    <a href="demos/tutorial_vqt.html">
                        <div class="card">
                          <img class="card-img-top" src="_static/thumbs/thumbnail.png" alt="Card image cap">
                          <div class="card-body">
                            <h4 class="card-title">The variational quantum thermalizer</h4>
                            <p class="card-text">Learn about the variational quantum thermalizer algorithm, an extension of VQE.</p>
                          </div>
                        </div>
                    </a>
                </li>
                <li>
                    <a href="demos/tutorial_rosalin.html">
                        <div class="card">
                          <img class="card-img-top" src="_static/thumbs/rosalin_thumb.png" alt="Card image cap">
                          <div class="card-body">
                            <h4 class="card-title">Frugal shot optimization with Rosalin </h4>
                            <p class="card-text">Optimize variational quantum algorithms with a minimized number of shots by using the Rosalin optimizer.</p>
                          </div>
                        </div>
                    </a>
                </li>
            </ul>

            <p class="grey-text mx-auto mt-5" style="font-size: small;margin-top:-10px;">
                All content above is free, open-source, and available as executable code downloads. If you would like to contribute a demo, please make a pull request over at our <a href="https://github.com/PennyLaneAI/qml">GitHub repository</a>.
            </p>

        </section>
    </div>

.. toctree::
    :maxdepth: 2
    :caption: QML Demos
    :hidden:

    demos_basics
    demos_research
    demos_community
    demos_submission
