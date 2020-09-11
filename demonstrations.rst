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
                    <a href="demos_learning.html">
                    <div class="card rounded-lg">
                        <div class="text-center d-flex align-items-center pb-2">
                            <div>
                                <h3 class="card-title">
                                <img src="_static/flask.png" class="img-fluid" style="max-width: 53px;"></img>
                                <br>
                                <strong>Learning</strong>
                                </h3>
                                <p class="mb-1 grey-text px-3">Find out how the principles of quantum computing and machine learning can be united to create something new.</p>
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
                                <img src="_static/flask.png" class="img-fluid" style="max-width: 53px;"></img>
                                <br>
                                <strong>Research</strong>
                                </h3>
                                <p class="mb-1 grey-text px-3">Find out how the principles of quantum computing and machine learning can be united to create something new.</p>
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
                                <img src="_static/flask.png" class="img-fluid" style="max-width: 53px;"></img>
                                <br>
                                <strong>Community</strong>
                                </h3>
                                <p class="mb-1 grey-text px-3">Find out how the principles of quantum computing and machine learning can be united to create something new.</p>
                                <div class="white-text d-flex justify-content-center"><h5>Read more <i class="fas fa-angle-double-right"></i></h5></div>
                            </div>
                        </div>
                    </div>
                </a>
                </div>
            </div>

            <h2 class="text-center mx-auto my-0">Featured</h2>

            <ul class="light-slider" id="featured-demos">
                <li>
                    <a href="demos/tutorial_vqls.html">
                        <div class="card">
                          <img class="card-img-top img-fluid" src="_images/vqls_circuit.png" alt="Card image cap" style="min-width: 500px!important;">
                          <div class="card-body">
                            <h4 class="card-title">Variational quantum linear solver</h4>
                            <p class="card-text">Explore how variational quantum circuits can be used to solve systems of linear equations. Here, we solve a system of 8 linear equations using 3 qubits and an ancilla.</p>
                          </div>
                        </div>
                    </a>
                </li>
                <li>
                    <a href="demos/tutorial_quantum_transfer_learning.html">
                        <div class="card">
                          <img class="card-img-top img-fluid" src="_images/transfer_images.png" alt="Card image cap" style="min-width: 350px!important;">
                          <div class="card-body">
                            <h4 class="card-title">Quantum transfer learning</h4>
                            <p class="card-text">Learn how to apply a machine learning method, known as transfer learning, to a hybrid classical-quantum image classifier.</p>
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
                    <a href="demos/tutorial_doubly_stochastic.html">
                        <div class="card">
                          <img class="card-img-top img-fluid" src="_images/sphx_glr_tutorial_doubly_stochastic_002.png" alt="Card image cap">
                          <div class="card-body">
                            <h4 class="card-title">Doubly stochastic gradient descent</h4>
                            <p class="card-text">Quantum gradient descent with finite number of shots is a form of stochastic gradient descent. By sampling from terms in the VQE Hamiltonian, we get "doubly stochastic gradient descent".</p>
                          </div>
                        </div>
                    </a>
                </li>
                <li>
                    <a href="demos/barren_plateaus.html">
                        <div class="card">
                          <img class="card-img-top" src="_static/thumbs/surface.png" alt="Card image cap">
                          <div class="card-body">
                            <h4 class="card-title">Barren plateaus in QNNs</h4>
                            <p class="card-text">We show how variational quantum circuits face the problem of barren plateaus. We partly reproduce some of the findings in McClean et al. (2018) with just a few lines of code.</p>
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

    demos_learning
    demos_research
    demos_community