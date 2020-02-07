.. raw:: html

    <style>
    h1 {
    text-align: center;
    }
    </style>

Quantum machine learning
========================

.. raw:: html

    <link href="https://cdnjs.cloudflare.com/ajax/libs/mdbootstrap/4.5.14/css/mdb.min.css" rel="stylesheet">
    <div class="container">
        <!-- Section: Features v.1 -->
            <p class="lead grey-text text-center mx-auto mb-5">We're entering an exciting time in quantum physics and quantum computation: <span class="teal-text">near-term quantum devices</span> are rapidly becoming a reality, accessible to everyone over the internet.
            <br><br>
            This, in turn, is driving the development of quantum machine learning and variational quantum circuits.</p>

        <section class="my-5">
            <div class="row" id="main-cards">
                <div class="col-lg-4 mb-5">
                    <a href="whatisqml.html">
                    <div class="card rounded-lg">
                        <div class="text-center d-flex align-items-center pb-2">
                            <div>
                                <h3 class="card-title">
                                <img src="_static/dummy.png" class="img-fluid" style="max-width: 86px;"></img>
                                <br>
                                <strong>What is QML?</strong>
                                </h3>
                                <p class="mb-1 grey-text px-3">Find out how the principles of quantum computing and machine learning can be united to create something new.</p>
                                <div class="white-text d-flex justify-content-center"><h5>Read more <i class="fas fa-angle-double-right"></i></h5></div>
                            </div>
                        </div>
                    </div>
                </a>
                </div>
                <div class="col-lg-4 mb-5">
                    <a href="trainingcircuits.html">
                    <div class="card rounded-lg">
                        <div class="text-center d-flex align-items-center pb-2">
                            <div>
                                <h3 class="card-title">
                                <img src="_static/board.png" class="img-fluid" style="max-width: 86px;"></img>
                                <br>
                                <strong>Training circuits</strong>
                                </h3>
                                <p class="mb-1 grey-text px-3">Understand how variational quantum circuits are trained using the idea of quantum gradients.</p>
                                <div class="white-text d-flex justify-content-center"><h5>Read more <i class="fas fa-angle-double-right"></i></h5></div>
                            </div>
                        </div>
                    </div>
                </a>
                </div>
                <div class="col-lg-4 mb-5">
                    <a href="concepts.html">
                        <div class="card rounded-lg">
                            <div class="text-center d-flex align-items-center pb-2">
                                <div>
                                    <h3 class="card-title">
                                    <img src="_static/key.png" class="img-fluid" style="max-width: 80px;"></img>
                                    <br>
                                    <strong>Key concepts</strong>
                                    </h3>
                                    <p class="mb-1 grey-text px-3">Explore different concepts underpinning variational quantum circuits and quantum machine learning.</p>
                                    <div class="white-text d-flex justify-content-center"><h5>Read more <i class="fas fa-angle-double-right"></i></h5></div>
                                    <!-- <a href="concepts.html" class="btn peach-gradient">Get started</a> -->
                                </div>
                            </div>
                        </div>
                    </a>
                </div>
                <div class="col-lg-4 mb-5">
                    <a href="demonstrations.html">
                    <div class="card rounded-lg">
                        <div class="text-center d-flex align-items-center pb-2">
                            <div>
                                <h3 class="card-title">
                                <img src="_static/flask.png" class="img-fluid" style="max-width: 53px;"></img>
                                <br>
                                <strong>Demos</strong>
                                </h3>
                                <p class="mb-1 grey-text px-3">Take a dive into quantum machine learning by exploring cutting-edge algorithms on near-term quantum hardware.</p>
                                <div class="white-text d-flex justify-content-center"><h5>Read more <i class="fas fa-angle-double-right"></i></h5></div>
                            </div>
                        </div>
                    </div>
                </a>
                </div>
            </div>
            <h2 class="text-center mx-auto my-0">Featured</h2>

            <ul id="light-slider">
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
                    <a href="demos/tutorial_barren_plateaus.html">
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
                All content above is free, open-source, and available as executable code downloads. If you would like to contribute a tutorial or QML demonstration, please make a pull request over at our <a href="https://github.com/XanaduAI/qml">GitHub repository</a>.
            </p>

        </section>
    </div>

    <hr class="pb-5">

    <p class="lead grey-text text-center mx-auto mt-3 mb-5">Our aim is to bring together a community focused on quantum machine learning, and provide a leading resource hub for quantum computing education and research.</p>
    <!-- Grid row -->
    <div class="row">
        <!-- Grid column -->
        <div class="col-md-3">
            <div class="text-center">
                <img src="_static/research.png" class="img-fluid"></img>
                <br>
                <h5 class="font-weight-bold my-4">Research focused</h5>
            </div>
            <p class="grey-text mb-0">Xanadu is not just a software company; we also perform high-impact research and build quantum hardware. Our software gets used internally across the company &mdash; check out some of the <a href="https://xanadu.ai/research">papers released</a> using our open-source software tools.
            </p>
        </div>
        <!-- Grid column -->
        <!-- Grid column -->
        <div class="col-md-3">
            <div class="text-center">
                <img src="_static/book.png" class="img-fluid"></img>
                <br>
                <h5 class="font-weight-bold my-4">Documented</h5>
            </div>
            <p class="grey-text mb-md-0 mb-5">Code documentation is important; it encourages everyone to dive straight in and begin tinkering with the code. We have the highest standard for documentation &mdash; our aim is to make everything accessible, from code to theory.
            </p>
        </div>
        <!-- Grid column -->
        <!-- Grid column -->
        <div class="col-md-3">
            <div class="text-center">
                <img src="_static/fork.png" class="img-fluid"></img>
                <br>
                <h5 class="font-weight-bold my-4">Open source</h5>
            </div>
            <p class="grey-text mb-md-0 mb-5">The physics community is embracing open-source software, bringing transparency and reproducibility to physics research. All of our software is open-source, and we are excited to be along for the ride.
            </p>
        </div>
        <!-- Grid column -->
        <!-- Grid column -->
        <div class="col-md-3">
            <div class="text-center">
                <img src="_static/community.png" class="img-fluid"></img>
                <br>
                <h5 class="font-weight-bold my-4">Community driven</h5>
            </div>
                <p class="grey-text mb-0">When we release software, we have the community in mind. Development takes place publicly on GitHub, and members of our team are available to chat on our <a href="https://u.strawberryfields.ai/slack">Strawberry Fields Slack</a> and <a href="https://discuss.pennylane.ai">PennyLane forum</a>.
            </p>
        </div>
    <!-- Grid column -->
    </div>
    <!-- Grid row -->

.. toctree::
    :maxdepth: 2
    :hidden:

    whatisqml
    trainingcircuits
    concepts
    demonstrations
