 .. role:: html(raw)
   :format: html

QML demos
=========

Take a deeper dive into quantum machine learning by exploring cutting-edge
algorithms using PennyLane and near-term quantum hardware.


.. raw:: html

    <link href="https://cdnjs.cloudflare.com/ajax/libs/mdbootstrap/4.8.10/css/mdb.min.css" rel="stylesheet">
    <div class="tags">
        <input type="radio" class="controls" id="All" name="categories" value="All" checked>
        <input type="radio" class="controls" id="tensorflow" name="categories" value="tensorflow">
        <input type="radio" class="controls" id="pytorch" name="categories" value="pytorch">
        <input type="radio" class="controls" id="autograd" name="categories" value="autograd">
        <input type="radio" class="controls" id="forest" name="categories" value="forest">
        <input type="radio" class="controls" id="cirq" name="categories" value="cirq">
        <input type="radio" class="controls" id="qiskit" name="categories" value="qiskit">
        <input type="radio" class="controls" id="strawberryfields" name="categories" value="strawberryfields">
        <ol class="filters">
            <li>
                <label for="All">All</label>
            </li>
            <li>
                <label for="tensorflow">TensorFlow</label>
            </li>
            <li>
                <label for="pytorch">PyTorch</label>
            </li>
            <li>
                <label for="autograd">NumPy/Autograd</label>
            </li>
            <li>
                <label for="forest">Rigetti Forest</label>
            </li>
            <li>
                <label for="cirq">Cirq</label>
            </li>
            <li>
                <label for="qiskit">Qiskit</label>
            </li>
            <li>
                <label for="strawberryfields">Strawberry Fields</label>
            </li>
        </ol>

:html:`<div class="gallery-grid row">`

.. customgalleryitem::
    :tooltip: Do arbitrary state preparation on a real quantum computer.
    :figure: implementations/state_preparation/NOON.png
    :description: :doc:`app/tutorial_state_preparation`
    :tags: forest pytorch

.. customgalleryitem::
    :tooltip: Ising model example with PennyLane PyTorch interface.
    :figure: implementations/Ising_model/isingspins.png
    :description: :doc:`app/tutorial_isingmodel_PyTorch`
    :tags: pytorch autograd

.. customgalleryitem::
    :tooltip: Create a simple QGAN with Cirq and TensorFlow.
    :figure: implementations/QGAN/qgan3.png
    :description: :doc:`app/tutorial_QGAN`
    :tags: cirq tensorflow

.. customgalleryitem::
    :tooltip: A quantum variational classifier
    :figure: implementations/variational_classifier/classifier_output_59_0.png
    :description: :doc:`app/tutorial_variational_classifier`
    :tags: autograd

.. customgalleryitem::
    :tooltip: Fit one dimensional noisy data with a quantum neural network.
    :figure: implementations/quantum_neural_net/qnn_output_28_0.png
    :description: :doc:`app/quantum_neural_net`
    :tags: autograd strawberryfields

.. customgalleryitem::
    :tooltip: Find the ground state of a Hamiltonian.
    :figure: implementations/variational_quantum_eigensolver/pes_h2.png
    :description: :doc:`app/tutorial_vqe`
    :tags: autograd

.. customgalleryitem::
    :tooltip: Universal Quantum Classifier with data-reuploading
    :figure: implementations/data_reuploading/universal_dnn.png
    :description: :doc:`app/tutorial_data_reuploading_classifier`
    :tags: autograd

.. customgalleryitem::
    :tooltip: Faster optimization convergence using quantum natural gradient
    :figure: implementations/quantum_natural_gradient/qng_optimization.png
    :description: :doc:`app/tutorial_quantum_natural_gradient`
    :tags: autograd

.. customgalleryitem::
    :tooltip: Perform QAOA for MaxCut
    :figure: implementations/qaoa_maxcut/qaoa_maxcut_partition.png
    :description: :doc:`app/tutorial_qaoa_maxcut`
    :tags: autograd

.. customgalleryitem::
    :tooltip: Barren plateaus in quantum neural networks
    :figure: implementations/barren_plateaus/surface.png
    :description: :doc:`app/tutorial_barren_plateaus`
    :tags: autograd

.. customgalleryitem::
    :tooltip: Rotoselect algorithm
    :figure: implementations/rotoselect/rotoselect_structure.png
    :description: :doc:`app/tutorial_rotoselect`
    :tags: autograd

.. customgalleryitem::
    :tooltip: Doubly stochastic gradient descent
    :figure: implementations/doubly_stochastic/single_shot.png
    :description: :doc:`Doubly stochastic gradient descent <app/tutorial_doubly_stochastic>`
    :tags: autograd

.. customgalleryitem::
    :tooltip: Variational Quantum Linear Solver
    :figure: implementations/vqls/vqls_zoom.png
    :description: :doc:`app/tutorial_vqls`
    :tags: autograd

.. customgalleryitem::
    :tooltip: Coherent implementation of a variational quantum linear solver
    :figure: implementations/coherent_vqls/cvqls_zoom.png
    :description: :doc:`app/tutorial_coherent_vqls`
    :tags: autograd

.. customgalleryitem::
    :tooltip: Quantum transfer learning
    :figure: implementations/quantum_transfer_learning/transfer_images.png
    :description: :doc:`app/tutorial_quantum_transfer_learning`
    :tags: autograd pytorch

.. customgalleryitem::
    :tooltip: Training an embedding to perform metric learning
    :figure: implementations/embedding_metric_learning/training.png
    :description: :doc:`app/tutorial_embeddings_metric_learning`
    :tags: autograd

:html:`</div></div><div style='clear:both'></div>`


.. toctree::
    :maxdepth: 2
    :caption: QML Implementations
    :hidden:

    app/tutorial_state_preparation
    app/tutorial_isingmodel_PyTorch
    app/tutorial_QGAN
    app/tutorial_variational_classifier
    app/quantum_neural_net
    app/tutorial_vqe
    app/tutorial_data_reuploading_classifier
    app/tutorial_quantum_natural_gradient
    app/tutorial_qaoa_maxcut
    app/tutorial_barren_plateaus
    app/tutorial_rotoselect
    app/tutorial_doubly_stochastic
    app/tutorial_vqls
    app/tutorial_coherent_vqls
    app/tutorial_quantum_transfer_learning
    app/tutorial_embeddings_metric_learning
