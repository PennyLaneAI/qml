 .. role:: html(raw)
   :format: html

QML Research
============

.. meta::
   :property="og:description": Take a deeper dive into quantum machine learning by exploring cutting-edge algorithms using PennyLane and near-term quantum hardware.
   :property="og:image": https://pennylane.ai/qml/_static/demos_card.png


Research demos page. Note, the demos below aren't yet sorted into the correct
sections, and some of them might need to be moved to the Learning QML page.

.. raw:: html

    <link href="https://cdnjs.cloudflare.com/ajax/libs/mdbootstrap/4.8.10/css/mdb.min.css" rel="stylesheet">


Extending classical machine learning
------------------------------------
Add a descriptive text here (and update the header above if needed)

:html:`<div class="gallery-grid row">`

.. customgalleryitem::
    :tooltip: Ising model example with PennyLane PyTorch interface.
    :figure: demonstrations/Ising_model/isingspins.png
    :description: :doc:`demos/tutorial_isingmodel_PyTorch`
    :tags: pytorch autograd

.. customgalleryitem::
    :tooltip: Create a simple QGAN with Cirq and TensorFlow.
    :figure: demonstrations/QGAN/qgan3.png
    :description: :doc:`demos/tutorial_QGAN`
    :tags: cirq tensorflow

.. customgalleryitem::
    :tooltip: A quantum variational classifier
    :figure: demonstrations/variational_classifier/classifier_output_59_0.png
    :description: :doc:`demos/tutorial_variational_classifier`
    :tags: autograd

.. customgalleryitem::
    :tooltip: Fit one dimensional noisy data with a quantum neural network.
    :figure: demonstrations/quantum_neural_net/qnn_output_28_0.png
    :description: :doc:`demos/quantum_neural_net`
    :tags: autograd strawberryfields photonics

.. customgalleryitem::
    :tooltip: Find the ground state of a Hamiltonian.
    :figure: demonstrations/variational_quantum_eigensolver/pes_h2.png
    :description: :doc:`demos/tutorial_vqe`
    :tags: autograd chemistry

.. customgalleryitem::
    :tooltip: Universal Quantum Classifier with data-reuploading
    :figure: demonstrations/data_reuploading/universal_dnn.png
    :description: :doc:`demos/tutorial_data_reuploading_classifier`
    :tags: autograd

.. customgalleryitem::
    :tooltip: Faster optimization convergence using quantum natural gradient
    :figure: demonstrations/quantum_natural_gradient/qng_optimization.png
    :description: :doc:`demos/tutorial_quantum_natural_gradient`
    :tags: autograd

:html:`</div></div><div style='clear:both'>`

Exploring quantum effects/data
------------------------------
Add a descriptive text here (and update the header above if needed)

:html:`<div class="gallery-grid row">`

.. customgalleryitem::
    :tooltip: Perform QAOA for MaxCut
    :figure: demonstrations/qaoa_maxcut/qaoa_maxcut_partition.png
    :description: :doc:`demos/tutorial_qaoa_maxcut`
    :tags: autograd

.. customgalleryitem::
    :tooltip: Barren plateaus in quantum neural networks
    :figure: demonstrations/barren_plateaus/surface.png
    :description: :doc:`demos/tutorial_barren_plateaus`
    :tags: autograd

.. customgalleryitem::
    :tooltip: Rotoselect algorithm
    :figure: demonstrations/rotoselect/rotoselect_structure.png
    :description: :doc:`demos/tutorial_rotoselect`
    :tags: autograd

.. customgalleryitem::
    :tooltip: Doubly stochastic gradient descent
    :figure: demonstrations/doubly_stochastic/single_shot.png
    :description: :doc:`Doubly stochastic gradient descent <demos/tutorial_doubly_stochastic>`
    :tags: autograd

.. customgalleryitem::
    :tooltip: Variational Quantum Linear Solver
    :figure: demonstrations/vqls/vqls_zoom.png
    :description: :doc:`demos/tutorial_vqls`
    :tags: autograd

.. customgalleryitem::
    :tooltip: Coherent implementation of a variational quantum linear solver
    :figure: demonstrations/coherent_vqls/cvqls_zoom.png
    :description: :doc:`demos/tutorial_coherent_vqls`
    :tags: autograd

.. customgalleryitem::
    :tooltip: Quantum transfer learning
    :figure: demonstrations/quantum_transfer_learning/transfer_images.png
    :description: :doc:`demos/tutorial_quantum_transfer_learning`
    :tags: autograd pytorch

.. customgalleryitem::
    :tooltip: Training an embedding to perform metric learning
    :figure: demonstrations/embedding_metric_learning/training.png
    :description: :doc:`demos/tutorial_embeddings_metric_learning`
    :tags: autograd

.. customgalleryitem::
    :tooltip: Pre-process images with a quantum convolution
    :figure: demonstrations/quanvolution/zoom.png
    :description: :doc:`demos/tutorial_quanvolution`
    :tags: tensorflow

.. customgalleryitem::
    :tooltip: Implement a multiclass variational classifier using PyTorch, PennyLane, and the iris dataset
    :figure: demonstrations/multiclass_classification/margin_2.png
    :description: :doc:`demos/tutorial_multiclass_classification`
    :tags: pytorch

.. customgalleryitem::
    :tooltip: Frugal shot optimization with the Rosalin optimizer
    :figure: demonstrations/rosalin/rosalin_thumb.png
    :description: :doc:`demos/tutorial_rosalin`
    :tags: autograd

.. customgalleryitem::
    :tooltip: Differentiate any qubit gate with the stochastic parameter-shift rule
    :figure: demonstrations/stochastic_parameter_shift/stochastic_parameter_shift_thumbnail.png
    :description: :doc:`demos/tutorial_stochastic_parameter_shift`
    :tags: autograd

.. customgalleryitem::
    :tooltip: VQE optimization using quantum natural gradient
    :figure: demonstrations/vqe_qng/vqe_qng_thumbnail.png
    :description: :doc:`demos/tutorial_vqe_qng`
    :tags: chemistry

.. customgalleryitem::
    :tooltip: Optimizing measurement protocols with variational methods
    :figure: demonstrations/quantum_metrology/illustration.png
    :description: :doc:`demos/tutorial_quantum_metrology`
    :tags: cirq metrology autograd

.. customgalleryitem::
    :tooltip: Learn about the variational quantum thermalizer algorithm, an extension of VQE.
    :figure: demonstrations/vqt/thumbnail.png
    :description: :doc:`demos/tutorial_vqt`
    :tags: chemistry

.. customgalleryitem::
    :tooltip: Using a quantum graph recurrent neural network to learn quantum dynamics
    :figure: demonstrations/qgrnn/qgrnn_thumbnail.png
    :description: :doc:`demos/qgrnn`
    :tags: autograd

.. customgalleryitem::
    :tooltip: Optimize a Quantum Optical Neural Network using NLopt.
    :figure: demonstrations/qonn/qonn_thumbnail.png
    :description: :doc:`demos/qonn`
    :tags: autograd photonics

.. customgalleryitem::
    :tooltip: Understand the link between variational quantum models and Fourier series.
    :figure: demonstrations/expressivity_fourier_series/expressivity_thumbnail.png
    :description: :doc:`demos/tutorial_expressivity_fourier_series`
    :tags: autograd

.. customgalleryitem::
    :tooltip: Understand the difference between local and global cost functions
    :figure: demonstrations/local_cost_functions/Local_Thumbnail.png
    :description: :doc:`demos/tutorial_local_cost_functions`
    :tags: autograd

:html:`</div></div><div style='clear:both'>`

.. toctree::
    :maxdepth: 2
    :hidden:
