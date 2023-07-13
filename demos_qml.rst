 .. role:: html(raw)
   :format: html

Quantum machine learning
========================

.. meta::
   :property="og:description": Implementations of the latest cutting-edge ideas and research from quantum machine learning using PennyLane.
   :property="og:image": https://pennylane.ai/qml/_static/demos_card.png


Delve into the latest exciting research and cutting-edge ideas in
quantum machine learning. Implement and run a vast array of different QML
applications on your own computer—using simulators from Xanadu,
IBM, Google, Rigetti, and many more—or on real hardware devices.

.. raw:: html

    <link href="https://cdnjs.cloudflare.com/ajax/libs/mdbootstrap/4.8.10/css/mdb.min.css" rel="stylesheet">

:html:`<div class="gallery-grid row">`

.. gallery-item::
    :tooltip: Understand the link between variational quantum models and Fourier series.
    :figure: demonstrations/expressivity_fourier_series/expressivity_thumbnail.png
    :description: :doc:`demos/tutorial_expressivity_fourier_series`
    :tags: autograd

.. gallery-item::
    :tooltip: Kernels and alignment training with PennyLane.
    :figure: demonstrations/kernels_module/QEK_thumbnail.png
    :description: :doc:`demos/tutorial_kernels_module`
    :tags: kernels alignment classification

.. gallery-item::
    :tooltip: Kernel-based training with scikit-learn.
    :figure: demonstrations/kernel_based_training/scaling.png
    :description: :doc:`demos/tutorial_kernel_based_training`
    :tags: pytorch sklearn kernels

.. gallery-item::
    :tooltip: A quantum variational classifier.
    :figure: demonstrations/variational_classifier/classifier_output_59_0.png
    :description: :doc:`demos/tutorial_variational_classifier`
    :tags: autograd

.. gallery-item::
    :tooltip: Universal Quantum Classifier with data-reuploading.
    :figure: demonstrations/data_reuploading/universal_dnn.png
    :description: :doc:`demos/tutorial_data_reuploading_classifier`
    :tags: autograd

.. gallery-item::
    :tooltip: Quantum transfer learning.
    :figure: demonstrations/quantum_transfer_learning/transfer_images.png
    :description: :doc:`demos/tutorial_quantum_transfer_learning`
    :tags: autograd pytorch

.. gallery-item::
    :tooltip: Create a simple QGAN with Cirq and TensorFlow.
    :figure: demonstrations/QGAN/qgan3.png
    :description: :doc:`demos/tutorial_QGAN`
    :tags: cirq tensorflow

.. gallery-item::
    :tooltip: Fit one-dimensional noisy data with a quantum neural network.
    :figure: demonstrations/quantum_neural_net/qnn_output_28_0.png
    :description: :doc:`demos/quantum_neural_net`
    :tags: autograd strawberryfields photonics

.. gallery-item::
    :tooltip: Using a quantum graph recurrent neural network to learn quantum dynamics.
    :figure: demonstrations/qgrnn/qgrnn_thumbnail.png
    :description: :doc:`demos/tutorial_qgrnn`
    :tags: autograd

.. gallery-item::
    :tooltip: Meta-learning technique for variational quantum algorithms.
    :figure: demonstrations/learning2learn/l2l_thumbnail.png
    :description: :doc:`demos/learning2learn`
    :tags: tensorflow

.. gallery-item::
    :tooltip: Pre-process images with a quantum convolution.
    :figure: demonstrations/quanvolution/zoom.png
    :description: :doc:`demos/tutorial_quanvolution`
    :tags: tensorflow

.. gallery-item::
    :tooltip: Use multiple QPUs to improve classification.
    :figure: demonstrations/ensemble_multi_qpu/ensemble_diagram.png
    :description: :doc:`demos/ensemble_multi_qpu`
    :tags: pytorch rigetti qiskit

.. gallery-item::
    :tooltip: Generate images with Quantums GANs.
    :figure: demonstrations/quantum_gans/patch.jpeg
    :description: :doc:`demos/tutorial_quantum_gans`
    :tags: pytorch 

.. gallery-item::
    :tooltip: Estimate a classical kernel function on a quantum computer.
    :figure: demonstrations/classical_kernels/classical_kernels_flow_chart.png
    :description: :doc:`demos/tutorial_classical_kernels`
    :tags: kernels approximation
    
.. gallery-item::
    :tooltip: Tensor network quantum circuits
    :figure: demonstrations/tn_circuits/thumbnail_tn_circuits.png
    :description: :doc:`demos/tutorial_tn_circuits`
    :tags: tensor network

.. gallery-item::
    :tooltip: Quantum advantage in learning from experiments
    :figure: demonstrations/learning_from_experiments/learning_from_exp_thumbnail.png
    :description: :doc:`demos/tutorial_learning_from_experiments`
    :tags: advantage experiments
    
.. gallery-item::
    :tooltip: Machine learning for quantum many-body problems
    :figure: demonstrations/ml_classical_shadows/ml_classical_shadow.png
    :description: :doc:`demos/tutorial_ml_classical_shadows`
    :tags: kernels manybodyphysics classicalml

.. gallery-item::
    :tooltip: Train polynomial approximations to functions using QSP.
    :figure: demonstrations/function_fitting_qsp/cover.png
    :description: :doc:`demos/function_fitting_qsp`
    :tags: pytorch

.. gallery-item::
    :tooltip: Generalization in quantum machine learning from few training data
    :figure: demonstrations/learning_few_data/few_data_thumbnail.png
    :description: :doc:`demos/tutorial_learning_few_data`
    :tags: qcnn advantage 

.. gallery-item::
    :tooltip: Learn how to use symmetries to improve training with equivariant learning
    :figure: demonstrations/geometric_qml/equivariant_thumbnail.jpeg
    :description: :doc:`demos/tutorial_geometric_qml`
    :tags: pytorch geometric qml

.. gallery-item::
    :tooltip: Learn how to quantumly detect anomalous behaviour in time series data with the help of Covalent.
    :figure: demonstrations/univariate_qvr/thumbnail_tutorial_univariate_qvr.jpg
    :description: :doc:`demos/tutorial_univariate_qvr`
    :tags: covalent pytorch

:html:`</div></div><div style='clear:both'>`


.. toctree::
    :maxdepth: 2
    :hidden:

    demos/tutorial_expressivity_fourier_series
    demos/tutorial_kernels_module
    demos/tutorial_kernel_based_training
    demos/tutorial_variational_classifier
    demos/tutorial_data_reuploading_classifier
    demos/tutorial_quantum_transfer_learning
    demos/tutorial_QGAN
    demos/quantum_neural_net
    demos/tutorial_qgrnn
    demos/learning2learn
    demos/tutorial_quanvolution
    demos/ensemble_multi_qpu
    demos/tutorial_quantum_gans
    demos/tutorial_classical_kernels
    demos/tutorial_tn_circuits
    demos/tutorial_learning_from_experiments
    demos/tutorial_ml_classical_shadows
    demos/function_fitting_qsp
    demos/tutorial_learning_few_data
    demos/tutorial_geometric_qml
    demos/tutorial_univariate_qvr

