 .. role:: html(raw)
   :format: html

Basics
======

.. meta::
   :property="og:description": Begin your journey into quantum machine learning using PennyLane by exploring tutorials and basic applications.
   :property="og:image": https://pennylane.ai/qml/_static/demos_card.png

Begin your journey into PennyLane and Quantum Machine Learning (QML) by
exploring the tutorials below.

The `Getting started`_ section is a great place to start if you've just
discovered PennyLane and QML and want to learn more about the basics. Or venture
straight into the `Applications`_ section and explore how to implement trainable
circuits for popular applications such as Variational Quantum Eigensolvers and
Quantum Chemistry, using either simulators or near-term quantum hardware.

.. raw:: html

    <link href="https://cdnjs.cloudflare.com/ajax/libs/mdbootstrap/4.8.10/css/mdb.min.css" rel="stylesheet">


Getting started
---------------
Here you can discover the basic tools needed to use PennyLane through simple demonstrations. Learn about training a circuit to rotate a qubit, machine
learning tools to optimize quantum circuits, and introductory examples of
photonic quantum computing.

:html:`<div class="gallery-grid row">`

.. customgalleryitem::
    :tooltip: Use quantum machine learning to rotate a qubit.
    :figure: demonstrations/qubit_rotation/bloch.png
    :description: :doc:`demos/tutorial_qubit_rotation`
    :tags: autograd

.. customgalleryitem::
    :tooltip: Use quantum machine learning to tune a beamsplitter.
    :figure: demonstrations/gaussian_transformation/gauss-circuit.png
    :description: :doc:`demos/tutorial_gaussian_transformation`
    :tags: autograd photonics

.. customgalleryitem::
    :tooltip: Use quantum machine learning in a multi-device quantum algorithm.
    :figure: demonstrations/plugins_hybrid/photon_redirection.png
    :description: :doc:`demos/tutorial_plugins_hybrid`
    :tags: autograd photonics strawberryfields

.. customgalleryitem::
    :tooltip: Compare the parameter-shift rule with backpropagation.
    :figure: demonstrations/tutorial_backprop_thumbnail.png
    :description: :doc:`demos/tutorial_backprop`
    :tags: tensorflow autograd

.. customgalleryitem::
    :tooltip: Simulate noisy quantum computations.
    :figure: demonstrations/noisy_circuits/N-Nisq.png
    :description: :doc:`demos/tutorial_noisy_circuits`
    :tags: beginner
       
:html:`</div></div><div style='clear:both'>`

Applications
------------
Get familiar with more advanced applications of PennyLane and quantum machine
learning in this section. Learn how to implement a variational quantum
eigensolver, play around with quantum chemistry simulations, solve graph problems such as
MaxCut, or implement quantum machine learning circuits on real near-term
hardware.

:html:`<div class="gallery-grid row">`

.. customgalleryitem::
    :tooltip: Explore quantum chemistry in PennyLane.
    :figure: demonstrations/quantum_chemistry/water_structure.png
    :description: :doc:`demos/tutorial_quantum_chemistry`
    :tags: chemistry

.. customgalleryitem::
    :tooltip: Find the ground state of a Hamiltonian.
    :figure: demonstrations/variational_quantum_eigensolver/pes_h2.png
    :description: :doc:`demos/tutorial_vqe`
    :tags: autograd chemistry

.. customgalleryitem::
    :tooltip: A quantum variational classifier.
    :figure: demonstrations/variational_classifier/classifier_output_59_0.png
    :description: :doc:`demos/tutorial_variational_classifier`
    :tags: autograd

.. customgalleryitem::
    :tooltip: Learn how to implement QAOA workflows with PennyLane
    :figure: demonstrations/qaoa_module/qaoa_layer.png
    :description: :doc:`demos/tutorial_qaoa_intro`
    :tags: autograd beginner

.. customgalleryitem::
    :tooltip: Perform QAOA for MaxCut.
    :figure: demonstrations/qaoa_maxcut/qaoa_maxcut_partition.png
    :description: :doc:`demos/tutorial_qaoa_maxcut`
    :tags: autograd

.. customgalleryitem::
    :tooltip: Do arbitrary state preparation on a real quantum computer.
    :figure: demonstrations/state_preparation/NOON.png
    :description: :doc:`demos/tutorial_state_preparation`
    :tags: pytorch

.. customgalleryitem::
    :tooltip: Ising model example with PennyLane PyTorch interface.
    :figure: demonstrations/Ising_model/isingspins.png
    :description: :doc:`demos/tutorial_isingmodel_PyTorch`
    :tags: pytorch autograd

.. customgalleryitem::
    :tooltip: Extend PyTorch with real quantum computing power.
    :figure: demonstrations/pytorch_noise/bloch.gif
    :description: :doc:`demos/pytorch_noise`
    :tags: forest pytorch

.. customgalleryitem::
    :tooltip: Learn how noise can affect the optimization and training of quantum computations.
    :figure: demonstrations/noisy_circuit_optimization/noisy_circuit_optimization_thumbnail.png
    :description: :doc:`demos/tutorial_noisy_circuit_optimization`
    :tags: cirq

.. customgalleryitem::
    :tooltip: Evaluate the potential energy surface of H2 with parallel QPUs
    :figure: demonstrations/vqe_parallel/vqe_diagram.png
    :description: :doc:`demos/tutorial_vqe_parallel`
    :tags: chemistry

.. customgalleryitem::
    :tooltip: Use multiple QPUs to improve classification.
    :figure: demonstrations/ensemble_multi_qpu/ensemble_diagram.png
    :description: :doc:`demos/tutorial_ensemble_multi_qpu`
    :tags: pytorch forest qiskit

.. customgalleryitem::
    :tooltip: VQE in different spin sectors
    :figure: demonstrations/vqe_uccsd_obs/thumbnail_spectra_h2.png
    :description: :doc:`demos/tutorial_vqe_uccsd_obs`
    :tags: chemistry
    
.. customgalleryitem::
    :tooltip: Learn how to create hybrid ML models using Keras
    :figure: _static/Keras_logo.png
    :description: :doc:`demos/tutorial_qnn_module_tf`
    :tags: tensorflow

.. customgalleryitem::
    :tooltip: Learn how to create hybrid ML models using Torch
    :figure: _static/PyTorch_icon.png
    :description: :doc:`demos/tutorial_qnn_module_torch`
    :tags: pytorch

.. customgalleryitem::
    :tooltip: Learn how to use JAX with PennyLane.
    :figure: demonstrations/jax_logo/jax.png
    :description: :doc:`demos/tutorial_jax_transformations`
    :tags: beginner

.. customgalleryitem::
    :tooltip: Parallelize gradient calculations with Amazon Braket
    :figure: _static/pl-braket.png
    :description: :doc:`demos/braket-parallel-gradients`
    :tags: braket

.. customgalleryitem::
    :tooltip: Givens rotations
    :figure: demonstrations/givens_rotations/Givens_rotations.png
    :description: :doc:`demos/tutorial_givens_rotations`
    :tags: chemistry

:html:`</div></div><div style='clear:both'>`

.. toctree::
    :maxdepth: 2
    :caption: Basics
    :hidden:

    demos/tutorial_qubit_rotation
    demos/tutorial_gaussian_transformation
    demos/tutorial_plugins_hybrid
    demos/tutorial_backprop
    demos/tutorial_quantum_chemistry
    demos/tutorial_vqe
    demos/tutorial_variational_classifier
    demos/tutorial_qaoa_maxcut
    demos/tutorial_state_preparation
    demos/tutorial_isingmodel_PyTorch
    demos/pytorch_noise
    demos/tutorial_noisy_circuit_optimization
    demos/tutorial_vqe_parallel
    demos/tutorial_ensemble_multi_qpu
    demos/tutorial_qaoa_intro
    demos/tutorial_qnn_module_tf
    demos/tutorial_qnn_module_torch
    demos/tutorial_jax_transformations
    demos/tutorial_noisy_circuits
    demos/braket-parallel-gradients
    demos/tutorial_givens_rotations
