.. role:: html(raw)
   :format: html

Getting Started
===============

.. meta::
   :property="og:description": Begin your journey into quantum machine learning using PennyLane by exploring tutorials and basic applications.
   :property="og:image": https://pennylane.ai/qml/_static/demos_card.png

Here you can discover the basic tools needed to use PennyLane through simple demonstrations. Learn about training a circuit to rotate a qubit, machine learning tools to optimize quantum circuits, and introductory examples of photonic quantum computing.

.. raw:: html

    <link href="https://cdnjs.cloudflare.com/ajax/libs/mdbootstrap/4.8.10/css/mdb.min.css" rel="stylesheet">

:html:`<div class="gallery-grid row">`

.. gallery-item::
    :tooltip: Use quantum machine learning to rotate a qubit.
    :figure: demonstrations/qubit_rotation/bloch.png
    :description: :doc:`demos/tutorial_qubit_rotation`
    :tags: autograd

.. gallery-item::
    :tooltip: Compare the parameter-shift rule with backpropagation.
    :figure: demonstrations/tutorial_backprop_thumbnail.png
    :description: :doc:`demos/tutorial_backprop`
    :tags: tensorflow autograd

.. gallery-item::
    :tooltip: Learn how to compute gradients of quantum circuits with the adjoint method.
    :figure: demonstrations/adjoint_diff/icon.png
    :description: :doc:`demos/tutorial_adjoint_diff`
    :tags: 

.. gallery-item::
    :tooltip: Use quantum machine learning in a multi-device quantum algorithm.
    :figure: demonstrations/plugins_hybrid/photon_redirection.png
    :description: :doc:`demos/plugins_hybrid`
    :tags: autograd photonics strawberryfields

.. gallery-item::
    :tooltip: Simulate noisy quantum computations.
    :figure: demonstrations/noisy_circuits/N-Nisq.png
    :description: :doc:`demos/tutorial_noisy_circuits`
    :tags: beginner

.. gallery-item::
    :tooltip: Use quantum machine learning to tune a beamsplitter.
    :figure: demonstrations/gaussian_transformation/gauss-circuit.png
    :description: :doc:`demos/tutorial_gaussian_transformation`
    :tags: autograd photonics

.. gallery-item::
    :tooltip: Parallelize gradient calculations with Amazon Braket
    :figure: _static/pl-braket.png
    :description: :doc:`demos/braket-parallel-gradients`
    :tags: braket

.. gallery-item::
    :tooltip: Learn how to use JAX with PennyLane.
    :figure: demonstrations/jax_logo/jax.png
    :description: :doc:`demos/tutorial_jax_transformations`
    :tags: beginner

.. gallery-item::
    :tooltip: Learn how to create hybrid ML models using Keras
    :figure: _static/Keras_logo.png
    :description: :doc:`demos/tutorial_qnn_module_tf`
    :tags: tensorflow

.. gallery-item::
    :tooltip: Learn how to create hybrid ML models using Torch
    :figure: _static/PyTorch_icon.png
    :description: :doc:`demos/tutorial_qnn_module_torch`
    :tags: pytorch

.. gallery-item::
    :tooltip: Do arbitrary state preparation on a real quantum computer.
    :figure: demonstrations/state_preparation/NOON.png
    :description: :doc:`demos/tutorial_state_preparation`
    :tags: pytorch

.. gallery-item::
    :tooltip: Extend PyTorch with real quantum computing power.
    :figure: demonstrations/pytorch_noise/bloch.gif
    :description: :doc:`demos/pytorch_noise`
    :tags: rigetti pytorch

.. gallery-item::
    :tooltip: Learn how noise can affect the optimization and training of quantum computations.
    :figure: demonstrations/noisy_circuit_optimization/noisy_circuit_optimization_thumbnail.png
    :description: :doc:`demos/tutorial_noisy_circuit_optimization`
    :tags: cirq

.. gallery-item::
    :tooltip: Implement basic arithmetic operations using the quantum Fourier transform (QFT)
    :figure: demonstrations/qft_arithmetics/qft_arithmetics_thumbnail.png
    :description: :doc:`demos/tutorial_qft_arithmetics`
    :tags: qft qc short

.. gallery-item::
    :tooltip: Use phase kickback to create an unbreakable quantum lock.
    :figure: demonstrations/phase_kickback/thumbnail_tutorial_phase_kickback.png
    :description: :doc:`demos/tutorial_phase_kickback`

.. gallery-item::
    :tooltip: Use IBM devices with PennyLane through the pennylane-qiksit plugin
    :figure: demonstrations/ibm_pennylane/thumbnail_tutorial_ibm_pennylane.png
    :description: :doc:`demos/ibm_pennylane`
    :tags: IBM qiskit pennylane superconducting device runtime IBMQ hybrid algorithm

.. gallery-item::
    :tooltip: Learn with this interactive, code-free introduction to the idea of quantum circuits as Fourier series.
    :figure: demonstrations/circuits_as_fourier_series/thumbnail_circuits_as_fourier_series.png
    :description: :doc:`demos/circuits_as_fourier_series`
    :tags: fourier

.. gallery-item::
    :tooltip: Run variational algorithms on QPUs with Amazon Braket and PennyLane 
    :figure: demonstrations/getting_started_with_hybrid_jobs/thumbnail_getting_started_with_hybrid_jobs.png
    :description: :doc:`demos/getting_started_with_hybrid_jobs`
    :tags: braket


:html:`</div></div><div style='clear:both'>`

.. toctree::
    :maxdepth: 2
    :caption: Getting Started
    :hidden:

    demos/tutorial_qubit_rotation
    demos/tutorial_backprop
    demos/tutorial_adjoint_diff
    demos/plugins_hybrid
    demos/tutorial_noisy_circuits
    demos/tutorial_gaussian_transformation
    demos/braket-parallel-gradients
    demos/tutorial_jax_transformations
    demos/tutorial_qnn_module_tf
    demos/tutorial_qnn_module_torch
    demos/tutorial_state_preparation
    demos/pytorch_noise
    demos/tutorial_noisy_circuit_optimization
    demos/tutorial_qft_arithmetics
    demos/tutorial_phase_kickback
    demos/ibm_pennylane
    demos/getting_started_with_hybrid_jobs
