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

.. customgalleryitem::
    :tooltip: Extend PyTorch with real quantum computing power.
    :figure: demonstrations/pytorch_noise/bloch.gif
    :description: :doc:`demos/pytorch_noise`
    :tags: forest pytorch

.. customgalleryitem::
    :tooltip: Do arbitrary state preparation on a real quantum computer.
    :figure: demonstrations/state_preparation/NOON.png
    :description: :doc:`demos/tutorial_state_preparation`
    :tags: pytorch

.. customgalleryitem::
    :tooltip: Learn how noise can affect the optimization and training of quantum computations.
    :figure: demonstrations/noisy_circuit_optimization/noisy_circuit_optimization_thumbnail.png
    :description: :doc:`demos/tutorial_noisy_circuit_optimization`
    :tags: cirq

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

:html:`</div></div><div style='clear:both'>`

.. toctree::
    :maxdepth: 2
    :caption: Getting Started
    :hidden:

    demos/tutorial_qubit_rotation
    demos/tutorial_gaussian_transformation
    demos/tutorial_plugins_hybrid
    demos/tutorial_backprop
    demos/tutorial_state_preparation
    demos/pytorch_noise
    demos/tutorial_noisy_circuit_optimization
    demos/tutorial_qnn_module_tf
    demos/tutorial_qnn_module_torch
    demos/tutorial_jax_transformations
    demos/tutorial_noisy_circuits
    demos/braket-parallel-gradients
