 .. role:: html(raw)
   :format: html

Learn QML/PennyLane
===================

.. meta::
   :property="og:description": Take a deeper dive into quantum machine learning by exploring cutting-edge algorithms using PennyLane and near-term quantum hardware.
   :property="og:image": https://pennylane.ai/qml/_static/demos_card.png


Learn QML/PennyLane demos page. Note, the demos below aren't yet sorted into the
correct sections, and some of them might need to be moved to the Research page.

.. raw:: html

    <link href="https://cdnjs.cloudflare.com/ajax/libs/mdbootstrap/4.8.10/css/mdb.min.css" rel="stylesheet">


Basic algorithms
----------------
Add a descriptive text here (and update the header above if needed)

:html:`<div class="gallery-grid row">`

.. customgalleryitem::
    :tooltip: Use quantum machine learning to rotate a qubit.
    :figure: demonstrations/qubit_rotation/bloch.png
    :description: :doc:`demos/tutorial_qubit_rotation`
    :tags: autograd beginner

.. customgalleryitem::
    :tooltip: Use quantum machine learning to tune a beamsplitter.
    :figure: demonstrations/gaussian_transformation/gauss-circuit.png
    :description: :doc:`demos/tutorial_gaussian_transformation`
    :tags: autograd photonics beginner

.. customgalleryitem::
    :tooltip: Use quantum machine learning in a multi-device quantum algorithm.
    :figure: demonstrations/plugins_hybrid/photon_redirection.png
    :description: :doc:`demos/tutorial_plugins_hybrid`
    :tags: autograd photonics beginner

:html:`</div></div><div style='clear:both'>`

Using near term devices
-----------------------
Add a descriptive text here (and update the header above if needed)

:html:`<div class="gallery-grid row">`

.. customgalleryitem::
    :tooltip: Multiple expectation values, Jacobians, and keyword arguments.
    :description: :doc:`demos/tutorial_advanced_usage`
    :tags: autograd beginner

.. customgalleryitem::
    :tooltip: Extend PyTorch with real quantum computing power.
    :figure: demonstrations/pytorch_noise/bloch.gif
    :description: :doc:`demos/pytorch_noise`
    :tags: forest pytorch beginner

.. customgalleryitem::
    :tooltip: Explore quantum chemistry in PennyLane.
    :figure: demonstrations/quantum_chemistry/water_structure.png
    :description: :doc:`demos/tutorial_quantum_chemistry`
    :tags: chemistry beginner

.. customgalleryitem::
    :tooltip: Do arbitrary state preparation on a real quantum computer.
    :figure: demonstrations/state_preparation/NOON.png
    :description: :doc:`demos/tutorial_state_preparation`
    :tags: pytorch

:html:`</div></div><div style='clear:both'>`

Learning qml
------------
Add a descriptive text here (and update the header above if needed)

:html:`<div class="gallery-grid row">`

.. customgalleryitem::
    :tooltip: Evaluate the potential energy surface of H2 with parallel QPUs
    :figure: demonstrations/vqe_parallel/vqe_diagram.png
    :description: :doc:`demos/tutorial_vqe_parallel`
    :tags: chemistry

.. customgalleryitem::
    :tooltip: Use multiple QPUs to improve classification
    :figure: demonstrations/ensemble_multi_qpu/ensemble_diagram.png
    :description: :doc:`demos/tutorial_ensemble_multi_qpu`
    :tags: pytorch forest qiskit

.. customgalleryitem::
    :tooltip: Learn how noise can affect the optimization and training of quantum computations
    :figure: demonstrations/noisy_circuit_optimization/noisy_circuit_optimization_thumbnail.png
    :description: :doc:`demos/tutorial_noisy_circuit_optimization`
    :tags: cirq beginner

.. customgalleryitem::
    :tooltip: Compare the parameter-shift rule with backpropagation
    :figure: demonstrations/tutorial_backprop_thumbnail.png
    :description: :doc:`demos/tutorial_backprop`
    :tags: tensorflow autograd:html:`</div></div><div style='clear:both'>`

:html:`</div></div><div style='clear:both'>`

.. toctree::
    :maxdepth: 2
    :caption: Learning
    :hidden:
