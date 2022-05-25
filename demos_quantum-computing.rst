 .. role:: html(raw)
   :format: html

Quantum Computing
=================

.. meta::
   :property="og:description": Explore the applications of PennyLane to general quantum computing tasks such as benchmarking and characterizing quantum processors.
   :property="og:image": https://pennylane.ai/qml/_static/demos_card.png


Explore the applications of PennyLane to general quantum computing tasks
such as benchmarking and characterizing quantum processors.

.. raw:: html

    <link href="https://cdnjs.cloudflare.com/ajax/libs/mdbootstrap/4.8.10/css/mdb.min.css" rel="stylesheet">

:html:`<div class="gallery-grid row">`

.. customgalleryitem::
    :tooltip: Learn how to compute the quantum volume of a quantum processor.
    :figure: demonstrations/quantum_volume/quantum_volume_thumbnail.png
    :description: :doc:`demos/quantum_volume`
    :tags: characterization qiskit

.. customgalleryitem::
    :tooltip: Learn how to sample quantum states uniformly at random
    :figure: demonstrations/haar_measure/spherical_int_dtheta.png
    :description: :doc:`demos/tutorial_haar_measure`
    :tags: quantumcomputing

.. customgalleryitem::
   :tooltip: Explore the amazing applications of unitary t-designs.
   :figure: demonstrations/unitary_designs/fano.png
   :description: :doc:`demos/tutorial_unitary_designs`
   :tags: quantumcomputing

.. customgalleryitem::
    :tooltip: Approximate quantum states with classical shadows.
    :figure: demonstrations/classical_shadows/atom_shadow.png
    :description: :doc:`demos/tutorial_classical_shadows`
    :tags: quantumcomputing characterization

.. customgalleryitem::
    :tooltip: Making a quantum machine learning model using neutral atoms
    :figure: demonstrations/pasqal/pasqal_thumbnail.png
    :description: :doc:`demos/tutorial_pasqal`
    :tags: cirq tensorflow

.. customgalleryitem::
    :tooltip: Beyond classical computing with qsim.
    :figure: demonstrations/qsim_beyond_classical/sycamore.png
    :description: :doc:`demos/qsim_beyond_classical`
    :tags: cirq qsim

.. customgalleryitem::
   :tooltip: Construct and simulate a Gaussian Boson Sampler.
   :figure: demonstrations/tutorial_gbs_thumbnail.png
   :description: :doc:`demos/tutorial_gbs`
   :tags: photonics strawberryfields

.. customgalleryitem::
    :tooltip: Quantum computing using trapped ions
    :figure: demonstrations/trapped_ions/trapped_ions_tn.png
    :description: :doc:`demos/tutorial_trapped_ions`
    :tags: quantumcomputing

.. customgalleryitem::
    :tooltip: Error mitigation with Mitiq and PennyLane
    :figure: demonstrations/error_mitigation/laptop.png
    :description: :doc:`demos/tutorial_error_mitigation`
    :tags: quantumcomputing

.. customgalleryitem::
    :tooltip: Quantum computing with superconducting qubits
    :figure: demonstrations/sc_qubits/sc_qubits_tn.png
    :description: :doc:`demos/tutorial_sc_qubits`
    :tags: quantumcomputing

:html:`</div></div><div style='clear:both'>`

.. toctree::
    :maxdepth: 2
    :hidden:

    demos/quantum_volume
    demos/tutorial_haar_measure
    demos/tutorial_unitary_designs
    demos/tutorial_classical_shadows
    demos/tutorial_pasqal
    demos/qsim_beyond_classical
    demos/tutorial_gbs
    demos/tutorial_trapped_ions
    demos/tutorial_error_mitigation
  	demos/tutorial_sc_qubits
