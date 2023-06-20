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

.. gallery-item::
    :tooltip: Learn how to compute the quantum volume of a quantum processor.
    :figure: demonstrations/quantum_volume/quantum_volume_thumbnail.png
    :description: :doc:`demos/quantum_volume`
    :tags: characterization qiskit

.. gallery-item::
    :tooltip: Learn how to sample quantum states uniformly at random
    :figure: demonstrations/haar_measure/spherical_int_dtheta.png
    :description: :doc:`demos/tutorial_haar_measure`
    :tags: quantumcomputing

.. gallery-item::
   :tooltip: Explore the amazing applications of unitary t-designs.
   :figure: demonstrations/unitary_designs/fano.png
   :description: :doc:`demos/tutorial_unitary_designs`
   :tags: quantumcomputing

.. gallery-item::
    :tooltip: Approximate quantum states with classical shadows.
    :figure: demonstrations/classical_shadows/atom_shadow.png
    :description: :doc:`demos/tutorial_classical_shadows`
    :tags: quantumcomputing characterization

.. gallery-item::
    :tooltip: Making a quantum machine learning model using neutral atoms
    :figure: demonstrations/pasqal/pasqal_thumbnail.png
    :description: :doc:`demos/tutorial_pasqal`
    :tags: cirq tensorflow

.. gallery-item::
    :tooltip: Beyond classical computing with qsim.
    :figure: demonstrations/qsim_beyond_classical/sycamore.png
    :description: :doc:`demos/qsim_beyond_classical`
    :tags: cirq qsim

.. gallery-item::
   :tooltip: Construct and simulate a Gaussian Boson Sampler.
   :figure: demonstrations/gbs_thumbnail.png
   :description: :doc:`demos/gbs`
   :tags: photonics strawberryfields

.. gallery-item::
    :tooltip: Quantum computing using trapped ions
    :figure: demonstrations/trapped_ions/trapped_ions_tn.png
    :description: :doc:`demos/tutorial_trapped_ions`
    :tags: quantumcomputing

.. gallery-item::
    :tooltip: Error mitigation with Mitiq and PennyLane
    :figure: demonstrations/error_mitigation/laptop.png
    :description: :doc:`demos/tutorial_error_mitigation`
    :tags: quantumcomputing

.. gallery-item::
    :tooltip: Quantum computing with superconducting qubits
    :figure: demonstrations/sc_qubits/sc_qubits_tn.png
    :description: :doc:`demos/tutorial_sc_qubits`
    :tags: quantumcomputing

.. gallery-item::
    :tooltip: Photonic quantum computers
    :figure: demonstrations/photonics/photonics_tn.png
    :description: :doc:`demos/tutorial_photonics`
    :tags: quantumcomputing

.. gallery-item::
    :tooltip: Learn about the toric code and its excitations
    :figure: demonstrations/toric_code/types_of_loops.png
    :description: :doc:`demos/tutorial_toric_code`
    :tags: errorcorrection
    
.. gallery-item::
    :tooltip: Learn how to simulate a large quantum circuits with smaller ones
    :figure: demonstrations/quantum_circuit_cutting/cutqc_logo.png
    :description: :doc:`demos/tutorial_quantum_circuit_cutting`
    :tags: quantumcomputing

.. gallery-item::
    :tooltip: Compare simultaneously measuring qubit-wise-commuting observables with classical shadows
    :figure: demonstrations/diffable_shadows/pauli_shadows.jpg
    :description: :doc:`demos/tutorial_diffable_shadows`
    :tags: classical shadows qubit wise commuting observables

.. gallery-item::
    :tooltip: Measurement-based quantum computation
    :figure: demonstrations/mbqc/thumbnail_mbqc.png
    :description: :doc:`demos/tutorial_mbqc`
    :tags: quantumcomputing MBQC
    
.. gallery-item::
    :tooltip: Test if a system possesses discrete symmetries
    :figure: demonstrations/testing_symmetry/thumbnail_tutorial_testing_symmetry.png
    :description: :doc:`demos/tutorial_testing_symmetry`
    :tags: quantumcomputing symmetry

.. gallery-item::
    :tooltip: Simulate differentiable pulse programs with qubits in PennyLane
    :figure: demonstrations/pulse_programming101/thumbnail_tutorial_pulse_programming.png
    :description: :doc:`demos/tutorial_pulse_programming101`
    :tags: jax pulses pulse programming gate quantum optimal control

.. gallery-item::
    :tooltip: Neutral atom-based quantum hardware
    :figure: demonstrations/neutral_atoms/thumbnail_tutorial_neutral_atoms.png
    :description: :doc:`demos/tutorial_neutral_atoms`
    :tags: quantumcomputing symmetry
    
.. gallery-item::
    :tooltip: Create and run a pulse program on neutral atom hardware
    :figure: demonstrations/ahs_aquila/thumbnail_tutorial_pulse_on_hardware.png
    :description: :doc:`demos/ahs_aquila`
    :tags: pulses pulse programming neutral atom hardware

.. gallery-item::
    :tooltip: Learn how to interpret the Bernstein-Vazirani algorithm with qutrits
    :figure: demonstrations/qutrits_bernstein_vazirani/thumbnail_tutorial_qutrits_bernstein_vazirani.png
    :description: :doc:`demos/tutorial_qutrits_bernstein_vazirani`
    :tags: qutrits algorithm

.. gallery-item::
    :tooltip: Master the basics of the quantum singular value transformation
    :figure: demonstrations/intro_qsvt/thumbnail_intro_qsvt.png
    :description: :doc:`demos/tutorial_intro_qsvt`
    :tags: qsvt quantumcomputing algorithms

.. gallery-item::
    :tooltip: Learn about circuit transformations and quantum circuit compilation with PennyLane
    :figure: demonstrations/circuit_compilation/thumbnail_tutorial_circuit_compilation.png
    :description: :doc:`demos/tutorial_circuit_compilation`
    :tags: quantumcomputing 
    
.. gallery-item::
    :tooltip: ZX calculus
    :figure: demonstrations/zx_calculus/thumbnail_tutorial_zx_calculus.png
    :description: :doc:`demos/tutorial_zx_calculus`
    :tags: quantumcomputing ZX calculus ZXH parameter shif

.. gallery-item::
    :tooltip: Learn about noise-aware zero noise extrapolation
    :figure: demonstrations/mitigation_advantage/thumbnail_tutorial_mitigation_advantage.png
    :description: :doc:`demos/tutorial_mitigation_advantage`
    :tags: quantumcomputing ZNE PEC zero noise extrapolation quantum advantage

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
    demos/gbs
    demos/tutorial_trapped_ions
    demos/tutorial_error_mitigation
    demos/tutorial_sc_qubits
    demos/tutorial_photonics
    demos/tutorial_toric_code
    demos/tutorial_quantum_circuit_cutting
    demos/tutorial_testing_symmetry
    demos/tutorial_diffable_shadows
    demos/tutorial_mbqc
    demos/tutorial_zx_calculus
    demos/tutorial_pulse_programming101
    demos/tutorial_neutral_atoms
    demos/ahs_aquila
    demos/tutorial_qutrits_bernstein_vazirani
    demos/tutorial_circuit_compilation
    demos/tutorial_intro_qsvt

