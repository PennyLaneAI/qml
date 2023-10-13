 .. role:: html(raw)
   :format: html

Quantum Chemistry
=================

.. meta::
   :property="og:description": Explore and master quantum algorithms for quantum chemistry.
   :property="og:image": https://pennylane.ai/qml/_static/demos_card.png

Quantum chemistry is one of the leading application areas of quantum computers. Master the use of PennyLane to construct molecular Hamiltonians, design quantum circuits for quantum chemistry, and simulate properties of molecules and materials.

.. raw:: html

    <link href="https://cdnjs.cloudflare.com/ajax/libs/mdbootstrap/4.8.10/css/mdb.min.css" rel="stylesheet">

:html:`<div class="gallery-grid row">`

.. gallery-item::
    :tooltip: Explore quantum chemistry in PennyLane.
    :figure: demonstrations/quantum_chemistry/water_structure.png
    :description: :doc:`demos/tutorial_quantum_chemistry`
    :tags: chemistry

.. gallery-item::
    :tooltip: Find the ground state of a Hamiltonian.
    :figure: demonstrations/variational_quantum_eigensolver/pes_h2.png
    :description: :doc:`demos/tutorial_vqe`
    :tags: autograd chemistry

.. gallery-item::
    :tooltip: Discover the building blocks of quantum circuits for quantum chemistry
    :figure: demonstrations/givens_rotations/Givens_rotations.png
    :description: :doc:`demos/tutorial_givens_rotations`
    :tags: chemistry

.. gallery-item::
    :tooltip: Differentiable Hartree-Fock Solver.
    :figure: demonstrations/differentiable_HF/h2.png
    :description: :doc:`demos/tutorial_differentiable_HF`
    :tags: chemistry

.. gallery-item::
    :tooltip: Adaptive circuits for quantum chemistry.
    :figure: demonstrations/adaptive_circuits/thumbnail_adaptive_circuits.png
    :description: :doc:`demos/tutorial_adaptive_circuits`
    :tags: chemistry

.. gallery-item::
    :tooltip: Study chemical reactions using VQE.
    :figure: demonstrations/vqe_bond_dissociation/reaction.png
    :description: :doc:`demos/tutorial_chemical_reactions`

.. gallery-item::
    :tooltip: Optimizing the geometry of molecules.
    :figure: demonstrations/mol_geo_opt/fig_thumbnail.png
    :description: :doc:`demos/tutorial_mol_geo_opt`
    :tags: chemistry

.. gallery-item::
    :tooltip: VQE in different spin sectors
    :figure: demonstrations/vqe_spin_sectors/thumbnail_spectra_h2.png
    :description: :doc:`demos/tutorial_vqe_spin_sectors`
    :tags: chemistry

.. gallery-item::
    :tooltip: Optimize and reduce the number of measurements required to evaluate a variational algorithm cost function.
    :figure: demonstrations/measurement_optimize/meas_optimize_thumbnail.png
    :description: :doc:`demos/tutorial_measurement_optimize`
    :tags: chemistry

.. gallery-item::
    :tooltip: Evaluate the potential energy surface of H2 with parallel QPUs
    :figure: demonstrations/vqe_parallel/vqe_diagram.png
    :description: :doc:`demos/vqe_parallel`
    :tags: chemistry

.. gallery-item::
    :tooltip: Qubit tapering with symmetries
    :figure: demonstrations/qubit_tapering/qubit_tapering.png
    :description: :doc:`demos/tutorial_qubit_tapering`
    :tags: chemistry

.. gallery-item::
    :tooltip: Classically-boosted Variational Quantum Eigensolver
    :figure: demonstrations/classically_boosted_vqe/CB_VQE.png
    :description: :doc:`demos/tutorial_classically_boosted_vqe`
    :tags: chemistry

.. gallery-item::
    :tooltip: Quantum Resource Estimation.
    :figure: demonstrations/resource_estimation/resource_estimation.jpeg
    :description: :doc:`demos/tutorial_resource_estimation`
    :tags: chemistry

.. gallery-item::
    :tooltip: Using PennyLane with PySCF and OpenFermion
    :figure: demonstrations/external_libs/thumbnail_tutorial_external_libs.png
    :description: :doc:`demos/tutorial_qchem_external`
    :tags: chemistry

.. gallery-item::
    :tooltip: Fermionic Operators
    :figure: demonstrations/fermionic_operators/thumbnail_tutorial_fermionic_operators.png
    :description: :doc:`demos/tutorial_fermionic_operators`
    :tags: chemistry

.. gallery-item::
    :tooltip: Initial State Preparation for Quantum Chemistry
    :figure: demonstrations/resource_estimation/resource_estimation.jpeg
    :description: :doc:`demos/tutorial_initial_state_preparation`
    :tags: chemistry

:html:`</div></div><div style='clear:both'>`


.. toctree::
    :maxdepth: 2
    :hidden:

    demos/tutorial_quantum_chemistry
    demos/tutorial_vqe
    demos/tutorial_givens_rotations
    demos/tutorial_differentiable_HF
    demos/tutorial_adaptive_circuits
    demos/tutorial_chemical_reactions
    demos/tutorial_fermionic_operators
    demos/tutorial_mol_geo_opt
    demos/tutorial_vqe_spin_sectors
    demos/tutorial_measurement_optimize
    demos/vqe_parallel
    demos/tutorial_qubit_tapering
    demos/tutorial_classically_boosted_vqe
    demos/tutorial_qchem_external
    demos/tutorial_resource_estimation
    demos/tutorial_initial_state_preparation

