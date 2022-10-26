.. raw:: html

    <style>
        h1 {
            text-align: center;
        }
    </style>

Quantum Chemistry Datasets
===========================

One of the most promising directions for the current research in quantum algorithms is quantum chemistry, a field of critical importance as it allows one to enquire about the quantum properties of matter. To foster this momentum in the ongoing effort by the scientific community further, we share the first offering of our quantum chemistry dataset, which contains data related to some popularly examined molecular systems.

Molecules
~~~~~~~~~~

.. image:: /_static/datasets/bondlength.jpeg
    :align: right
    :width: 45%
    :target: javascript:void(0);

Through this dataset, it would be possible to access electronic structure data for the 42 different geometries of the following molecules:

* Linear hydrogen chains - H\ :sub:`2`, H\ :sub:`4`, H\ :sub:`6`, H\ :sub:`8`.
* Metallic and Non-metallic hydrides - LiH, BeH\ :sub:`2`, BH\ :sub:`3`, CH\ :sub:`4`, NH\ :sub:`3`, H\ :sub:`2`\ O, HF.
* Charged species - HeH\ :sup:`+`, H\ :sub:`3`\ :sup:`+`, OH\ :sup:`-`.

For the smaller molecules such as H\ :sub:`2`, HeH\ :sup:`+`, and H\ :sub:`3`\ :sup:`+`, data has been obtained for both minimal basis-set `STO-3G` and the split-valence double-zeta basis set `6-31G`. Whereas, for the rest of the other molecules, available data is for the former basis-set only. Moreover, for each molecule, the geometry is defined by equidistantly varying bond length around their central atom between :math:`[0.5 - 2.5]` Angstroms in 41 steps. In addition to these, we also include data for the optimal ground-state geometry of each molecule. 

Data Features
~~~~~~~~~~~~~~

For each of the molecules mentioned above, users will be able to extract the following characteristics for a geometry:

#. **Molecular Data**

    .. table::
        :widths: auto 

        +----------------------------+-----------------------------------------------------------------------------------+
        | `molecule`                 | PennyLane Molecule object                                                         |
        +----------------------------+-----------------------------------------------------------------------------------+
        | `hf_state`                 | Hartree-Fock state of the molecule                                                |
        +----------------------------+-----------------------------------------------------------------------------------+    
        | `fci_energy`               | Classical energy of the molecule from exact diagonalization.                      |
        +----------------------------+-----------------------------------------------------------------------------------+
        | `fci_spectrum`             | First few eigenvalues obtained from exact diagonalization.                        |
        +----------------------------+-----------------------------------------------------------------------------------+

#. **Hamiltonian Data**

    .. table::
        :widths: auto 
      
        +----------------------------+-----------------------------------------------------------------------------------+
        | `hamiltonian`              | PennyLane Hamiltonian in string format                                            |
        +----------------------------+-----------------------------------------------------------------------------------+
        | `meas_groupings`           | Measurement groupings for the Hamiltonian                                         |
        +----------------------------+-----------------------------------------------------------------------------------+
        | `wire_map`                 | Wire map for the Hamiltonian                                                      |
        +----------------------------+-----------------------------------------------------------------------------------+

#. **Auxillary Observables**
 
    .. table::
        :widths: auto

        +----------------------------+-----------------------------------------------------------------------------------+
        | `dipole_op`                | Dipole moment operators                                                           |
        +----------------------------+-----------------------------------------------------------------------------------+
        | `number_op`                | Number operator                                                                   |
        +----------------------------+-----------------------------------------------------------------------------------+
        | `spin2_op`                 | Total spin operator                                                               |
        +----------------------------+-----------------------------------------------------------------------------------+
        | `spinz_op`                 | Spin projection operator                                                          |
        +----------------------------+-----------------------------------------------------------------------------------+

#. **Tapering Data**

    .. table::
        :widths: auto

        +----------------------------+-----------------------------------------------------------------------------------+
        | `symmetries`               | Symmetries required for tapering molecular Hamiltonian                            |
        +----------------------------+-----------------------------------------------------------------------------------+
        | `paulix_ops`               | Supporting PauliX ops required to build Clifford U for tapering                   |
        +----------------------------+-----------------------------------------------------------------------------------+
        | `optimal_sector`           | Eigensector of the tapered qubits that would contain the ground state             |
        +----------------------------+-----------------------------------------------------------------------------------+


#. **Tapered Observables Data**

    .. table::
        :widths: auto

        +----------------------------+-----------------------------------------------------------------------------------+
        | `tapered_hamiltonian`      | Tapered Hamiltonian                                                               |
        +----------------------------+-----------------------------------------------------------------------------------+
        | `tapered_hf_state`         | Hartree-Fock state of the molecule                                                |
        +----------------------------+-----------------------------------------------------------------------------------+
        | `tapered_dipole_op`        | Tapered dipole moment operator                                                    |
        +----------------------------+-----------------------------------------------------------------------------------+
        | `tapered_num_op`           | Tapered number operator                                                           |
        +----------------------------+-----------------------------------------------------------------------------------+
        | `tapered_spin2_op`         | Tapered total spin operator                                                       |
        +----------------------------+-----------------------------------------------------------------------------------+
        | `tapered_spinz_op`         | Tapered spin projection operator                                                  |
        +----------------------------+-----------------------------------------------------------------------------------+

#. **VQE Data**

    .. table::
        :widths: auto

        +----------------------------+-----------------------------------------------------------------------------------+
        | `vqe_circuit`              | Circuit structure for AdaptiveGivens ansatz                                       |
        +----------------------------+-----------------------------------------------------------------------------------+
        | `vqe_params`               | Parameters for the AdaptiveGiven ansatz                                           |
        +----------------------------+-----------------------------------------------------------------------------------+
        | `vqe_energy`               | Energy obtained from VQE with the AdaptiveGivens ansatz                           |
        +----------------------------+-----------------------------------------------------------------------------------+


.. toctree::
    :maxdepth: 2
    :hidden:

