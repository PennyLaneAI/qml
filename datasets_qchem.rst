.. raw:: html

    <style>
        h1 {
            text-align: center;
        }
    </style>

Quantum Chemistry Datasets
==========================

.. meta::
   :property="og:description": Browse our collection of quantum datasets, and import them into PennyLane directly from your code.
   :property="og:image": https://pennylane.ai/qml/_static/datasets.png

One of the most promising directions for current research in quantum algorithms is quantum chemistry, a field of critical importance, as it allows one to inquire about the quantum properties of matter.

Explore our available quantum chemistry datasets below, providing data related to some popularly examined molecular systems.

Molecules
---------

.. image:: /_static/datasets/bondlength.jpeg
    :align: right
    :width: 45%
    :target: javascript:void(0);

Through this dataset, it is possible to access electronic structure data for the 42 different geometries of the following molecules:

* Linear hydrogen chains - H\ :sub:`2`, H\ :sub:`4`, H\ :sub:`6`, H\ :sub:`8`.
* Metallic and non-metallic hydrides - LiH, BeH\ :sub:`2`, BH\ :sub:`3`, CH\ :sub:`4`, NH\ :sub:`3`, H\ :sub:`2`\ O, HF.
* Charged species - HeH\ :sup:`+`, H\ :sub:`3`\ :sup:`+`, OH\ :sup:`-`.

For the smaller molecules such as H\ :sub:`2`, HeH\ :sup:`+`, and H\ :sub:`3`\ :sup:`+`, data has been obtained for both minimal basis-set `STO-3G` and the split-valence double-zeta basis set `6-31G`. 
For the remaining molecules, available data is for the former basis-set only. Moreover, for each molecule, the geometry is defined by equidistantly varying bond lengths around their central atom in 41 steps. 
In addition to these, we also include data for the optimal ground-state geometry of each molecule. 
We summarise all of this information for all the molecules in the table below:

.. raw:: html

     <style>
        .docstable tr.row-even th, .docstable tr.row-even td {
            text-align: center;
        }
        .docstable tr.row-odd th, .docstable tr.row-odd td {
            text-align: center;
        }
    </style>
    <div class="d-flex justify-content-center">

.. rst-class:: docstable
    :widths: auto
    :align: center

    +------------------------------+-------------------------------------+--------------+---------------------------------------------------------------------------+----------------------------------------------------------------------------+
    | **Molecule**                 | **Basis Set(s)**                    | **#Qubits**  | **Bondlength (Å)**                                                        | **Optimal Geometry (Å)**                                                   |
    +==============================+=====================================+==============+===========================================================================+============================================================================+
    | H\ :math:`_2`                | | STO\ :math:`-`\3G /               | 4 / 8        | H\ :math:`_A-`\ H\ :math:`_B` :math:`\in\ [0.5, 2.1]`                     | H\ :math:`_A-`\ H\ :math:`_B = 0.742`                                      |
    |                              | | 6\ :math:`-`\31G                  |              |                                                                           |                                                                            |   
    +------------------------------+-------------------------------------+--------------+---------------------------------------------------------------------------+----------------------------------------------------------------------------+
    | HeH\ :math:`^+`              | | STO\ :math:`-`\3G /               | 4 / 8        | He\ :math:`-`\ H :math:`\in\ [0.5, 2.1]`                                  | He\ :math:`-`\ H\ :math:`= 0.7748`                                         |
    |                              | | 6\ :math:`-`\31G                  |              |                                                                           |                                                                            |           
    +------------------------------+-------------------------------------+--------------+---------------------------------------------------------------------------+----------------------------------------------------------------------------+
    | H\ :math:`_3^+`              | | STO\ :math:`-`\3G /               | 6 / 12       | | H\ :math:`_A-`\ H\ :math:`_B` :math:`\in\ [0.5, 2.1]`,                  | | H\ :math:`_A-`\ H\ :math:`_B = 0.8737`,                                  |
    |                              | | 6\ :math:`-`\31G                  |              | | H\ :math:`_A-`\ H\ :math:`_B-`\ H\ :math:`_C = 60^{\circ}`              | | H\ :math:`_A-`\ H\ :math:`_B-`\ H\ :math:`_C = 60^{\circ}`               |
    +------------------------------+-------------------------------------+--------------+---------------------------------------------------------------------------+----------------------------------------------------------------------------+
    | H\ :math:`_4`                | STO\ :math:`-`\3G                   | 8            | | H\ :math:`_A-`\ H\ :math:`_B` :math:`\in\ [0.5, 1.3]`,                  |          N/A                                                               |
    |                              |                                     |              | | H\ :math:`_A-`\ H\ :math:`_B-\ldots-`\ H\ :math:`_D = 180^{\circ}`      |                                                                            |
    +------------------------------+-------------------------------------+--------------+---------------------------------------------------------------------------+----------------------------------------------------------------------------+
    | LiH                          | STO\ :math:`-`\3G                   | 12           | Li\ :math:`-`\ H :math:`\in\ [0.9, 2.1]`                                  | Li\ :math:`-`\ H :math:`= 1.5699`                                          |
    +------------------------------+-------------------------------------+--------------+---------------------------------------------------------------------------+----------------------------------------------------------------------------+
    | HF                           | STO\ :math:`-`\3G                   | 12           | H\ :math:`-`\ F :math:`\in\ [0.5, 2.1]`                                   | H\ :math:`-`\ F :math:`= 0.9167`                                           |
    +------------------------------+-------------------------------------+--------------+---------------------------------------------------------------------------+----------------------------------------------------------------------------+
    | OH\ :math:`^-`               | STO\ :math:`-`\3G                   | 12           | [O\ :math:`-`\ H]\ :math:`^-` :math:`\in\ [0.5, 2.1]`                     | [O\ :math:`-`\ H]\ :math:`^- = 0.9638`                                     |
    +------------------------------+-------------------------------------+--------------+---------------------------------------------------------------------------+----------------------------------------------------------------------------+
    | H\ :math:`_6`                | STO\ :math:`-`\3G                   | 12           | | H\ :math:`_A-`\ H\ :math:`_B` :math:`\in\ [0.5, 1.3]`                   |          N/A                                                               |
    |                              |                                     |              | | H\ :math:`_A-`\ H\ :math:`_B-\ldots-`\ H\ :math:`_F = 180^{\circ}`      |                                                                            |
    +------------------------------+-------------------------------------+--------------+---------------------------------------------------------------------------+----------------------------------------------------------------------------+
    | BeH\ :math:`_2`              | STO\ :math:`-`\3G                   | 14           | | Be\ :math:`-`\ H :math:`\in\ [0.5, 2.1]`,                               | | Be\ :math:`-`\ H :math:`=1.3295`,                                        |
    |                              |                                     |              | | H\ :math:`_A`\ :math:`-`\ Be\ :math:`-`\ H\ :math:`_B = 180^{\circ}`    | | H\ :math:`_A`\ :math:`-`\ Be\ :math:`-`\ H\ :math:`_B = 180^{\circ}`     |
    +------------------------------+-------------------------------------+--------------+---------------------------------------------------------------------------+----------------------------------------------------------------------------+
    | H\ :math:`_2`\ O             | STO\ :math:`-`\3G                   | 16           | | H\ :math:`-`\ O :math:`\in [0.5, 2.1]`,                                 | | H\ :math:`-`\ O :math:`=0.9575`,                                         |
    |                              |                                     |              | | H\ :math:`_A`\ :math:`-`\ O\ :math:`-`\ H\ :math:`_B = 104.5^{\circ}`   | | H\ :math:`_A`\ :math:`-`\ O\ :math:`-`\ H\ :math:`_B = 104.5^{\circ}`    |
    +------------------------------+-------------------------------------+--------------+---------------------------------------------------------------------------+----------------------------------------------------------------------------+
    | H\ :math:`_8`                | STO\ :math:`-`\3G                   | 16           | | H\ :math:`_A-`\ H\ :math:`_B` :math:`\in\ [0.5, 1.3]`,                  |          N/A                                                               |
    |                              |                                     |              | | H\ :math:`_A-`\ H\ :math:`_B-\ldots-`\ H\ :math:`_H = 180^{\circ}`      |                                                                            |
    +------------------------------+-------------------------------------+--------------+---------------------------------------------------------------------------+----------------------------------------------------------------------------+
    | BH\ :math:`_3`               | STO\ :math:`-`\3G                   | 18           | | B\ :math:`-`\ H :math:`\in\ [0.5, 2.1]`                                 | | B\ :math:`-`\ H :math:`=1.1893`,                                         |
    |                              |                                     |              | | H\ :math:`_A`\ :math:`-`\ B\ :math:`-`\ H\ :math:`_B = 120^{\circ}`     | | H\ :math:`_A`\ :math:`-`\ B\ :math:`-`\ H\ :math:`_B = 120^{\circ}`      |
    +------------------------------+-------------------------------------+--------------+---------------------------------------------------------------------------+----------------------------------------------------------------------------+
    | NH\ :math:`_3`               | STO\ :math:`-`\3G                   | 18           | | N\ :math:`-`\ H :math:`\in\ [0.5, 2.1]`                                 | | N\ :math:`-`\ H :math:`=1.1096`,                                         |
    |                              |                                     |              | | H\ :math:`_A`\ :math:`-`\ N\ :math:`-`\ H\ :math:`_B = 106.8^{\circ}`   | | H\ :math:`_A`\ :math:`-`\ N\ :math:`-`\ H\ :math:`_B = 106.8`^{\circ}`   |
    +------------------------------+-------------------------------------+--------------+---------------------------------------------------------------------------+----------------------------------------------------------------------------+
    | H\ :math:`_3`\ O\ :math:`^+` | STO\ :math:`-`\3G                   | 18           | | [H\ :math:`-`\ O]\ :math:`^+` :math:`\in\ [0.5, 2.1]`                   | | [H\ :math:`-`\ O]\ :math:`^+ = 2.5`                                      |
    |                              |                                     |              | | [H\ :math:`_A`\ :math:`-`\ O\ :math:`-`\ H\ :math:`_B]^+=111.3^{\circ}` | | [H\ :math:`_A`\ :math:`-`\ O\ :math:`-`\ H\ :math:`_B]^+=111.3^{\circ}`  |
    +------------------------------+-------------------------------------+--------------+---------------------------------------------------------------------------+----------------------------------------------------------------------------+

Data Features
-------------

For each of the molecules mentioned above, the following characteristics can be extracted for each geometries:

Molecular Data
~~~~~~~~~~~~~~

Information regarding the molecule, including its complete classical description and the Harteee Fock state.

.. rst-class:: docstable
    :widths: auto 

    +----------------------------+--------------------------------+-----------------------------------------------------------------------------------+
    | ``molecule``               |  :class:`~.pennylane.Molecule` | PennyLane Molecule object containing description for the system and basis set.    |
    +----------------------------+--------------------------------+-----------------------------------------------------------------------------------+
    | ``hf_state``               |  ``numpy.array``               | Hartree-Fock state of the chemical system represented by a binary vector.         |
    +----------------------------+--------------------------------+-----------------------------------------------------------------------------------+

Hamiltonian Data
~~~~~~~~~~~~~~~~

Hamiltonian for the molecular system under Jordan-Wigner transformation and its properties. 

.. rst-class:: docstable
    :widths: auto 
    
    +----------------------------+------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------+
    | ``hamiltonian``            |  :class:`~.pennylane.Hamiltonian`                                                  | Hamiltonian of the system in the Pauli basis.                                     |
    +----------------------------+------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------+
    | ``sparse_hamiltonian``     |  ``scipy.sparse.csr_array``                                                        | Sparse matrix representation of a Hamiltonian in the computational basis.         |
    +----------------------------+------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------+
    | ``meas_groupings``         | list[list[list[\ :class:`~.pennylane.operation.Operator`]], list[``tensor_like``]] | List of grouped qubit-wise commuting Hamiltonian terms.                           |
    +----------------------------+------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------+
    | ``fci_energy``             | ``float``                                                                          | Classical energy of the molecule obtained from exact diagonalization.             |
    +----------------------------+------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------+
    | ``fci_spectrum``           | ``numpy.array``                                                                    | First :math:`2\times`\ #qubits eigenvalues obtained from exact diagonalization.   |
    +----------------------------+------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------+

Auxiliary Observables
~~~~~~~~~~~~~~~~~~~~~

Supplementary operators required to obtain additional properties of the molecule such as its dipole moment, spin, etc. 

.. rst-class:: docstable
    :widths: auto

    +----------------------------+----------------------------------+-----------------------------------------------------------------------------------+
    | ``dipole_op``              | :class:`~.pennylane.Hamiltonian` | Qubit dipole moment operators for the chemical system.                            |
    +----------------------------+----------------------------------+-----------------------------------------------------------------------------------+
    | ``number_op``              | :class:`~.pennylane.Hamiltonian` | Qubit particle number operator for the chemical system.                           |
    +----------------------------+----------------------------------+-----------------------------------------------------------------------------------+
    | ``spin2_op``               | :class:`~.pennylane.Hamiltonian` | Qubit operator for computing total spin :math:`S^2` for the chemical system.      |
    +----------------------------+----------------------------------+-----------------------------------------------------------------------------------+
    | ``spinz_op``               | :class:`~.pennylane.Hamiltonian` | Qubit operator for computing total spin's projection in :math:`Z` direction.      |
    +----------------------------+----------------------------------+-----------------------------------------------------------------------------------+

Tapering Data
~~~~~~~~~~~~~

Features based on :math:`Z_2` symmetries of the molecular Hamiltonian for performing `tapering <https://docs.pennylane.ai/en/stable/code/api/pennylane.taper.html>`_. 

.. rst-class:: docstable
    :widths: auto

    +----------------------------+------------------------------------------+-----------------------------------------------------------------------------------+
    | ``symmetries``             | list[\ :class:`~.pennylane.Hamiltonian`] | Symmetries required for tapering molecular Hamiltonian                            |
    +----------------------------+------------------------------------------+-----------------------------------------------------------------------------------+
    | ``paulix_ops``             | list[\ :class:`~.pennylane.PauliX`]      | Supporting PauliX ops required to build Clifford :math:`U` for tapering           |
    +----------------------------+------------------------------------------+-----------------------------------------------------------------------------------+
    | ``optimal_sector``         | ``numpy.array``                          | Eigensector of the tapered qubits that would contain the ground state             |
    +----------------------------+------------------------------------------+-----------------------------------------------------------------------------------+

Tapered Observables Data
~~~~~~~~~~~~~~~~~~~~~~~~

Tapered observables and Hartree-Fock state based on the on :math:`Z_2` symmetries of the molecular Hamiltonian. 

.. rst-class:: docstable
    :widths: auto

    +----------------------------+----------------------------------+-----------------------------------------------------------------------------------+
    | ``tapered_hamiltonian``    | :class:`~.pennylane.Hamiltonian` | Tapered Hamiltonian                                                               |
    +----------------------------+----------------------------------+-----------------------------------------------------------------------------------+
    | ``tapered_hf_state``       | ``numpy.array``                  | Tapered Hartree-Fock state of the molecule                                        |
    +----------------------------+----------------------------------+-----------------------------------------------------------------------------------+
    | ``tapered_dipole_op``      | :class:`~.pennylane.Hamiltonian` | Tapered dipole moment operator                                                    |
    +----------------------------+----------------------------------+-----------------------------------------------------------------------------------+
    | ``tapered_num_op``         | :class:`~.pennylane.Hamiltonian` | Tapered number operator                                                           |
    +----------------------------+----------------------------------+-----------------------------------------------------------------------------------+
    | ``tapered_spin2_op``       | :class:`~.pennylane.Hamiltonian` | Tapered total spin operator                                                       |
    +----------------------------+----------------------------------+-----------------------------------------------------------------------------------+
    | ``tapered_spinz_op``       | :class:`~.pennylane.Hamiltonian` | Tapered spin projection operator                                                  |
    +----------------------------+----------------------------------+-----------------------------------------------------------------------------------+

VQE Data
~~~~~~~~

Variational data obtained using :class:`~.pennylane.AdaptiveOptimizer` for minimzing ground state energy.

.. rst-class:: docstable
    :widths: auto

    +----------------------------+--------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------+
    | ``vqe_gates``              | list[\ :class:`~.pennylane.operation.Operation`] | :class:`~.pennylane.SingleExcitation` and :class:`~.pennylane.DoubleExcitation` gates for the optimized circuit         |
    +----------------------------+--------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------+
    | ``vqe_params``             | ``numpy.array``                                  | Optimal parameters for the gates that prepares ground state                                                             |
    +----------------------------+--------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------+
    | ``vqe_energy``             | ``float``                                        | Energy obtained from the state prepared by the optimized circuit                                                        |
    +----------------------------+--------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------+

.. toctree::
    :maxdepth: 2
    :hidden:

