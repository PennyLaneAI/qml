.. raw:: html

    <style>
        h1 {
            text-align: center;
        }
    </style>

Quantum chemistry datasets
==========================

.. meta::
   :property="og:description": Browse our collection of quantum datasets and import them into PennyLane directly from your code.
   :property="og:image": https://pennylane.ai/qml/_static/datasets.png

Quantum chemistry is one of the most promising directions for research in quantum algorithms. Here you can explore our available quantum chemistry datasets for some common molecular systems.

Molecules
---------

.. image:: /_static/datasets/bondlength.jpeg
    :align: right
    :width: 45%
    :target: javascript:void(0);

We provide the electronic structure data for 42 different geometries of the following molecules:

* **Linear hydrogen chains:** H\ :sub:`2`, H\ :sub:`4`, H\ :sub:`6`, H\ :sub:`8`.
* **Metallic and non-metallic hydrides:** - LiH, BeH\ :sub:`2`, BH\ :sub:`3`, CH\ :sub:`4`, NH\ :sub:`3`, H\ :sub:`2`\ O, HF.
* **Charged species:** HeH\ :sup:`+`, H\ :sub:`3`\ :sup:`+`, OH\ :sup:`-`.

For the smaller molecules such as H\ :sub:`2`, HeH\ :sup:`+`, and H\ :sub:`3`\ :sup:`+`, data has been obtained for both the minimal basis-set `STO-3G` and the split-valence double-zeta basis set `6-31G`. 
For the remaining molecules, data is only available for the former basis-set. The geometries for each molecule are defined by the bond lengths between atoms, with the available bondlengths
given as 41 equally spaced values within a range (see the table below).
In addition to these, we also include data for the optimal ground-state geometry of each molecule. 
We summarise all of this information for all the molecules in the table below.

Accessing chemistry datasets
----------------------------

The quantum chemistry datasets can be downloaded and loaded to memory using the :func:`~pennylane.data.load` function as follows:

>>> data = qml.data.load("qchem", molname="H2", basis="STO-3G", bondlength=1.1)[0]
>>> print(data)
<pennylane.data.dataset.Dataset object at 0x7f14e4369640>

Here, the positional argument ``"qchem"`` denotes that we are loading a chemistry dataset,
while the keyword arguments ``molname``, ``basis``, and ``bondlength`` specify the requested dataset.
The possible values for these keyword arguments are included in the table below. For more information on using PennyLane functions
please see the `PennyLane Documentation <https://docs.pennylane.ai/en/latest/introduction/data.html>`_.

.. raw:: html

     <style>
        .docstable tr.row-even th, .docstable tr.row-even td {
            vertical-align: baseline;
            text-align: center;
            padding-bottom: 0;
        }
        .docstable tr.row-odd th, .docstable tr.row-odd td {
            vertical-align: baseline;
            text-align: center;
            padding-bottom: 0;
        }
        .docstable thead th {
            vertical-align: baseline;
            padding-bottom: 0;
        }
    </style>
    <div class="d-flex justify-content-center">

.. rst-class:: docstable
    :widths: auto
    :align: center

    +------------------------------+------------------------------+--------------+---------------------------------------------------------------------------+----------------------------------------------------------------------------+
    | **Molecule**                 | **Basis set(s)**             | **#Qubits**  | **Bond length (Å), Bond angle (°)**                                       | **Optimal geometry (Å, °)**                                                |
    +==============================+==============================+==============+===========================================================================+============================================================================+
    | H\ :math:`_2`                | | STO\ :math:`\text{-}`\3G / | 4 / 8        | H\ :math:`_A-`\ H\ :math:`_B` :math:`\in\ [0.5, 2.1]`                     | H\ :math:`_A-`\ H\ :math:`_B = 0.742`                                      |
    |                              | | 6\ :math:`\text{-}`\31G    |              |                                                                           |                                                                            |   
    +------------------------------+------------------------------+--------------+---------------------------------------------------------------------------+----------------------------------------------------------------------------+
    | HeH\ :math:`^+`              | | STO\ :math:`\text{-}`\3G / | 4 / 8        | He\ :math:`-`\ H :math:`\in\ [0.5, 2.1]`                                  | He\ :math:`-`\ H\ :math:`= 0.7748`                                         |
    |                              | | 6\ :math:`\text{-}`\31G    |              |                                                                           |                                                                            |           
    +------------------------------+------------------------------+--------------+---------------------------------------------------------------------------+----------------------------------------------------------------------------+
    | H\ :math:`_3^+`              | | STO\ :math:`\text{-}`\3G / | 6 / 12       | | H\ :math:`_A-`\ H\ :math:`_B` :math:`\in\ [0.5, 2.1]`,                  | | H\ :math:`_A-`\ H\ :math:`_B = 0.8737`,                                  |
    |                              | | 6\ :math:`\text{-}`\31G    |              | | :math:`\measuredangle` HHH :math:`= 60^{\circ}`                         | | :math:`\measuredangle` HHH :math:`= 60^{\circ}`                          |
    +------------------------------+------------------------------+--------------+---------------------------------------------------------------------------+----------------------------------------------------------------------------+
    | H\ :math:`_4`                | STO\ :math:`\text{-}`\3G     | 8            | | H\ :math:`_A-`\ H\ :math:`_B` :math:`\in\ [0.5, 1.3]`,                  |          N/A                                                               |
    |                              |                              |              | | :math:`\measuredangle` HHH :math:`= 180^{\circ}`                        |                                                                            |
    +------------------------------+------------------------------+--------------+---------------------------------------------------------------------------+----------------------------------------------------------------------------+
    | LiH                          | STO\ :math:`\text{-}`\3G     | 12           | Li\ :math:`-`\ H :math:`\in\ [0.9, 2.1]`                                  | Li\ :math:`-`\ H :math:`= 1.5699`                                          |
    +------------------------------+------------------------------+--------------+---------------------------------------------------------------------------+----------------------------------------------------------------------------+
    | HF                           | STO\ :math:`\text{-}`\3G     | 12           | H\ :math:`-`\ F :math:`\in\ [0.5, 2.1]`                                   | H\ :math:`-`\ F :math:`= 0.9167`                                           |
    +------------------------------+------------------------------+--------------+---------------------------------------------------------------------------+----------------------------------------------------------------------------+
    | OH\ :math:`^-`               | STO\ :math:`\text{-}`\3G     | 12           | O\ :math:`-`\ H :math:`\in\ [0.5, 2.1]`                                   | O\ :math:`-`\ H :math:`= 0.9638`                                           |
    +------------------------------+------------------------------+--------------+---------------------------------------------------------------------------+----------------------------------------------------------------------------+
    | H\ :math:`_6`                | STO\ :math:`\text{-}`\3G     | 12           | | H\ :math:`_A-`\ H\ :math:`_B` :math:`\in\ [0.5, 1.3]`                   |          N/A                                                               |
    |                              |                              |              | | :math:`\measuredangle` HHH :math:`= 180^{\circ}`                        |                                                                            |
    +------------------------------+------------------------------+--------------+---------------------------------------------------------------------------+----------------------------------------------------------------------------+
    | BeH\ :math:`_2`              | STO\ :math:`\text{-}`\3G     | 14           | | Be\ :math:`-`\ H :math:`\in\ [0.5, 2.1]`,                               | | Be\ :math:`-`\ H :math:`=1.3295`,                                        |
    |                              |                              |              | | :math:`\measuredangle` HBeH :math:`= 180^{\circ}`                       | | :math:`\measuredangle` HBeH :math:`= 180^{\circ}`                        |
    +------------------------------+------------------------------+--------------+---------------------------------------------------------------------------+----------------------------------------------------------------------------+
    | H\ :math:`_2`\ O             | STO\ :math:`\text{-}`\3G     | 16           | | H\ :math:`-`\ O :math:`\in [0.5, 2.1]`,                                 | | H\ :math:`-`\ O :math:`=0.9575`,                                         |
    |                              |                              |              | | :math:`\measuredangle` HOH :math:`= 104.5^{\circ}`                      | | :math:`\measuredangle` HOH :math:`= 104.5^{\circ}`                       |
    +------------------------------+------------------------------+--------------+---------------------------------------------------------------------------+----------------------------------------------------------------------------+
    | H\ :math:`_8`                | STO\ :math:`\text{-}`\3G     | 16           | | H\ :math:`_A-`\ H\ :math:`_B` :math:`\in\ [0.5, 1.3]`,                  |          N/A                                                               |
    |                              |                              |              | | :math:`\measuredangle` HHH :math:`= 180^{\circ}`                        |                                                                            |
    +------------------------------+------------------------------+--------------+---------------------------------------------------------------------------+----------------------------------------------------------------------------+
    | BH\ :math:`_3`               | STO\ :math:`\text{-}`\3G     | 18           | | B\ :math:`-`\ H :math:`\in\ [0.5, 2.1]`                                 | | B\ :math:`-`\ H :math:`=1.1893`,                                         |
    |                              |                              |              | | :math:`\measuredangle` HBH :math:`= 120^{\circ}`                        | | :math:`\measuredangle` HBH :math:`= 120^{\circ}`                         |
    +------------------------------+------------------------------+--------------+---------------------------------------------------------------------------+----------------------------------------------------------------------------+
    | NH\ :math:`_3`               | STO\ :math:`\text{-}`\3G     | 18           | | N\ :math:`-`\ H :math:`\in\ [0.5, 2.1]`                                 | | N\ :math:`-`\ H :math:`=1.1096`,                                         |
    |                              |                              |              | | :math:`\measuredangle` HNH :math:`= 106.8^{\circ}`                      | | :math:`\measuredangle` HNH :math:`= 106.8^{\circ}`                       |
    +------------------------------+------------------------------+--------------+---------------------------------------------------------------------------+----------------------------------------------------------------------------+
    | H\ :math:`_3`\ O\ :math:`^+` | STO\ :math:`\text{-}`\3G     | 18           | | H\ :math:`-`\ O :math:`\in\ [0.5, 2.1]`                                 | | H\ :math:`-`\ O :math:`= 2.5`                                            |
    |                              |                              |              | | :math:`\measuredangle` HOH :math:`= 111.3^{\circ}`                      | | :math:`\measuredangle` HOH :math:`= 111.3^{\circ}`                       |
    +------------------------------+------------------------------+--------------+---------------------------------------------------------------------------+----------------------------------------------------------------------------+

Data features
-------------

For each of the molecules mentioned above, the following characteristics can be extracted for each geometry:

Molecular data
~~~~~~~~~~~~~~

Information regarding the molecule, including its complete classical description and the Hartree Fock state.

.. rst-class:: docstable
    :widths: auto 

    +----------------------------+--------------------------------+-----------------------------------------------------------------------------------+
    | **Name**                   | **Type**                       | **Description**                                                                   | 
    +============================+================================+===================================================================================+
    | ``molecule``               |  :class:`~.pennylane.Molecule` | PennyLane Molecule object containing description for the system and basis set.    |
    +----------------------------+--------------------------------+-----------------------------------------------------------------------------------+
    | ``hf_state``               |  ``numpy.array``               | Hartree-Fock state of the chemical system represented by a binary vector.         |
    +----------------------------+--------------------------------+-----------------------------------------------------------------------------------+

Hamiltonian data
~~~~~~~~~~~~~~~~

Hamiltonian for the molecular system under Jordan-Wigner transformation and its properties. 

.. rst-class:: docstable
    :widths: auto 
    
    +----------------------------+------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------+
    | **Name**                   | **Type**                                                                           | **Description**                                                                   | 
    +============================+====================================================================================+===================================================================================+
    | ``hamiltonian``            |  :class:`~.pennylane.Hamiltonian`                                                  | Hamiltonian of the system in the Pauli basis.                                     |
    +----------------------------+------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------+
    | ``sparse_hamiltonian``     |  ``scipy.sparse.csr_array``                                                        | Sparse matrix representation of a Hamiltonian in the computational basis.         |
    +----------------------------+------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------+
    | ``meas_groupings``         | list[list[list[\ :class:`~.pennylane.operation.Operator`]], list[``tensor_like``]] | List of grouped qubit-wise commuting Hamiltonian terms.                           |
    +----------------------------+------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------+
    | ``fci_energy``             | ``float``                                                                          | Ground state energy of the molecule obtained from exact diagonalization.          |
    +----------------------------+------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------+
    | ``fci_spectrum``           | ``numpy.array``                                                                    | First :math:`2\times`\ #qubits eigenvalues obtained from exact diagonalization.   |
    +----------------------------+------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------+

Auxiliary observables
~~~~~~~~~~~~~~~~~~~~~

The supplementary operators required to obtain additional properties of the molecule such as its dipole moment, spin, etc. 

.. rst-class:: docstable
    :widths: auto

    +----------------------------+----------------------------------+-----------------------------------------------------------------------------------+
    | **Name**                   | **Type**                         | **Description**                                                                   | 
    +============================+==================================+===================================================================================+
    | ``dipole_op``              | :class:`~.pennylane.Hamiltonian` | Qubit dipole moment operators for the chemical system.                            |
    +----------------------------+----------------------------------+-----------------------------------------------------------------------------------+
    | ``number_op``              | :class:`~.pennylane.Hamiltonian` | Qubit particle number operator for the chemical system.                           |
    +----------------------------+----------------------------------+-----------------------------------------------------------------------------------+
    | ``spin2_op``               | :class:`~.pennylane.Hamiltonian` | Qubit operator for computing total spin :math:`S^2` for the chemical system.      |
    +----------------------------+----------------------------------+-----------------------------------------------------------------------------------+
    | ``spinz_op``               | :class:`~.pennylane.Hamiltonian` | Qubit operator for computing total spin's projection in :math:`Z` direction.      |
    +----------------------------+----------------------------------+-----------------------------------------------------------------------------------+

Tapering data
~~~~~~~~~~~~~

Features based on :math:`Z_2` symmetries of the molecular Hamiltonian for performing `tapering <https://docs.pennylane.ai/en/stable/code/api/pennylane.taper.html>`_. 

.. rst-class:: docstable
    :widths: auto

    +----------------------------+------------------------------------------+-----------------------------------------------------------------------------------+
    | **Name**                   | **Type**                                 | **Description**                                                                   | 
    +============================+==========================================+===================================================================================+
    | ``symmetries``             | list[\ :class:`~.pennylane.Hamiltonian`] | Symmetries required for tapering molecular Hamiltonian                            |
    +----------------------------+------------------------------------------+-----------------------------------------------------------------------------------+
    | ``paulix_ops``             | list[\ :class:`~.pennylane.PauliX`]      | Supporting PauliX ops required to build Clifford :math:`U` for tapering           |
    +----------------------------+------------------------------------------+-----------------------------------------------------------------------------------+
    | ``optimal_sector``         | ``numpy.array``                          | Eigensector of the tapered qubits that would contain the ground state             |
    +----------------------------+------------------------------------------+-----------------------------------------------------------------------------------+

Tapered observables data
~~~~~~~~~~~~~~~~~~~~~~~~

Tapered observables and Hartree-Fock state based on the on :math:`Z_2` symmetries of the molecular Hamiltonian. 

.. rst-class:: docstable
    :widths: auto

    +----------------------------+----------------------------------+-----------------------------------------------------------------------------------+
    | **Name**                   | **Type**                         | **Description**                                                                   | 
    +============================+==================================+===================================================================================+
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

VQE data
~~~~~~~~

Variational data obtained by using :class:`~.pennylane.AdaptiveOptimizer` to minimize ground state energy.

.. rst-class:: docstable
    :widths: auto

    +----------------------------+--------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------+
    | **Name**                   | **Type**                                         | **Description**                                                                                                         | 
    +============================+==================================================+=========================================================================================================================+
    | ``vqe_gates``              | list[\ :class:`~.pennylane.operation.Operation`] | :class:`~.pennylane.SingleExcitation` and :class:`~.pennylane.DoubleExcitation` gates for the optimized circuit         |
    +----------------------------+--------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------+
    | ``vqe_params``             | ``numpy.array``                                  | Optimal parameters for the gates that prepares ground state                                                             |
    +----------------------------+--------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------+
    | ``vqe_energy``             | ``float``                                        | Energy obtained from the state prepared by the optimized circuit                                                        |
    +----------------------------+--------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------+

.. toctree::
    :maxdepth: 2
    :hidden:

