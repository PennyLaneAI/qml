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
   :property="og:image": https://pennylane.ai/qml/_static/datasets/datasets.png

Quantum chemistry is one of the most promising directions for research in quantum algorithms. Here you can explore our available quantum chemistry datasets for some common molecular systems.

Molecules
---------

.. image:: /_static/datasets/h2o.png
    :align: right
    :width: 45%
    :target: javascript:void(0);

We provide the electronic structure data for different geometries of the following molecules:

* **Linear hydrogen chains:** H\ :sub:`2`, H\ :sub:`4`, H\ :sub:`5`, H\ :sub:`6`, H\ :sub:`7`, H\ :sub:`8`, H\ :sub:`10`.
* **Metallic and non-metallic hydrides:** LiH, BeH\ :sub:`2`, BH\ :sub:`3`, NH\ :sub:`3`, H\ :sub:`2`\ O, HF.
* **Metallic and non-metallic dimers:** He\ :sub:`2`, Li\ :sub:`2`, C\ :sub:`2`, N\ :sub:`2`, O\ :sub:`2`.
* **Charged species:** HeH\ :sup:`+`, H\ :sub:`3`\ :sup:`+`, OH\ :math:`^-`, NeH\ :math:`^+`.
* **Inorganic molecules:**  CO, CO\ :sub:`2`, N\ :sub:`2`\ H\ :sub:`2`, N\ :sub:`2`\ H\ :sub:`4`, H\ :sub:`2`\ O\ :sub:`2`, O\ :sub:`3`.
* **Organic molecules:** CH\ :sub:`4`, HCN, C\ :sub:`2`\ H\ :sub:`2`, C\ :sub:`2`\ H\ :sub:`4`, C\ :sub:`2`\ H\ :sub:`6`.

For H\ :sub:`2` and HeH\ :sup:`+`, data is provided for the minimal basis-set `STO-3G`, the split-valence double-zeta basis set `6-31G`,
and the correlation-consistent polarized valence double zeta basis set `CC-PVDZ`. While, for H\ :sub:`3`\ :sup:`+`, data is provided for both `STO-3G` and `6-31G`,
for He\ :sub:`2`, data is present just for `6-31G`. For the remaining molecules, data is only available for the minimal basis set, `STO-3G`.
The molecular geometries are defined by bond lengths and bond angles. For each molecule, the available bond lengths and bond lengths are given in the table below.
These are written as ``[minimum bond length, maximum bond length, number of bond lengths]`` and contain `number of bond lengths` equispaced values in the given range.
In addition to these, we also include the data for the optimal ground-state geometry of each molecule. While for the molecule that require between 22 and 30 qubits,
we do not provide VQE or sampling data for any of the geometries, for ones that require between 24 and 30 qubits, we consider only the optimal ground-state geometries.
We summarise all of this information for all the molecules below.

Accessing chemistry datasets
----------------------------

The quantum chemistry datasets can be downloaded and loaded to memory using the :func:`~pennylane.data.load` function as follows:

>>> data = qml.data.load("qchem", molname="H2", basis="STO-3G", bondlength=1.1)[0]
>>> print(data)
<Dataset = description: qchem/H2/STO-3G/1.1, attributes: ['molecule', 'hamiltonian', ...]>

Here, the positional argument ``"qchem"`` denotes that we are loading a chemistry dataset,
while the keyword arguments ``molname``, ``basis``, and ``bondlength`` specify the requested dataset.
The possible values for these keyword arguments are included in the table below. For more information on using PennyLane functions
please see the `PennyLane Documentation <https://docs.pennylane.ai/en/stable/introduction/data.html>`_.

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
    | H\ :math:`_2`                | | STO\ :math:`\text{-}`\3G / | | 4 /        | | H\ :math:`_A-`\ H\ :math:`_B` :math:`\in\ [0.5, 2.1, 41]` Å             | H\ :math:`_A-`\ H\ :math:`_B = 0.742` Å                                    |
    |                              | | 6\ :math:`\text{-}`\31G /  | | 8 /        | | H\ :math:`_A-`\ H\ :math:`_B` :math:`\in\ [0.5, 2.1, 41]` Å             |                                                                            |   
    |                              | | CC\ :math:`\text{-}`\PVDZ  | | 20         | | H\ :math:`_A-`\ H\ :math:`_B` :math:`\in\ [0.5, 2.5, 11]` Å             |                                                                            |   
    +------------------------------+------------------------------+--------------+---------------------------------------------------------------------------+----------------------------------------------------------------------------+
    | HeH\ :math:`^+`              | | STO\ :math:`\text{-}`\3G / | | 4 /        | | He\ :math:`-`\ H :math:`\in\ [0.5, 2.1, 41]` Å                          | He\ :math:`-`\ H\ :math:`= 0.775` Å                                        |
    |                              | | 6\ :math:`\text{-}`\31G /  | | 8 /        | | He\ :math:`-`\ H :math:`\in\ [0.5, 2.1, 41]` Å                          |                                                                            |           
    |                              | | CC\ :math:`\text{-}`\PVDZ  | | 20         | | He\ :math:`-`\ H :math:`\in\ [0.5, 2.5, 11]` Å                          |                                                                            |   
    +------------------------------+------------------------------+--------------+---------------------------------------------------------------------------+----------------------------------------------------------------------------+
    | H\ :math:`_3^+`              | | STO\ :math:`\text{-}`\3G / | 6 / 12       | | H\ :math:`_A-`\ H\ :math:`_B` :math:`\in\ [0.5, 2.1, 41]` Å,            | | H\ :math:`_A-`\ H\ :math:`_B = 0.874` Å,                                 |
    |                              | | 6\ :math:`\text{-}`\31G    |              | | :math:`\measuredangle` HHH :math:`= 60^{\circ}`                         | | :math:`\measuredangle` HHH :math:`= 60^{\circ}`                          |
    +------------------------------+------------------------------+--------------+---------------------------------------------------------------------------+----------------------------------------------------------------------------+
    | H\ :math:`_4`                | STO\ :math:`\text{-}`\3G     | 8            | | H\ :math:`_A-`\ H\ :math:`_B` :math:`\in\ [0.5, 1.3, 41]` Å,            | | H\ :math:`_A-`\ H\ :math:`_B = 0.88` Å,                                  |
    |                              |                              |              | | :math:`\measuredangle` HHH :math:`= 180^{\circ}`                        | | :math:`\measuredangle` HHH :math:`= 180^{\circ}`                         |
    +------------------------------+------------------------------+--------------+---------------------------------------------------------------------------+----------------------------------------------------------------------------+
    | He\ :math:`_2`               | 6\ :math:`\text{-}`\31G      | 8            | He\ :math:`-`\ He :math:`\in\ [0.5, 6.5, 41]` Å                           | He\ :math:`-`\ He\ :math:`= 5.200` Å                                       |
    +------------------------------+------------------------------+--------------+---------------------------------------------------------------------------+----------------------------------------------------------------------------+
    | H\ :math:`_5`                | STO\ :math:`\text{-}`\3G     | 10           | | H\ :math:`_A-`\ H\ :math:`_B` :math:`\in\ [0.5, 1.5, 11]` Å,            | | H\ :math:`_A-`\ H\ :math:`_B = 1.0` Å,                                   |
    |                              |                              |              | | :math:`\measuredangle` HHH :math:`= 180^{\circ}`                        | | :math:`\measuredangle` HHH :math:`= 180^{\circ}`                         |
    +------------------------------+------------------------------+--------------+---------------------------------------------------------------------------+----------------------------------------------------------------------------+
    | LiH                          | STO\ :math:`\text{-}`\3G     | 12           | Li\ :math:`-`\ H :math:`\in\ [0.9, 2.1, 41]` Å                            | Li\ :math:`-`\ H :math:`= 1.57` Å                                          |
    +------------------------------+------------------------------+--------------+---------------------------------------------------------------------------+----------------------------------------------------------------------------+
    | HF                           | STO\ :math:`\text{-}`\3G     | 12           | H\ :math:`-`\ F :math:`\in\ [0.5, 2.1, 41]` Å                             | H\ :math:`-`\ F :math:`= 0.917` Å                                          |
    +------------------------------+------------------------------+--------------+---------------------------------------------------------------------------+----------------------------------------------------------------------------+
    | OH\ :math:`^-`               | STO\ :math:`\text{-}`\3G     | 12           | O\ :math:`-`\ H :math:`\in\ [0.5, 2.1, 41]` Å                             | O\ :math:`-`\ H :math:`= 0.964` Å                                          |
    +------------------------------+------------------------------+--------------+---------------------------------------------------------------------------+----------------------------------------------------------------------------+
    | NeH\ :math:`^+`              | STO\ :math:`\text{-}`\3G     | 12           | Ne\ :math:`-`\ H :math:`\in\ [0.5, 2.5, 11]` Å                            | Ne\ :math:`-`\ H :math:`= 0.991` Å                                         |
    +------------------------------+------------------------------+--------------+---------------------------------------------------------------------------+----------------------------------------------------------------------------+
    | H\ :math:`_6`                | STO\ :math:`\text{-}`\3G     | 12           | | H\ :math:`_A-`\ H\ :math:`_B` :math:`\in\ [0.5, 1.3, 41]` Å,            | | H\ :math:`_A-`\ H\ :math:`_B = 0.92` Å,                                  |
    |                              |                              |              | | :math:`\measuredangle` HHH :math:`= 180^{\circ}`                        | | :math:`\measuredangle` HHH :math:`= 180^{\circ}`                         |
    +------------------------------+------------------------------+--------------+---------------------------------------------------------------------------+----------------------------------------------------------------------------+
    | BeH\ :math:`_2`              | STO\ :math:`\text{-}`\3G     | 14           | | Be\ :math:`-`\ H :math:`\in\ [0.5, 2.1, 41]` Å,                         | | Be\ :math:`-`\ H :math:`=1.330` Å,                                       |
    |                              |                              |              | | :math:`\measuredangle` HBeH :math:`= 180^{\circ}`                       | | :math:`\measuredangle` HBeH :math:`= 180^{\circ}`                        |
    +------------------------------+------------------------------+--------------+---------------------------------------------------------------------------+----------------------------------------------------------------------------+
    | H\ :math:`_2`\ O             | STO\ :math:`\text{-}`\3G     | 14           | | H\ :math:`-`\ O :math:`\in [0.5, 2.1, 41]` Å,                           | | H\ :math:`-`\ O :math:`=0.958` Å,                                        |
    |                              |                              |              | | :math:`\measuredangle` HOH :math:`= 104.5^{\circ}`                      | | :math:`\measuredangle` HOH :math:`= 104.5^{\circ}`                       |
    +------------------------------+------------------------------+--------------+---------------------------------------------------------------------------+----------------------------------------------------------------------------+
    | H\ :math:`_7`                | STO\ :math:`\text{-}`\3G     | 14           | | H\ :math:`_A-`\ H\ :math:`_B` :math:`\in\ [0.5, 1.5, 11]` Å,            | | H\ :math:`_A-`\ H\ :math:`_B = 1.0` Å,                                   |
    |                              |                              |              | | :math:`\measuredangle` HHH :math:`= 180^{\circ}`                        | | :math:`\measuredangle` HHH :math:`= 180^{\circ}`                         |
    +------------------------------+------------------------------+--------------+---------------------------------------------------------------------------+----------------------------------------------------------------------------+
    | BH\ :math:`_3`               | STO\ :math:`\text{-}`\3G     | 16           | | B\ :math:`-`\ H :math:`\in\ [0.5, 2.1, 41]` Å,                          | | B\ :math:`-`\ H :math:`=1.189` Å,                                        |
    |                              |                              |              | | :math:`\measuredangle` HBH :math:`= 120^{\circ}`                        | | :math:`\measuredangle` HBH :math:`= 120^{\circ}`                         |
    +------------------------------+------------------------------+--------------+---------------------------------------------------------------------------+----------------------------------------------------------------------------+
    | NH\ :math:`_3`               | STO\ :math:`\text{-}`\3G     | 16           | | N\ :math:`-`\ H :math:`\in\ [0.5, 2.1, 41]` Å,                          | | N\ :math:`-`\ H :math:`=1.110` Å,                                        |
    |                              |                              |              | | :math:`\measuredangle` HNH :math:`= 106.8^{\circ}`                      | | :math:`\measuredangle` HNH :math:`= 106.8^{\circ}`                       |
    +------------------------------+------------------------------+--------------+---------------------------------------------------------------------------+----------------------------------------------------------------------------+
    | H\ :math:`_8`                | STO\ :math:`\text{-}`\3G     | 16           | | H\ :math:`_A-`\ H\ :math:`_B` :math:`\in\ [0.5, 0.9, 41]` Å,            |          N/A                                                               |
    |                              |                              |              | | :math:`\measuredangle` HHH :math:`= 180^{\circ}`                        |                                                                            |
    +------------------------------+------------------------------+--------------+---------------------------------------------------------------------------+----------------------------------------------------------------------------+
    | CH\ :math:`_4`               | STO\ :math:`\text{-}`\3G     | 18           | | C\ :math:`-`\ H :math:`\in\ [0.5, 2.5, 11]` Å,                          | | C\ :math:`-`\ H :math:`=1.086` Å,                                        |
    |                              |                              |              | | :math:`\measuredangle` HCH :math:`= 109.5^{\circ}`                      | | :math:`\measuredangle` HCH :math:`= 109.5^{\circ}`                       |
    +------------------------------+------------------------------+--------------+---------------------------------------------------------------------------+----------------------------------------------------------------------------+
    | Li\ :math:`_2`               | STO\ :math:`\text{-}`\3G     | 20           | Li\ :math:`-`\ Li :math:`\in\ [1.5, 3.5, 11]` Å,                          | Li\ :math:`-`\ Li :math:`=2.679` Å                                         |
    +------------------------------+------------------------------+--------------+---------------------------------------------------------------------------+----------------------------------------------------------------------------+
    | C\ :math:`_2`                | STO\ :math:`\text{-}`\3G     | 20           | C\ :math:`-`\ C :math:`\in\ [0.5, 2.5, 11]` Å,                            | C\ :math:`-`\ C :math:`=1.246` Å                                           |
    +------------------------------+------------------------------+--------------+---------------------------------------------------------------------------+----------------------------------------------------------------------------+
    | N\ :math:`_2`                | STO\ :math:`\text{-}`\3G     | 20           | N\ :math:`-`\ N :math:`\in\ [0.5, 2.5, 11]` Å,                            | N\ :math:`-`\ N :math:`=1.120` Å                                           |
    +------------------------------+------------------------------+--------------+---------------------------------------------------------------------------+----------------------------------------------------------------------------+
    | O\ :math:`_2`                | STO\ :math:`\text{-}`\3G     | 20           | O\ :math:`-`\ O :math:`\in\ [0.5, 2.5, 11]` Å,                            | O\ :math:`-`\ O :math:`=1.220` Å                                           |
    +------------------------------+------------------------------+--------------+---------------------------------------------------------------------------+----------------------------------------------------------------------------+
    | CO                           | STO\ :math:`\text{-}`\3G     | 20           | C\ :math:`-`\ O :math:`\in\ [0.5, 2.5, 11]` Å,                            | C\ :math:`-`\ O :math:`=1.128` Å                                           |
    +------------------------------+------------------------------+--------------+---------------------------------------------------------------------------+----------------------------------------------------------------------------+
    | H\ :math:`_{10}`             | STO\ :math:`\text{-}`\3G     | 20           | | H\ :math:`_A-`\ H\ :math:`_B` :math:`= 1.0` Å,                          |          N/A                                                               |
    |                              |                              |              | | :math:`\measuredangle` HHH :math:`= 180^{\circ}`                        |                                                                            |
    +------------------------------+------------------------------+--------------+---------------------------------------------------------------------------+----------------------------------------------------------------------------+
    | HCN                          | STO\ :math:`\text{-}`\3G     | 22           | | :math:`\measuredangle` HCN :math:`\in\ [0, \pi]^{\circ}`,               | | :math:`\measuredangle` HCN :math:`= \pi^{\circ}`,                        |
    |                              |                              |              | | C\ :math:`-`\ N :math:`= 1.156` Å                                       | | C\ :math:`-`\ N :math:`=1.156` Å                                         |
    +------------------------------+------------------------------+--------------+---------------------------------------------------------------------------+----------------------------------------------------------------------------+
    | H\ :math:`_2`\ CO            | STO\ :math:`\text{-}`\3G     | 24           | | C\ :math:`-`\ O :math:`= 0.9167` Å,                                     | | C\ :math:`-`\ O :math:`= 0.9167` Å,                                      |
    |                              |                              |              | | :math:`\measuredangle` OCH :math:`= 102.3^{\circ}`                      | | :math:`\measuredangle` OCH :math:`= 102.3^{\circ}`                       |
    +------------------------------+------------------------------+--------------+---------------------------------------------------------------------------+----------------------------------------------------------------------------+
    | H\ :math:`_2`\ O\ :math:`_2` | STO\ :math:`\text{-}`\3G     | 24           | | O\ :math:`_A-`\ O\ :math:`_B` :math:`= 1.475` Å,                        | | O\ :math:`_A-`\ O\ :math:`_B` :math:`= 1.475` Å,                         |
    |                              |                              |              | | :math:`\measuredangle` OOH :math:`= 94.8^{\circ}`                       | | :math:`\measuredangle` OOH :math:`= 94.8^{\circ}`                        |
    +------------------------------+------------------------------+--------------+---------------------------------------------------------------------------+----------------------------------------------------------------------------+
    | N\ :math:`_2`\ H\ :math:`_2` | STO\ :math:`\text{-}`\3G     | 24           | | N\ :math:`_A-`\ N\ :math:`_B` :math:`= 1.247` Å,                        | | N\ :math:`_A-`\ N\ :math:`_B` :math:`= 1.247` Å,                         |
    |                              |                              |              | | :math:`\measuredangle` NNH :math:`= 106.9^{\circ}`                      | | :math:`\measuredangle` NNH :math:`= 106.9^{\circ}`                       |
    +------------------------------+------------------------------+--------------+---------------------------------------------------------------------------+----------------------------------------------------------------------------+
    | C\ :math:`_2`\ H\ :math:`_2` | STO\ :math:`\text{-}`\3G     | 24           | | C\ :math:`_A-`\ C\ :math:`_B` :math:`= 1.203` Å,                        | | C\ :math:`_A-`\ C\ :math:`_B` :math:`= 1.203` Å,                         |
    |                              |                              |              | | :math:`\measuredangle` HCC :math:`= 180.0^{\circ}`                      | | :math:`\measuredangle` HCC :math:`= 180.0^{\circ}`                       |
    +------------------------------+------------------------------+--------------+---------------------------------------------------------------------------+----------------------------------------------------------------------------+
    | C\ :math:`_2`\ H\ :math:`_4` | STO\ :math:`\text{-}`\3G     | 28           | | C\ :math:`_A-`\ C\ :math:`_B` :math:`= 1.339` Å,                        | | C\ :math:`_A-`\ C\ :math:`_B` :math:`= 1.339` Å,                         |
    |                              |                              |              | | :math:`\measuredangle` CCH :math:`= 121.2^{\circ}`                      | | :math:`\measuredangle` CCH :math:`= 121.2^{\circ}`                       |
    |                              |                              |              | | :math:`\measuredangle` HCH :math:`= 117.6^{\circ}`                      | | :math:`\measuredangle` HCH :math:`= 117.6^{\circ}`                       |
    +------------------------------+------------------------------+--------------+---------------------------------------------------------------------------+----------------------------------------------------------------------------+
    | N\ :math:`_2`\ H\ :math:`_4` | STO\ :math:`\text{-}`\3G     | 28           | | N\ :math:`_A-`\ N\ :math:`_B` :math:`= 1.446` Å,                        | | N\ :math:`_A-`\ N\ :math:`_B` :math:`= 1.446` Å,                         |
    |                              |                              |              | | :math:`\measuredangle` NNH :math:`= 108.9^{\circ}`                      | | :math:`\measuredangle` NNH :math:`= 108.9^{\circ}`                       |
    |                              |                              |              | | :math:`\measuredangle` HNH :math:`= 106.0^{\circ}`                      | | :math:`\measuredangle` HNH :math:`= 106.0^{\circ}`                       |
    +------------------------------+------------------------------+--------------+---------------------------------------------------------------------------+----------------------------------------------------------------------------+
    | C\ :math:`_2`\ H\ :math:`_6` | STO\ :math:`\text{-}`\3G     | 30           | | C\ :math:`_A-`\ C\ :math:`_B` :math:`= 1.535` Å,                        | | C\ :math:`_A-`\ C\ :math:`_B` :math:`= 1.535` Å,                         |
    |                              |                              |              | | :math:`\measuredangle` CCH :math:`= 110.9^{\circ}`                      | | :math:`\measuredangle` CCH :math:`= 110.9^{\circ}`                       |
    |                              |                              |              | | :math:`\measuredangle` HCH :math:`= 108.0^{\circ}`                      | | :math:`\measuredangle` HCH :math:`= 108.0^{\circ}`                       |
    +------------------------------+------------------------------+--------------+---------------------------------------------------------------------------+----------------------------------------------------------------------------+
    | CO\ :math:`_2`               | STO\ :math:`\text{-}`\3G     | 30           | | C\ :math:`-`\ O :math:`= 1.162` Å,                                      | | C\ :math:`-`\ O :math:`= 1.162` Å,                                       |
    |                              |                              |              | | :math:`\measuredangle` OCO :math:`= 180.0^{\circ}`                      | | :math:`\measuredangle` OCO :math:`= 180^{\circ}`                         |
    +------------------------------+------------------------------+--------------+---------------------------------------------------------------------------+----------------------------------------------------------------------------+
    | O\ :math:`_3`                | STO\ :math:`\text{-}`\3G     | 30           | | O\ :math:`_A-`\ O\ :math:`_B`\ :math:`= 1.278` Å,                       | | O\ :math:`_A-`\ O\ :math:`_B` :math:`= 1.278` Å,                         |
    |                              |                              |              | | :math:`\measuredangle` OOO :math:`= 116.8^{\circ}`                      | | :math:`\measuredangle` OOO :math:`= 116.8^{\circ}`                       |
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
    | ``molecule``               |  :class:`~.pennylane.Molecule` | PennyLane Molecule object containing description for the system and basis set     |
    +----------------------------+--------------------------------+-----------------------------------------------------------------------------------+
    | ``hf_state``               |  ``numpy.ndarray``             | Hartree-Fock state of the chemical system represented by a binary vector          |
    +----------------------------+--------------------------------+-----------------------------------------------------------------------------------+

Hamiltonian data
~~~~~~~~~~~~~~~~

Hamiltonian for the molecular system under Jordan-Wigner transformation and its properties. 

.. rst-class:: docstable
    :widths: auto 
    
    +----------------------------+-----------------------------------+------------------------------------------------------------------------------------------+
    | **Name**                   | **Type**                          | **Description**                                                                          | 
    +============================+===================================+==========================================================================================+
    | ``hamiltonian``            |  :class:`~.pennylane.Hamiltonian` | Hamiltonian of the system in the Pauli basis                                             |
    +----------------------------+-----------------------------------+------------------------------------------------------------------------------------------+
    | ``sparse_hamiltonian``     |  ``scipy.sparse.csr_array``       | Sparse matrix representation of a Hamiltonian in the computational basis                 |
    +----------------------------+-----------------------------------+------------------------------------------------------------------------------------------+
    | ``fci_energy``             | ``float``                         | Ground-state energy of the molecule obtained from exact diagonalization                  |
    +----------------------------+-----------------------------------+------------------------------------------------------------------------------------------+
    | ``fci_spectrum``           | ``numpy.ndarray``                 | First :math:`2\times`\ ``num_qubits`` eigenvalues obtained from exact diagonalization    |
    +----------------------------+-----------------------------------+------------------------------------------------------------------------------------------+

Groupings data
~~~~~~~~~~~~~~

Groupings of the Hamiltonian terms for facilitating simultaneous measurements of all observables within a group.

.. rst-class:: docstable
    :widths: auto

    +----------------------------+--------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------+
    | **Name**                   | **Type**                                                                                                     | **Description**                                                                                                        | 
    +============================+==============================================================================================================+========================================================================================================================+
    | ``qwc_groupings``          | tuple(list[``tensor_like``], list[list[\ :class:`~.pennylane.operation.Operator`]], list[``tensor_like``]])  | List of grouped qubit-wise commuting Hamiltonian terms obtained using :func:`~.pennylane.pauli.optimize_measurements`  |
    +----------------------------+--------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------+
    | ``basis_rot_groupings``    | tuple(list[``tensor_like``], list[list[\ :class:`~.pennylane.operation.Operator`]], list[``tensor_like``]])  | List of grouped Hamiltonian terms obtained using :func:`~.pennylane.qchem.basis_rotation`                              |
    +----------------------------+--------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------+

Auxiliary observables
~~~~~~~~~~~~~~~~~~~~~

The supplementary operators required to obtain additional properties of the molecule such as its dipole moment, spin, etc. 

.. rst-class:: docstable
    :widths: auto

    +----------------------------+----------------------------------+-----------------------------------------------------------------------------------+
    | **Name**                   | **Type**                         | **Description**                                                                   | 
    +============================+==================================+===================================================================================+
    | ``dipole_op``              | :class:`~.pennylane.Hamiltonian` | Qubit dipole moment operators for the chemical system                             |
    +----------------------------+----------------------------------+-----------------------------------------------------------------------------------+
    | ``number_op``              | :class:`~.pennylane.Hamiltonian` | Qubit particle number operator for the chemical system                            |
    +----------------------------+----------------------------------+-----------------------------------------------------------------------------------+
    | ``spin2_op``               | :class:`~.pennylane.Hamiltonian` | Qubit operator for computing total spin :math:`S^2` for the chemical system       |
    +----------------------------+----------------------------------+-----------------------------------------------------------------------------------+
    | ``spinz_op``               | :class:`~.pennylane.Hamiltonian` | Qubit operator for computing total spin's projection in the :math:`Z` direction   |
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
    | ``optimal_sector``         | ``numpy.ndarray``                        | Eigensector of the tapered qubits that would contain the ground state             |
    +----------------------------+------------------------------------------+-----------------------------------------------------------------------------------+

Tapered observables data
~~~~~~~~~~~~~~~~~~~~~~~~

Tapered observables and Hartree-Fock state based on the :math:`Z_2` symmetries of the molecular Hamiltonian. 

.. rst-class:: docstable
    :widths: auto

    +----------------------------+----------------------------------+-----------------------------------------------------------------------------------+
    | **Name**                   | **Type**                         | **Description**                                                                   | 
    +============================+==================================+===================================================================================+
    | ``tapered_hamiltonian``    | :class:`~.pennylane.Hamiltonian` | Tapered Hamiltonian                                                               |
    +----------------------------+----------------------------------+-----------------------------------------------------------------------------------+
    | ``tapered_hf_state``       | ``numpy.ndarray``                | Tapered Hartree-Fock state of the molecule                                        |
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

.. note::
    This data is only available for molecules with basis sets that require 20 or fewer qubits.

.. rst-class:: docstable
    :widths: auto

    +----------------------------+--------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------+
    | **Name**                   | **Type**                                         | **Description**                                                                                                         | 
    +============================+==================================================+=========================================================================================================================+
    | ``vqe_gates``              | list[\ :class:`~.pennylane.operation.Operation`] | :class:`~.pennylane.SingleExcitation` and :class:`~.pennylane.DoubleExcitation` gates for the optimized circuit         |
    +----------------------------+--------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------+
    | ``vqe_params``             | ``numpy.ndarray``                                | Optimal parameters for the gates that prepares ground state                                                             |
    +----------------------------+--------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------+
    | ``vqe_energy``             | ``float``                                        | Energy obtained from the state prepared by the optimized circuit                                                        |
    +----------------------------+--------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------+

Samples data
~~~~~~~~~~~~~

Samples data obtained from the optimized variational circuit with available Hamiltonian groupings.

.. note::
    This data is only available for molecules with basis sets that require 20 or fewer qubits.

.. rst-class:: docstable
    :widths: auto

    +----------------------------+----------------+---------------------------------------------------------------------------------+
    | **Name**                   | **Type**       | **Description**                                                                 | 
    +============================+================+=================================================================================+
    | ``qwc_samples``            | list[``dict``] | List of samples for each grouping of the qubit-wise commuting Hamiltonian terms |
    +----------------------------+----------------+---------------------------------------------------------------------------------+
    | ``basis_rot_samples``      | list[``dict``] | List of samples for each grouping of the basis-rotated Hamiltonian terms        |
    +----------------------------+----------------+---------------------------------------------------------------------------------+

.. toctree::
    :maxdepth: 2
    :hidden:

