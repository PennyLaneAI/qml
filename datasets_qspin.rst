.. raw:: html

    <style>
        h1 {
            text-align: center;
        }
    </style>

Quantum many-body physics datasets
===================================

.. meta::
   :property="og:description": Browse our collection of quantum datasets and import them into PennyLane directly from your code.
   :property="og:image": https://pennylane.ai/qml/_static/datasets/datasets.png

Simulating quantum many-body physics is an area of research with the potential for practical quantum advantage.
This field investigates spin models displaying quantum correlations.
Here you can explore our available quantum spin datasets for some common spin systems.

Spin systems
------------

.. image:: /_static/datasets/spin.png
    :align: right
    :width: 45%
    :target: javascript:void(0);

These datasets provide access to the following spin systems, with up to 16 particles:

* | **Transverse-field Ising model**
  | Parameterized by energy prefactor :math:`J`, and external field :math:`h`.
  | Hamiltonian: :math:`J\sum_{\langle i,j\rangle} \sigma_i^z\sigma_j^z + h\sum_i \sigma_i^x`
  | Order parameter: :math:`\langle M_z \rangle =\langle |\sum_i \sigma_i^z|\rangle`

* | **XXZ Heisenberg model**
  | Parameterized by coupling term :math:`J_{xy}` and :math:`J_z`.
  | Hamiltonian: :math:`J_{xy}\sum_{\langle i,j\rangle}(\sigma_i^x\sigma_j^x+\sigma_i^y\sigma_j^y) + J_z\sum_{\langle i,j\rangle} \sigma_i^z \sigma_j^z`
  | Order parameter: :math:`\langle M_z \rangle =\langle |\sum_i \sigma_i^z|\rangle`

* | **Fermi-Hubbard model**
  | Parameterized by hopping term :math:`t`, on-site interaction term :math:`U` and spin direction :math:`\sigma \in \{ \uparrow, \downarrow \}`.
  | Hamiltonian: :math:`-t(\sum_{\langle i, j\rangle, \sigma} \hat{c}^\dagger_i\hat{c}_j + h.c.) + U \sum_i \hat{n}_{i\uparrow} \hat{n}_{i\downarrow}`

* | **Bose-Hubbard model**
  | Parameterized by hopping term :math:`t`, and on-site interaction term :math:`U` with Fock space truncation of :math:`4`. 
  | Hamiltonian: :math:`-t ( \sum_{\langle i, j\rangle} \hat{b}^\dagger_i\hat{b}_j + h.c.) + U \sum_i \hat{n}_{i}\hat{n}_{i}`

For each spin system, datasets are available for 1-D lattices (linear chain) and 2-D lattices (rectangular grid) with and without periodic boundary conditions.
Each dataset contains results for 100 different values of a tunable parameter such as the external magnetic field, coupling constants, etc.
Additionally, each dataset contains classical shadows obtained with 1000-shot randomized measurements in the Pauli basis.

Accessing spin datasets
-----------------------

The spin datasets can be downloaded and loaded to memory using the :func:`~pennylane.data.load` function as follows:

>>> data = qml.data.load(
...     "qspin", sysname="Ising", periodicity="closed", lattice="chain", layout=(1, 4)
... )[0]
>>> print(data)
<Dataset = description: qspin/Ising/closed/chain/1x4, attributes: ['spin_system', 'hamiltonians', ...]>

Here, the positional argument ``"qspin"`` denotes that we are loading a spin dataset,
while the keyword arguments ``sysname``, ``periodicity``, ``lattice``, and ``layout`` specify the requested dataset.
The values for these keyword arguments are included in the table below. For more information on using PennyLane functions
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

+--------------------------------+---------------+------------------+------------------------------+---------------------------------+
| **Spin system (sysname)**      | **Lattices**  | **Periodicity**  | **Layout**                   | **Description**                 |
+================================+===============+==================+==============================+=================================+
| | Transverse-field Ising model | | Chain       | Open, Closed     | | (1, 4), (1, 8), (1, 16)    | | Varied Parameter - :math:`h`  |
| | (`Ising`)                    | | Rectangular |                  | | (2, 2), (2, 4), (2, 8)     | | Order Parameter - :math:`M_z` |
+--------------------------------+---------------+------------------+------------------------------+---------------------------------+
| | XXZ-Heisenberg model         | | Chain       | Open, Closed     | | (1, 4), (1, 8), (1, 16)    | | Varied Parameter - :math:`J_z`|
| | (`Heisenberg`)               | | Rectangular |                  | | (2, 2), (2, 4), (2, 8)     | | Order Parameter - :math:`M_z` |
+--------------------------------+---------------+------------------+------------------------------+---------------------------------+
| | Fermi Hubbard model          | | Chain       | Open, Closed     | | (1, 4), (1, 8)             | | Varied Parameter - :math:`U`  |
| | (`FermiHubbard`)             | | Rectangular |                  | | (2, 2), (2, 4)             | | Order Parameter - N/A         |
+--------------------------------+---------------+------------------+------------------------------+---------------------------------+
| | Bose Hubbard model           | | Chain       | Open, Closed     | | (1, 4), (1, 8)             | | Varied Parameter - :math:`U`  |
| | (`BoseHubbard`)              | | Rectangular |                  | | (2, 2), (2, 4)             | | Order Parameter - N/A         |
+--------------------------------+---------------+------------------+------------------------------+---------------------------------+


Data features
-------------

For each spin system, we can obtain the following characteristics for each of the 100 different system configurations:

Spin systems data
~~~~~~~~~~~~~~~~~

Information regarding the spin system, including a text description and parameters for each configuration.

.. rst-class:: docstable
    :widths: auto 

    +----------------------------+--------------------------------+-----------------------------------------------------------------------------------+
    | **Name**                   | **Type**                       | **Description**                                                                   | 
    +============================+================================+===================================================================================+
    | ``spin_system``            |  ``dict``                      | Basic description of the spin system including its name, Hamiltonian string, etc. |
    +----------------------------+--------------------------------+-----------------------------------------------------------------------------------+
    | ``parameters``             |  ``numpy.ndarray``             | Tunable parameters that determine the spin system configuration                   |
    +----------------------------+--------------------------------+-----------------------------------------------------------------------------------+

Hamiltonians and ground-state data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Hamiltonians for the spin systems (under the Jordan-Wigner transformation for the Fermi Hubbard model and `Binary Bosonic mapping <https://arxiv.org/abs/2105.12563>`__ for the Bose Hubbard Model). 

.. rst-class:: docstable
    :widths: auto 

    +----------------------------+------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------+
    | **Name**                   | **Type**                                                                           | **Description**                                                                   | 
    +============================+====================================================================================+===================================================================================+
    | ``hamiltonian``            |  list[:class:`~.pennylane.Hamiltonian`]                                            | Hamiltonian of the system in the Pauli basis                                      |
    +----------------------------+------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------+
    | ``ground_energies``        | ``numpy.ndarray``                                                                  | Ground state energies of each configuration of the spin system                    |
    +----------------------------+------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------+
    | ``ground_states``          | ``numpy.ndarray``                                                                  | Ground state of each configuration of the spin system                             |
    +----------------------------+------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------+

Phase transition data
~~~~~~~~~~~~~~~~~~~~~

Values of the order parameters, which can be used to investigate the phases of the spin systems.

.. rst-class:: docstable
    :widths: auto 

    +----------------------------+-------------------+-------------------------------------------------------------------+
    | **Name**                   | **Type**          | **Description**                                                   | 
    +============================+===================+===================================================================+
    | ``num_phases``             | ``int``           | Number of phases for the considered configurations                |
    +----------------------------+-------------------+-------------------------------------------------------------------+
    | ``order_params``           | ``numpy.ndarray`` | Value of order parameters for identifying phases                  |
    +----------------------------+-------------------+-------------------------------------------------------------------+    

Classical shadow data
~~~~~~~~~~~~~~~~~~~~~

Classical shadows measurement results and the randomized basis for each configuration using 1000 shots. 

.. rst-class:: docstable
    :widths: auto 

    +----------------------------+-------------------+-----------------------------------------------------------------+
    | **Name**                   | **Type**          | **Description**                                                 | 
    +============================+===================+=================================================================+
    | ``shadow_basis``           | ``numpy.ndarray`` | Randomized Pauli basis for the classical shadow measurements    |
    +----------------------------+-------------------+-----------------------------------------------------------------+
    | ``shadow_meas``            | ``numpy.ndarray`` | Results from the classical shadow measurements                  |
    +----------------------------+-------------------+-----------------------------------------------------------------+  

.. toctree::
    :maxdepth: 2
    :hidden:
