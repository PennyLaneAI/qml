.. raw:: html

    <style>
        h1 {
            text-align: center;
        }
    </style>

Quantum many-body physics datasets
===================================

.. meta::
   :property="og:description": Browse our collection of quantum datasets, and import them into PennyLane directly from your code.
   :property="og:image": https://pennylane.ai/qml/_static/datasets.png

Simulating quantum many-body physics is an area of research with the potential for practical quantum advantage.
This field investigates spin models displaying quantum correlations.

Explore our available quantum spin datasets below, providing data related to some popularly examined spin systems.

Spin systems
------------

.. image:: /_static/datasets/spin.png
    :align: right
    :width: 45%
    :target: javascript:void(0);

These datasets provide access to data for the following spin systems, with up to 16 particles:

* Transverse-Field Ising model.
* XXZ Heisenberg model
* Fermi-Hubbard model.
* Bose-Hubbard model.

For each spin system, datasets are available for 1-D lattices (linear chain) and 2-D lattices (rectangular grid) with and without periodic boundary conditions.
Each dataset contains results for 100 different values of a tunable parameter such as the external magnetic field or the coupling constant.
Additionally, each dataset contains classical shadows obtained with 1000-shot randomized measurements in the Pauli basis.

Accessing spin datasets
-----------------------

The spin datasets can be downloaded and loaded to memory using the :func:`~pennylane.data.load` function as follows:

>>> data = qml.data.load(
...     "qspin", sysname="Ising", periodicity="closed", lattice="chain", layout=(1, 4)
... )[0]
>>> print(data)
<pennylane.data.dataset.Dataset object at 0x7f14e4369640>

Here, the positional argument ``"qspin"`` denotes that we are loading a spin dataset,
while the keyword arguments ``sysname``, ``periodicity``, ``lattice``, and ``layout`` specify the requested dataset.
The values for these keyword arguments are included in the table below. For more information on using PennyLane functions
please see the `PennyLane Documentation <https://docs.pennylane.ai/en/latest/introduction/data.html>`_.

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

+-------------------------------+---------------+------------------+------------------------------+-------------------------------------+
| **Spin system**               | **Lattices**  | **Periodicity**  | **Layout**                   | **Description**                     |
+===============================+===============+==================+==============================+=====================================+
| Transverse-field Ising model  | | Chain       | Open, Closed     | | (1, 4), (1, 8), (1, 16)    | | Varied Parameter - :math:`h`      |
|                               | | Rectangular |                  | | (2, 2), (2, 4), (2, 8)     | | Order Parameter - :math:`M_z`     |
+-------------------------------+---------------+------------------+------------------------------+-------------------------------------+
| XXZ-Heisenberg model          | | Chain       | Open, Closed     | | (1, 4), (1, 8), (1, 16)    | | Varied Parameter - :math:`\delta` |
|                               | | Rectangular |                  | | (2, 2), (2, 4), (2, 8)     | | Order Parameter - :math:`M_z`     |
+-------------------------------+---------------+------------------+------------------------------+-------------------------------------+
| Fermi Hubbard model           | | Chain       | Open, Closed     | | (1, 4), (1, 8)             | | Varied Parameter - :math:`U`      |
|                               | | Rectangular |                  | | (2, 2), (2, 4)             | | Order Parameter - N/A             |
+-------------------------------+---------------+------------------+------------------------------+-------------------------------------+
| Bose Hubbard model            | | Chain       | Open, Closed     | | (1, 4), (1, 8)             | | Varied Parameter - :math:`U`      |
|                               | | Rectangular |                  | | (2, 2), (2, 4)             | | Order Parameter - N/A             |
+-------------------------------+---------------+------------------+------------------------------+-------------------------------------+


Data features
-------------

For each spin system, we can obtain the following characteristics for each of the `100` different system configurations:

Spin systems data
~~~~~~~~~~~~~~~~~

Information regarding the spin system, including a text description and parameters for each configuration.

.. rst-class:: docstable
    :widths: auto 

    +----------------------------+--------------------------------+-----------------------------------------------------------------------------------+
    | ``spin_system``            |  ``dict``                      | Basic description of the spin system including its name, Hamiltonian string, etc. |
    +----------------------------+--------------------------------+-----------------------------------------------------------------------------------+
    | ``parameters``             |  ``numpy.array``               | Tunable parameters that determine the spin system configuration.                  |
    +----------------------------+--------------------------------+-----------------------------------------------------------------------------------+

Hamiltonians and ground-state data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Hamiltonians for the spin systems (under the Jordan-Wigner transformation for the Fermi Hubbard model and `Binary Bosonic mapping <https://arxiv.org/abs/2105.12563>`__ for the Bose Hubbard Model). 

.. rst-class:: docstable
    :widths: auto 
    
    +----------------------------+------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------+
    | ``hamiltonian``            |  list[:class:`~.pennylane.Hamiltonian`]                                            | Hamiltonian of the system in the Pauli basis.                                     |
    +----------------------------+------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------+
    | ``ground_energies``        | ``numpy.array``                                                                    | Ground state energies of each configuration of the spin system.                   |
    +----------------------------+------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------+
    | ``ground_states``          | ``numpy.array``                                                                    | Ground state of each configuration of the spin system                             |
    +----------------------------+------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------+

Phase transition data
~~~~~~~~~~~~~~~~~~~~~

Value of the order parameters that can be used to investigate the phases of the spin systems.

.. rst-class:: docstable
    :widths: auto 

    +----------------------------+-----------------+-------------------------------------------------------------------+
    | ``num_phases``             | ``int``         | Number of phases for the considered configurations                |
    +----------------------------+-----------------+-------------------------------------------------------------------+
    | ``order_parameters``       | ``numpy.array`` | Value of order paramteres for identifying phases                  |
    +----------------------------+-----------------+-------------------------------------------------------------------+    

Classical shadow data
~~~~~~~~~~~~~~~~~~~~~

Classical shadows measurement results and the randomized basis for each configuration using 1000 shots. 

.. rst-class:: docstable
    :widths: auto 

    +----------------------------+-----------------+-----------------------------------------------------------------+
    | ``shadow_basis``           | ``numpy.array`` | Randomized Pauli basis for the classical shadow measurements    |
    +----------------------------+-----------------+-----------------------------------------------------------------+
    | ``shadow_meas``            | ``numpy.array`` | Results from the classical shadow measurements                  |
    +----------------------------+-----------------+-----------------------------------------------------------------+  

.. toctree::
    :maxdepth: 2
    :hidden:
