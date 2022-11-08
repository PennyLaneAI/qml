.. raw:: html

    <style>
        h1 {
            text-align: center;
        }
    </style>

Quantum Many-Body Physics Datasets
===================================

.. meta::
   :property="og:description": Browse our collection of quantum datasets, and import them into PennyLane directly from your code.
   :property="og:image": https://pennylane.ai/qml/_static/datasets.png

Simulating quantum many-body physics with quantum computation is an important area of research with potential for practical quantum advantage. It involves exploring quantum spin models that --- while more straightforward than simulating molecular Hamiltonian --- foster similar quantum-correlations-enabled phenomena.

Spin Systems
------------

.. image:: /_static/datasets/spin.png
    :align: right
    :width: 45%
    :target: javascript:void(0);

Through this dataset, it would be possible to access data for the following spin systems having up to 16 particles:

* Transverse-Field Ising model.
* XXZ Heisenberg model
* Fermi-Hubbard model.
* Bose-Hubbard model.

We vary a tunable parameter in their Hamiltonian for each spin system to obtain 100 different configurations.
For each such configuration, data is being made available for 1-D lattices (linear chain) and 2-D lattices (rectangular grid)
with and without the periodic boundary conditions. Additionally, we offer classical shadows for each configuration obtained
with a 1000-shot randomized measurement in the Pauli basis.

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
| **Spin System**               | **Lattices**  | **Periodicity**  | **Layout**                   | **Description**                     |
+===============================+===============+==================+==============================+=====================================+
| Transverse-field Ising Model  | | 1-D         | Open, Closed     | | (1, 4), (1, 8), (1, 16)    | | Varied Parameter - :math:`h`      |
|                               | | 2-D         |                  | | (2, 2), (2, 4), (2, 8)     | | Order Parameter - :math:`M_z`     |
+-------------------------------+---------------+------------------+------------------------------+-------------------------------------+
| XXZ-Heisenberg Model          | | 1-D         | Open, Closed     | | (1, 4), (1, 8), (1, 16)    | | Varied Parameter - :math:`\delta` |
|                               | | 2-D         |                  | | (2, 2), (2, 4), (2, 8)     | | Order Parameter - :math:`M_z`     |
+-------------------------------+---------------+------------------+------------------------------+-------------------------------------+
| Fermi Hubbard Model           | | 1-D         | Open, Closed     | | (1, 4), (1, 8)             | | Varied Parameter - :math:`U`      |
|                               | | 2-D         |                  | | (2, 2), (2, 4)             | | Order Parameter - N/A             |
+-------------------------------+---------------+------------------+------------------------------+-------------------------------------+
| Bose Hubbard Model            | | 1-D         | Open, Closed     | | (1, 4), (1, 8)             | | Varied Parameter - :math:`U`      |
|                               | | 2-D         |                  | | (2, 2), (2, 4)             | | Order Parameter - N/A             |
+-------------------------------+---------------+------------------+------------------------------+-------------------------------------+


Data Features
-------------

For each spin system, we can obtain the following characteristics for each of the `100` different system configuration:

Spin Systems Data
~~~~~~~~~~~~~~~~~

Information regarding the spin system, including its description in text and parameters for each configuration.

.. rst-class:: docstable
    :widths: auto 

    +----------------------------+--------------------------------+-----------------------------------------------------------------------------------+
    | ``spin_system``            |  ``dict``                      | Basic description of the spin system inlcuding its name, Hamiltonian string, etc. |
    +----------------------------+--------------------------------+-----------------------------------------------------------------------------------+
    | ``parameters``             |  ``numpy.array``               | Tunable parameters that determine the spin system configuration.                  |
    +----------------------------+--------------------------------+-----------------------------------------------------------------------------------+

Hamiltonian Data
~~~~~~~~~~~~~~~~

Hamiltonian for the molecular system under Jordan-Wigner transformation and `Binary Bosonic mapping <https://arxiv.org/abs/2105.12563>`__ (for Bose Hubbard Model). 

.. rst-class:: docstable
    :widths: auto 
    
    +----------------------------+------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------+
    | ``hamiltonian``            |  list[:class:`~.pennylane.Hamiltonian`]                                            | Hamiltonian of the system in the Pauli basis.                                     |
    +----------------------------+------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------+
    | ``ground_energies``        | ``numpy.array``                                                                    | Ground state energies of each configuration of the spin system.                   |
    +----------------------------+------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------+
    | ``ground_states``          | ``numpy.array``                                                                    | Ground state of each configuration of the spin system                             |
    +----------------------------+------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------+

Phase Transition Data
~~~~~~~~~~~~~~~~~~~~~

.. rst-class:: docstable
    :widths: auto 

    +----------------------------+-----------------+-------------------------------------------------------------------+
    | `num_phases`               | int             | Number of phases for the considered configurations                |
    +----------------------------+-----------------+-------------------------------------------------------------------+
    | `order_parameters`         | ``numpy.array`` | Value of order paramteres for identifying phases                  |
    +----------------------------+-----------------+-------------------------------------------------------------------+    

Classical Shadow Data
~~~~~~~~~~~~~~~~~~~~~

.. rst-class:: docstable
    :widths: auto 

    +----------------------------+-----------------+-----------------------------------------------------------------+
    | `shadow_basis`             | ``numpy.array`` | Randomized Pauli basis for the classical shadow measurements    |
    +----------------------------+-----------------+-----------------------------------------------------------------+
    | `shadow_meas`              | ``numpy.array`` | Results from the classical shadow measurements                  |
    +----------------------------+-----------------+-----------------------------------------------------------------+  

.. toctree::
    :maxdepth: 2
    :hidden:
