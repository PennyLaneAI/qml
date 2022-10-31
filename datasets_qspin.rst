.. raw:: html

    <style>
        h1 {
            text-align: center;
        }
    </style>

Quantum many-body Physics Datasets
===================================

.. meta::
   :property="og:description": Browse our collection of quantum datasets, and import them into PennyLane directly from your code.
   :property="og:image": https://pennylane.ai/qml/_static/datasets.png

Simulating quantum many-body physics with quantum computation is an important area of research with potential for practical quantum advantage. It involves exploring quantum spin models that --- while more straightforward than simulating molecular Hamiltonian --- foster similar quantum-correlations-enabled phenomena.

Spin Systems
~~~~~~~~~~~~~

.. image:: /_static/datasets/spin.png
    :align: right
    :width: 45%
    :target: javascript:void(0);

Through this dataset, it would be possible to access data for the following spin systems having up to 16 particles:

* Transverse-Field Ising model.
* XXZ Heisenberg model
* Fermi-Hubbard model.
* Bose-Hubbard model.

We vary a tunable parameter in their Hamiltonian for each spin system to obtain 100 different configurations. For each such configuration, data is being made available for 1-D lattices (linear chain) and 2-D lattices (rectangular grid) with and without the periodic boundary conditions. Additionally, we offer classical shadows for each configuration obtained with a 1000-shot randomized measurement in the Pauli basis.


Data Features
~~~~~~~~~~~~~~

For each spin system, we obtain the following data for `100` different `parameters`.


#. **Spin Systems Data**
    .. rst-class:: docstable
        :widths: auto 

        +----------------------------+-----------------------------------------------------------------------------------+
        | `spin_system`              | Basic description of the spin system                                              |
        +----------------------------+-----------------------------------------------------------------------------------+
        | `parameters`               | Tunable parameters that determine the spin system                                 |
        +----------------------------+-----------------------------------------------------------------------------------+    

#. **Hamiltonian and Ground-State Data**
    .. rst-class:: docstable
        :widths: auto 

        +----------------------------+-----------------------------------------------------------------------------------+
        | `hamiltonians`             | PennyLane Hamiltonian for the spin system                                         |
        +----------------------------+-----------------------------------------------------------------------------------+
        | `ground_energies`          | Ground state energies of each system                                              |
        +----------------------------+-----------------------------------------------------------------------------------+
        | `ground_states`            | Ground state of each system                                                       |
        +----------------------------+-----------------------------------------------------------------------------------+   

#. **Phase Transition Data**
    .. rst-class:: docstable
        :widths: auto 

        +----------------------------+-----------------------------------------------------------------------------------+
        | `num_phases`               | Number of phases for the considered configurations                                |
        +----------------------------+-----------------------------------------------------------------------------------+
        | `order_parameters`         | Observables and their values identifying phases                                   |
        +----------------------------+-----------------------------------------------------------------------------------+    

#. **Classical Shadow Data**
    .. rst-class:: docstable
        :widths: auto 

        +----------------------------+-----------------------------------------------------------------------------------+
        | `shadow_basis`             | Randomized Pauli basis for the classical shadow measurements                      |
        +----------------------------+-----------------------------------------------------------------------------------+
        | `shadow_meas`              | Results from the classical shadow measurements                                    |
        +----------------------------+-----------------------------------------------------------------------------------+  

.. toctree::
    :maxdepth: 2
    :hidden:
