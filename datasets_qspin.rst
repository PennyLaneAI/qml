.. raw:: html

    <style>
        h1 {
            text-align: center;
        }
    </style>

Quantum many-body Physics Datasets
===================================

Another area for exploring practical quantum advantage is quantum many-body physics, which involves exploring quantum spin models that while being more straightforward than molecular Hamiltonian, foster similar quantum-correlations-enabled phenomena. Therefore, in the first offering of our datasets, we include a quantum many-body physics dataset containing data related to some popularly studied spin systems.

Spin Systems
~~~~~~~~~~~~~

.. image:: /_static/datasets/spin.png
    :align: right
    :width: 45%
    :target: javascript:void(0);

Through this dataset, it would be possible to access data for the following spin systems having up to 16 particles:

* Transeverse-Field Ising model.
* XXZ Heisenberg model
* Fermi-Hubbard model.
* Bose-Hubbard model.

We vary a tunable parameter in their Hamiltonian for each spin system to obtain 100 different configurations. For each such configuration, data is being made available for 1-D lattices (linear chain) and 2-D lattices (rectangular grid) with and without the periodic boundary conditions. Additionally, we also offer classical shadows for each configuration obtained with a 1000-shot randomized measurement in the Pauli basis.


Data Features
~~~~~~~~~~~~~~

For each spin system, we obtain the following data for `100` different `parameters`.


#. **Spin Systems Data**

    .. table::
        :widths: auto 

        +----------------------------+-----------------------------------------------------------------------------------+
        | `spin_system`              | Basic description of the spin system                                              |
        +----------------------------+-----------------------------------------------------------------------------------+
        | `parameters`               | Tunable parameters that determine the spin system                                 |
        +----------------------------+-----------------------------------------------------------------------------------+    

#. **Hamiltonian and Ground-state Data**

    .. table::
        :widths: auto 

        +----------------------------+-----------------------------------------------------------------------------------+
        | `hamiltonians`             | PennyLane Hamiltonian for the spin system                                         |
        +----------------------------+-----------------------------------------------------------------------------------+
        | `ground_energies`          | Ground state energies of each system                                              |
        +----------------------------+-----------------------------------------------------------------------------------+
        | `ground_states`            | Ground state of each system                                                       |
        +----------------------------+-----------------------------------------------------------------------------------+   

#. **Phase Transition Data**

    .. table::
        :widths: auto 

        +----------------------------+-----------------------------------------------------------------------------------+
        | `num_phases`               | Number of phases for the considered configurations                                |
        +----------------------------+-----------------------------------------------------------------------------------+
        | `order_parameters`         | Observables and their values identifying phases                                   |
        +----------------------------+-----------------------------------------------------------------------------------+    

#. **Classical Shadow Data**

    .. table::
        :widths: auto 

        +----------------------------+-----------------------------------------------------------------------------------+
        | `shadow_basis`             | Randomized Pauli basis for the classical shadow measurements                      |
        +----------------------------+-----------------------------------------------------------------------------------+
        | `shadow_meas`              | Results from the classical shadow measurements                                    |
        +----------------------------+-----------------------------------------------------------------------------------+  

.. toctree::
    :maxdepth: 2
    :hidden:
