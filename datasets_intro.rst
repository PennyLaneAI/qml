.. raw:: html

    <style>
        h1 {
            text-align: center;
        }
    </style>

    <link href="https://cdnjs.cloudflare.com/ajax/libs/mdbootstrap/4.5.14/css/mdb.min.css" rel="stylesheet">

.. |quantum_function| raw:: html

   <a href="https://docs.pennylane.ai/en/stable/introduction/circuits.html#quantum-functions" target="_blank">quantum function</a>

What is a Quantum Dataset?
==========================

.. meta::
   :property="og:description": Browse our collection of quantum datasets and import them into PennyLane directly from your code.
   :property="og:image": https://pennylane.ai/qml/_static/datasets/datasets.png


A **quantum dataset** is a collection of data that describes quantum systems and their evolution.
We refer to such data features as **quantum data**, which in the context of quantum programming,
can be realized as the input arguments and outputs of a |quantum_function| that defines a quantum system.

Examples of quantum data
~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: /_static/datasets/data_module.png
    :width: 85%
    :target: javascript:void(0);
    :align: center

Examples of quantum data include the following:

#. **Hamiltonian** of the system and any other observables for other relevant properties of interest.
#. **Quantum state** of interest, such as the ground state, and an efficient **state-preparation circuit** for it.
#. Any useful **unitary transformations** such as the Clifford operator required for :doc:`qubit tapering <demos/tutorial_qubit_tapering.html>`.
#. **Measurement or projection operators** and any resulting distributions and expectation values that can be extracted for the system.
#. Control parameters for **system evolution** and data related to **noise descriptions** such as the Kraus operators for channels.


Motivation for Quantum Datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. **A lesson from the success of machine learning:** easily-accessible, high-quality datasets catalyze the development of new algorithms and the improvement of older ones. 
#. **Challenge of achieving quantum advantage:** learning from quantum data is more intuitive for quantum computers, leading to an ideal candidate for quantum computational advantage.
#. **Never-ending quest for useful research:** readily available data reduces the work required for collaboration between different disciplines, fostering advancements in algorithmic techniques. 


Exploring quantum datasets using PennyLane
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PennyLane is an **open-source software framework built around the concept of quantum differentiable programming**, seamlessly integrating quantum simulators and hardware with machine-learning libraries.

PennyLane datasets allow you to develop novel algorithms and benchmark them against problems in applied quantum computation, such as the simulation of physical systems.

For more information on accessing hosted data from PennyLane, please see the `PennyLane Documentation <https://docs.pennylane.ai/en/stable/introduction/data.html>`_.

.. figure:: /_static/whatisqml/jigsaw.png
    :align: center
    :width: 70%
    :target: javascript:void(0);

.. toctree::
    :maxdepth: 2
    :hidden:
