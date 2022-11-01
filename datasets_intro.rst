.. raw:: html

    <style>
        h1 {
            text-align: center;
        }
    </style>

    <link href="https://cdnjs.cloudflare.com/ajax/libs/mdbootstrap/4.5.14/css/mdb.min.css" rel="stylesheet">


What is a Quantum Dataset?
==========================

.. meta::
   :property="og:description": Browse our collection of quantum datasets, and import them into PennyLane directly from your code.
   :property="og:image": https://pennylane.ai/qml/_static/datasets.png


A quantum dataset is a collection of **quantum data** obtained from various quantum systems that describes it and its evolution.
In the context of quantum programming, this can be realized as the input and output of a quantum function that describes a quantum system.

Examples of quantum data
~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: /_static/datasets/qdata-illustration.png
    :align: right
    :width: 30%
    :target: javascript:void(0);

Examples of quantum data include the following:

#. **Hamiltonian** of the system and any other auxiliary observables for other relevant properties of interest.
#. **Quantum state** of interest for the system such as the ground state, and an efficient **state-preparation circuit** for it.
#. Any useful **unitary transformations** of the system such as the Clifford operator required for :doc:`qubit tapering <demos/tutorial_qubit_tapering.html>`.
#. **Measurement or projection operators** for the systems and any resulting distributions and expectation values that can be extracted for the system.
#. Control parameters for **system evolution** and data related to **noise descriptions** such as the Kraus operators for channels.


Motivating Quantum Datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: /_static/datasets/qdata-motivation.png
    :align: left
    :width: 25%
    :target: javascript:void(0);

#. **A lesson from the success of machine learning:** Availability of multi-scale, high-quality accessible datasets for training and benchmarking acts as a catalysis of both developing new algorithms, and improving older ones. 
#. **Challenge of achieving quantum advantage:** Learning from quantum data might be more intuitive for quantum computers than classical ones, leading to an ideal candidate for quantum computational advantage.
#. **Never-ending quest for useful research:** Readily available data reduces the work required for cross-collaboration among different disciplines, ultimately fostering better advancements in current algorithmic techniques. 


Exploring quantum datasets using PennyLane
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PennyLane is an **open-source software framework built around the concept of quantum differentiable programming**, seamlessly integrating quantum simulators and hardware with machine-learning libraries.

PennyLane datasets give you the power to use the provided datasets to develop novel algorithms, and benchmark them against problems in applied quantum computation, such as the simulation of physical systems.

For more information on accessing hosted data from PennyLane, please see the :doc:`datasets <introduction/quantum_datasets>` documentation.

.. figure:: /_static/whatisqml/jigsaw.png
    :align: center
    :width: 70%
    :target: javascript:void(0);

.. toctree::
    :maxdepth: 2
    :hidden:
