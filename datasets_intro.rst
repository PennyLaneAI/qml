.. raw:: html

    <style>
        h1 {
            text-align: center;
        }
    </style>

    <link href="https://cdnjs.cloudflare.com/ajax/libs/mdbootstrap/4.5.14/css/mdb.min.css" rel="stylesheet">


What is a Quantum Dataset?
=====================

.. meta::
   :property="og:description": Every second spent in training parameterized circuits is a second spent not doing greater things in quantum computing.
   :property="og:image": https://pennylane.ai/qml/_static/datasets.png


A quantum dataset is a collection of **quantum data** obtained from various quantum systems that describes it and its evolution.

Quantum Data
~~~~~~~~~~~~~~

.. image:: /_static/datasets/qdata-illustration.png
    :align: right
    :width: 30%
    :target: javascript:void(0);

In a more general sense, we consider the quantum data to encompass the following things:

#. **Hamiltonian** of the system and any other auxiliary observables for other relevant properties of interest.
#. **Quantum state** of interest for the system such as the ground state, and an efficient **state-preparation circuit** for it.
#. Any useful **unitary transformations** of the system such as the Clifford operator required for tapering.
#. **Measurement or projection operators** for the systems and any resulting distributions and expectation values.
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

PennyLane is an **open-source software framework built around the concept of quantum differentiable programming**. It seamlessly integrates quantum simulators and hardware with machine-learning libraries, giving users the power to use the provided datasets to develop novel algorithms and benchmark them for problems in applied quantum computation, such as the simulation of physical systems. We also provide the **``data`` module** in PennyLane to access the hosted dataset readily and store and manipulate them locally.

To find out more, visit the `PennyLane Documentation <https://pennylane.readthedocs.io>`_, or check out the `Accessing Datasets` section of :doc:`quantum datasets <datasets>`.

.. figure:: /_static/whatisqml/jigsaw.png
    :align: center
    :width: 70%
    :target: javascript:void(0);

.. toctree::
    :maxdepth: 2
    :hidden:
