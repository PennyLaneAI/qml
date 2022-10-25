.. raw:: html

    <style>
        h1 {
            text-align: center;
        }
    </style>

    <link href="https://cdnjs.cloudflare.com/ajax/libs/mdbootstrap/4.5.14/css/mdb.min.css" rel="stylesheet">


What is Quantum Data?
=====================

.. meta::
   :property="og:description": Every second spent in training parameterized circuits is a second spent not doing further greater things in quantum computing.
   :property="og:image": https://pennylane.ai/qml/_static/datasets.png


Quantum dataset is just a collection of **quantum data** obtained from various quantum systems.

Quantum data can be defined as the **information** that describes a quantum system and its evolution.


Describing Quantum Data
~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: /_static/datasets/qdata-illustration.png
    :align: right
    :width: 30%
    :target: javascript:void(0);


#. Hamiltonian of the system and any other auxillary observables
#. System's quantum state and ab efficient state-preparation circuit
#. Any useful unitary transformations of the system
#. Measurement operators for the systems, and obtained measurement distributions and expectation values
#. Control parameters for system evolution and data related to noise description


Motivating Quantum Datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: /_static/datasets/qdata-motivation.png
    :align: left
    :width: 25%
    :target: javascript:void(0);

#. **Lesson from the success of machine learning:** Availability of multi-scale, high quality accessible datasets acts as a catalysis of developing new algorithms and benchmarks. 


#. **Challenge of finding quantum advantage:** Learning from quantum data might be more intuitive for quantum computers than classical ones


#. **Never-ending quest for useful research:** Enables cross-collaboration among different disciplines much easier which can foster better advancements in current techniques. 

|

Exploring quantum datasets using PennyLane
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PennyLane is an **open-source software framework built around the concept of quantum differentiable programming**. It seamlessly integrates classical machine learning libraries
with quantum simulators and hardware, giving users the power to use the provided datasets as a benchmark data for their novel algorithms and also to use it to solve problems in
the areas of applied quantum computation, such as simulation of physical systems. We provide the `data` module in PennyLane to readily access the hosted dataset, and also to store 
and manipulate them locally.

To find out more, visit the `PennyLane Documentation <https://pennylane.readthedocs.io>`_, or
check out the `Accessing Datasets` section of :doc:`quantum datasets <datasets>`.

.. figure:: /_static/whatisqml/jigsaw.png
    :align: center
    :width: 70%
    :target: javascript:void(0);

.. toctree::
    :maxdepth: 2
    :hidden:

    datasets
    datasets_intro
    datasets_module
    datasets_qchem
    datasets_qspin
