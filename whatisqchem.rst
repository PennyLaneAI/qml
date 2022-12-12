.. role:: html(raw)
   :format: html

What is Quantum Chemistry?
==========================

.. meta::
   :property="og:description": Quantum chemistry is a research area focused on addressing classically intractable chemistry problems with quantum computing.
   :property="og:image": https://pennylane.ai/qml/_static/whatisqchem/quantum_chemistry.svg

Quantum chemistry is an area of research focused on **calculating properties of molecules, and the materials built out of them, using quantum mechanics**. 
As an application of quantum computing, it is paramount to the commercial adoption of quantum computers, because the task of simulating various properties of
matter is a ubiquitous task in many industries.

Research stemming from quantum chemistry could be used to calculate the time evolution of a complex system, estimate the ground state energy of a Hamiltonian, or determine the electronic band structure of an exotic material, all of which go *beyond classical physics*. Since quantum computers are also quantum-mechanical, they offer a potential computational advantage in simulating the quantum properties of matter. It is also possible that we can exploit quantum computers to learn new classical methods, using techniques such as `quantum machine learning <https://pennylane.ai/qml/whatisqml.html>`_.

Nature is quantum-mechanical
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Richard Feynman famously said that "Nature isn't classical, dammit, and if you want to make a simulation of nature, you'd better make it quantum mechanical." To simulate a general quantum system on a classical computer, the available resources need to *scale exponentially* with the size of the system. Therefore, a classical computer cannot efficiently simulate a general quantum system.

.. image:: /_static/whatisqchem/quantum_chemistry.svg
    :align: right
    :width: 45%
    :target: javascript:void(0);

Following Feynman's suggestion, we note that **nature is quantum**. This suggests that if our simulation platform is also quantum, we should be able to simulate nature efficiently, with resources scaling as the size of the target system. For quantum chemistry, this offers a dramatic improvement on traditional, exponentially-costly methods.

Quantum chemistry is the leading quantum computing application 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Understanding the quantum properties of materials is of growing importance in a wide range of industries. As a result, quantum chemistry is viewed as the **leading candidate for a practical application of quantum computing**. Current quantum devices are small, and are therefore limited to *small systems*, even if they simulate them efficiently.

.. image:: /_static/whatisqchem/computational_quantum_chemistry.svg
    :align: left
    :width: 60%
    :target: javascript:void(0);

Hence, there is a push to develop **scalable algorithms** that will work on larger devices, which we expect to become available as current hardware limitations pass. Progress in this direction is being made in earnest with the ability to	`simulate chemical reactions <https://pennylane.ai/qml/demos/tutorial_chemical_reactions.html>`_, `optimize molecular geometries <https://pennylane.ai/qml/demos/tutorial_mol_geo_opt.html>`_, and `retrieve low-energy states <https://pennylane.ai/qml/demos/tutorial_vqe.html>`_ of small molecules to a high precision.


Quantum machine learning for quantum chemistry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Many quantum chemistry algorithms require *optimization* as a critical step; an algorithm may contain parameters that must be 
fine-tuned for a specific application. What we know about optimization algorithms within quantum computing falls back on 
another area of research within quantum computing; quantum machine learning (QML).

.. image:: /_static/whatisqchem/QChem_circuit.svg
    :align: right
    :width: 55%
    :target: javascript:void(0);

Generally, QML algorithms are a non-trivial topic. Luckily, we are still able to transfer concepts from QML to quantum 
chemistry algorithms like the `variational quantum eigensolver <https://pennylane.ai/qml/demos/tutorial_vqe.html>`_.

Quantum chemistry for quantum machine learning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Oftentimes, when research techniques are applied to other areas, it is mutually beneficial for both areas of research; QML has
provided incredible advancements in quantum chemistry, and vice versa. 

.. figure:: /_static/whatisqchem/QChem_applications.svg
    :align: center
    :width: 65%
    :target: javascript:void(0);

The advent of machine learning in the physical sciences has spurred countless generalizable techniques for the simulation of
matter. Inversely, employing such techniques in practice has led to invaluable discoveries in QML, like 
`barren plateaus <https://pennylane.ai/qml/demos/tutorial_barren_plateaus.html>_`, 
`quantum-inspired optimization routines <https://pennylane.ai/qml/demos/qnspsa.html>_`, and 
`hardware-compatible differentiation methods <https://pennylane.ai/qml/demos/tutorial_general_parshift.html>_`, 
all of which are relevant in quantum chemistry algorithms.

PennyLane for quantum chemistry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PennyLane is an **open-source software framework built around the concept of quantum differentiable programming**. 
Within PennyLane, the quantum chemistry module gives users the power to implement and develop state-of-the-art 
quantum chemistry algorithms.

To find out more, visit the `PennyLane Documentation <https://pennylane.readthedocs.io>`_, or
check out the gallery of hands-on :doc:`quantum chemistry demonstrations <demonstrations>`.

.. figure:: /_static/whatisqchem/PennyLane_applications.svg
    :align: center
    :width: 70%
    :target: javascript:void(0);
