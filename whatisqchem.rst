.. role:: html(raw)
   :format: html

What is Quantum Chemistry?
==========================

.. meta::
   :property="og:description": Quantum chemistry is an area of research focused on addressing classically intractable chemistry problems with quantum computing.
   :property="og:image": 

Quantum chemistry is an area of research focused on calculating properties of molecules, and the materials built out of them, using quantum mechanics. 
As an application of quantum computing, it is paramount to the commercial adoption of quantum computers, as simulating properties of
matter is a ubiquitous task in many industries.

Examples include simulating the time evolution of a complex system, estimating the ground state energy of a Hamiltonian, or determining the electronic band structure of an exotic material, all of which go beyond classical physics. Since quantum computers are also quantum-mechanical, they offer a potential computational advantage in simulating the quantum properties of matter. It is also possible that, using `quantum machine learning <https://pennylane.ai/qml/whatisqml.html>`_, we can exploit quantum computers to learn new classical methods.

Nature is quantum mechanical
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: /_static/whatisqchem/quantum_chemistry.svg
    :align: right
    :width: 45%
    :target: javascript:void(0);


Richard Feynman famously said that "Nature isn't classical, dammit, and if you want to make a simulation of nature, you'd better make it quantum mechanical." To simulate a general quantum system on a classical computer, the resources need to scale exponentially with the size of the system. Therefore, a classical computer cannot efficiently simulate a general quantum system.

Following Feynman's suggestion, we note that **nature is quantum**. This suggests that if our simulation platform is also quantum, we should be able to simulate nature efficiently, with resources scaling as the size of the target system. For quantum chemistry, this offers a dramatic improvement on traditional, exponentially-costly methods.


Quantum chemistry is the leading quantum computing application 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: /_static/whatisqchem/computational_quantum_chemistry.svg
    :align: left
    :width: 60%
    :target: javascript:void(0);


Understanding the quantum properties of materials is of growing importance in a wide range of industries. As a result, quantum chemistry is viewed as the leading candidate for a practical application of quantum computing. Current quantum devices are small, and are therefore limited to small systems even if they simulate them efficiently. Hence, there is a push to develop scalable algorithms that will work on larger devices which we expect to become available as current hardware limitations pass.
Progress in this direction is being made in earnest with the ability to
	`simulate chemical reactions <https://pennylane.ai/qml/demos/tutorial_chemical_reactions.html>`_, 
	`optimize molecular geometries <https://pennylane.ai/qml/demos/tutorial_mol_geo_opt.html>`_, and 
	`retrieve low-energy states <https://pennylane.ai/qml/demos/tutorial_vqe.html>`_ 
	of small molecules to a high precision.


Leveraging Quantum Machine Learning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: /_static/whatisqchem/QChem_circuit.svg
    :align: right
    :width: 55%
    :target: javascript:void(0);


Many quantum chemistry algorithms require optimization as a step; an algorithm may contain parameters that must be 
fine-tuned for a specific application. Optimization within quantum computing is a non-trivial topic, 
but luckily we can transfer concepts from quantum machine learning (QML), another research area of quantum computing.

.. figure:: /_static/whatisqchem/QChem_applications.svg
    :align: center
    :width: 65%
    :target: javascript:void(0);

The advent of machine learning in the physical sciences has spurred countless generalizable techniques for simulating
matter. Applying such techniques to trainable quantum circuits has led to invaluable discoveries, like 
`barren plateaus <https://pennylane.ai/qml/demos/tutorial_barren_plateaus.html>_`, 
`quantum-inspired optimization routines <https://pennylane.ai/qml/demos/qnspsa.html>_`,
and `hardware-compatible differentiation methods <https://pennylane.ai/qml/demos/tutorial_general_parshift.html>_`, 
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
