.. role:: html(raw)
   :format: html

What is Quantum Chemistry?
==========================

.. meta::
   :property="og:description": Quantum chemistry is an area of research focused on addressing classically intractable chemistry problems with quantum computing.
   :property="og:image": 

Quantum chemistry is an area of research focused on addressing classically intractable chemistry problems with 
quantum computing. It is paramount to the commercial adoption of quantum computing, as simulating properties of
matter is a ubiquitous task in many industries.

Traditional methods for simulating the properties of matter are hindered, in some way, by representational capacity. 
Tried-and-true methods like Density Functional Theory (DFT) don't provide an obvious path forward for improving 
accuracy, while methods that are based on wave function approximations are limited to very small systems. By 
simulating properties of matter on quantum computers, we hope to alleviate both of these pain points in the near-term 
and long-term.

Nature is quantum mechanical
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: /_static/whatisqchem/???.png
    :align: right
    :width: 45%
    :target: javascript:void(0);


Classical computers are simply not able to simulate general quantum systems efficiently due to the computational 
complexity of nature. Richard Feynman famously said that "If you want to make a simulation of nature, you'd better make 
it quantum mechanical..."

**Nature is quantum, which is why using quantum computers as a simulation platform will enable us to simulate nature
efficiently.** We can leverage the physical laws of nature to tell us its secrets.


Quantum chemistry is the leading quantum computing application 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: /_static/whatisqchem/???.png
    :align: left
    :width: 60%
    :target: javascript:void(0);


Current devices limit our simulations to small molecules, but demonstrating quantum chemistry's use cases
on small systems only increases the appeal for quantum computing applications when hardware size constraints 
inevitably pass.

The omnipresent nature of simulating properties of materials in many industries means that quantum chemistry
is viewed as the leading candidate for applications of quantum computing.
As such, effort is being put into designing scalable algorithms that can be run on current devices.
This is proceeding in earnest, with the ability to 
`simulate chemical reactions <https://pennylane.ai/qml/demos/tutorial_chemical_reactions.html>_`, 
`optimize molecular geometries <https://pennylane.ai/qml/demos/tutorial_mol_geo_opt.html>_`, and 
`retrieve low-energy states <https://pennylane.ai/qml/demos/tutorial_vqe.html>_` 
of small molecules to a high-precision.


Leverage Quantum Machine Learning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: /_static/whatisqchem/???.png
    :align: right
    :width: 55%
    :target: javascript:void(0);


Many quantum chemistry algorithms require optimization as a step; an algorithm may contain parameters that must be 
fine-tuned for a specific application. Optimization within quantum computing is a non-trivial topic, 
but luckily we can transfer concepts from quantum machine learning (QML), another research area of quantum computing.

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

.. figure:: /_static/whatisqchem/???.png
    :align: center
    :width: 70%
    :target: javascript:void(0);
