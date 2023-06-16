.. role:: html(raw)
   :format: html

What Is Quantum Computing?
==========================

.. meta::
   :property="og:description": Quantum computing is a research area that extends the set of physical laws classical computers operate on by accessing quantum aspects of the physical world, opening up new ways of processing information.
   :property="og:image": https://pennylane.ai/qml/_static/whatisqc/quantum_computer.svg

Quantum computing is a research area that extends the set of physical laws classical computers operate on by 
accessing **quantum aspects of the physical world**, opening up new ways of processing information.
As a highly interdisciplinary field of quantum science and technology, quantum computing sits at the meeting point of algorithm design, complexity theory, system architecture, and hardware development.

Applications of quantum computing come in a plethora of flavors and span the gamut from low-level hardware
problems to higher-level and abstract problems. On the one hand, various phenomena of quantum mechanics are used
to implement physical schemes to control and stabilize quantum units of information—**qubits**—or to synthesize 
a quantum circuit as efficiently as possible. On the other hand, more conceptual problems of quantum computing 
involve the breaking of classical encryption protocols, simulation of quantum systems, or even training of 
`quantum neural networks <https://pennylane.ai/qml/what-is-quantum-machine-learning.html>`_.


Advantage and challenge of quantum computing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: /_static/whatisqc/quantum_computer.svg
    :align: right
    :width: 42%
    :target: javascript:void(0);


To explore **what quantum computers can do and how to make them do it**, many of their aspects are researched and applied.
For example, the quantum phenomenon of *superposition* allows for quantum computers to explore many possibilities 
at once and operate entirely differently from their classical counterparts.

The original motivation for studying quantum computers was the efficient simulation of quantum systems. But a 
sequence of breakthroughs in algorithm research in the ‘90s demonstrated that quantum computers could also 
**outperform their classical counterparts** at specific tasks like searching lists and factoring numbers. 
Subsequently, research in this field began in earnest.

The art and science of quantum computing lies in the design of algorithms that can encode useful information into 
a computational model, modify it in specific ways, and read out the result. But this is far from trivial, as the 
branches of a quantum algorithm cannot be observed simultaneously—they are *collapsed* into a single number by 
the act of measuring.

Quantum computing on near-term quantum devices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: /_static/whatisqc/NISQ_computing.svg
    :align: left
    :width: 42%
    :target: javascript:void(0);

Perhaps the primary ambition of quantum computing nowadays is to develop a **scalable, fault-tolerant device** to 
implement any and all quantum algorithms. Research in specialized fields like quantum machine learning and quantum 
chemistry is already underway, but existing quantum hardware is limited to devices that are either small, noisy, 
or non-universal—a computing paradigm known as noisy intermediate-scale quantum, or **NISQ**.

On top of that, real devices that can carry out useful quantum algorithms consist of **many interacting components**, 
which must be able to correct errors that inevitably arise from **device–environment interactions**. Architectures 
where errors can be corrected faster than they occur are referred to as *fault-tolerant*.

The bigger picture: quantum advantage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: /_static/whatisqc/Quantum_advantage.svg
    :align: right
    :width: 63%
    :target: javascript:void(0);

Whether practical algorithmic speedups are possible in the NISQ regime remains an open problem, but some instances 
have already been demonstrated of `quantum devices solving computational problems <https://www.nature.com/articles/s41586-022-04725-x>`_ that would take classical 
computers an unfeasible amount of time. **Quantum computational advantage is already a reality**, 
with an :doc:`increasing number <gbs>` of commercial and research organizations announcing their :doc:`breakthroughs <qsim_beyond_classical>`—some even 
making their devices publicly available for further research.

Quantum computational advantage does not necessarily need to be demonstrated on problems that are thought of as 
useful or practical, but it shows a clear sign toward the bright future of quantum computing. 
**Scalable and error-resilient** quantum computers remain a central goal in the trek toward universal, 
fault-tolerant quantum computing for the post-NISQ era.

A multifaceted technology to solve real-world problems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Few technological advances hold a real potential to let us speed through entire stages of scientific development, 
but quantum computers are likely to be one of the cornerstone technologies of the 21st century, 
**changing the way we do research, protect our data, communicate, and understand the world around us**. 
Initial progress has already shown that—with further improvements to quantum hardware setups—quantum computing 
will be used widely and applied to an ever-growing variety of problems across the globe.

.. figure:: /_static/whatisqc/QC_applications.svg
    :align: center
    :width: 63%
    :target: javascript:void(0);

For example, quantum computers could eventually be used to speedrun the `development of new chemical compounds <https://pennylane.ai/qml/what-is-quantum-chemistry.html>`_ for 
medicine or agriculture, enable a perfectly secure exchange of private messages, optimize and enhance existing 
computational algorithms for image classification, traffic management, or product design, and for 
**thousands of other uses we haven’t thought of yet**.

PennyLane for quantum computing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PennyLane is an **open-source cross-platform Python library** that supports a wide array of tasks in quantum computing, quantum machine learning, and quantum chemistry. Its capabilities for the differentiable programming of quantum computers have been designed to seamlessly integrate with classical machine learning libraries, quantum simulators and hardware, giving users the power to train quantum circuits.

To find out more, visit the `PennyLane Documentation <https://docs.pennylane.ai>`_ or check out the gallery of hands-on :doc:`demonstrations <demonstrations>`.

.. figure:: /_static/whatisqc/PennyLane_applications.svg
    :align: center
    :width: 77%
    :target: javascript:void(0);

