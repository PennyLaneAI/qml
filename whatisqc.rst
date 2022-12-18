.. role:: html(raw)
   :format: html

What Is Quantum Computing?
==========================

.. meta::
   :property="og:description": Quantum computing is a research area that explores what quantum computers can do and how to make them do it.
   :property="og:image": https://pennylane.ai/qml/_static/whatisqc/quantumt_computer.svg

Quantum computing is a research area that explores **what quantum computers can do and how to make them do it**. It is a highly interdisciplinary field of quantum science and technology, sitting at the meeting point of algorithm design, complexity theory, system architecture, and hardware development.

Examples span the gamut from low-level hardware problems that engage a variety of phenomena in quantum mechanics, such as the implementation of physical schemes to control and stabilize quantum units of information, qubits, or the best way to synthesize a quantum circuit. On the other hand, higher-level problems in quantum computing address the breaking of classical protocols and devising of post-quantum approaches to encryption schemes, simulating quantum systems, or even training quantum neural networks.


Advantage and challenge of quantum computing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: /_static/whatisqc/quantum_computer.svg
    :align: right
    :width: 42%
    :target: javascript:void(0);


Quantum computers **extend the set of physical laws** classical computers operate on by accessing quantum aspects of the physical world and, as a result, gain new ways of processing information. For example, quantum computers have the ability to explore many possibilities at once, a phenomenon known as *superposition*.

The original motivation for studying quantum computers was the efficient simulation of quantum systems. But a sequence of breakthroughs in algorithm research in the ‘90s demonstrated that quantum computers could also **outperform their classical counterparts** at specific tasks like searching lists and factoring numbers. Subsequently, research in this field began in earnest.

The art and science of quantum computing lies in the design of algorithms that can encode useful information into a computational model, modify it in specific ways, and read out the result. But this is far from trivial, as the branches of a quantum algorithm cannot be observed simultaneously—they are *collapsed* into a single number by the act of measuring.

Quantum computing on near-term quantum devices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: /_static/whatisqc/NISQ_computing.svg
    :align: left
    :width: 42%
    :target: javascript:void(0);

Perhaps the primary ambition of the field of quantum computing nowadays is to develop a **scalable, fault-tolerant device** on which any and all quantum algorithms can be implemented. Research in specialized fields like quantum machine learning and quantum chemistry is already underway, but existing quantum hardware is currently limited to devices that are either small, noisy, or non-universal—a computing paradigm known as noisy intermediate-scale quantum, or **NISQ**.

On top of that, real devices that can carry out useful quantum algorithms consist of **many interacting components** and they must be able to correct errors that inevitably arise from **device–environment interactions**. Architectures that support error correction are referred to as *fault-tolerant*.

The bigger picture: quantum advantage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: /_static/whatisqc/Borealis_quantum_advantage.svg
    :align: right
    :width: 63%
    :target: javascript:void(0);

Whether practical algorithmic speedups are possible in the NISQ regime remains an open problem, but some instances of quantum devices solving computational problems that would take classical computers an unfeasible amount of time have already been demonstrated. For example, `Borealis <https://www.xanadu.ai/products/borealis/>`_, **Xanadu’s flagship 216-qubit photonic quantum computer**, was shown to be able to achieve `quantum computational advantage <https://xanadu.ai/blog/beating-classical-computers-with-Borealis>`_, and was the first of its kind to been made accessible to everyone on `Xanadu Cloud <https://pennylane.xanadu.ai/>`_.

Quantum computational advantage does not necessarily need to be demonstrated on problems that are thought of as useful or practical, but it shows a clear sign toward the bright future of quantum computing. **Scalable and error-resilient** photonic quantum computers remain Xanadu’s central goal in the trek toward universal, fault-tolerant quantum computing.

A general-purpose technology to solve real-world problems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Few technological advances hold a real potential to let us speed through entire stages of scientific development, but quantum computers are likely to be the cornerstone *general-purpose technology* of the 21st century that will **change the way we do research, protect our data, communicate, and understand the world around us**. Initial progress has already shown that—with further improvements to quantum hardware setups—quantum computing will be used widely and applied to an ever-growing variety of problems across the globe.

.. figure:: /_static/whatisqc/QC_applications.svg
    :align: center
    :width: 63%
    :target: javascript:void(0);

For example, quantum computers can be used to speedrun the development of new chemical compounds for medicine or agriculture, enable a perfectly secure exchange of private messages, optimize and enhance existing computational algorithms for image classification, traffic management, or product design, and for **thousands of other uses we haven’t thought of yet**.

PennyLane for quantum computing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PennyLane is an **open-source cross-platform Python library** that supports a wide array of tasks in quantum computing, quantum machine learning, and quantum chemistry. Its capabilities for the differentiable programming of quantum computers have been designed to seamlessly integrate with classical machine learning libraries, quantum simulators and hardware, giving users the power to train quantum circuits.

To find out more, visit the `PennyLane Documentation <https://pennylane.readthedocs.io>`_ or check out the gallery of hands-on :doc:`demonstrations <demonstrations>`.

.. figure:: /_static/whatisqc/PennyLane_applications.svg
    :align: center
    :width: 77%
    :target: javascript:void(0);

