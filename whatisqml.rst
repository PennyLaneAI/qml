.. role:: html(raw)
   :format: html

What is Quantum Machine Learning?
=================================

.. meta::
   :property="og:description": Quantum machine learning is a research area that explores the interplay of ideas from quantum computing and machine learning.
   :property="og:image": https://pennylane.ai/qml/_static/whatisqml/gpu_to_qpu.png

Quantum machine learning is a research area that **explores the interplay of ideas from quantum computing and machine learning.**

For example, we might want to find out whether quantum computers can speed up the
time it takes to train or evaluate a machine learning model. On the other hand, we can leverage techniques from machine learning to help us uncover quantum error-correcting codes, estimate the properties of quantum systems, or develop new quantum algorithms.


Quantum computers as AI accelerators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: /_static/whatisqml/gpu_to_qpu.png
    :align: right
    :width: 45%
    :target: javascript:void(0);


The limits of what machines can learn have always been defined by the computer hardware
we run our algorithms on—for example, the success of modern-day deep learning with neural networks is
enabled by parallel GPU clusters.

**Quantum machine learning extends the pool of hardware for machine learning by an entirely
new type of computing device—the quantum computer.** Information processing with quantum computers
relies on substantially different laws of physics known as *quantum theory*.


Machine learning on near-term quantum devices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: /_static/whatisqml/quantum_devices_ai.png
    :align: left
    :width: 60%
    :target: javascript:void(0);

Some research focuses on ideal, universal quantum computers ("fault-tolerant QPUs")
which are still years away. But there is rapidly-growing interest in **quantum machine learning on near-term** `quantum devices <https://www.cornell.edu/video/john-preskill-quantum-computing-nisq-era-beyond>`_.

We can `understand these devices <https://medium.com/xanaduai/quantum-machine-learning-1-0-76a525c8cf69>`_
as special-purpose hardware
like Application-Specific Integrated Circuits (ASICs) and
Field-Programmable Gate Arrays (FPGAs), which are more limited in their functionality.


Using quantum computers like neural networks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: /_static/whatisqml/trainable_circuit.png
    :align: right
    :width: 55%
    :target: javascript:void(0);

In the modern viewpoint, **quantum computers can be used and trained like neural networks**. 
We can systematically adapt the physical control parameters,
such as an electromagnetic field strength or a laser pulse frequency, to solve a problem.

For example, a trained circuit can be used to classify the content of images, by encoding
the image into the physical state of the device and taking measurements. 

The bigger picture: differentiable programming
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

But the story is bigger than just using quantum computers to tackle machine learning problems. 
Quantum circuits are *differentiable*, and a quantum computer
itself can compute the change in control parameters needed to become better at a given task.

`Differentiable programming <https://en.wikipedia.org/wiki/Differentiable_programming>`_
is the very basis of deep learning, implemented in software libraries such as TensorFlow and PyTorch.
**Differentiable programming is more than deep learning: it is a programming paradigm where the algorithms are not hand-coded, but learned.**

.. figure:: /_static/whatisqml/applications.png
    :align: center
    :width: 65%
    :target: javascript:void(0);


Similarly, the idea of training quantum computers is larger than quantum machine learning. Trainable quantum circuits can be leveraged in other fields like **quantum chemistry** or **quantum optimization**. It can help in a variety of applications such as the **design of quantum algorithms**, the discovery of **quantum error correction** schemes, and the **understanding of physical systems**.

PennyLane for quantum differentiable programming
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PennyLane is an **open-source software framework built around the concept of quantum differentiable programming**. It seamlessly integrates classical machine learning libraries with quantum simulators and hardware, giving users the power to train quantum circuits.

To find out more, visit the `PennyLane Documentation <https://pennylane.readthedocs.io>`_, or
check out the gallery of hands-on :doc:`quantum machine learning demonstrations <demonstrations>`.

.. figure:: /_static/whatisqml/jigsaw.png
    :align: center
    :width: 70%
    :target: javascript:void(0);
