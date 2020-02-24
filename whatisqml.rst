.. role:: html(raw)
   :format: html

What is Quantum Machine Learning?
=================================

Quantum machine learning investigates the **consequences of using quantum computers for machine learning**.

For example, we might want to find out whether quantum computers can speed up the
time it takes to train or evaluate a machine learning model, or whether using quantum information
can improve the generalization performance on unseen data.


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


Machine learning with near-term quantum devices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: /_static/whatisqml/quantum_devices_ai.png
    :align: left
    :width: 60%
    :target: javascript:void(0);

While a lot of research focuses on ideal, universal quantum processing units ("fault-tolerant QPUs")
whose development is still further in the future, **a large share of quantum machine learning
focuses on**
`near-term quantum devices <https://www.cornell.edu/video/john-preskill-quantum-computing-nisq-era-beyond>`_.

One can `understand these devices <https://medium.com/xanaduai/quantum-machine-learning-1-0-76a525c8cf69>`_
as a form of special-purpose hardware
like Application-Specific Integrated Circuits (ASICs) and
Field-Programmable Gate Arrays (FPGAs), as they are limited in the number and type of operations
that can be executed in a single run.


Using quantum devices like neural networks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: /_static/whatisqml/trainable_circuit.png
    :align: right
    :width: 55%
    :target: javascript:void(0);

In the modern viewpoint on quantum machine learning,
**near-term quantum devices are used and trained like neural networks**.

This is done by systematically adapting the physical control parameters,
such as an electromagnetic field strength or a laser pulse frequency, to solve a machine learning problem.

For example, the trained circuit can be used to classify the content of images by encoding
the image into the physical state of the device and taking measurements.

The bigger picture: Making computers differentiable
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

But the story does not end here. Quantum circuits turn out to be *differentiable*, which means that a quantum computer
itself can compute the change in control parameters needed to become better at a given task.

`Differentiable programming <https://en.wikipedia.org/wiki/Differentiable_programming>`_
is the very basis of deep learning, implemented in software libraries such as TensorFlow and PyTorch.
**Differentiable programming is also more than deep learning: it is a programming paradigm where steps of an
algorithm are not hand-coded, but learned.**

.. figure:: /_static/whatisqml/applications.png
    :align: center
    :width: 65%
    :target: javascript:void(0);


Similarly, the idea of trainable quantum computations is larger than quantum machine learning. It includes,
and in fact originates from, other fields like **quantum chemistry** :cite:`peruzzo2014variational`
:cite:`mcclean2016theory`, **quantum optimization** :cite:`farhi2014quantum`, and extends to a variety of
applications such as the **design of quantum algorithms** :cite:`anschuetz2018variational`
or **quantum error correction** :cite:`johnson2017qvector`.

PennyLane for differentiable quantum computing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PennyLane is a **software framework that is built around the concept of
differentiable quantum computation**. It seamlessly integrates classical machine learning libraries with
and quantum simulators and hardware to give users the power to train quantum circuits themselves.

To find out more, visit the `PennyLane Documentation <https://pennylane.readthedocs.io/en/stable/>`_, or
check out the gallery of `hands-on demonstrations <https://pennylane.ai/qml/demos.html>`_.

.. figure:: /_static/whatisqml/jigsaw.png
    :align: center
    :width: 70%
    :target: javascript:void(0);
