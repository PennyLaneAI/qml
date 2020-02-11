
What is Quantum Machine Learning?
=================================

The limits of what machines can learn have always been defined by the computer hardware
we run our algorithms on - for example, the success of modern-day deep learning  with neural networks is
enabled by parallel GPU clusters.

:html:`<br>`

.. figure:: ../_static/whatisml/Tesla-NVIDIA_GPU_cluster.jpg
    :align: left
    :width: 30%
    :target: javascript:void(0);

:html:`<br>`

Quantum machine learning extends the pool of hardware for machine learning by an entirely
new type of computing device - the quantum computer. Information processing with quantum computers
underlies substantially different laws of physics, namely quantum theory.

:html:`<br>`

.. figure:: ../_static/whatisml/quantum_hardware.png
    :align: left
    :width: 30%
    :target: javascript:void(0);

:html:`<br>`

While a lot of research focuses on ideal, universal quantum processing units (QPUs) whose development
is still a thing of the future [REFs], a large share of quantum machine learning - and the share
PennyLane caters for - is interested in near term quantum devices [REFs]. One can understand these devices
as special purpose hardware in between Application-Specific Integrated Circuits (ASICs) and
Field-Programmable Gate Arrays (FPGAs).

:html:`<br>`

.. figure:: ../_static/whatisml/quantum_devices_ai.svg
    :align: left
    :width: 80%
    :target: javascript:void(0);

:html:`<br>`

The quantum device is used like a neural network: Data gets "loaded" into the quantum state of the
hardware and then processed. A measurement reveals the outcome of the machine learning model, for example
if an image showed a cat or a dog. The crucial point is that near-term quantum computers are just physical
machines that are programmed by control parameters, such as an electromagnetic field. By systematically
adapting these parameters, the quantum computation - and hence the machine learning algorithm - becomes
*trainable*.

:html:`<br>`

.. figure:: ../_static/whatisml/trainable_quantum_device.png
    :align: left
    :width: 50%
    :target: javascript:void(0);

:html:`<br>`

But the story does not end here. Quantum circuits turn out to be *differentiable*, which means that a quantum computer
can compute the change in control parameters needed to become better at a given task. Differentiable programming
is the very basis of deep learning, implemented in software libraries such as TensorFlow and PyTorch.
Differentiable programming is also more than deep learning, it is a programming paradigm where steps of an
algorithm are not hand-coded but learnt for a given task.

Similarly, the idea of trainable quantum computations is larger than quantum machine learning. It includes,
and in fact originates, from a field called *quantum chemistry* [REF], in which adaptable quantum circuits are
used to find ground state energies of atoms and molecules. Trainable circuits also feature
in *quantum optimization* [REF], where optimization problems are solved using quantum computers. It can be used
to design quantum algorithms [REF] and correct errors [REF].

:html:`<br>`

.. figure:: ../_static/whatisml/applications.png
    :align: left
    :width: 50%
    :target: javascript:void(0);

:html:`<br>`

PennyLane is a software framework that is built around the concept of trainable quantum circuits or
*differentiable quantum computation*, to fully exploit the power quantum machine learning and beyond.