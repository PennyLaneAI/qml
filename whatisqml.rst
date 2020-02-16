.. role:: html(raw)
   :format: html

What is Quantum Machine Learning?
=================================

Quantum machine learning is a field at the intersection of artificial intelligence and physics,
which investigates a new way in which machines can learn from data.

Replacing classical hardware by quantum computers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:html:`<br>`

.. figure:: /_static/whatisml/gpu_to_qpu.png
    :align: center
    :width: 60%
    :target: javascript:void(0);

    Replacing GPUs by quantum computers changes the way we do machine learning.

:html:`<br>`


The limits of what machines can learn have always been defined by the computer hardware
we run our algorithms on - for example, the success of modern-day deep learning  with neural networks is
enabled by parallel GPU clusters.

Quantum machine learning extends the pool of hardware for machine learning by an entirely
new type of computing device --- the quantum computer. Information processing with quantum computers
underlies substantially different laws of physics, namely quantum theory.


Near-term quantum devices
~~~~~~~~~~~~~~~~~~~~~~~~~

:html:`<br>`

.. figure:: /_static/whatisml/quantum_devices_ai.png
    :align: center
    :width: 80%
    :target: javascript:void(0);

    Current day quantum computers are special purpose chips.

:html:`<br>`

While a lot of research focuses on ideal, universal quantum processing units (QPUs) whose development
is still a thing of the future, a large share of quantum machine learning
is interested in near term quantum devices :cite:`farhi2018classification`
:cite:`schuld2018circuit` :cite:`grant2018hierarchical` :cite:`liu2018differentiable`.
One can understand these devices
as special purpose hardware in between Application-Specific Integrated Circuits (ASICs) and
Field-Programmable Gate Arrays (FPGAs).

Using quantum devices like neural networks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:html:`<br>`

.. figure:: /_static/whatisml/trainable_circuit.png
    :align: center
    :width: 70%
    :target: javascript:void(0);

    Quantum algorithms can be trained and used like neural networks.

:html:`<br>`

In the near-term branch of quantum machine learning,
**quantum devices are used and trained like neural networks**.
This is done by systematically adapting the physical control parameters,
such as an electromagnetic field strength or a laser pulse frequency, to solve a machine learning problem.
Afterwards, the trained circuit can for example be used to classify the content of images - by encoding
the image into the physical state of the device and taking measurements.

The bigger picture: Making computers differentiable
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:html:`<br>`

.. figure:: /_static/whatisml/applications.png
    :align: center
    :width: 50%
    :target: javascript:void(0);

:html:`<br>`


But the story does not end here. Quantum circuits turn out to be *differentiable*, which means that a quantum computer
can compute the change in control parameters needed to become better at a given task. Differentiable programming
is the very basis of deep learning, implemented in software libraries such as TensorFlow and PyTorch.
Differentiable programming is also more than deep learning: it is a programming paradigm where steps of an
algorithm are not hand-coded but learnt.

Similarly, the idea of trainable quantum computations is larger than quantum machine learning. It includes,
and in fact originates from, a field called *quantum chemistry* :cite:`peruzzo2014variational`
:cite:`mcclean2016theory`, in which adaptable quantum circuits are
used to find ground state energies of atoms and molecules. Trainable circuits also feature
in *quantum optimization* :cite:`farhi2014quantum` and can be used to
design quantum algorithms :cite:`anschuetz2018variational`
or correct errors :cite:`johnson2017qvector`.

PennyLane is a software framework that is built around the concept of
*differentiable quantum computation*, and allows users to fully exploit the power of quantum machine learning and beyond.
