r"""Intro to Qubitization
=========================

Qubitization is a Block Encoding technique with very particular properties. This operator opens up a range of
applications such as Qubitization-based QPE that are not possible with other encoding techniques.
In this demo we will introduce this operator, and we will code it in PennyLane via :class:`~.pennylane.Qubitization`.

Qubitization
------------

Encoding a Hamiltonian into a quantum computer is a fundamental task for many applications, but the way to do it is not unique.
One method that is gaining special importance is known as **Qubitization** and the operator is defined as:

.. math::

    Q =  \text{Prep}_{\mathcal{H}}^{\dagger} \text{Sel}_{\mathcal{H}} \text{Prep}_{\mathcal{H}} (2|0\rangle\langle 0| - I).

This operator is a particular Block Encoding technique with a key property: **the eigenvalues of :math:`\mathcal{H}`
are encoded within the eigenvalues of :math:`Q`. **

Going in detail, if :math:`E` is an eigenvalue of :math:`\mathcal{H}`, then :math:`e^{i\arccos(E/\lambda)}` is an
eigenvalue of :math:`Q`, where :math:`\lambda` is a known normalization factor. This is really useful since this allows
to apply Quantum Phase Estimation to obtain the phase :math:`\arccos(E/\lambda)`, which we use to clear the :math:`E` value.

Sounds good, but why is this so? Why is that the decomposition of :math:`Q`? and that arccosine, where does it come from? 🤔
Below we will show you how to reach these conclusions in a simple way.

Block Encoding
----------------

Given a Hamiltonian :math:`\mathcal{H}`, we can define as Block Encoding any operator which embeds :math:`\mathcal{H}`
inside the matrix associated to the circuit:

.. math::
    :math:`\text{BE}_\mathcal{H}` \rightarrow \begin{bmatrix} \mathcal{H} / \lambda & \cdot \\ \cdot & \cdot \end{bmatrix}.

By using that :math:`\mathcal{H}| \phi \rangle = E | \phi \rangle`, we can deduce that this operator applied to an
eigenvector of :math:`\mathcal{H}`, generates the state:

.. math::
    |\Psi\rangle :=\text{BE}_\mathcal{H}|0\rangle |\phi\rangle = \frac{E}{\lambda}|0\rangle |\phi\rangle + \sqrt{1 - \left( \frac{E}{\lambda}\right)^2} |\phi^{\perp}\rangle,

where :math:`|\phi^{\perp}\rangle` is a state orthogonal to :math:`|0\rangle |\phi\rangle`,
and :math:`E` is the eigenvalue.

The advantage of expressing :math:`|\Psi\rangle` as the sum of two orthogonal states is that it can be represented
in a two-dimensional space  — idea that we have exploited in our `Amplitude Amplification <https://pennylane.ai/qml/demos/tutorial_intro_amplitude_amplification/>`_ demo.

.. figure:: ../_static/demonstration_assets/qubitization/qubitization_lcu.jpeg
    :align: center
    :width: 40%
    :target: javascript:void(0)

    Representation of the state which forms an angle of :math:`\theta = \arccos {\frac{E}{\lambda}}` with the x-axis.

The walk operator
----------------------

Any Block Encoding operator, manages to move the state :math:`|0\rangle |\phi\rangle` into the state :math:`|\Psi\rangle`.
The Qubitization operator does this by just applying a :math:`\theta` rotation in the subspace, in the literature
referred as the **walk operator**.

.. figure:: ../_static/demonstration_assets/qubitization/block_encodings.jpeg
    :align: center
    :width: 65%
    :target: javascript:void(0)

In order to build this rotation we will follow the same idea of Amplitude Amplification:
two reflections are equivalent to one rotation.

Let's use :math:`|\Psi\rangle` as an arbitrary initial state to visualize the rotation.
The first reflection we make is with respect to the x-axis:

.. figure:: ../_static/demonstration_assets/qubitization/qubitization_reflection1.jpeg
    :align: center
    :width: 60%
    :target: javascript:void(0)

This could be built by making a reflection on the state :math:`|0\rangle` restricted to the first register:

.. math::

    R_{|0\rangle} = 2|0\rangle\langle 0| - I.

Now, if we want the initial state to rotate a total of :math:`\theta` degrees, we must reflect over the bisector of the initial
state and the x-axis:

.. figure:: ../_static/demonstration_assets/qubitization/qubitization_reflection2.jpeg
    :align: center
    :width: 60%
    :target: javascript:void(0)

But how are we supposed to find this operator?

Fortunately, we don't have to look very far: the block encoding :math:`\text{PSP}_{\mathcal{H}}:=\text{Prep}_{\mathcal{H}}^{\dagger} \text{Sel}_{\mathcal{H}} \text{Prep}_{\mathcal{H}}` is
exactly that reflection 🤯. To learn more about this technique, take a look to `this demo <https://pennylane.ai/qml/demos/tutorial_lcu_blockencoding/>`_.


To prove that this is the reflection we are looking for, firstly we have to check that :math:`\text{PSP}_{\mathcal{H}}^2 = \mathbb{I}`,
definition of a reflection. This property is fulfilled by the construction of the operator.
After that we can know with respect to which axis, by taking the midpoint between an input state and
its output. Using :math:`|0\rangle|\phi\rangle` as input and knowing that :math:`|\Psi\rangle` is the output,
we note that the reflection is indeed over that bisector.

The union of these two reflections defines our rotation.

.. math::

    Q = \text{Prep}^{\dagger} \cdot \text{Sel} \cdot \text{Prep} \cdot (2|0\rangle \langle 0| - \mathbb{I}).

The advantage of the rotations is that they have the angle of rotation encoded in their eigenvalues, i.e., the
eigenvalues of the rotation in the subspace are :math:`\{1, e^{\pm i\theta}\}`. By using that
:math:`\theta = \arccos {\frac{E}{\lambda}}`, we can conclude that the eigenvalues
of :math:`Q` are :math:`\{1, e^{\pm i\arccos {\frac{E}{\lambda}}}\}`. 🎉

Qubitization in PennyLane
--------------------------

Now that we know how to encode the energy in the eigenvalues, let's see how we can use Quantum Phase Estimation (QPE)
to retrieve the encoded information. In `this demo <https://pennylane.ai/qml/demos/tutorial_qpe/>`_  you can learn more about
how QPE works.

In PennyLane, the Qubitization operator can be built by making use of :class:`~.pennylane.Qubitization`. We just have to pass the
Hamiltonian and the control qubits characteristic of the Block Encoding.

.. note::
    The number of control qubits should be :math:`⌈\log_2 k⌉` where :math:`k` is the number of terms in the Hamiltonian.

"""

import pennylane as qml

H = -0.4 * qml.Z(0) + 0.3 * qml.Z(1) + 0.4 * qml.Z(0) @ qml.Z(1)
control_wires = [2, 3]

print(qml.matrix(H))

##############################################################################
# In this example we have chosen a diagonal matrix since it is easy to identify eigenvalues and eigenvectors but it
# would work with any Hamiltonian. We are going to take :math:`|\phi_k\rangle = |11\rangle` and we will try to
# get its eigenvalue :math:`E_k = 0.5` using this technique.

estimation_wires = [4, 5, 6, 7, 8, 9]

dev = qml.device("default.qubit")

@qml.qnode(dev)
def circuit():
    # Initialize the eigenstate |11⟩
    for wire in [0, 1]:
        qml.X(wire)

    # Apply QPE with the walk operator
    for wire in estimation_wires:
        qml.Hadamard(wires=wire)

    qml.ControlledSequence(qml.Qubitization(H, control_wires),
                           control=estimation_wires)

    qml.adjoint(qml.QFT)(wires=estimation_wires)

    return qml.probs(wires=estimation_wires)

##############################################################################
# Let's run the circuit and plot the estimated :math:`\theta` values:

import matplotlib.pyplot as plt
plt.style.use('pennylane.drawer.plot')

results = circuit()

bit_strings = [f"0.{x:0{len(estimation_wires)}b}" for x in range(len(results))]

plt.bar(bit_strings, results)
plt.xlabel("theta")
plt.ylabel("probability")
plt.xticks(range(0, len(results), 3),
           bit_strings[::3],
           rotation="vertical")

plt.subplots_adjust(bottom=0.3)
plt.show()

##############################################################################
# The two peaks obtained refer to values of :math:`|\theta\rangle` and :math:`|-\theta\rangle`.
#
# Finally, by doing some post-processing, we can obtain the value of :math:`E_k`:

import numpy as np

lambda_ = sum([abs(coeff) for coeff in H.terms()[0]])

# Simplification by estimating theta with the peak value
print("E_k = ", lambda_ * np.cos(2 * np.pi * np.argmax(results) / 2** (len(estimation_wires))))

##############################################################################
# Great, we have managed to approximate the real value of :math:`E_k = 0.5`. 🚀
#
# Conclusion
# ----------------
#
# In this demo we have seen the concept of Qubitization and one of its applications.
# For this, we have combined concepts such as Block Encoding, Quantum Phase Estimation, and Amplitude Amplification.
# This algorithm is the precursor of more advanced algorithms such as QSVT so we invite you to continue studying these
# techniques and apply them in your research.
#
# About the author
# ----------------

