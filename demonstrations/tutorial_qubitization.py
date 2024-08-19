r"""Intro to Qubitization
=========================

Encoding a Hamiltonian into a quantum computer is a fundamental task for many applications, but the way to do it is not unique.
One method that has gained special status is known as **Qubitization**. In this demo, we will introduce the qubitization operator and explain how to view it as a rotation operator. We then explain how to combine it with quantum phase estimation in an example application, illustrated with example code in PennyLane using the :class:`~.pennylane.Qubitization` template.

.. figure:: ../_static/demonstration_assets/qubitization/OGthumbnail_large_Qubitization.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

Qubitization operator
----------------------

For a Hamiltonian 'math:`\mathcal{H}`, the Qubitization operator is defined as:

.. math::

    Q =  \text{Prep}_{\mathcal{H}}^{\dagger} \text{Sel}_{\mathcal{H}} \text{Prep}_{\mathcal{H}}\cdot (2|0\rangle\langle 0| - I),

where :math:`\text{Prep}_{\mathcal{H}}` and :math:`\text{Sel}_{\mathcal{H}}` are the preparation and selection operators. They are explained `this demo <https://pennylane.ai/qml/demos/tutorial_lcu_blockencoding/>`_, which you can check out for more details.
The operator :math:`Q` is a block encoding operator with a key property: its eigenvalues encode the eigenvalues of the Hamiltonian.
As we will soon explain in detail, if :math:`E` is an eigenvalue of :math:`\mathcal{H}`, then :math:`e^{i\arccos(E/\lambda)}` is an
eigenvalue of :math:`Q`, where :math:`\lambda` is a known normalization factor. This property is very useful because it means that we can use the Quantum Phase Estimation  algorithm to estimate the eigenvalues of  :math:`Q`, and then use them to retrieve the eigenvalues of the encoded Hamiltonian.

This is the essence of why qubitization is attractive for applications: it provides a method to exactly encode eigenvalues of a Hamiltonian into a unitary operator that can be used inside the quantum phase estimation algorithm to sample Hamiltonian eigenvalues. But where this decomposition come from? Why are the eigenvalues are encoded in this way? 🤔 We explain these concepts below.

Block Encoding
----------------

First, we introduce some useful concepts about block encodings.
Given a Hamiltonian :math:`\mathcal{H}`, we define as a block encoding any operator that embeds :math:`\mathcal{H}`
inside the matrix associated to the circuit (up to a normalization factor):

.. math::
    \text{Block Encoding}_\mathcal{H} \rightarrow \begin{bmatrix} \mathcal{H} / \lambda & \cdot \\ \cdot & \cdot \end{bmatrix}.

Given an eigenvector of :math:`\mathcal{H}` such that :math:`\mathcal{H}| \phi \rangle = E | \phi \rangle`, we deduce that any :math:`\text{Block Encoding}_\mathcal{H}`
generates a state:

.. math::
    |\Psi\rangle = \frac{E}{\lambda}|0\rangle |\phi\rangle + \sqrt{1 - \left( \frac{E}{\lambda}\right)^2} |\phi^{\perp}\rangle,

where :math:`|\phi^{\perp}\rangle` is a state orthogonal to :math:`|0\rangle |\phi\rangle`,
and :math:`E` is the eigenvalue. The advantage of expressing :math:`|\Psi\rangle` as the sum of two orthogonal states is that it can be represented
in a two-dimensional space  — an idea that we have explored in our `Amplitude Amplification <https://pennylane.ai/qml/demos/tutorial_intro_amplitude_amplification/>`_ demo. The state :math:`|\Psi\rangle` forms an angle :math:`\theta =\arccos {\frac{E}{\lambda}}` with respect to the axis defined by the initial state :math:`|0\rangle |\phi\rangle`, as shown in the image below.

.. figure:: ../_static/demonstration_assets/qubitization/qubitization0.jpeg
    :align: center
    :width: 50%
    :target: javascript:void(0)

    Representation of the state which forms an angle of :math:`\theta = \arccos {\frac{E}{\lambda}}` with the x-axis.

Qubitization as a rotation
---------------------------

Any Block Encoding operator manages to transform the state :math:`|0\rangle |\phi\rangle` into the state :math:`|\Psi\rangle`.
The Qubitization operator does this by applying a rotation in that two-dimensional subspace by an angle of :math:`\theta=\arccos {\frac{E}{\lambda}}`.
The advantage of a rotation operator is that the angle :math:`\theta` appears directly in its eigenvalues as the phases  :math:`e^{\pm i\theta}`. Therefore, if the rotation angle encodes useful information, it can be retrieved by estimating the phase of the rotation operator, for example using QPE.


We now show that the qubitization operator is a rotation by using the same idea of Amplitude Amplification:
two reflections are equivalent to one rotation.

Let's use for instance :math:`|\Psi\rangle` as an initial state to visualize the rotation.
The first reflection we make is with respect to the x-axis:

.. figure:: ../_static/demonstration_assets/qubitization/qubitization2.jpeg
    :align: center
    :width: 50%
    :target: javascript:void(0)

This is built by making a reflection on the state :math:`|0\rangle` restricted to the first register:

.. math::

    R_{|0\rangle} = 2|0\rangle\langle 0| - I.

The second reflection of interest will be over the bisector of :math:`|\Psi\rangle`
and the x-axis:

.. figure:: ../_static/demonstration_assets/qubitization/qubitization3.jpeg
    :align: center
    :width: 60%
    :target: javascript:void(0)

As we can see, this reflection achieves a rotation of :math:`\theta` degrees with respect to the initial state.
But how are we supposed to find this particular reflection? Fortunately, we don't have to look very far: the block encoding :math:`\text{Prep}_{\mathcal{H}}^{\dagger} \text{Sel}_{\mathcal{H}} \text{Prep}_{\mathcal{H}}` ,or just :math:`\text{PSP}_{\mathcal{H}}`, is
exactly that reflection 🤯. To learn more about this technique, take a look to `this demo <https://pennylane.ai/qml/demos/tutorial_lcu_blockencoding/>`_.


To prove that this is the reflection we are looking for, firstly we have to check that :math:`\text{PSP}_{\mathcal{H}}^2 = \mathbb{I}`,
definition of a reflection. This property is fulfilled by the construction of the operator.
After that, we can know the reflection axis, by taking the midpoint between an input state and
its output. Using :math:`|0\rangle|\phi\rangle` as input and knowing that :math:`|\Psi\rangle` is the output,
we note that the reflection is indeed over that bisector.

The union of these two reflections defines our :math:`\theta` rotation in the subspace.


Qubitization in PennyLane
--------------------------

Now that we know how to encode the Hamiltonian in this useful way, let's see how we can use Quantum Phase Estimation (QPE)
to retrieve the encoded eigenvalue. In `this demo <https://pennylane.ai/qml/demos/tutorial_qpe/>`_  you can learn more about
how QPE works. The algorithm is expressed in the following way:

.. figure:: ../_static/demonstration_assets/qubitization/QPE_qubitization.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

Applying QPE with the Qubitization operator (i.e., the :math:`\theta` rotation in subspace), we will get the :math:`\theta` and :math:`-\theta` values.

Let's define the Hamiltonian to see an example:

"""

import pennylane as qml

H = -0.4 * qml.Z(0) + 0.3 * qml.Z(1) + 0.4 * qml.Z(0) @ qml.Z(1)

print(qml.matrix(H))

##############################################################################
# We have chosen a diagonal matrix since it is easy to identify eigenvalues and eigenvectors but it
# would work with any Hamiltonian. We are going to take :math:`|\phi\rangle = |11\rangle` and we will try to
# estimate its eigenvalue :math:`E = 0.5` using this technique.
#
# In PennyLane, the Qubitization operator can be built by making use of :class:`~.pennylane.Qubitization`. We just have to pass the
# Hamiltonian and the control qubits characteristic of the Block Encoding, where the number of control wires
# is :math:`⌈\log_2 k⌉` where :math:`k` is the number of terms in the Hamiltonian LCU representation.
#
# .. note::
#
#    Qubitization rotation in pennylane is clockwise defined as the adjunct of the operator described above.
#
#

control_wires = [2, 3]
estimation_wires = [4, 5, 6, 7, 8, 9]

dev = qml.device("default.qubit")

@qml.qnode(dev)
def circuit():
    # Initialize the eigenstate |11⟩
    for wire in [0, 1]:
        qml.X(wire)

    # Apply QPE with the Qubitization operator
    qml.QuantumPhaseEstimation(qml.Qubitization(H, control_wires), estimation_wires = estimation_wires)

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
# Finally, by doing some post-processing, we can obtain the value of :math:`E`:

import numpy as np

lambda_ = sum([abs(coeff) for coeff in H.terms()[0]])

# Simplification by estimating theta with the peak value
print("E = ", lambda_ * np.cos(2 * np.pi * np.argmax(results) / 2** (len(estimation_wires))))

##############################################################################
# Great, we have managed to approximate the real value of :math:`E = 0.5`. 🚀
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

