r"""Intro to Qubitization
=========================

Encoding a Hamiltonian into a quantum computer is a fundamental task for many applications, but the way to do it is not unique.
One method that has gained special status is known as **Qubitization**. In this demo, we will introduce the Qubitization operator and explain how to view it as a rotation operator. We then explain how to combine it with Quantum Phase Estimation in an example application, illustrated with example code in PennyLane using the :class:`~.pennylane.Qubitization` template.

.. figure:: ../_static/demonstration_assets/qubitization/OGthumbnail_large_Qubitization.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

Qubitization operator
----------------------

For a Hamiltonian :math:`\mathcal{H}` given in `LCU representation <https://pennylane.ai/qml/demos/tutorial_lcu_blockencoding/>`_, the Qubitization operator is defined as:

.. math::

    Q =  \text{PSP}_{\mathcal{H}}\cdot (2|0\rangle\langle 0| - I),

where :math:`\text{PSP}_{\mathcal{H}}` refers to the block encoding :math:`\text{Prep}_{\mathcal{H}}^{\dagger} \text{Sel}_{\mathcal{H}} \text{Prep}_{\mathcal{H}}`, as explained in `this demo <https://pennylane.ai/qml/demos/tutorial_lcu_blockencoding/>`_ which you can check it out for more details.
The operator :math:`Q` is also a block encoding operator but with a key property: its eigenvalues encode the eigenvalues of the Hamiltonian.
As we will soon explain in detail, if :math:`E` is an eigenvalue of :math:`\mathcal{H}`, then :math:`e^{i\arccos(E/\lambda)}` is an
eigenvalue of :math:`Q`, where :math:`\lambda` is a known normalization factor. This property is very useful because it means that we can use the `Quantum Phase Estimation algorithm <https://pennylane.ai/qml/demos/tutorial_qpe/>`_  to estimate the eigenvalues of  :math:`Q`, and then use them to retrieve the eigenvalues of the encoded Hamiltonian.

This is the essence of why qubitization is attractive for applications: it provides a method to exactly encode eigenvalues of a Hamiltonian into a unitary operator that can be used inside the quantum phase estimation algorithm to sample Hamiltonian eigenvalues. But where does this decomposition come from? Why are the eigenvalues encoded in this way? 🤔 We explain these concepts below.

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

Any block encoding operator manages to transform the state :math:`|0\rangle |\phi\rangle` into the state :math:`|\Psi\rangle`.
The Qubitization operator does this by applying a rotation in that two-dimensional subspace by an angle of :math:`\theta=\arccos {\frac{E}{\lambda}}`.
The advantage of a rotation operator is that the angle :math:`\theta` appears directly in its eigenvalues as the phases  :math:`e^{\pm i\theta}`. Therefore, if the rotation angle encodes useful information, it can be retrieved by estimating the phase of the rotation operator, for example using QPE.

We now show that the Qubitization operator is a rotation by using the same idea of Amplitude Amplification:
two reflections are equivalent to one rotation. These reflections correspond to :math:`2|0\rangle\langle 0| - I` and :math:`\text{PSP}_{\mathcal{H}}`, which together define our Qubitization operator.

.. note::

    To verify that an operator is indeed a reflection, one must square the operator and check if the result is the identity operator.

As an example, let’s take :math:`|\Psi\rangle` as the initial state to visualize the rotation. The first reflection, given by :math:`2|0\rangle\langle 0| - I`, mirrors the state around the x-axis. This occurs because the operator flips the sign of any component not aligned with :math:`|0\rangle` in the first register.

.. figure:: ../_static/demonstration_assets/qubitization/qubitization2.jpeg
    :align: center
    :width: 50%
    :target: javascript:void(0)

Given the effect that :math:`\text{PSP}_{\mathcal{H}}` has on the state :math:`|0\rangle|\phi\rangle`, it can be verified that the second reflection occurs along the line that bisects the angle between the state :math:`|\Psi\rangle` and the x-axis. Let's now examine the effect this reflection has on the previous state:

.. figure:: ../_static/demonstration_assets/qubitization/qubitization3.jpeg
    :align: center
    :width: 60%
    :target: javascript:void(0)

After applying the reflection, the new state is rotated by :math:`\theta` degrees relative to the initial state. This shows that the Qubitization operator successfully creates a rotation of :math:`\theta` degrees within the subspace.

The term "qubitization" comes from the fact that this process occurs within a two-dimensional subspace, which can be viewed as a qubit. For each eigenstate of the Hamiltonian, the Qubitization operator acts within this two-dimensional space, effectively treating it as a qubit. This is why we say the system has been "qubitized".

Qubitization in PennyLane
--------------------------

We now describe how to implement qubitization-based `Quantum Phase Estimation (QPE) <https://pennylane.ai/qml/demos/tutorial_qpe/>`_  to sample eigenvalues of a Hamiltonian.

First, let's define a Hamiltonian to use as an example:
"""

import pennylane as qml

H = -0.4 * qml.Z(0) + 0.3 * qml.Z(1) + 0.4 * qml.Z(0) @ qml.Z(1)

print(qml.matrix(H))

##############################################################################
# We have chosen an operator that is diagonal in the computational basis because it is easy to identify
# eigenvalues and eigenvectors, but this would work with any Hamiltonian. We are going to take
# the eigenstate  :math:`|\phi\rangle = |11\rangle` as input and try to estimate its eigenvalue :math:`E = 0.5` using
# this technique.
#
# In PennyLane, the Qubitization operator can be easily constructed using the built-in :class:`~.pennylane.Qubitization` operation. You simply need to provide the Hamiltonian and the control qubits that defines the block encoding. The number of control wires is :math:`⌈\log_2 k⌉`, where :math:`k` is the number of terms in the Hamiltonian's `LCU representation <https://pennylane.ai/qml/demos/tutorial_lcu_blockencoding/>`_.
#
# .. note::
#
#    The Qubitization rotation in PennyLane is defined clockwise as the adjoint of the operator described above.
#
# We will make use of the built-in :class:`~.pennylane.QuantumPhaseEstimation` operator to easily apply the algorithm:

control_wires = [2, 3]
estimation_wires = [4, 5, 6, 7, 8, 9]

dev = qml.device("default.qubit")


@qml.qnode(dev)
def circuit():
    # Initialize the eigenstate |11⟩
    for wire in [0, 1]:
        qml.X(wire)

    # Apply QPE with the Qubitization operator
    qml.QuantumPhaseEstimation(
        qml.Qubitization(H, control_wires), estimation_wires=estimation_wires
    )

    return qml.probs(wires=estimation_wires)


##############################################################################
# Let's run the circuit and plot the estimated eigenvalue:

import matplotlib.pyplot as plt

plt.style.use("pennylane.drawer.plot")

results = circuit()

bit_strings = [f"0.{x:0{len(estimation_wires)}b}" for x in range(len(results))]

plt.bar(bit_strings, results)
plt.xlabel("theta")
plt.ylabel("probability")
plt.xticks(range(0, len(results), 3), bit_strings[::3], rotation="vertical")

plt.subplots_adjust(bottom=0.3)
plt.show()

##############################################################################
# The two peaks obtained correspond to the values :math:`|\theta\rangle` and :math:`|-\theta\rangle`.
# Finally, with some post-processing, we can determine the value of :math:`E`:

import numpy as np

lambda_ = sum([abs(coeff) for coeff in H.terms()[0]])

# Simplification by estimating theta with the peak value
print("E = ", lambda_ * np.cos(2 * np.pi * np.argmax(results) / 2 ** (len(estimation_wires))))

##############################################################################
# Great! We successfully approximated the real value of :math:`E = 0.5`. 🚀
#
# Conclusion
# ----------------
#
# In this demo, we explored the concept of Qubitization and one of its applications. To achieve this, we combined several key concepts, including `Block Encoding <https://pennylane.ai/qml/demos/tutorial_lcu_blockencoding/>`_, `Quantum Phase Estimation <https://pennylane.ai/qml/demos/tutorial_qpe/>`_, and `Amplitude Amplification <https://pennylane.ai/qml/demos/tutorial_intro_amplitude_amplification/>`_. This algorithm serves as a foundation for more advanced techniques like `Quantum Singular Value Transformation <https://pennylane.ai/qml/demos/tutorial_intro_qsvt/>`_. We encourage you to continue studying these methods and apply them in your research.
#
# About the authors
# ------------------
