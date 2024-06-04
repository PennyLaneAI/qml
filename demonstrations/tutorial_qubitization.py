r"""Intro to Qubitization
=========================

Qubitization is a Block Encodinig technique with very particular properties. This operator opens up a range of
applications such as Qubitization-based QPE that are not possible with other encoding techniques.
In this demo we will introduce this operator, and we will code it in PennyLane via :class:`~.pennylane.Qubitization`.


Block Encoding
----------------

A standard approach to encode this Hamiltonian is **Block Encoding**,
which embeds the Hamiltonian inside the matrix representing the circuit:

.. math::
    \begin{bmatrix} \frac{\mathcal{H}}{\lambda} & \cdot \\ \cdot & \cdot \end{bmatrix},

where :math:`\lambda` is a known normalization factor.
The most popular technique is :math:`\text{Prep}^{\dagger}\text{Sel}\text{Prep}`, which you can learn more about in `this demo <https://pennylane.ai/qml/demos/tutorial_lcu_blockencoding/>`_.
This encoding creates an operator :math:`\text{BE}_\mathcal{H}` such that, applied to an eigenvector
of :math:`\mathcal{H}`, generates the state:

.. math::
    \text{BE}_\mathcal{H}|0\rangle \otimes |\phi_k\rangle = \frac{E_k}{\lambda}|0\rangle \otimes|\phi_k\rangle + \sqrt{1 - \left( \frac{E_k}{\lambda}\right)^2} |\psi^{\perp}\rangle,

where :math:`|\psi^{\perp}\rangle` is a state orthogonal to :math:`|0\rangle \otimes |\phi_k\rangle`
and :math:`E_k` is the eigenvalue.
This allows us to represent the initial state in a two-dimensional space  — idea that we have exploited in our `Amplitude Amplification <https://pennylane.ai/qml/demos/tutorial_intro_amplitude_amplification/>`_ demo:

.. figure:: ../_static/demonstration_assets/qubitization/qubitization_lcu.jpeg
    :align: center
    :width: 60%
    :target: javascript:void(0)

    Representation of the state which forms an angle of :math:`\theta = \arccos {\frac{E_k}{\lambda}}` with the x-axis.



Qubitization Operator
----------------------

Any Block Encoding technique, manages to transform the state :math:`|0\rangle \otimes |\phi_k\rangle` into state :math:`\text{BE}_\mathcal{H}|0\rangle \otimes |\phi_k\rangle`. But the way to do it is not unique.
Working in the above subspace, there are two ways to do it, through a reflection or through a rotation.

.. figure:: ../_static/demonstration_assets/qubitization/block_encodings.jpeg
    :align: center
    :width: 80%
    :target: javascript:void(0)

The :math:`\text{Prep}^{\dagger}\text{Sel}\text{Prep}` subroutine is the reflection Block Encoding and Qubitization is the rotation Block Encoding.
This simple difference will play a key role in the choice of technique for particular algorithms.
In order to find the construction of this rotation, we will follow the same idea of Amplitude Amplification:
two reflections are equivalent to one rotation.

The first reflection is made with respect to the x-axis, which, for our initial state :math:`|0\rangle \otimes |\phi_k\rangle`, has no effect, and
the second reflection is the :math:`\text{Prep}^{\dagger}\text{Sel}\text{Prep}` reflection:

.. figure:: ../_static/demonstration_assets/qubitization/reflections_qubitization.jpeg
    :align: center
    :width: 100%
    :target: javascript:void(0)

From this we can deduce the expression of the Qubitization operator:

.. math::

    Q = \text{Prep}^{\dagger}\text{Sel}\text{Prep}(2|0\rangle \langle 0| - \mathbb{I}),

Where the reflection on zero is applied only in the first register of the state.










The two eigenstates of this rotation are :math:`\frac{1}{\sqrt{2}}|0\rangle|\phi_k\rangle \pm \frac{i}{\sqrt{2}}|\psi^{\perp}\rangle`
and, in general, they are not trivial to prepare. This is where the second major observation of the algorithm is born:
the :math:`|0\rangle|\phi_k\rangle` state is the uniform superposition of the two eigenstates. Therefore,
applying QPE to that state, we obtain the two eigenvalues superposition,
from which we extract :math:`\theta`.

.. figure:: ../_static/demonstration_assets/qubitization/qubitization_qpe.jpeg
    :align: center
    :width: 60%
    :target: javascript:void(0)


Qubitization in PennyLane
--------------------------

The great advantage of doing a rotation Block Encoding, is that it encodes the rotation angle in its eigenvalues and
this can be used to calculate the eigenvalues of :math:`\mathcal{H}`.
Therefore, we will use Quantum Phase Estimation to obtain these values. In `this demo <https://pennylane.ai/qml/demos/tutorial_qpe/>`_  you can learn more about
how QPE works.

In PennyLane, Qubitization operator can be built by making use of :class:`~.pennylane.Qubitization`. We just have to pass the
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
# Great, we have managed to approximate the real value of :math:`E_k = 0.5`. 🎉
#
# Conclusion
# ----------------
#
# In this demo we have seen one of the most advanced techniques in quantum computing for energy calculation.
# For this, we have combined concepts such as Block Encoding, Quantum Phase Estimation, and Amplitude Amplification.
# This algorithm is the precursor of more advanced algorithms such as QSVT so we invite you to continue studying these
# techniques and apply them in your research.
#
# About the author
# ----------------

