r"""Intro to Qubitization
=========================

Qubitization is a technique that together with Quantum Phase Estimation (QPE) solves a simple task: given an
eigenstate of a Hamiltonian, find its eigenvalue. This demo explains the basics behind this idea and how to implement
it in PennyLane through :class:`~.pennylane.Qubitization`.


Qubitization-based QPE
------------------------

Let's look at the problem in detail. We are given a Hamiltonian :math:`\mathcal{H}` and an eigenvector :math:`|\phi_k\rangle`.
We look for the value :math:`E_k` such that:

.. math::
    \mathcal{H}|\phi_k\rangle = E_k|\phi_k\rangle.

The first step to solve this task on a quantum computer is to encode the Hamiltonian in it, which we cannot do directly
since :math:`\mathcal{H}` may not be a unitary operator. One of the most popular techniques to encode this Hamiltonian
is **Block Encoding**, which you can learn more about in `this demo <https://pennylane.ai/qml/demos/tutorial_lcu_blockencoding/>`_.
This encoding creates an operator :math:`\text{BE}_\mathcal{H}` such that:

.. math::
    \text{BE}_\mathcal{H}|0\rangle \otimes \mathbb{I} = |0\rangle \otimes \frac{\mathcal{H}}{\lambda} + \sum_{i>0} |i\rangle \otimes U_i,

where :math:`\lambda` is a known normalization factor and :math:`U_i` are operators that will not be of interest. Our
challenge is to design a quantum algorithm that calculates :math:`E_k` using that useful codification. How could we do this? 🤔

Part 1. Problem reduction
~~~~~~~~~~~~~~~~~~~~~~~~~~

Firstly, let's see what happens if we apply the operator :math:`\text{BE}_\mathcal{H}` to the eigenstate:

.. math::
    \text{BE}_\mathcal{H}|0\rangle \otimes |\phi_k\rangle = |0\rangle \otimes \frac{\mathcal{H}}{\lambda}|\phi_k\rangle + \sum_{i>0} |i\rangle \otimes U_i|\phi_k\rangle.

Since the terms on the right are not relevant in this problem, we just denote the resulting state as :math:`|\psi^{\perp}\rangle`.
This leaves the expression as:

.. math::
    \text{BE}_\mathcal{H}|0\rangle \otimes |\phi_k\rangle = \frac{E_k}{\lambda}|0\rangle \otimes|\phi_k\rangle + \sqrt{1 - \left( \frac{E_k}{\lambda}\right)^2} |\psi^{\perp}\rangle,

where we are using that :math:`\mathcal{H}|\phi_k\rangle = E_k|\phi_k\rangle`.
This allows us to represent the initial state in a two-dimensional space  — idea that we have exploited in our `Amplitude Amplification <https://pennylane.ai/qml/demos/tutorial_intro_amplitude_amplification/>`_ demo:

.. figure:: ../_static/demonstration_assets/qubitization/qubitization_lcu.jpeg
    :align: center
    :width: 60%
    :target: javascript:void(0)

The state forms an angle of :math:`\theta = \arccos {\frac{E_k}{\lambda}}`
with the x-axis. Therefore, if we obtain that angle, we could calculate :math:`E_k`.
This technique is commonly seen as a reduction of a large system to a single qubit (two orthogonal states), hence the name **Qubitization**.


Part 2. Quantum Phase Estimation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The big idea behind this algorithm lies in the realization that it would be enough to find :math:`R(\theta)` —
the :math:`\theta`-rotation in this two-dimensional space. The reason for this is that the two eigenvalues of this
operator are :math:`e^{\pm 2 \pi i\theta}`. Thus, if we apply Quantum Phase Estimation (QPE) on an eigenvector,
we get :math:`\theta`. In `this demo <https://pennylane.ai/qml/demos/tutorial_qpe/>`_  you can learn more about
how QPE works.

The two eigenstates of this rotation are :math:`\frac{1}{\sqrt{2}}|0\rangle|\phi_k\rangle \pm \frac{i}{\sqrt{2}}|\psi^{\perp}\rangle`
and, in general, they are not trivial to prepare. This is where the second major observation of the algorithm is born:
the :math:`|0\rangle|\phi_k\rangle` state is the uniform superposition of the two eigenstates. Therefore,
applying QPE to that state, we obtain the two eigenvalues superposition,
from which we extract :math:`\theta`.

.. figure:: ../_static/demonstration_assets/qubitization/qubitization_qpe.jpeg
    :align: center
    :width: 60%
    :target: javascript:void(0)

Part 3: The quantum walk operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


There is only one thing left to do: build the :math:`R(\theta)` rotation, often referred to as the **walk operator**.
For this, we will follow the same idea of Amplitude Amplification: two reflections are equivalent to one rotation.

The first reflection we make is with respect to the x-axis:

.. figure:: ../_static/demonstration_assets/qubitization/qubitization_reflection1.jpeg
    :align: center
    :width: 60%
    :target: javascript:void(0)

This could be built by making a reflection on the state :math:`|0\rangle` restricted to the first register.
Now, if we want the initial state to rotate a total of :math:`\theta` degrees, we must reflect over the bisector of the initial
state and the x-axis:

.. figure:: ../_static/demonstration_assets/qubitization/qubitization_reflection2.jpeg
    :align: center
    :width: 60%
    :target: javascript:void(0)

But how are we supposed to find this operator?

Fortunately, we don't have to look very far: :math:`\text{BE}_\mathcal{H}` is
exactly that reflection 🤯. To prove this, firstly we have to check that :math:`\text{BE}_\mathcal{H}^2 = \mathbb{I}`,
definition of a reflection. This property is fulfilled by the `construction <https://pennylane.ai/qml/demos/tutorial_lcu_blockencoding/>`_ of the operator.
Once we know that it is a reflection, we can know with respect to which axis, taking the midpoint between a vector and
its output. Taking :math:`|0\rangle|\phi\rangle` and :math:`\text{BE}_\mathcal{H}|0\rangle|\phi\rangle`, we note that the reflection is indeed over the bisector.
The union of these two reflections defines our walk operator.

Qubitization in PennyLane
--------------------------

In PennyLane, the walk operator can be built by making use of :class:`~.pennylane.Qubitization`. We just have to pass the
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

