r"""

Givens rotations for quantum chemistry
======================================

.. meta::
    :property="og:description": Discover the building blocks of quantum circuits for
        quantum chemistry

    :property="og:image": https://pennylane.ai/qml/_images/Givens_rotations.png

.. related::
    tutorial_quantum_chemistry Quantum chemistry with PennyLane
    tutorial_vqe A brief ofverview of VQE


*Author: PennyLane dev team. Posted:  2021. Last updated: *

In the book `“Sophie’s world” <https://en.wikipedia.org/wiki/Sophie%27s_World>`_, the young
protagonist receives a white envelope containing a letter
with an intriguing question: “Why is Lego the most ingenious toy in the world?” At first baffled by
this curious message, she decides to reflect on the question. As told by the book's narrator,
she arrives at a conclusion:

“*The best thing about them was that with Lego she could construct any kind of object. And then
she could separate the blocks and construct something new. What more could one ask of a toy?
Sophie decided that Lego really could be called the most ingenious toy in the world.*”

In this tutorial, you will learn about the building blocks of quantum circuits for quantum
chemistry: Givens rotations. These are operations that can be used to construct any kind of
particle-conserving circuit. We discuss single and double excitation gates, which are particular
types of Givens rotations that play an important role in quantum chemistry. Notably,
controlled single excitation gates are universal for particle-conserving unitaries. You will also
learn how to use these gates to build arbitrary states of a fixed number of particles.

Particle-conserving unitaries
-----------------------------

Arguably the central problem in quantum chemistry is understanding the electronic structure of
molecules. Quantum computers tackle this problem by using systems of qubits to
represent the quantum states of the electrons. One method is to consider a
collection of `molecular orbitals <https://en.wikipedia.org/wiki/Molecular_orbital>`_, which
capture the three-dimensional region of space occupied by the electrons. Each orbital can be
occupied by at most two electrons, each with a different spin orientation. In this case we refer to
*spin orbitals* that can be occupied by a single electron.

The state of electrons in a molecule can then be described by specifying how the
orbitals are occupied. The `Jordan-Wigner representation
<https://en.wikipedia.org/wiki/Jordan%E2%80%93Wigner_transformation>`_  provides a
convenient way to do this: we associate a qubit with each spin orbital and
use its states to represent occupied (:math:`|1\rangle`) or unoccupied (
:math:`|0\rangle`) spin orbitals. An :math:`n`-qubit state with Hamming weight :math:`k`, i.e.,
with :math:`k` qubits in state :math:`|1\rangle`, represents a state of :math:`k` electrons in
:math:`n` spin orbitals. For example :math:`|1010\rangle` is a state of two electrons in two spin
spin orbitals. More generally, superpositions over all basis states with a fixed
number of particles are valid states of the electrons in a molecule. These are states such as

.. math::

    |\psi\rangle = c_1|1100\rangle + c_2|1010\rangle + c_3|1001\rangle + c_4|0110\rangle +
    c_5|0101\rangle + c_6|0011\rangle,

for some coefficients :math:`c_1,\ldots, c_6`. Because the number of electrons is
fixed, any transformation must conserve the number particles. We refer to these as
particle-conserving unitaries. When designing quantum circuits and algorithms for quantum
chemistry, it is desirable to employ only particle-conserving gates that guarantee that the
states of the system remain valid. This raises the questions: what are the simplest
particle-conserving unitaries? Like Legos, can they be used to construct any quantum circuit for
quantum chemistry?

.. figure:: ../demonstrations/givens_rotations/orbitals+states.png
    :align: center
    :width: 70%

    A system with six spin orbitals and three electrons. Qubit states can be used to represent
    different electronic configurations.

Givens rotations
----------------

Consider single-qubit gates. In their most general form, they perform the transformation

.. :math:

    U|0\rangle &= a |0\rangle + b |1\rangle,\\
    U|1\rangle &= c |1\rangle + d |0\rangle,

where :math:`|a|^2+|b|^2=|c|^2+|d|^2=1` and :math:`ab^* + cd^*=0`. This gate is
particle-conserving only if :math:`b=d=0`, which means that the only single-qubit gates that
preserve particle number are diagonal gates of the form

.. :math:

    \begin{pmatrix}
    e^{i\theta} & 0\\
    0 & e^{i\phi}
    \end{pmatrix}.

On their own. these gates are not very interesting. They can only be used to change the
relative phases of states in a superposition; they cannot be used to create and control such
superpositions. So let's take a look at two-qubit gates.

We can divide basis states depending on
their number of particles. We have :math:`|00\rangle` with zero particles, :math:`|01\rangle,
|10\rangle` with one particle, and :math:`|11\rangle` with two particles. Particle-conserving
unitaries can only apply a phase to the states
:math:`|00\rangle, |11\rangle`, but we can now consider transformations that couple the remaining
states :math:`|01\rangle,|10\rangle`. These are gates of the form

.. :math:

    U|01\rangle &= a |01\rangle + b |10\rangle\\
    U|10\rangle &= c |10\rangle + d |01\rangle.

This should be familiar: it has the same form as a single-qubit gate, where the states
:math:`|01\rangle, |10\rangle` respectively take the place of :math:`|0\rangle, |1\rangle`. This
correspondence has a name: the `dual-rail qubit
<https://en.wikipedia.org/wiki/Optical_cluster_state>`, where a two-level system is constructed
by specifying in which of two possible orbitals the single particle is located. The
difference compared to single-qubit gates is that any values of the unitary's parameters :math:`a,
b,c,d` give rise to a valid particle-conserving unitary. Take for instance the two-qubit gate

.. :math:

    G(\theta)=\begin{pmatrix}
    1 & 0 & 0 & 0\\
    0 & \cos (\theta/2) & -\sin (\theta/2) & 0\\
    0 & \sin(\theta/2) & \cos(\theta/2) & 0\\
    0 & 0 & 0 & 1
    \end{pmatrix}.


.. figure:: ../demonstrations/givens_rotations/Givens_rotaions1.png
    :align: center
    :width: 70%

    A Givens rotation can be used to couple states that differ by a single excitation.

This is an example of a `Givens rotation <https://en.wikipedia.org/wiki/Givens_rotation>`_: a
rotation in a two-dimensional subspace of a larger Hilbert space. In this case, we are performing a
Givens rotation in the four-dimensional space of two-qubit states. This gate is more
interesting: it allows us to create superpositions by exchanging the particle between
the two qubits. Such transformations can be interpreted as an *single excitation*,
where we view the exchange from :math:`|10\rangle` to :math:`|10\rangle` as exciting the electron
from the first to the second qubit. This gate is implemented in PennyLane as the
:func:`~.pennylane.ops.SingleExcitation` operation. We can use it to prepare an equal
superposition of three-qubit states with a single particle:
"""

import pennylane as qml
import numpy as np
dev = qml.device('default.qubit', wires=3)


@qml.qnode(dev)
def circuit(x, y):
    qml.BasisState(np.array([1, 0, 0]), wires=[0, 1, 2])
    qml.SingleExcitation(x, wires=[0, 1])
    qml.SingleExcitation(y, wires=[0, 2])

    return qml.state()


x = -2 * np.arccos(np.sqrt(2/3))
y = -np.pi/2
print(circuit(x, y))

##############################################################################
# The components of the output state are ordered according to their binary
# representation, so entry 1 is :math:`|001\rangle`, entry 2 is :math:`|010\rangle`, and entry 4 is
# :math:`|100\rangle`, meaning we indeed prepare the desired state.
#
# We can also study *double excitations* involving the transfer of two particles. These
# are four-qubit transformations where particles are excited from two initial qubits to another
# pair of qubits. For example, consider a Givens rotation in the subspace spanned by the states
# :math:`|1100\rangle` and :math:`|0011\rangle`. These
# states differ by a double excitation since we can map :math:`|1100\rangle` to
# :math:`|0011\rangle` by exciting the particles form the first two qubits to the last two.
# Mathematically, this gate can be represented by a unitary :math:`G^{(2)}(\theta)` that performs
# the mapping
#
# .. :math:
#
#   G^{(2)}|0011\rangle &= \cos (\theta/2)|0011\rangle + \sin (\theta/2)|1100\rangle\\
#   G^{(2)}|1100\rangle &= \cos (\theta/2)|1100\rangle - \sin (\theta/2)|0011\rangle,
#
# while leaving all other basis states unchanged. This gate is implemented in PennyLane as the
# :func:`~.pennylane.ops.DoubleExcitation` operation. We can also consider analogous Givens
# rotations in the other two-dimensional subspaces of states that differ by a double excitation,
# for example :math:`|1010\rangle` and :math:`|0101\rangle`.
#
#
# .. figure:: ../demonstrations/givens_rotations/Givens_rotaions2.png
#     :align: center
#     :width: 70%
#
#     A Givens rotation can also be used to couple states that differ by a double excitation.
#
#
# In the context of quantum chemistry, it is common to consider excitations on a fixed reference
# state, typically the `Hartree-Fock state
# <https://en.wikipedia.org/wiki/Hartree%E2%80%93Fock_method>`_, and include only the excitations
# that preserve the spin orientation of the electron. PennyLane allows you to obtain all such
# excitations using the function :func:`~.pennylane_qchem.qchem.excitations`. Let's use it to
# build a circuit that includes all single and double excitations acting on a reference state of
# three particles in six qubits. We apply a random rotation for each gate:

nr_particles = 3
nr_qubits = 6

singles, doubles = qml.qchem.excitations(3, 6)
print(f"Single excitations = {singles}")
print(f"Double excitations = {doubles}")

dev2 = qml.device('default.qubit', wires=6)


@qml.qnode(dev2)
def circuit2(x, y):
    qml.BasisState(np.array([1, 1, 1, 0, 0, 0]), wires=[0, 1, 2, 3, 4, 5])
    for i, s in enumerate(singles):
        qml.SingleExcitation(x[i], wires=s)
    for j, d in enumerate(doubles):
        qml.DoubleExcitation(y[j], wires=d)

    return qml.state()


x = np.random.normal(0, 0.1, len(singles))
y = np.random.normal(0, 0.1, len(doubles))

output = circuit2(x, y)

##############################################################################
# We can check which basis states appear in the resulting superposition to confirm that they
# involve only states with three particles.
#

states = [np.binary_repr(i, width=6) for i in range(len(output)) if output[i] != 0]
print(states)

##############################################################################
# Besides these Givens rotations, there are other versions that have been
# reported in the literature and used to construct circuits for quantum chemistry. For instance,
# Ref. [#anselmetti]_ considers a different sign convention for single-excitation gates,
#
# .. :math:
#     G(\theta)=\begin{pmatrix}
#     1 & 0 & 0 & 0\\
#     0 & \cos (\theta/2) & \sin (\theta/2) & 0\\
#     0 & -\sin(\theta/2) & \cos(\theta/2) & 0\\
#     0 & 0 & 0 & 1
#     \end{pmatrix},
#
# and Ref. [#barkoutsos]_ introduces the particle-conserving gates listed below, which are all
# Givens rotations
#
# .. :math:
#      U_1(\theta, phi) &= \begin{pmatrix}
#      1 & 0 & 0 & 0\\
#      0 & \cos (\theta) & e^{i\phi}\sin (\theta) & 0\\
#      0 & e^{-1\phi}\sin(\theta) & -\cos(\theta) & 0\\
#      0 & 0 & 0 & 1
#      \end{pmatrix},\\
#
#      U_2(\theta) &= \begin{pmatrix}
#      1 & 0 & 0 & 0\\
#      0 & \cos (2\theta) & -i\sin (2\theta) & 0\\
#      0 & -i\sin(2\theta) & \cos(2\theta) & 0\\
#      0 & 0 & 0 & 1
#      \end{pmatrix}.
#
# Givens rotations are a powerful abstraction for understanding
# quantum circuits for quantum chemistry. Instead of thinking of single-qubit gates and CNOTs as the
# building-blocks of quantum circuits, we can select two-dimensional subspaces spanned by states
# with an equal number of particles, and use Givens rotations in that subspace to construct the
# circuits.
#
# Controlled excitation gates
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# If Givens rotations as analogous to single-qubit gates, then controlled Givens rotations are
# analogous to two-qubit gates. Single-qubit gates and CNOT gates are universal for quantum
# computing: they can be used to implement any conceivable quantum computation. In these
# constructions, the ability to control operations based on the states of other qubits is
# essential, so it's natural to study controlled Givens rotations. The simplest of these are
# controlled single-excitation gates, which are three-qubit gates that perform the mapping
#
# .. :math:
#   CG(\theta) |101\rangle = \cos (\theta/2)|101\rangle + \sin (\theta/2)|110\rangle
#   CG(\theta) |110\rangle = \cos (\theta/2)|110\rangle - \sin (\theta/2)|101\rangle,
#
# while leaving all other basis states unchanged. This gate only excites a particle
# from the second to third qubit if the first (control) qubit is in state :math:`\ket{1}`. This
# is a useful property: as the name suggests, this provides us with better control over the
# transformations we want to apply. Suppose we aim to prepare the state
#
# .. :math:
#   |psi\rangle = \frac{1}{2}(\ket{110000} + \ket{001100} + \ket{000011} + \ket{100100}).
#
# Some inspection is sufficient to see that the states :math:`\ket{001100}` and :math`\ket{000011}`
# differ by a double excitation from the reference state :math:`\ket{110000}`, while the state
# :math:`\ket{100100}` differs by a single excitation. It is tempting to think that applying
# two double-excitation gates and a single-excitation gate can be used to prepare the target state.
# It won't work! Applying the single-excitation gate on qubits two and four will also lead to a
# contribution for the state :math:`|011000\rangle` through a coupling with :math:`\ket{001100}`.
# Let's check that this is the case:

dev = qml.device('default.qubit', wires=6)


@qml.qnode(dev)
def circuit3(x, y, z):
    qml.BasisState(np.array([1, 1, 0, 0, 0, 0]), wires=[i for i in range(6)])
    qml.DoubleExcitation(x, wires=[0, 1, 2, 3])
    qml.DoubleExcitation(y, wires=[0, 1, 4, 5])
    qml.SingleExcitation(z, wires=[1, 3])

    return qml.state()


x = -2 * np.arcsin(np.sqrt(1/4))
y = -2 * np.arcsin(np.sqrt(1/3))
z = -2 * np.arcsin(np.sqrt(1/2))

output = circuit3(x, y, z)
states = [np.binary_repr(i, width=6) for i in range(len(output)) if output[i] != 0]
print(states)

##############################################################################
# To address this problem, we can instead apply the single-excitation gate *controlled* on the
# state of the first qubit. This ensures that there is no coupling with the state :math:`\ket{
# 001100}` since here the first qubit is in state :math:`|0\rangle`. Let's implement the circuit
# above, this time controlling on the state of the first qubit and verify that we can prepare the
# desired state, for which we use the :func:`~.pennylane.ctrl` transform:


# define a function for the single excitation gate
def op(param, wires):
    qml.SingleExcitation(param, wires=wires)


# perform a transform using qml.ctrl
ctrl_single0 = qml.ctrl(op, control=0)

dev = qml.device('default.qubit', wires=4)


@qml.qnode(dev)
def circuit4(x, y, z):
    qml.BasisState(np.array([1, 1, 0, 0, 0, 0]), wires=[i for i in range(6)])
    qml.DoubleExcitation(x, wires=[0, 1, 2, 3])
    qml.DoubleExcitation(y, wires=[0, 1, 4, 5])
    ctrl_single0(z, wires=[1, 3])

    return qml.state()


##############################################################################
# It was proven in Ref. [#arrazola]_ that controlled single-excitation gates, performing arbitrary
# unitary transformations on the subspace spanned by :math:`\ket{01}, \ket{01}`, are universal for
# particle-conserving unitaries. With enough ingenuity, you can use these operations to construct
# any kind of circuit for quantum chemistry applications. What more could you ask from a gate?
#
# State preparation
# -----------------
#
# We can bring all these pieces together and implement a circuit capable of preparing
# four-qubit states of two particles with real coefficients. The main idea is that we can
# perform the construction one basis state at a time by applying a suitable excitation gate,
# which may need to be controlled.
#
#
# .. figure:: ../demonstrations/givens_rotations/circuit.png
#     :align: center
#     :width: 70%
#
#     A circuit for preparing four-qubit states with two particles.
#
#
# Starting from the reference state :math:`\ket{1100}`, create a superposition
# with the state :math:`\ket{1010}` by applying a single-excitation gate on qubits 2 and 3.
# Similarly, create a superposition with the state :math:`\ket{1001}` with a single
# excitation between qubits 2 and 4. This leaves us with a state of the form
#
# .. :math:
#   |\psi\rangle = a \ket{1100} + b \ket{1010} + c \ket{1001}.
#
# We can now perform two single excitations from qubit 1 to qubits 3 and 4. These will have
# to be controlled on the state of qubit 2. Finally, applying a double-excitation gate on all qubits
# can create a superposition of the form
#
#  .. :math:
#   |\psi\rangle = c_1 \ket{1100} + c_2 \ket{1010} + c_3 \ket{1001} + c_4 \ket{0110} + c_5 \ket{
#   0101} + c_6 \ket{0011},
#
# which is our intended outcome. Let's use this strategy to create an equal superposition over
# all two-particle states on four qubits. We follow the same strategy as before, setting the angle
# of the :math:`i`-th Givens rotation as :math:`-2 \arcsin(`1/\sqrt{n-i}`, where :math:`n` is the
# number of basis states in the superposition.

ctrl_single1 = qml.ctrl(op, control=1)


@qml.qnode(dev)
def state_preparation(params):
    qml.BasisState(np.array([1, 1, 0, 0]), wires=[0, 1, 2, 3])
    qml.SingleExcitation(params[0], wires=[1, 2])
    qml.SingleExcitation(params[1], wires=[1, 3])
    ctrl_single1(params[2], wires=[0, 2])
    ctrl_single1(params[3], wires=[0, 3])
    qml.DoubleExcitation(params[4], wires=[0, 1, 2, 3])
    return qml.state()


n = 6
params = np.array([-2 * np.arcsin(1/np.sqrt(n-i)) for i in range(n-1)])

output = state_preparation(params)
output[np.abs(output) < 1e-6] = 0
print(output)


##############################################################################
# When it comes to quantum circuits for quantum chemistry, there is a wide variety of different
# architectures that have been proposed. Researchers in the field are faced with the apparent
# choice of selecting one of these circuits to conduct their computations and benchmark new
# algorithms. Ultimately, the aim of this tutorial is to provide you
# with the conceptual and software tools to implement any of these proposed circuits,
# but *also to design your own*. It's not only fun to play with toys; it's also fun to make them.
#
#
# References
# ----------
#
# .. [#anselmetti]
#
#     G-L. R. Anselmetti, D. Wierichs, C. Gogolin, Christian, and R. M. Parrish,
#     "Local, Expressive, Quantum-Number-Preserving VQE Ansatze
#     for Fermionic Systems", arXiv:2104.05695, (2021).
#
# .. [#barkoutsos]
#
#     P. KL. Barkoutsos, Panagiotis, et al., "Quantum algorithms for electronic structure
#     calculations: Particle-hole Hamiltonian and optimized wave-function expansions", Physical
#     Review A 98(2), 022322, (2018).
#
# .. [#arrazola]
#
#   J. M. Arrazola, et al., "Universal quantum circuits for quantum chemistry", arXiv: , (2021)
#
#

