r"""

Givens rotations for quantum chemistry
======================================

.. meta::
    :property="og:description": Discover the building blocks of particle-conserving unitaries for
        quantum chemistry

    :property="og:image":

.. related::



*Author: PennyLane dev team. Posted:  2021. Last updated: *

In the book `“Sophie’s world” <https://en.wikipedia.org/wiki/Sophie%27s_World>`_, the young
protagonist receives a white envelope containing a letter
with an intriguing question: “Why is Lego the most ingenious toy in the world?” At first baffled by
this curious piece of mail, she decides to reflect on the question, and finally concludes:

“*The best thing about them was that with Lego she could construct any kind of object. And then
she could separate the blocks and construct something new. What more could one ask of a toy?
Sophie decided that Lego really could be called the most ingenious toy in the world.*”

In this tutorial, you will learn about the Lego of quantum circuits for quantum chemistry: Givens
rotations. These are building blocks that can be used to construct any kind of
particle-conserving circuit. They can also be rearranged to construct new circuits. What more
could one ask of a quantum gate? You will also learn how to use these gates to build arbitrary
states of a fixed number of particles.

Particle-conserving unitaries
-----------------------------

Arguably the central problem in quantum chemistry is understanding how electrons
behave in molecules. Quantum computers tackle this problem by using systems of many qubits to
represent the quantum states of the electrons. One method for doing this is to consider a
collection of `molecular orbitals <https://en.wikipedia.org/wiki/Molecular_orbital>`_ that
describe wave functions that are approximate solutions to the Schrödinger equation. These orbitals
capture the three-dimensional region of space occupied by the electrons. Each orbital can be
occupied by at most two electrons, each with a different spin orientation. In this case we refer to
spin orbitals that can be occupied by a single electron.

.. figure:: ../demonstrations/Givens_rotations/molecular_orbitals.jpg
    :align: center
    :width: 50%

    Different molecular orbitals for the carbon dioxide molecule. Each orbital has a different
    associated energy. Red regions correspond to positive values of the wavefunction,
    blue regions to negative values. Image credit: `here <https://www.flickr.com/photos/69057297@N04/26568303629>`.

    ..

The state of electrons in a molecule can then be described by specifying which
orbitals are occupied. The `Jordan-Wigner
<https://en.wikipedia.org/wiki/Jordan%E2%80%93Wigner_transformation>`_ representation provides a
convenient way of using qubit states to do this: we associate a qubit with each spin orbital and
use its states to represent occupied (:math:`|1\rangle`) or unoccupied (
:math:`|0\rangle`) spin orbitals. An :math:`n`-qubit state with Hamming weight :math:`k`, i.e.,
with :math:`k` qubits in state :math:`|1\rangle` represents a state of :math:`k` electrons in
:math:`n` spin orbitals. For example :math:`|1010\rangle` is a state of two electrons in two spin
spin orbitals. More generally, superpositions over all basis states with a fixed
number of particles are valid states of the electrons in a molecule. These are states such as

.. math::

    |\psi\rangle = c_1|1100\rangle + c_2|1010\rangle + c_3|1001\rangle + c_4|0110\rangle +
    c_5|0101\rangle + c_6|0011\rangle,

for some coefficients :math:`c_1,\ldots, c_6`. Because the number of electrons in a molecule is
fixed, any transformation on its states must conserve the number particles. We refer to these as
particle-conserving unitaries. When designing quantum circuits and algorithms for quantum
chemistry, it is desirable to employ only particle-conserving gates that guarantee that the
states of the system remain valid. This raises the questions: what are the simplest
particle-conserving unitaries? Like Legos, can they be used to construct any quantum circuit for
quantum chemistry?

Givens rotations
----------------

Consider single-qubit gates. In their most general form, they perform the transformation

.. :math:

    U|0\rangle &= a |0\rangle + b |1\rangle\\
    U|1\rangle &= c |1\rangle + d |0\rangle,

where :math:`|a|^2+|b|^2=|c|^2+|d|^2=1` and :math:`ab^* + cd^*=0`. This gate is
particle-conserving only if :math:`b=d=0`, which means that the only single-qubit gates that
preserve particle number are diagonal gates of the form

.. :math:

    \begin{pmatrix}
    e^{i\theta} & 0\\
    0 & e^{i\phi}
    \end{pmatrix}.

On their own these gates are not very interesting since they can only be used to change the
relative phases of states in a superposition; they cannot be used to create and control such
superpositions. So let's take a look at two-qubit gates. We can divide basis states depending on
their number of particles. We have :math:`|00\rangle` with zero particles, :math:`|01\rangle,
|10\rangle` with one particle, and :math:`|11\rangle` with two particles. A similar argument as
before shows that any particle-conserving unitary can only apply a phase to the states
:math:`|00\rangle, |11\rangle`, but we can now consider transformations that couple the remaining
states. These are gates of the form

.. :math:

    U|01\rangle &= a |01\rangle + b |10\rangle\\
    U|10\rangle &= c |10\rangle + d |01\rangle.

This should be familiar: it has the same form as a single-qubit gate, where the states
:math:`|01\rangle, |10\rangle` respectively take the place of :math:`|0\rangle, |1\rangle`. In
fact, this correspondence has a name: the `dual-rail qubit
<https://en.wikipedia.org/wiki/Optical_cluster_state>`. The
difference is that now any values of the unitary's parameters :math:`a,b,c,d` give rise to a valid
particle-conserving unitary. Take for instance the two-qubit gate

.. :math:

    G(\theta)=\begin{pmatrix}
    1 & 0 & 0 & 0\\
    0 & \cos (\theta/2) & -\sin (\theta/2) & 0\\
    0 & \sin(\theta/2) & \cos(\theta/2) & 0\\
    0 & 0 & 0 & 1
    \end{pmatrix}.

This is an example of a `Givens rotation <https://en.wikipedia.org/wiki/Givens_rotation>`_: a
rotation in a two-dimensional subspace of a larger space. In this case, we are performing a
Givens rotation in the four-dimensional space of two-qubit states. This gate is more
interesting: it allows us to create superpositions by essentially exchanging the particle between
the two qubits. Such transformations can be interpreted as an *excitation operation*, where we
view the exchange from :math:`|10\rangle` to :math:`|10\rangle` as exciting the electron from the
first to the second qubit. The difference compared to fermionic excitations is that Givens
rotations do not keep track of phase changes arising from anti-commutation relations. This
gate is implemented in PennyLane as the :func:`~.pennylane.ops.SingleExcitation` operation. We
can use it to prepare an equal superposition of three-qubit states with a single particle:
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
# Note that the components of the output state are ordered according to their binary
# representation, so entry 1 is :math:`|001\rangle`, entry 2 is :math:`|010\rangle`, and entry 4 is
# :math:`|100\rangle`.
#
# We can also consider higher-order transformations where more particles are excited between
# different qubits. Remember that these represent electrons being excited to spin-orbitals with
# higher energy. We can thus study *double excitations* involving the transfer of two particles,
# which are four-qubit transformations. For example, consider a Givens rotation performing a
# transformation spanned by the states :math:`|1100\rangle` and :math:`|0011\rangle`. These
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
# In the context of quantum chemistry, it is common to consider excitations on a fixed reference
# state, typically the Hartree-Fock state, and to include only the excitations that preserve the
# spin orientation of the electron. PennyLane allows you to obtain all such excitations using the
# :func:`~.pennylane_qchem.qchem.excitations`. Let's use this to build a circuit that includes
# all single and double excitations acting on a reference state of three particles in six qubits,
# and apply a random rotation for each gate:

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
# Besides these
#
#
#
#
#
#
#
#
#
#
# References
# ----------
#
# .. [#NandC2000]
#
#     M. A. Nielsen, and I. L. Chuang (2000) "Quantum Computation and Quantum Information",
#     Cambridge University Press.
#
# .. [#deGuise2018]
#
#     H. de Guise, O. Di Matteo, and L. L. Sánchez-Soto. (2018) "Simple factorization
#     of unitary transformations", `Phys. Rev. A 97 022328
#     <https://journals.aps.org/pra/abstract/10.1103/PhysRevA.97.022328>`__.
#     (`arXiv <https://arxiv.org/abs/1708.00735>`__)
#
# .. [#Clements2016]
#
#     W. R. Clements, P. C. Humphreys, B. J. Metcalf, W. S. Kolthammer, and
#     I. A. Walmsley (2016) “Optimal design for universal multiport
#     interferometers”, \ `Optica 3, 1460–1465
#     <https://www.osapublishing.org/optica/fulltext.cfm?uri=optica-3-12-1460&id=355743>`__.
#     (`arXiv <https://arxiv.org/abs/1603.08788>`__)
#
# .. [#Reck1994]
#
#    M. Reck, A. Zeilinger, H. J. Bernstein, and P. Bertani (1994) “Experimental
#    realization of any discrete unitary operator”, `Phys. Rev. Lett.73, 58–61
#    <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.73.58>`__.
#
# .. [#Mezzadri2006]
#
#     F. Mezzadri (2006) "How to generate random matrices from the classical compact groups".
#     (`arXiv <https://arxiv.org/abs/math-ph/0609050>`__)
#
# .. [#Meckes2014]
#
#     E. Meckes (2019) `"The Random Matrix Theory of the Classical Compact Groups"
#     <https://case.edu/artsci/math/esmeckes/Haar_book.pdf>`_, Cambridge University Press.
#
# .. [#Gerken2013]
#
#     M. Gerken (2013) "Measure concentration: Levy's Lemma"
#     (`lecture notes <http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.679.2560>`__).
#
#
# .. [#Hayden2006]
#
#     P. Hayden, D. W. Leung, and A. Winter (2006) "Aspects of generic
#     entanglement", `Comm. Math. Phys. Vol. 265, No. 1, pp. 95-117
#     <https://link.springer.com/article/10.1007%2Fs00220-006-1535-6>`__.
#     (`arXiv <https://arxiv.org/abs/quant-ph/0407049>`__)
#
# .. [#McClean2018]
#
#     J. R. McClean, S. Boixo, V. N. Smelyanskiy, R. Babbush, and H. Neven
#     (2018) "Barren plateaus in quantum neural network training
#     landscapes", `Nature Communications, 9(1)
#     <http://dx.doi.org/10.1038/s41467-018-07090-4>`__.
#     (`arXiv <https://arxiv.org/abs/1803.11173>`__)
#
# .. [#Holmes2021]
#
#     Z. Holmes, K. Sharma, M. Cerezo, and P. J. Coles (2021) "Connecting ansatz
#     expressibility to gradient magnitudes and barren plateaus". (`arXiv
#     <https://arxiv.org/abs/2101.02138>`__)
#
#
