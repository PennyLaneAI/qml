r"""

Mapping Fermionic operators to Qubit operators
==============================================

.. meta::
    :property="og:description": Learn how to map fermionic operators to qubit operators.
    :property="og:image": 

.. related::
    tutorial_quantum_chemistry Building molecular Hamiltonians
    tutorial_vqe A brief overview of VQE

*Author: Diksha Dhawan — Posted: April 2024. Last updated: April 2024.*

Simulating quantum systems stands as one of the most eagerly anticipated applications of quantum
chemistry with the potential to transform our understanding of chemical and physical systems. These
simulations typically require mapping schemes that transform fermionic systems into qubit
representations. There are a variety of mapping schemes used in quantum computing but the
conventional ones are the Jordan-Wigner, Parity, and Bravyi-Kitaev transformations [#Tranter]. In
this demo, you will learn about these mapping schemes and their implementation in PennyLane. You
will also learn how to use these mappings in the context of computing ground state energy
calculations of a molecular system.

Jordan-Wigner Mapping
---------------------
The state of a quantum system in the `second quantized <https://en.wikipedia.org/wiki/Second_quantization>`__
formalism is typically represented in the occupation-number basis. For fermions, the occupation
number is either :math:`1` or :math:`0` as a result of the Pauli exclusion principle. The
occupation-number basis states can be represented by a vector that is typically constructed by
applying the fermionic creation operators to a vacuum state. Similarly, electrons can be removed
from a state by applying the fermionic annihilation operators. An intuitive way to represent a
fermionic systems in the qubit basis is to store the fermionic occupation numbers in qubit states.
This requires constructing qubit creation and annihilation operators that can be applied to an
initial :math:`| 0 \rangle` state to provide the desired occupation number state. These operators
are defined as

.. math::

    Q^{\dagger}_j = \frac{1}{2}(X_j - iY_j)

and

.. math::

    Q_j = \frac{1}{2}(X_j + iY_j)

However, an important property of fermionic creation and annihilation operators is the
anti-commutation relations between them, which is not preserved by directly using the analogous
qubit operators. These relations are essential for capturing the Pauli exclusion principle which
requires the fermionic wave function to be antisymmetric. The anti-commutation relations between
fermionic operators can be incorporated by adding a sequence of Pauli :math:`Z` operators to the
qubit operators. These :math:`Z` strings introduce a phase :math:`-1` if the parity of preceding
qubits is odd. According to these, the fermionic creation and annihilation operators can be
represented as:

.. math::

    a_{j}^{\dagger} = \otimes_{i<j} Z_{i} \frac{1}{2}(X_j - iY_j),

and

.. math::

    a_{j} = \otimes_{i<j} Z_{i} \frac{1}{2}(X_j + iY_j) .

This representation is called the **Jordan-Wigner** mapping where the parity information is stored
and accessed non-locally by operating with a long sequence of Pauli :math:`Z` operations.

Let's now look at an example using PennyLane: to map a simple fermionic operator to a qubit operator
using the Jordan-Wigner mapping. First, we define our fermionic operator [#fermionicOptutorial]_
:math:`a_{10}^{\dagger}`, which creates an electron in the :math:`10`-th qubit of a :math:`20`
qubit system. One way to do this in PennyLane is to use :func:`~.pennylane.fermi.from_string`. We
then mapp the operator using :func:`~.pennylane.fermi.jordan_wigner`.
"""

import pennylane as qml
from pennylane.fermi import from_string, jordan_wigner

qubits = 10
fermi_op = from_string("5+")
pauli_jw = jordan_wigner(fermi_op, ps=True)
pauli_jw

###############################################################################
# The long sequence of :math:`Z` operations in this operator can significantly increase the
# resources needed to implement the operator in quantum hardware as it may require using entangling
# operations across multiple qubits, which can be challenging to implement efficiently. One way to
# avoid having such long tails of :math:`Z` operations is to work in the parity basis where the
# fermionic state stores the parity instead of the occupation number.
#
# Parity Mapping
# --------------
# Parity mapping solves the non-locality problem, of the parity information, by storing
# the parity of spin orbital :math:`j` in qubit :math:`j` while the occupation information for the
# orbital is stored non-locally. In this representation, the state of a fermionic system is
# represented through a binary vector, where each element corresponds to the parity of the spin
# orbitals. Let's look at an example using the PennyLane function func:`~.pennylane.qchem.hf_state`
# for a system with :math:`4` spin-orbitals and :math:`2` electrons.

orbitals = 10
electrons = 5
state_number = qml.qchem.hf_state(electrons, orbitals)
state_parity = qml.qchem.hf_state(electrons, orbitals, basis="parity")

print("State in occupation number basis:\n", state_number)
print("State in parity basis:\n", state_parity)

##############################################################################
# In the parity basis we cannot represent the creation or annihilation of a particle in orbital
# :math:`j` by simply operating with qubit creation or annihilation operators. In fact, the state of
# the :math:`(j − 1)`-th qubit provides information about the occupation  state of qubit :math:`j`
# and whether we need to act with a creation or annihilation operator. Similarly, the creation or
# annihilation of a particle in qubit :math:`j` changes the parity of all qubits following it.
# As a result, the operator that is equivalent to creation and annihilation operators in
# the parity basis is a two-qubit operator acting on qubits :math:`j` and :math:`j − 1`, and
# an update operator which updates the parity of all qubits with index greater than j.
#
# .. math::
#
#     a_{j}^{\dagger} = \frac{1}{2}(Z_{j-1} \otimes X_j - iY_j) \otimes_{i>j} X_{i}
#
# and
#
# .. math::
#
#     a_{j} = \frac{1}{2}(Z_{j-1} \otimes X_j + iY_j) \otimes_{i>j} X_{i}
#
# Let's now look at an example where we map our fermionic operator :math:`a_{10}^{\dagger}` with
# Parity mapping using :func:`~.pennylane.fermi.parity_transform` in PennyLane.

pauli_pr = qml.parity_transform(fermi_op, qubits, ps=True)
pauli_pr

##############################################################################
# It is evident from this example that the Parity transform doesn't improve upon the
# scaling of the Jordan-Wigner mapping as the :math:`Z` strings are now replaced by :math:`X`
# strings. However, a very important advantage of using parity mapping is the ability to taper two
# qubits by leveraging symmetries of molecular Hamiltonians. Let's look at an example. You can find
# more information about qubit tapering in [#taperingtutorial]_.

from pennylane import numpy as np

pw = []
generators = []

pw.append(qml.pauli.PauliWord(dict(zip(range(0, qubits-1), ["Z"] * (qubits-1)))))
pw.append(qml.pauli.PauliWord({**{qubits-1:"Z"}}))

for sym in pw:
    ham = qml.pauli.PauliSentence({sym:1.0})
    ham = ham.operation(pauli_pr.wires)
    generators.append(ham)

paulixops = qml.paulix_ops(generators, qubits)
paulix_sector = qml.qchem.optimal_sector(pauli_pr, generators, electrons)

op_tapered = qml.taper(pauli_pr, generators, paulixops, paulix_sector)
coeffs = op_tapered.terms()[0]
ops = op_tapered.terms()[1]
op_tapered = qml.Hamiltonian(np.real(coeffs), ops)

print(op_tapered)

###############################################################################
# Note that the tapered operator doesn't have any Paulis on qubit :math:`18` and :math:`19`.
#
# Bravyi-Kitaev Mapping
# ---------------------
# Bravyi-Kitaev mapping aims to improve the linear scaling of Jordan-Wigner and Parity mappings by
# storing both the occupation number and the parity non-locally. In this formalism, even labelled
# qubits store the occupation number of spin orbitals whereas odd labelled qubits store parity
# through partial sums of occupation numbers. The corresponding creation and annhilation operators
# for when :math:`j` is even can be represented as
#
# .. math::
#
#     a^{\dagger}_n = \frac{1}{2} \left ( X_{U(n)} \otimes X_n \otimes Z_{P(n)} -iX_{U(n)} \otimes
#     Y_{n} \otimes Z_{P(n)}\right ),
#
# and
#
# .. math::
#
#     a_n = \frac{1}{2} \left ( X_{U(n)} \otimes X_n \otimes Z_{P(n)} +iX_{U(n)} \otimes Y_{n}
#     \otimes Z_{P(n)}\right ).
#
# Similarly, the Bravyi-Kitaev mapped creation and annhilation operators for odd-labelled orbitals
# are represented as
#
# .. math::
#
#     a^{\dagger}_n = \frac{1}{2} \left ( X_{U(n)} \otimes X_n \otimes Z_{P(n)} -iX_{U(n)} \otimes
#     Y_{n} \otimes Z_{R(n)}\right ),
#
# and
#
# .. math::
#
#     a_n = \frac{1}{2} \left ( X_{U(n)} \otimes X_n \otimes Z_{P(n)} +iX_{U(n)} \otimes Y_{n}
#     \otimes Z_{R(n)}\right ).
#
# where :math:`U(n)`, :math:`P(n)` and :math:`R(n)` represent the update, parity and remainder sets,
# respectively [#Tranter]_. An example of how to use :func:`~.pennylane.fermi.bravyi_kitaev` in
# PennyLane is as follows:

pauli_bk = qml.bravyi_kitaev(fermi_op, qubits, ps=True)
pauli_bk

##############################################################################
# A closer look at the qubit operators obtained with different mappings makes it clear that the
# local nature of the transformations in the Bravyi-Kitaev mapping helps improve the scaling for
# this transformation.
#
# VQE Calculations
# ----------------
# Let's now put all these together in an example of VQE calculations to find the ground state of
# :math:`H_3^{+}`.
# A VQE calculation typically requires several components: molecular Hamiltonian, reference state
# and a quantum circuit that encodes the ansatz. All these components need to comply with the same
# mappings in order for a VQE calculation to work. For this example, we will use only Bravyi-Kitaev
# transformation but similar calculations can be run with the other two schemes as well.
# First, let's build the molecular Hamiltonian [#QCtutorial]_, this requires defining the molecular
# structure.

from pennylane import qchem

symbols = ["H", "H", "H"]
geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4], [0.0, 0.0, 2.8]], requires_grad=False)

mol = qchem.Molecule(symbols, geometry, charge=1)

##############################################################################
# This is followed by the use of :func:~pennylane.qchem.fermionic_hamiltonian function to build
# the molecular Hamiltonian for the above molecule.

h_ferm = qchem.fermionic_hamiltonian(mol)()

##############################################################################
# In the previous sections, we learnt how to obtain qubit representation of an operator
# using different mapping schemes. We make use of the above discussed
# :func:~pennylane.fermi.bravyi_kitaev function to transform the molecular Hamiltonian to its qubit
# representation.

active_electrons = 2
qubits = len(h_ferm.wires)
h_bk = qml.bravyi_kitaev(h_ferm, qubits,ps=True, tol=1e-16).hamiltonian()

##############################################################################
# Next, we discuss the mapping of reference state. For the reference state of VQE,
# we use Hartree-Fock state, which can be obtained in the user defined basis
# by using :func:`~.pennylane.qchem.hf_state` function in PennyLane. For example:

hf_state = qchem.hf_state(active_electrons, qubits, basis="bravyi_kitaev")

##############################################################################
# Lastly, we can discuss the generation of quantum circuit, here, we choose to use the UCCSD ansatz
# and start by obtaining electronic excitations for :math:H_{3}^{+} molecule.

singles, doubles = qchem.excitations(active_electrons, qubits)

##############################################################################
# The fermionic operators for the singles and doubles excitations in the chosen
# ansatz [#Yordanov]_ can be defined using respective equations:
#
# .. math::
#
#     T_i^k(\theta) = \theta(a_k^{\dagger}a_i - a_i^{\dagger}a_k)
#
# and
#
# .. math::
#
#     T_{ij}^{kl}(\theta) = \theta(a_k^{\dagger}a_l^{\dagger}a_i a_j -
#     a_i^{\dagger}a_j^{\dagger}a_k a_l)
#
# These can be obtained in PennyLane using the given code

from pennylane.fermi import from_string

fermi_op_singles = []
for ex in singles:
    fermi_op_singles.append(from_string(str(ex[1])+ "+ " + str(ex[0]) + "-")
                          - from_string(str(ex[0])+ "+ " + str(ex[1]) + "-"))

fermi_op_doubles = []
for ex in doubles:
    fermi_op_doubles.append(from_string(str(ex[3])+ "+ " + str(ex[2]) + "+ "
                                        + str(ex[1])+ "- " + str(ex[0]) + "-")
                            - from_string(str(ex[0])+ "+ " + str(ex[1]) + "+ "
                                          + str(ex[2])+ "- " + str(ex[3]) + "-"))
    
##############################################################################
# These fermionic operators can now be mapped to qubit operators using the user defined mapping. We
# show here an example for Bravyi-Kitaev mapping.

op_singles_bk = []
for op in fermi_op_singles:
    op_singles_bk.append(qml.bravyi_kitaev(op, qubits, ps=True))

op_doubles_bk = []
for op in fermi_op_doubles:
    op_doubles_bk.append(qml.bravyi_kitaev(op, qubits, ps=True))

##############################################################################
# Final step in generating the VQE circuit is exponentiating these excitations[#Yordanov]_ to
# obtain the unitary operators
#
# .. math::
#
#     U_{ki}(\theta) = exp^{T_i^k(\theta)}
#
#     U_{klij}(\theta) = exp^{T_{ij}^{kl}(\theta)}
#
# and hence the required gates for the quantum circuit.

dev = qml.device("default.qubit", wires=qubits)

@qml.qnode(dev)
def circuit(params):
    qml.BasisState(hf_state, wires=range(qubits))
    
    for i, excitation in enumerate(op_doubles_bk):
        # from Eq 4 of Ref. [#Yordanov]_
        qml.exp((excitation * params[i] / 2).operation()), range(qubits)
    
    for j, excitation in enumerate(op_singles_bk):
        # from Eq 3 of Ref. [#Yordanov]_
        qml.exp((excitation * params[i + j + 1] / 2).operation()), range(qubits)

    return qml.expval(h_bk)

params = [3.1415945, 0.14896247, 3.14157128, 7.8475722, 4.71722918, -3.45172302, 3.89213951, -0.46622931]
print(circuit(params))

##############################################################################
# Using the above circuit, we produce the ground state energy of :math:`H_3^{+}` molecule.
#
# Summary
# -------
# In this demo, we talked about various mapping schemes available in PennyLane. We also showed how
# these mappings can be used to convert fermionic operators to qubits operators in PennyLane and
# discussed the pros and cons associated with each scheme. The Jordan-Wigner mapping, despite its
# non-local transformations, preserves essential fermionic properties and provides an intuitive
# approach. Parity mapping, though unable to improve upon the scaling, offers a promising approach
# by exploiting symmetry properties of fermionic systems. Conversely, the Bravyi-Kitaev mapping
# emphasizes locality and resource efficiency, making it an attractive option for certain
# applications. Through this demonstration, we recognize the importance of choosing an appropriate
# mapping scheme tailored to the specific problem at hand and the available quantum
# resources. Lastly, we showed how a user can employ these different mappings in VQE calculations
# through an example. We would like to encourage the interested readers to run VQE calculations for
# different molecular systems and observe how the scaling is influenced by the chosen mapping
# techniques.
#
# References
# ----------
#
# .. [#Tranter]
#
#      A. Tranter, S. Sofia, *et al.*, "The Bravyi–Kitaev Transformation:
#      Properties and Applications". `International Journal of Quantum Chemistry 115.19 (2015).
#      <https://onlinelibrary.wiley.com/doi/10.1002/qua.24969>`__
#
# .. [#Yordanov]
#
#      Y. S. Yordanov, *et al.*, "Efficient quantum circuits for quantum computational chemistry".
#      `Physical Review A 102.6 (2020).
#      <https://journals.aps.org/pra/abstract/10.1103/PhysRevA.102.062612>`__
#
# About the author
# ----------------
# .. include:: ../_static/authors/diksha_dhawan.txt
