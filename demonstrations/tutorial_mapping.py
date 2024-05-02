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

Simulating quantum systems stands as one of the most eagerly anticipated applications of quantum chemistry
with the potential to transform our understanding of chemical and physical systems.
These simulations typically require applying mapping schemes to translate fermionic systems into qubit representations.
There are a variety of mapping schemes used in quantum computing but the conventional ones are the Jordan-Wigner,
Parity, and Bravyi-Kitaev transformations [#PJLove2015]. In this demo, we will discuss these mapping formalisms and show you how to use PennyLane to map fermionic operators to qubit
operators using these schemes. You will also learn how to use these mappings in the context of computing ground state energy calculations with VQE.

Jordan-Wigner Mapping
---------------------
The wave function of a molecular system in the second quantized formalism <https://en.wikipedia.org/wiki/Second_quantization>`__
can be represented by a vector that defines the occupation of the molecular spin-orbitals. These occupation-number vectors are
typically constructed by applying the fermionic creation operators to a vacuum state. Similarly, electrons can be removed from a
state by applying the fermionic annihilation operators. An intuitive way to represent molecular systems in the qubit basis is to
store the occupation state of the molecular spin-orbitals in a set of qubits. This requires constructing qubit creation and annihilation
operators that can be defined as

.. math::

    Q^{\dagger}_j = \frac{1}{2}(X_j - iY_j)

and

.. math::

    Q_j = \frac{1}{2}(X_j + iY_j)

However, an important property of fermionic creation and annihilation operators is the anti-commutation
relations between them, which is not preserved by directly using the analogous qubit operators. These
relations are essential for capturing the Pauli exclusion principle which requires the fermionic wave function to be antisymmetric.
The anti-commutation relations between fermionic operators can be incorporated in the qubit operators, applied to orbital :math:j,
by adding a sequence of :math:Z operators on the preceding qubits. The effect of :math:Z strings is to introduce a phase of
:math:-1, if parity of qubits with index less than j is odd. Accordingly, the fermionic creation and annihilation operators
can be represented as:


.. math::

    a_{j}^{\dagger} = \frac{1}{2}(X_j - iY_j) \otimes_{i<j} Z_{i}

and

.. math::

    a_{j} = \frac{1}{2}(X_j + iY_j) \otimes_{i<j} Z_{i}

This representation is called the Jordan-Wigner mapping. In this mapping the parity information is stored and accessed non-locally by
operating with a long sequence Pauli :math:`Z` operations.
As an example, we show here how to map a simple fermionic operator to a qubit operator
using Jordan-Wigner mapping in PennyLane. First, let's define a fermionic operator [#fermionicOptutorial]_,
say :math:`a_10^{\dagger}` in a :math:`20` qubit system, one way to do this in PennyLane is through :func:`~.pennylane.fermi.from_string`
function. This fermionic Hamiltonian can be then mapped to qubit Hamiltonian using :func:`~.pennylane.fermi.jordan_wigner`.
"""

import pennylane as qml
from pennylane import numpy as np
from pennylane import pauli
from pennylane.operation import active_new_opmath
from pennylane.fermi import from_string

qubits = 20
fermi_op = from_string("10+")
op_jw = qml.jordan_wigner(fermi_op, ps=True, tol=1e-16)
print("Jordan-Wigner transformed Hamiltonian: ", op_jw)

###############################################################################
# However, these Z strings in the qubit operators can significantly increase the resource requirements for implementing
# quantum algorithms on quantum hardware. Operations involving Z strings may involve entangling operations across multiple
# qubits, which can be challenging to implement efficiently.


###############################################################################
# Parity Mapping
# ----------------------
# Parity mapping solves the problem of non-locality of parity information in JW mapping by storing the parity of spin
# orbital :math:`j` in qubit :math:`j` while the occupation information for the orbital is stored non-locally. In this
# representation, state of a fermionic system is represented through a binary vector, where each element corresponds to
# the parity of the spin orbitals. We can look at an example state vector in parity basis by making use of
# :func:~.pennylane.qchem.hf_state function and choosing an arbitrary number of electrons and qubits:

from pennylane import qchem

active_electrons = 8
hf_state = qchem.hf_state(active_electrons, qubits, basis="parity")

print("HF state in parity formalism: ", hf_state)

##############################################################################
# However, unlike the Jordan-Wigner transformation, we cannot represent the creation or annihilation of a
# particle in orbital :math:`j` by simply operating with qubit creation or annihilation operators. Thus the
# state of :math:`(j − 1)`th qubit provides information about the occupation state of qubit :math:`j` and
# whether we need to act with creation or annihilation operator. Moreover, creation or annhilation of a particle
# in qubit :math:`j` changes the parity of all qubits following it. Therefore, the operator equivalent to creation and
# annhilation operators in the parity basis is a two-qubit operator acting on qubits :math:`j` and :math:`j − 1`, and
# an update operator which updates the parity of all qubits with index greater than j.

#.. math::

#    a_{j}^{\dagger} = \frac{1}{2}(Z_{j-1} \otimes X_j - iY_j) \otimes_{i>j} X_{i}

#and

#.. math::

#    a_{j} = \frac{1}{2}(Z_{j-1} \otimes X_j + iY_j) \otimes_{i>j} X_{i}

##############################################################################
# Let's look at an example where we map the above fermionic operator to a parity transformed qubit operator in PennyLane
# using :func:`~.pennylane.fermi.parity_transform`.

op_pr = qml.parity_transform(fermi_op, qubits, ps=True, tol=1e-16)
print("Parity transformed Hamiltonian: ", op_pr)

##############################################################################
# It is evident from this example that the change in locality of parity doesn't improve upon the scaling of                            
# JW mapping as the :math:`Z` strings are now replaced by :math:`X` strings. The advantage of using parity mapping
# however relies in the fact that it allows us to taper two qubits based on symmetries. We provide here the code to taper off
# last two qubits based on the spin symmetries in fermionic systems, more information on qubit tapering
# can be found in Ref. [#taperingtutorial]_.

pw         = []
generators = []

pw.append(pauli.PauliWord(dict(zip(range(0, qubits-1), ["Z"] * (qubits-1)))))
pw.append(pauli.PauliWord({**{qubits-1:"Z"}}))

for sym in pw:
    ham = pauli.PauliSentence({sym:1.0})
    ham = ham.operation(op_pr.wires) if active_new_opmath() else ham.hamiltonian(op_pr.wires)
    generators.append(ham)

paulixops = qml.paulix_ops(generators, qubits)
paulix_sector = qml.qchem.optimal_sector(op_pr, generators, active_electrons)
print(op_pr)

op_tapered = qml.taper(op_pr, generators, paulixops, paulix_sector)
coeffs = op_tapered.terms()[0]
ops = op_tapered.terms()[1]
op_tapered = qml.Hamiltonian(np.real(coeffs), ops)

print(op_tapered)

###############################################################################
# Note that the tapered operator doesn't have any Paulis on qubit :math:`18` and :math:`19`.

###############################################################################
# Bravyi-Kitaev Mapping
# --------------------------------
# Bravyi-Kitaev mapping aims to improve the linear scaling of Jordan-Wigner and Parity mappings, and uses
# a middle way between the two. In Bravyi-Kitaev mapping, both the occupation number and parity are stored
# non-locally. In this formalism, even labelled qubits store the occupation number of spin orbitals whereas
# odd labelled qubits store parity through partial sums of occupation numbers. The corresponding creation and
# annhilation operators for when :math:`j` is even can be represented as
# .. math::
#           a^{\dagger}_n &= \frac{1}{2} \left ( X_{U(n)} \otimes X_n \otimes Z_{P(n)} -iX_{U(n)} \otimes Y_{n} \otimes Z_{P(n)}\right ), \\\\
# and
# .. math::
#           a_n &= \frac{1}{2} \left ( X_{U(n)} \otimes X_n \otimes Z_{P(n)} +iX_{U(n)} \otimes Y_{n} \otimes Z_{P(n)}\right ). \\\\
# Similarly, the Bravyi-Kitaev mapped creation and annhilation operators for odd-labelled orbitals are represented as 
# .. math::
#           a^{\dagger}_n &= \frac{1}{2} \left ( X_{U(n)} \otimes X_n \otimes Z_{P(n)} -iX_{U(n)} \otimes Y_{n} \otimes Z_{R(n)}\right ), \\\\
# and
# .. math::
#           a_n &= \frac{1}{2} \left ( X_{U(n)} \otimes X_n \otimes Z_{P(n)} +iX_{U(n)} \otimes Y_{n} \otimes Z_{R(n)}\right ). \\\\
# where :math:`U(n)`, :math:`P(n)` and :math:`R(n)` represent the update, parity and remainder sets, respectively [#PJLove2015]_.
# An example of how to use :func:`~.pennylane.fermi.bravyi_kitaev` in PennyLane is as follows:

op_bk = qml.bravyi_kitaev(fermi_op, qubits, ps=True, tol=1e-16)
print("Bravyi-Kitaev transformed Hamiltonian: ", op_bk)

##############################################################################
# A closer look at the qubit Hamiltonians mapped through different mappings makes it clear that use of local transformations
# in Bravyi-Kitaev mapping, helps improve the scaling for this transformation.

###############################################################################
# VQE Calculations
# --------------------------------
# In this section, we want to explore the use of these mappings in quantum simulations.
# We show this through an example VQE calculation to find the ground state of :math:`H_3^{+}`.
# A VQE calculation typically requires several components: molecular Hamiltonian, reference state and
# a quantum circuit that encodes the ansatz. All these components need to comply with the same mappings
# in order for a VQE calculation to work. For this example, we will use only Bravyi-Kitaev transformation but
# similar calculations can be run with the other two schemes as well.
# First, let's build the molecular Hamiltonian [#QCtutorial]_, this requires defining the molecular structure.

symbols = ["H", "H", "H"]
geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4], [0.0, 0.0, 2.8]], requires_grad=False)

mol = qchem.Molecule(symbols, geometry, charge=1)

##############################################################################
# This is followed by the use of :func:~pennylane.qchem.fermionic_hamiltonian function to build
# the molecular Hamiltonian for the above molecule.

h_ferm = qchem.fermionic_hamiltonian(mol)()

##############################################################################
# In the previous sections, we learnt how to obtain qubit representation of an operator
# using different mapping schemes. We make use of the above discussed :func:~pennylane.fermi.bravyi_kitaev
# function to transform the molecular Hamiltonian to its qubit representation.

active_electrons = 2
qubits = len(h_ferm.wires)
h_bk = qml.bravyi_kitaev(h_ferm, qubits,ps=True, tol=1e-16).hamiltonian()


##############################################################################
# Next, we discuss the mapping of reference state. For the reference state of VQE,
# we use Hartree-Fock state, which can be obtained in the user defined basis
# by using :func:`~.pennylane.qchem.hf_state` function in PennyLane. For example:

hf_state = qchem.hf_state(active_electrons, qubits, basis="bravyi_kitaev")

##############################################################################
# Lastly, we can discuss the generation of quantum circuit, here, we choose to use the UCCSD ansatz and
# start by obtaining electronic excitations for :math:H_{3}^{+} molecule.

singles, doubles = qchem.excitations(active_electrons, qubits)

##############################################################################
# The fermionic operators for the singles and doubles excitations in the chosen ansatz[#HWBarnes2020]_ can be defined using
# respective equations:

#.. math::

#     T_i^k(\theta) = \theta(a_k^{\dagger}a_i - a_i^{\dagger}a_k)
     
#and

#.. math::

#     T_{ij}^{kl}(\theta) = \theta(a_k^{\dagger}a_l^{\dagger}a_i a_j- a_i^{\dagger}a_j^{\dagger}a_k a_l)

##############################################################################
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
# These fermionic operators can now be mapped to qubit operators using the user defined mapping. We show here
# an example for Bravyi-Kitaev mapping.

op_singles_bk = []
for op in fermi_op_singles:
    op_singles_bk.append(qml.bravyi_kitaev(op, qubits, ps=True))

op_doubles_bk = []
for op in fermi_op_doubles:
    op_doubles_bk.append(qml.bravyi_kitaev(op, qubits, ps=True))

##############################################################################
# Final step in generating the VQE circuit is exponentiating these excitations[#HWBarnes2020]_ to obtain the unitary
# operators

#.. math::
#     U_{ki}(\theta) = exp^{T_i^k(\theta)}

#     U_{klij}(\theta) = exp^{T_{ij}^{kl}(\theta)}
     
##############################################################################
# and hence the required gates for the quantum circuit.

dev = qml.device("default.qubit", wires=qubits)

@qml.qnode(dev)
def circuit(params):
    qml.BasisState(hf_state, wires=range(qubits))
    
    for i, excitation in enumerate(op_doubles_bk):
        # from Eq 4 of Ref. [#HWBarnes2020]_
        qml.exp((excitation * params[i] / 2).operation()), range(qubits)
    
    for j, excitation in enumerate(op_singles_bk):
        # from Eq 3 of Ref. [#HWBarnes2020]_
        qml.exp((excitation * params[i + j + 1] / 2).operation()), range(qubits)

    return qml.expval(h_bk)

params = [3.1415945, 0.14896247, 3.14157128, 7.8475722, 4.71722918, -3.45172302, 3.89213951, -0.46622931]
print(circuit(params))

##############################################################################
# Using the above circuit, we produce the ground state energy of :math:`H_3^{+}` molecule.

###############################################################################
# Summary
# -------
# In this demo, we talked about various mapping schemes available in PennyLane. We also showed how
# these mappings can be used to convert fermionic operators to qubits operators in PennyLane and discussed the pros and
# cons associated with each scheme. The Jordan-Wigner mapping, despite its non-local transformations, preserves essential
# fermionic properties and provides an intuitive approach. Parity mapping, though unable to improve upon the scaling,
# offers a promising approach by exploiting symmetry properties of fermionic systems. Conversely, the Bravyi-Kitaev mapping
# emphasizes locality and resource efficiency, making it an attractive option for certain applications. Through this demonstration,
# we recognize the importance of choosing an appropriate mapping scheme tailored to the specific problem at hand and the available
# quantum resources. Lastly, we showed how a user can employ these different mappings in VQE calculations through an example. We
# would like to encourage the interested readers to run VQE calculations for different molecular systems and observe how the scaling is
# influenced by the chosen mapping techniques.

###############################################################################
# References
# ----------
#
# .. [#PJLove2015]
#
# A. Tranter, S. Sofia, *et al.*, "The Bravyi–Kitaev Transformation:
# Properties and Applications". `International Journal of Quantum Chemistry 115.19 (2015).
# <https://onlinelibrary.wiley.com/doi/10.1002/qua.24969>`__
#
# .. [#HWBarnes2020]
#
# Y. S. Yordanov, *et al.*, "Efficient quantum circuits for quantum computational chemistry". `Physical Review A 102.6 (2020).
# <https://journals.aps.org/pra/abstract/10.1103/PhysRevA.102.062612>`__

# .. [#fermionicOptutorial]
#
#     :doc:`tutorial_fermionic_operators`

# .. [#QCtutorial]
#
#     :doc:`tutorial_quantum_chemistry`

# .. [#taperingtutorial]
#
#     :doc:`tutorial_qubit_tapering`

# About the author
# ----------------
# .. include:: ../_static/authors/diksha_dhawan.txt
