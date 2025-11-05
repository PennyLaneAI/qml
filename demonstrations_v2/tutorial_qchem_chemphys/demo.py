r"""

Molecular Hamiltonian Representations
=====================================

.. meta::
    :property="og:description": Learn how to use chemist, physicist and quantum notations
    :property="og:image": https://pennylane.ai/qml/_static/demonstration_assets/thumbnail_tutorial_external_libs.png


.. related::
    tutorial_quantum_chemistry Quantum chemistry with PennyLane

Molecular Hamiltonians can be constructed in different ways depending on the the arrangement of
the two-electron integral tensor. Here, we review the common ways to represent a molecular
Hamiltonian. We use three common two-electron integral notations that are typically referred to
as physicists', chemists' and quantum computing notation. The two-electron integrals computed with
one convention can be easily converted to the other notations. Such conversions allow constructing
different representations of the molecular Hamiltonian without re-calculating the integrals.
"""

##############################################################################
# Quantum computing notation
# --------------------------
# This notation is commonly used in the quantum computing literature and quantum computing
# software libraries.
#
# The two-electron integral tensor in this notation is defined as
#
# .. math::
#
#     \langle \langle pq | rs \rangle \rangle = \int dr_1 dr_2 \phi_p^*(r_1) \phi_q^*(r_2) \frac{1}{r_{12}} \phi_r(r_2) \phi_s(r_1).
#
# The corresponding Hamiltonian in second quantization is defined in terms of the fermionic
# creation and annihilation operators as
#
# .. math::
#
#     H = \sum_{pq} h_{pq} a_p^{\dagger} a_q + \frac{1}{2} \sum_{pqrs} h_{pqrs} a_p^{\dagger} a_q^{\dagger} a_r a_s.
#
# where :math:`h_{pq}` denotes the one-electron integral. We have skipped the spin indices and
# the core constant for brevity. Note that the order of the creation and annihilation operator
# indices matches the order of the coefficient indices for both :math:`h_{pq}` and h_{pqrs} terms.
#
# We will now construct the Hamiltonian using PennyLane. PennyLane employs the quantum computing
# convention for the two-electron integral tensor, which can be efficiently computed via the
# built-in electron_integrals function. We use the water molecule as an example.

import numpy as np
import pennylane as qml
from pennylane.fermi import FermiWord, FermiSentence, from_string

symbols = ["H", "O", "H"]
geometry = np.array([[-0.0399, -0.0038, 0.0000],
                     [1.5780, 0.8540, 0.0000],
                     [2.7909, -0.5159, 0.0000]])

mol = qml.qchem.Molecule(symbols, geometry)

core_constant, one_mo, two_mo = qml.qchem.electron_integrals(mol)()


##############################################################################
# These integrals are obtained over spatial molecular orbitals. That means, for water, we have
# only reported the integrals over the 1s, 2s, and the 2p_x, 2p_y and 2p_z orbitals, without
# accounting for the spin component. To construct the full
# Hamiltonian, the integrals objects need to be expanded to include spin orbitals, i.e., 1s_alpha,
# 1s_beta etc. Assuming an interleaved convention for the order of spin orbitals, i.e.,
# |alpha, beta, alpha, beta, ...>, the following functions give the expanded one-electron and
# two-electron objects.

def transform_one(h_mo: np.ndarray) -> np.ndarray:
    """
    """
    n_so = 2 * h_mo.shape[0]
    h_so = np.zeros((n_so, n_so))

    alpha, beta = slice(0, n_so, 2), slice(1, n_so, 2)

    h_so[alpha, alpha] = h_mo
    h_so[beta, beta] = h_mo

    return h_so


def transform_two(g_mo: np.ndarray) -> np.ndarray:
    """
    """
    n_so = 2 * g_mo.shape[0]

    g_so = np.zeros((n_so, n_so, n_so, n_so))

    alpha = slice(0, n_so, 2)
    beta = slice(1, n_so, 2)

    g_so[alpha, alpha, alpha, alpha] = g_mo
    g_so[alpha, beta, beta, alpha] = g_mo
    g_so[beta, alpha, alpha, beta] = g_mo
    g_so[beta, beta, beta, beta] = g_mo

    return g_so


##############################################################################
# Note that the transformation must respect the spin-selection rules. For instance, the integral
# :math:`\langle \chi_p | \hat{h} | \chi_q \rangle = \langle \phi_i | \hat{h}^{\text{spatial}} | \phi_j \rangle \cdot \langle \sigma_p | \sigma_q \rangle`
# must be zero if the spins of the initial and final spin orbitals are different. For the
# one-electron integrals, only αα or ββ combinations have a non-zero
# value. Similarly, the two-electron operator 1/r_{12} is spin-independent and if we use the
# compact notation '1221' to represent the order in the two-electron integral, the non-zero
# combinations are: αααα, αββα, βααβ and ββββ. The integrals computed using molecular orbitals
# can be expanded to include spin orbitals using these combination rules. We now obtain the full
# electron integrals in the basis of spin orbitals.

one_so = transform_one(one_mo)
two_so = transform_two(two_mo)

##############################################################################
# Having the electron integrals objects, computing the Hamiltonian is straightforward. We simply
# loop over the elements of the tensors and multiply them with the fermionic operators with the
# indices matching those of the integral elements. For better performance, we can skip the zero
# integral elements and just obtain the operator indices for the non-zero integrals.

one_operators = qml.math.argwhere(abs(one_so) >= 1e-12)
two_operators = qml.math.argwhere(abs(two_so) >= 1e-12)

##############################################################################
# Now we construct the Hamiltonian as a FermiSentence object.

sentence = FermiSentence({FermiWord({}): core_constant[0]})

for o in one_operators:
    sentence.update({from_string(f'{o[0]}+ {o[1]}-'): one_so[*o]})

for o in two_operators:
    sentence.update({from_string(f'{o[0]}+ {o[1]}+ {o[2]}- {o[3]}-'): two_so[*o] / 2})

sentence.simplify()

##############################################################################
# Note that the order of indices for the fermionic creation and annihilation operators matches
# the order of indices in the integral objects. We finally map the fermionic Hamiltonian to the
# qubit bases and compute its ground-state energy, which should match the value -75.01562736 a.u.

h = qml.jordan_wigner(sentence)
qml.eigvals(qml.SparseHamiltonian(h.sparse_matrix(), wires=h.wires))

##############################################################################
# We can further verify this value by using the PennyLane function molecular_hamiltonian, which
# automatically constructs the Hamiltonian.

h = qml.qchem.molecular_hamiltonian(mol)[0]
qml.eigvals(qml.SparseHamiltonian(h.sparse_matrix(), wires=h.wires))

##############################################################################
# Chemists' notation
# ------------------
# This notation is commonly used by quantum chemistry software libraries such as PySCF. The two-
# electron integral tensor in this notation is defined as
#
# .. math::
#
#     (pq | rs) = \int dr_1 dr_2 \phi_p^*(r_1) \phi_q(r_1) \frac{1}{r_{12}} \phi_r^*(r_2) \phi_s(r_2).
#
# The corresponding Hamiltonian in second quantization is then defined as
#
# .. math::
#
#     H = \sum_{pq} (h_{pq} - \frac{1}{2} \sum_s h_{pssq}) a_p^{\dagger} a_q + \frac{1}{2} \sum_{pqrs} h_{pqrs} a_p^{\dagger} a_q a_r^{\dagger} a_s.
#
# Note that a one-body correction term should be included. Also note that the order of indices of
# the creation and annihilation operators in the second term matches the order of the two-electron
# integral coefficients. See the appendix for a full derivation of the Hamiltonian.
#
# Let's now build the Hamiltonian using electron integrals computed by PySCF.

from pyscf import gto, ao2mo, scf

mol_pyscf = gto.M(atom='''H -0.02111417 -0.00201087  0.;
                            O  0.83504162  0.45191733  0.;
                            H  1.47688065 -0.27300252  0.''')
rhf = scf.RHF(mol_pyscf)
energy = rhf.kernel()

one_ao = mol_pyscf.intor_symmetric('int1e_kin') + mol_pyscf.intor_symmetric('int1e_nuc')
two_ao = mol_pyscf.intor('int2e_sph')

one_mo = np.einsum('pi,pq,qj->ij', rhf.mo_coeff, one_ao, rhf.mo_coeff)
two_mo = ao2mo.incore.full(two_ao, rhf.mo_coeff)

core_constant = np.array([rhf.energy_nuc()])


##############################################################################
# These integrals are also obtained for the spatial orbitals and we need to expand them to include
# spin orbitals. To do that, we need to slightly upgrade our transform_two_interleaved function
# because the allowed spin combination in the chemists' notation, also denoted by '1122', are
# αααα, ααββ, ββαα and ββββ.

def transform_two(g_mo: np.ndarray, notation) -> np.ndarray:
    """
    """
    n = g_mo.shape[0]
    n_so = 2 * n

    g_so = np.zeros((n_so, n_so, n_so, n_so))

    alpha = slice(0, n_so, 2)
    beta = slice(1, n_so, 2)

    g_so[alpha, alpha, alpha, alpha] = g_mo
    g_so[beta, beta, beta, beta] = g_mo

    if notation == '1221':
        g_so[alpha, beta, beta, alpha] = g_mo
        g_so[beta, alpha, alpha, beta] = g_mo
        return g_so

    if notation == '1122':
        g_so[alpha, alpha, beta, beta] = g_mo
        g_so[beta, beta, alpha, alpha] = g_mo
        return g_so


##############################################################################
# We can compute the full integrals objects and add the one-body correction term

one_so = transform_one(one_mo)
two_so = transform_two(two_mo, '1122')

one_so_corrected = one_so - 0.5 * np.einsum('pssq->pq', two_so)

##############################################################################
# Constructing the Hamiltonian is now straightforward.

one_operators = qml.math.argwhere(abs(one_so_corrected) >= 1e-12)
two_operators = qml.math.argwhere(abs(two_so) >= 1e-12)

sentence = FermiSentence({FermiWord({}): core_constant[0]})

for o in one_operators:
    sentence.update({from_string(f'{o[0]}+ {o[1]}-'): one_so_corrected[*o]})

for o in two_operators:
    sentence.update({from_string(f'{o[0]}+ {o[1]}- {o[2]}+ {o[3]}-'): two_so[*o] / 2})

sentence.simplify()

##############################################################################
# Note that the order of the indices for the fermionic operators match those of the electron
# integral coefficients. Also note that the two-body operator has the order
# :math:`a^{\dagger} a a^{\dagger} a'. Let's now validate the Hamiltonian by computing the
# ground-state energy.

h = qml.jordan_wigner(sentence)
qml.eigvals(qml.SparseHamiltonian(h.sparse_matrix(), wires=h.wires))

##############################################################################
# Physicists' notation
# --------------------
# The two-electron integral tensor in this notation is defined as
#
# .. math::
#
#     \langle pq | rs \rangle = \int dr_1 dr_2 \phi_p^*(r_1) \phi_q^*(r_2) \frac{1}{r_{12}} \phi_r(r_1) \phi_s(r_2).
#
# The corresponding Hamiltonian in second quantization is then constructed as
#
# .. math::
#
#     H = \sum_{pq} h_{pq} a_p^{\dagger} a_q + \frac{1}{2} \sum_{pqrs} h_{pqrs} a_p^{\dagger} a_q^{\dagger} a_s a_r,
#
# It is important to note the order of the creation and annihilation
# operators in the second term, which is :math:`a_p^{\dagger} a_q^{\dagger} a_s a_r`. This order
# must be preserved when the Hamiltonian is constracted with the two-electron integral tensor
# represented in the physicists' notation.
#
# Let's now build this Hamiltonian step-by-step. There is not a commonly-used software library
# to compute integrals in this notation but we can easily convert integrals obtained in other
# notations. For instance, we use PySCF integrals and convert them to physicists' notation. We can
# do this in two different ways: converting the integrals in the spatial orbital basis or
# converting them in the spin orbital basis. Let's do both.
#
# The conversion can be done by the transformation the transformation
# :math:`(pq | rs) \to (pr | qs)` where we have swapped the :math:`1,2`
# indices. This transformation can be done with

one_mo = one_mo.copy()
two_mo = two_mo.transpose(0, 2, 1, 3)


##############################################################################
# We intentionally converted the integrals obtained for the spatial orbitals, so we need to expand
# them to include spin orbitals. To do that, we need to again slightly upgrade our
# transform_two_interleaved function to include the allowed spin combination in the physicists'
# notation, denoted by '1212', as αααα, αβαβ, βαβα and ββββ.

def transform_two(g_mo: np.ndarray, notation) -> np.ndarray:
    """
    """
    n = g_mo.shape[0]
    n_so = 2 * n

    g_so = np.zeros((n_so, n_so, n_so, n_so))

    alpha = slice(0, n_so, 2)
    beta = slice(1, n_so, 2)

    g_so[alpha, alpha, alpha, alpha] = g_mo
    g_so[beta, beta, beta, beta] = g_mo

    if notation == '1221':
        g_so[alpha, beta, beta, alpha] = g_mo
        g_so[beta, alpha, alpha, beta] = g_mo
        return g_so

    if notation == '1122':
        g_so[alpha, alpha, beta, beta] = g_mo
        g_so[beta, beta, alpha, alpha] = g_mo
        return g_so

    if notation == '1212':
        g_so[alpha, beta, alpha, beta] = g_mo
        g_so[beta, alpha, beta, alpha] = g_mo
        return g_so

##############################################################################
# The new tensor can then be used to construct the Hamiltonian

one_so = transform_one(one_mo)
two_so = transform_two(two_mo, '1212')

one_operators = qml.math.argwhere(abs(one_so) >= 1e-12)
two_operators = qml.math.argwhere(abs(two_so) >= 1e-12)

sentence = FermiSentence({FermiWord({}): core_constant[0]})

for o in one_operators:
    sentence.update({from_string(f'{o[0]}+ {o[1]}-'): one_so[*o]})

for o in two_operators:
    sentence.update({from_string(f'{o[0]}+ {o[1]}+ {o[3]}- {o[2]}-'): two_so[*o] / 2})

sentence.simplify()

##############################################################################
# Note the order of the creation and annihilation operator indices. Now we compute the ground
# state energy to validate the Hamiltonian.

h = qml.jordan_wigner(sentence)
qml.eigvals(qml.SparseHamiltonian(h.sparse_matrix(), wires=h.wires))


##############################################################################
# Integral conversion
# -------------------
# The two-electron integrals computed with one convention can be easily converted to the other
# conventions as we already did for the chemist to physicist conversion. Such conversions allow
# constructing different representations of the molecular Hamiltonian without re-calculating the
# integrals. Here we provide the conversion rules for all three conventions.

def convert_integrals(two_body, in_notation, out_notation):
    """  """
    if in_notation == out_notation:
        return two_body

    if in_notation == "chemist" and out_notation == "physicist":
        return two_body.transpose(0, 2, 1, 3)

    if in_notation == "chemist" and out_notation == "quantum":
        return two_body.transpose(0, 2, 3, 1)

    if in_notation == "quantum" and out_notation == "chemist":
        return two_body.transpose(0, 3, 1, 2)

    if in_notation == "quantum" and out_notation == "physicist":
        return two_body.transpose(0, 1, 3, 2)

    if in_notation == "physicist" and out_notation == "chemist":
        return two_body.transpose(0, 2, 1, 3)

    if in_notation == "physicist" and out_notation == "quantum":
        return two_body.transpose(0, 1, 3, 2)


##############################################################################
# We can also create a versatile function that computes the Hamiltonian for each convention.

def hamiltonian(one_body, two_body, notation, cutoff=1e-12):
    if notation == "chemist":
        one_body = one_body - 0.5 * np.einsum('pssq->pq', two_body)

    op_one = qml.math.argwhere(abs(one_body) >= cutoff)
    op_two = qml.math.argwhere(abs(two_body) >= cutoff)

    sentence = FermiSentence({FermiWord({}): core_constant[0]})

    for o in op_one:
        sentence.update({FermiWord({(0, o[0]): "+", (1, o[1]): "-"}): one_body[*o]})

    if notation == "quantum":
        for o in op_two:
            sentence.update({FermiWord(
                {(0, o[0]): "+", (1, o[1]): "+", (2, o[2]): "-", (3, o[3]): "-"}): two_body[*o] / 2})

    if notation == "chemist":
        for o in op_two:
            sentence.update({FermiWord(
                {(0, o[0]): "+", (1, o[1]): "-", (2, o[2]): "+", (3, o[3]): "-"}): two_body[*o] / 2})

    if notation == "physicist":
        for o in op_two:
            sentence.update({FermiWord(
                {(0, o[0]): "+", (1, o[1]): "+", (2, o[3]): "-", (3, o[2]): "-"}): two_body[*o] / 2})

    sentence.simplify()

    return qml.jordan_wigner(sentence)


##############################################################################
# We now have all the necessary tools in our arsenal to convert the integrals to a desired
# notation and construct the corresponding Hamiltonian. Let's look at a few examples.
#
# First we convert the integrals obtained with PennyLane in 'quantum' notation to the 'chemist'
# notation and compute the corresponding Hamiltonian. This conversion can be done by the
# transformation :math:`<pq | rs> \to <pq | sr> \to <ps | qr)`. We
# have first swapped the :math:`2,3` indices to go from the quantum notation to the physicist
# notation and then swap :math:`1,2` indices to get the integrals in the chemist
# notation. This transformation can be done with :math:`transpose(0, 3, 1, 2)`.
# Let's now use our versatile functions for the full workflow.

core_constant, one_mo, two_mo = qml.qchem.electron_integrals(mol)()
one_so = transform_one(one_mo)
two_so = transform_two(two_mo, '1221')

two_so_converted = convert_integrals(two_so, 'quantum', 'physicist')

h = hamiltonian(one_so, two_so_converted, 'physicist')
qml.eigvals(qml.SparseHamiltonian(h.sparse_matrix(), wires=h.wires))

##############################################################################
# You can verify that the Hamiltonian and its ground state energy are correct. We now do convert
# to the physicist notation.

two_so_converted = convert_integrals(two_so, 'quantum', 'chemist')

h = hamiltonian(one_so, two_so_converted, 'chemist')
qml.eigvals(qml.SparseHamiltonian(h.sparse_matrix(), wires=h.wires))

##############################################################################
# We can have a similar example for converting from the chemist notation to the quantum notation.

one_mo = np.einsum('pi,pq,qj->ij', rhf.mo_coeff, one_ao, rhf.mo_coeff)
two_mo = ao2mo.incore.full(two_ao, rhf.mo_coeff)
core_constant = np.array([rhf.energy_nuc()])

one_so = transform_one(one_mo)
two_so = transform_two(two_mo, '1122')

two_so_converted = convert_integrals(two_so, 'chemist', 'quantum')

h = hamiltonian(one_so, two_so_converted, 'quantum')
qml.eigvals(qml.SparseHamiltonian(h.sparse_matrix(), wires=h.wires))

##############################################################################
# Similarly, we go from chemist notation to the physicist notation

two_so_converted = convert_integrals(two_so, 'chemist', 'physicist')

h = hamiltonian(one_so, two_so_converted, 'physicist')
qml.eigvals(qml.SparseHamiltonian(h.sparse_matrix(), wires=h.wires))

##############################################################################
# The other converstions follow a similar logic.
#
# References
# ----------
#
# .. [#szabo1996]
#
#     Attila Szabo, Neil S. Ostlund, "Modern Quantum Chemistry: Introduction to Advanced Electronic
#     Structure Theory". Dover Publications, 1996.
