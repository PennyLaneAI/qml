r"""

The Hamiltonian :math:`H` of molecular systems with :math:`N` orbitals in the second-quantized
form can be expressed as a sum of one-body (:math:`h_{pq}`) and two-body (:math:`g_{pqrs}`) terms
as follows:

.. math::  H = \text{core} + \sum_{\sigma, pq} h_{pq} a^\dagger_{\sigma, p} a_{\sigma, q} + \frac{1}{2} \sum_{\sigma \tau, pqrs} h_{pqrs} a^\dagger_{\sigma, p} a^\dagger_{\tau, q} a_{\tau, r} a_{\sigma, s},

where :math:`a^\dagger` (:math:`a`) is the creation (annihilation) operator,
:math:`\sigma \in {\uparrow, \downarrow}` represents the spin, and :math:`p, q, r, s` are the
orbital indices. In PennyLane, we can obtain the :math:`h_{pq}` and :math:`g_{pqrs}` using the
:func:`~pennylane.qchem.electron_integrals`:
"""

import pennylane as qml

symbols  = ['H', 'H', 'H', 'H']
geometry = qml.math.array([[0.0, 0.0, -0.2],
                           [0.0, 0.0, -0.1],
                           [0.0, 0.0, +0.1],
                           [0.0, 0.0, +0.2]])

mol = qml.qchem.Molecule(symbols, geometry)
nuc_core, one_body, two_body = qml.qchem.electron_integrals(mol)()

print(f"One-body and two-body tensor shapes: {one_body.shape}, {two_body.shape}")

######################################################################
# In the above expression, the two-body terms can be rearranged to :math:`V_{pqrs}` in the
# `chemist notation <http://vergil.chemistry.gatech.edu/notes/permsymm/permsymm.pdf>`_ to rewrite
# the Hamiltonian as :math:`H_{\text{C}}` with :math:`T_{pq} = h_{pq} - 0.5 \sum_{s} g_{pssq}`:
#
# .. math::  H_{\text{C}} = \text{core} + \sum_{\sigma, pq} T_{pq} a^\dagger_{\sigma, p} a_{\sigma, q} + \sum_{\sigma \tau, pqrs} V_{pqrs} a^\dagger_{\sigma, p} a_{\sigma, q} a_{\tau, r} a_{\tau, s}.
#
# We can easily obtain the modified terms for doing this using:
#

two_chem = 0.5 * qml.math.swapaxes(two_body, 1, 3) # V_pqrs
one_chem = one_body - qml.math.einsum("prrs", two_chem) # T_pq

######################################################################
# Constructing double factorized Hamiltonians
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# A key feature of the above representation is that the two-body terms can be factorized into
# sum of low-rank terms, which can be used to efficiently simulate the Hamiltonian. Hence, the
# double factorization methods can be described as Hamiltonian manipulation techniques based on
# decomposing the :math:`V_{pqrs}` to orthonormal core tensors (:math:`Z`) and symmetric leaf
# tensors (:math:`U`) tensors such that:
#
# .. math::  V_{pqrs} \approx \sum_t^T \sum_{ij} U_{pi}^{(t)} U_{pj}^{(t)} Z_{ij}^{(t)} U_{qk}^{(t)} U_{ql}^{(t)}
#
# We can do this as shown below using the :func:`~pennylane.qchem.factorize` method,
# which performs an eigenvalue or Cholesky decomposition to obtain the symmetric tensors
# :math:``L^{(t)}``. This decomposition is called the *explicit* double factorization as it exact,
# i.e, :math:`V_{pqrs} = \sum_t^T L_{pq}^{(t)} L_{rs}^{(t) {\dagger}}`, where the core and leaf
# tensors are obtained by further diagonalizing each term :math:``L^{(t)}``:
#

factors, _, _ = qml.qchem.factorize(two_chem, cholesky=True)
approx_two_chem = qml.math.tensordot(factors, factors, axes=([0], [0]))

assert qml.math.allclose(approx_two_chem, two_chem, atol=1e-5)
print("Shapes of the factors: ", factors.shape)

######################################################################
# We can further improve the above factorization by employing the block-invariant symmetry shift
# (BLISS) method [`arXiv:2304.13772 <https://arxiv.org/pdf/2304.13772>`_] to decrease the one-norm
# and the spectral range of the above Hamiltonian:
#

from pennylane.resource import DoubleFactorization as DF

core_shift, one_shift, two_shift = qml.qchem.symmetry_shift(nuc_core, one_chem, two_chem, n_elec=mol.n_electrons)

norm_shift = DF(one_chem, two_chem, chemist_notation=True).lamb - DF(one_shift, two_shift, chemist_notation=True).lamb
print(f"Decrease in one-norm: {norm_shift}")

######################################################################
# Moreover, in many practical cases, the number of terms :math:`T` in the above
# factorization can be truncated to give :math:`H^\prime`, such that the approximation error
# :math:`\epsilon \geq ||H_{\text{C}} - H_{\text{C}}^\prime||`. This limits the number of terms
# :math:`O(N)` from :math:`O(N^2)` and is referred to as the *compressed* double factorization.
# One way to perform this is to directly truncate the factorization with a threshold, while another
# is to start with random :math:`O(N)` orthornormal and symmetric tensors and optimizing them based
# on the following cost function :math:``\mathcal{L}`` in a greedy layered-wise manner:
#
# .. math::  \mathcal{L}(U, Z) = \frac{1}{2} \bigg|V_{pqrs} - \sum_t^T \sum_{ij} U_{pi}^{(t)} U_{pj}^{(t)} Z_{ij}^{(t)} U_{qk}^{(t)} U_{ql}^{(t)}\bigg|_{\text{F}} + \rho \sum_t^T \sum_{ij} \bigg|Z_{ij}^{(t)}\bigg|^{\gamma}.
#
# These can be done by using the ``tol_factor`` and ``compressed=True`` keyword arguments,
# respectively, in the :func:`~pennylane.qchem.factorize` method as shown below:
#

_, two_body_cores, two_body_leaves = qml.qchem.factorize(two_shift, tol_factor=1e-2, cholesky=True, compressed=True)

approx_two_shift = qml.math.einsum(
    "tpk,tqk,tkl,trl,tsl->pqrs", two_body_leaves, two_body_leaves, two_body_cores, two_body_leaves, two_body_leaves
)
assert qml.math.allclose(approx_two_shift, two_shift, atol=1e-2)

print(f"Shape of two-body core and leaf tensors: {two_body_cores.shape, two_body_leaves.shape}")

######################################################################
# We can obtain a similar decomposition for the one-body terms in terms of orthornormal and
# symmetric tensors to express the entire Hamiltonian in terms of the core and leaf tensors.
# This can be done using the:
#

one_body_eigvals, one_body_eigvecs = qml.math.linalg.eigh(one_shift)

one_body_cores = qml.math.expand_dims(qml.math.diag(one_body_eigvals), axis = 0)
one_body_leaves = qml.math.expand_dims(one_body_eigvecs, axis = 0)

print(f"Shape of one-body core and leaf tensors: {two_body_cores.shape, two_body_leaves.shape}")

######################################################################
# Simulating double factorized Hamiltonians
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# In general, the Suzuki-Trotter product formula provides a method to approximate the evolution
# operator :math:`e^{iHt}` for time :math:`t` with symmetrized products :math:`S_m(t/n)` of order
# :math:`m \in [1, 2, 4, 6, \ldots]`, repeated :math:`n` times. This can be easily implemented
# using the :func:`~pennylane.TrotterProduct` method, but its complexity scales exponentially with
# the number of terms in the Hamiltonian, making it inefficient for larger Hamiltonians.
#
# This could be solved to a great extent by employing the compressed double factorized form of the
# Hamiltonian to perform the above approximation of the evolution operator. While doing this is
# not directly supported in PennyLane in the form of a template, we can still implement the
# first-order Trotter approximation using the following :func:`CDFTrotter` function that uses
# the compressed double factorized form of the Hamiltonian:
#

import itertools as it

def CDFTrotter(nuc_core, one_body_cdf, two_body_cdf, time, steps):
    """Implements a first order Trotterized circuit for a Hamiltonian expressed as a CDF"""

    norbs = qml.math.shape(one_body_cdf[0])[1]
    cores = qml.math.concatenate((one_body_cdf[0], two_body_cdf[0]), axis=0)
    leaves = qml.math.concatenate((one_body_cdf[1], two_body_cdf[1]), axis=0)
    btypes = qml.math.array([1] * len(one_body_cdf[0]) + [2] * len(two_body_cdf[0]))

    norms = qml.math.linalg.norm(cores, axis=(1, 2))
    ranks = qml.math.argsort(norms)

    step = time / steps
    cores, leaves, btypes = cores[ranks], leaves[ranks], btypes[ranks]

    basis_mat = qml.math.eye(norbs)
    qml.BasisRotation(unitary_matrix=basis_mat, wires=range(0, 2*norbs, 2))
    qml.BasisRotation(unitary_matrix=basis_mat, wires=range(1, 2*norbs, 2))

    for _ in range(steps):
        for core, leaf, btype in zip(cores, leaves, btypes):
            qml.BasisRotation(unitary_matrix=basis_mat @ leaf, wires=range(0, 2*norbs, 2))
            qml.BasisRotation(unitary_matrix=basis_mat @ leaf, wires=range(1, 2*norbs, 2))
            basis_mat = leaf.T

            if btype == 1: # gates for one-body term
                for wire, cval in enumerate(qml.math.diag(core)):
                    for sigma in [0, 1]:
                        qml.RZ(-step * cval, wires=2*wire + sigma)
                qml.GlobalPhase(step * qml.math.sum(core), wires=range(2*norbs))

            else: # gates for two-body term
                for odx1, odx2 in it.product(range(norbs), repeat=2):
                    cval = core[odx1, odx2]
                    for sigma, tau in it.product(range(2), repeat=2):
                        if (odx1 != odx2 or sigma != tau):
                           qml.MultiRZ(step * cval / 4., wires=[2*odx1+sigma, 2*odx2+tau])
                qml.GlobalPhase(-step / 2. * (
                    qml.math.sum(core) - qml.math.trace(core) / 2), wires=range(2*norbs))

    qml.GlobalPhase(nuc_core[0] * time, wires=range(2*norbs))

######################################################################
# We can use this to simulate the evolution of the Hamiltonian for a given number of steps as shown below:
#

num_wires, time = 2 * mol.n_orbitals, 1.0
one_body_cdf, two_body_cdf = (one_body_cores, one_body_leaves), (two_body_cores, two_body_leaves)

hf_state = qml.qchem.hf_state(electrons=mol.n_electrons, orbitals=num_wires)

dev = qml.device("lightning.qubit", wires=num_wires)
@qml.qnode(dev)
def cdf_circuit(steps):
    qml.BasisState(hf_state, wires=range(num_wires))
    CDFTrotter(core_shift, one_body_cdf, two_body_cdf, time, steps=steps)
    return qml.state()

circuit_state = cdf_circuit(steps=20)

######################################################################
# Moreover, we can test it by evolving the Hartree-Fock state analytically ourselves and testing the fidelity:
# 

from scipy.linalg import expm

init_state_vec = qml.math.array([1] + [0] * (2 ** num_wires - 1))
hf_state_vec = qml.matrix(qml.BasisState(hf_state, wires=range(num_wires))) @ init_state_vec

H = qml.qchem.molecular_hamiltonian(mol)[0]
evolved_state = expm(1j * qml.matrix(H) * time) @ hf_state_vec

print(f"Fidelity of two states: {qml.math.fidelity_statevector(circuit_state, evolved_state)}")

######################################################################
# Conclusion
# ----------
# 
# Compressed Double Factorization (CDF)-based and standard Trotter simulation both aim to efficiently
# simulate quantum dynamics by approximating the Hamiltonian evolution, but they differ in how they
# reduce computational cost. Standard simulation approach can be inefficient when dealing with dense
# molecular Hamiltonians, where the number of Pauli terms scales exponentially. CDF-based Trotter, on
# the other hand, first applies a low-rank factorization to the two-electron integrals, significantly
# reducing the number of terms. This leads to fewer sampled operators per time step, reducing the
# overall circuit depth and improving feasibility for near-term quantum devices. While standard
# Trotter is more general, CDF-based Trotter leverages double factroization to achieve superior
# performance in molecular simulations, making it a promising approach for fault-tolerant algorithms.
# 

######################################################################
# About the author
# ----------------
# # .. include:: ../_static/authors/utkarsh_azad.txt
