r"""How to build compressed double factorized Hamiltonians
----------------------------------------------------------
"""

import pennylane as qml
import numpy as np

######################################################################
# Double factorization methods
# ----------------------------
# 
# The molecular Hamilotnians for a system with :math:`N` spatial orbitals are typically written as the
# sum of following one-body and two-body terms that scale as :math:``O(N^2)`` and :math:``O(N^4)``,
# respectively:
# 
# .. math::  H = \sum_{\sigma, pq} h_{pq} a^\dagger_{\sigma, p} a_{\sigma, q} + \frac12 \sum_{\sigma \tau, pqrs} h_{pqrs} a^\dagger_{\sigma, p} a^\dagger_{\tau, r} a_{\\tau, s} a_{\\sigma, q} + \text{constant} \tag{1}
# 

symbols  = ['H', 'H', 'H', 'H']
geometry = qml.numpy.array([[0.0, 0.0,  0.0],
                            [0.0, 0.0, -0.1],
                            [0.0, 0.0, +0.1],
                            [0.0, 0.0, +0.2]], requires_grad=False)
mol = qml.qchem.Molecule(symbols, geometry)
core, one, two = qml.qchem.electron_integrals(mol)()
one.shape, two.shape

######################################################################
# The above is referred in common terminology as the physicist notation. We can instead represent the
# above in the `chemist notation <http://vergil.chemistry.gatech.edu/notes/permsymm/permsymm.pdf>`__
# as
# 

one, two = qml.qchem.factorization._chemist_transform(one, two, spatial_basis=True)
two.shape

######################################################################
# Double factorization methods are Hamiltonian manipulation techniques that allow for an alternate
# representation that can be implemented more efficiently on the hardware. It is based on decomposing
# the above representation to orthonormal matrices :math:``U`` and symmetric matrices :math:``Z`` such
# that
# 
# .. math::  V_{ijkl} \approx \sum_r^R \sum_{pq} U_{ip}^{(r)} U_{jp}^{(r)} Z_{pq}^{(r)} U_{kq}^{(r)} U_{lq}^{(r)} \tag{2}
# 
# We can do this in pennylane by obtaining the two-electron integrals as shown below and then
# utilizing the ``qml.qchem.factorize`` method, which performs an eigenvalue or Cholesky decomposition
# to obtain symmetric matrices :math:``L^{(r)}`` such that
# :math:``V_{ijkl} = \sum_r^R L_{ij}^{(r)} L_{kl}^{(r) T}``, where core and leaf tensors are obtained
# by further diagonalizing each matrix :math:``L^{(r)}``, and truncating its eigenvalues (and the
# corresponding eigenvectors) at a threshold error.
# 

factors, cores, leaves = qml.qchem.factorize(two, 1e-5, 1e-5)
assert np.allclose(np.tensordot(factors, factors, axes=([0], [0])), two, atol=1e-4)

factors.shape[0], len(cores), len(leaves)

######################################################################
# As a first optional step, we begin by employing the block-invariant symmetry shift (BLISS) method
# [``arXiv:2304.13772 <https://arxiv.org/pdf/2304.13772>``\ \_] to decrease the one-norm and the
# spectral range of the above Hamiltonian, allowing a more efficient compression with double
# factorization.
# 

s_core, s_one, s_two = qml.qchem.symmetry_shift(core, one, two, n_elec=mol.n_electrons)
s_one.shape, s_two.shape

######################################################################
# As we see above, the performed decomposition is exact and is called explicit double factorization.
# However, in many cases, we could truncate the ``R`` such that the approximate error
# :math:`\epsilon \geq || H - H^\prime||`, and limit the the number of terms :math:`O(N)` from
# :math:`O(N^2)`. One straightforward way is to perform this truncation directly, other one is by
# starting with randomized :math:`O(N)` orthornormal and symmetric matrices and optimizing them based
# on the following cost function :math:``\mathcal{L}`` in a greedy layered-wise manner:
# 
# .. math::  \mathcal{L}(U, Z) = \frac{1}{2} \bigg|V_{ijkl} - \sum_r^R \sum_{pq} U_{ip}^{(r)} U_{jp}^{(r)} Z_{pq}^{(r)} U_{kq}^{(r)} U_{lq}^{(r)}\bigg|_{\text{F}} + \rho \sum_r^R \sum_{pq} \bigg|Z_{pq}^{(r)}\bigg|^{\gamma}, \tag{2}
# 

two_body_factors, two_body_cores, two_body_leaves = qml.qchem.factorize(s_two, 1e-4, cholesky=True, compressed=True)
assert np.allclose(
    np.einsum("tpk,tqk,tkl,trl,tsl->pqrs", two_body_leaves, two_body_leaves, two_body_cores, two_body_leaves, two_body_leaves),
    s_two,
    atol=1e-2
)

factors.shape[0], len(two_body_cores), len(two_body_leaves)

######################################################################
# We can obtain a similar deocmposition for one-body terms:
# 

def factorize_onebody(h1e):
    """Obtain the matrices U (special orthogonal) and Z (symmetric)"""
    # Diagonalize the one-electron integral matrix
    eigvals, eigvecs = np.linalg.eigh(h1e)
    U0 = np.expand_dims(eigvecs, axis = 0)
    Z0 = np.expand_dims(np.diag(eigvals), axis = 0)

    return U0, Z0

one_body_leaves, one_body_cores = factorize_onebody(s_one)

######################################################################
# Trotter Evolution
# -----------------
# 
# The Suzuki-Trotter product formula provides a method to approximate the matrix exponential of
# Hamiltonian expressed as a linear combination of operands which in general do not commute. For a
# Hamiltonian :math:``H = \Sigma^{N}_{j=0} O_{j}``, a :math:``m``\ th order, :math:``n``-step
# Suzuki-Trotter approximation is then defined as:
# 
# .. math:: e^{iHt} \approx \left [S_{m}(t / n)  \right ]^{n}, \quad \text{where}
# 
# .. math::
# 
# 
#    S_{m}(t) = S_{m-2}(p_{m}t)^{2} \cdot S_{m-2}((1-4p_{m})t) \cdot S_{m-2}(p_{m}t)^{2}, \quad \text{with}\ \ S_{1}(t) = \Pi_{j=0}^{N} \ e^{i t O_{j}}
# 

######################################################################
# For a CDF Hamiltonian defined as the above, we can use the following function to implement the
# trotter evolution:
# 

import itertools as it

def CDFTrotter(nuc_core, one_body_cdf, two_body_cdf, time, n):
    """Implements a first order Trotterized circuit for a Hamiltonian expressed as a CDF"""

    norbs = np.shape(one_body_cdf[0])[1]
    cores = np.concatenate((one_body_cdf[1], two_body_cdf[1]), axis=0)
    leaves = np.concatenate((one_body_cdf[0], two_body_cdf[0]), axis=0)
    btypes = np.array([1] * len(one_body_cdf[0]) + [2] * len(two_body_cdf[0]))

    norms = np.linalg.norm(cores, axis=(1, 2))
    ranks = np.argsort(norms)

    step = time / n
    gphase_vals = [np.sum(cores), np.trace(cores)]
    cores, leaves, btypes = cores[ranks], leaves[ranks], btypes[ranks]

    init_u = np.eye(norbs)
    qml.BasisRotation(unitary_matrix=init_u, wires=range(0, 2*norbs, 2))
    qml.BasisRotation(unitary_matrix=init_u, wires=range(1, 2*norbs, 2))

    for _ in range(n):
        for core, leaf, btype in zip(cores, leaves, btypes):
            qml.BasisRotation(unitary_matrix=init_u @ leaf, wires=range(0, 2*norbs, 2))
            qml.BasisRotation(unitary_matrix=init_u @ leaf, wires=range(1, 2*norbs, 2))
            init_u = leaf.T

            if btype == 1:
                for wire, cval in enumerate(np.diag(core)):
                    for sigma in [0, 1]:
                        qml.RZ(-step * cval, wires=2*wire + sigma)
                qml.GlobalPhase(step * np.sum(core), wires=range(2*norbs))

            else:
                for odx1, odx2 in it.permutations(range(norbs), r=2):
                    cval = core[odx1, odx2]
                    for sigma, tau in it.permutations(range(2), r=2):
                       qml.MultiRZ(step * cval / 4., wires=[2*odx1+sigma, 2*odx2+tau])
                qml.GlobalPhase(-step / 2. * (np.sum(core) - np.trace(core) / 2), wires=range(2*norbs))

    qml.GlobalPhase(nuc_core[0] * time, wires=range(2*norbs))

######################################################################
# We can then do the following
# 

time = 1
n = 10
one_body_cdf, two_body_cdf = (one_body_cores, one_body_leaves), (two_body_cores, two_body_leaves)
CDFTrotter(core, one_body_cdf, two_body_cdf, time, n)

######################################################################
# We can test this by comparing it with the PennyLane’s qml.Trotter
# 

H, num_wires = qml.qchem.molecular_hamiltonian(mol)
hf_state = qml.qchem.hf_state(electrons=mol.n_electrons, orbitals=num_wires)

dev = qml.device("default.qubit")
@qml.qnode(dev)
def my_circ1():
    qml.BasisState(hf_state, wires=range(num_wires))
    CDFTrotter(core, one_body_cdf, two_body_cdf, time=1.0, n=1)
    return qml.state()

H, num_wires = qml.qchem.molecular_hamiltonian(mol)

dev = qml.device("default.qubit")
@qml.qnode(dev)
def my_circ2():
    qml.BasisState(hf_state, wires=range(num_wires))
    qml.TrotterProduct(H, time=1.0, order=1, n=1)
    return qml.state()

qml.math.fidelity_statevector(my_circ1(), my_circ2())

######################################################################
# As we see, this gives the expected result.
# 

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
# performance in molecular simulations, making it a promising approach for fault-tolerant and
# variational quantum algorithms.
# 

######################################################################
# About the author
# ----------------
# # .. include:: ../_static/authors/utkarsh_azad.txt
