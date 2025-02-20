r"""How to build compressed double factorized Hamiltonians
==========================================================

The Hamiltonian :math:`H` of molecular systems with :math:`N` orbitals in the second-quantized
form can be expressed as a sum of one-body (:math:`h_{pq}`) and two-body (:math:`g_{pqrs}`) terms
as follows:

.. math::  H = \mu + \sum_{\sigma, pq} h_{pq} a^\dagger_{\sigma, p} a_{\sigma, q} + \frac{1}{2} \sum_{\sigma \tau, pqrs} h_{pqrs} a^\dagger_{\sigma, p} a^\dagger_{\tau, q} a_{\tau, r} a_{\sigma, s},

where :math:`a^\dagger` (:math:`a`) is the creation (annihilation) operator, :math:`\mu` is the
nuclear repulsion energy constant, :math:`\sigma \in {\uparrow, \downarrow}` represents the spin,
and :math:`p, q, r, s` are the orbital indices. In PennyLane, we can obtain the :math:`\mu`,
:math:`h_{pq}` and :math:`g_{pqrs}` using the :func:`~pennylane.qchem.electron_integrals` function:
"""

import pennylane as qml

symbols = ["H", "H", "H", "H"]
geometry = qml.math.array([[0., 0., -0.2], [0., 0., -0.1], [0., 0., 0.1], [0., 0., 0.2]])

mol = qml.qchem.Molecule(symbols, geometry)
nuc_core, one_body, two_body = qml.qchem.electron_integrals(mol)()

print(f"One-body and two-body tensor shapes: {one_body.shape}, {two_body.shape}")

######################################################################
# In the above expression, the two-body terms can be rearranged to :math:`V_{pqrs}` in the
# `chemist notation <http://vergil.chemistry.gatech.edu/notes/permsymm/permsymm.pdf>`_ to rewrite
# the above Hamiltonian as :math:`H_{\text{C}}` with :math:`T_{pq} = h_{pq} - 0.5 \sum_{s} g_{pssq}`:
#
# .. math::  H_{\text{C}} = \mu + \sum_{\sigma, pq} T_{pq} a^\dagger_{\sigma, p} a_{\sigma, q} + \sum_{\sigma \tau, pqrs} V_{pqrs} a^\dagger_{\sigma, p} a_{\sigma, q} a^\dagger_{\tau, r} a_{\tau, s}.
#
# We can easily obtain the modified one-body and two-body tensors for doing this with:
#

two_chem = 0.5 * qml.math.swapaxes(two_body, 1, 3)  # V_pqrs
one_chem = one_body - qml.math.einsum("prrs", two_chem)  # T_pq

######################################################################
# Constructing double factorized Hamiltonians
# -------------------------------------------
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
# :math:`L^{(t)}`. This decomposition is called the *explicit* double factorization as it exact,
# i.e, :math:`V_{pqrs} = \sum_t^T L_{pq}^{(t)} L_{rs}^{(t) {\dagger}}`, where the core and leaf
# tensors are obtained by further diagonalizing each term :math:`L^{(t)}`. In PennyLane, we can
# truncate the resulting terms by discarding the ones with individual contributions below a
# specified threshold by using the ``tol_factor`` keyword argument:
#

factors, _, _ = qml.qchem.factorize(two_chem, cholesky=True, tol_factor=1e-5)
print("Shape of the factors: ", factors.shape)

approx_two_chem = qml.math.tensordot(factors, factors, axes=([0], [0]))
assert qml.math.allclose(approx_two_chem, two_chem, atol=1e-5)

######################################################################
# We can further improve the above factorization by employing the block-invariant symmetry
# shift (BLISS) [#bliss]_ to decrease the one-norm and the spectral range of the above
# Hamiltonian using the :func:`~pennylane.qchem.symmetry_shift` method:
#

from pennylane.resource import DoubleFactorization as DF

core_shift, one_shift, two_shift = qml.qchem.symmetry_shift(
    nuc_core, one_chem, two_chem, n_elec = mol.n_electrons
)

norm_shift = (
    DF(one_chem, two_chem, chemist_notation=True).lamb
    - DF(one_shift, two_shift, chemist_notation=True).lamb
)
print(f"Decrease in one-norm: {norm_shift}")

######################################################################
# Moreover, in many practical scenarios, the number of terms :math:`T` in the above
# factorization can be truncated to give :math:`H^\prime`, such that the approximation error
# :math:`||H_{\text{C}} - H^\prime||` remains below a desired threshold. This is referred to as
# the *compressed* double factorization, as it reduces the number of terms in the factorization
# of the two-body term to :math:`O(N)` from :math:`O(N^2)`. One possible way to do this is to
# directly truncate the factorization with a threshold error tolerance, while another way is
# to begin with random :math:`O(N)` orthornormal and symmetric tensors and optimizing them based
# on the following cost function :math:`\mathcal{L}` in a greedy layered-wise manner:
#
# .. math::  \mathcal{L}(U, Z) = \frac{1}{2} \bigg|V_{pqrs} - \sum_t^T \sum_{ij} U_{pi}^{(t)} U_{pj}^{(t)} Z_{ij}^{(t)} U_{qk}^{(t)} U_{ql}^{(t)}\bigg|_{\text{F}} + \rho \sum_t^T \sum_{ij} \bigg|Z_{ij}^{(t)}\bigg|^{\gamma},
#
# where :math:`|\cdot|_{\text{F}}` computes the Frobenius norm, :math:`\rho`
# is a constant scaling factor, and :math:`|\cdot|^\gamma` specifies the optional
# L1 and L2 regularization [#cdf]_. In PennyLane, these can be achieved by using
# the ``compressed=True`` and ``regularization`` keyword arguments, respectively,
# in the :func:`~pennylane.qchem.factorize` method as shown below:
#

_, two_body_cores, two_body_leaves = qml.qchem.factorize(
    two_shift, tol_factor=1e-2, cholesky=True, compressed=True
)
print(f"Two-body tensors' shape: {two_body_cores.shape, two_body_leaves.shape}")

approx_two_shift = qml.math.einsum(
    "tpk,tqk,tkl,trl,tsl->pqrs",
    two_body_leaves, two_body_leaves, two_body_cores, two_body_leaves, two_body_leaves
)
assert qml.math.allclose(approx_two_shift, two_shift, atol=1e-2)

######################################################################
# We can clearly see that the number of terms in the factorization decreases from
# :math:`10` to :math:`6` in the above example, which is a significant reduction. Next,
# we can also decompose the one-body terms using the orthornormal and symmetric tensors.
# This can be done by first obtaining the one-body correction based on the above two-body
# terms and then performing an eigenvalue decomposition on the corrected one-body terms:
#

two_core_prime = (qml.math.eye(mol.n_orbitals) * two_body_cores.sum(axis=-1)[:, None, :])
one_body_extra = qml.math.einsum(
    'tpk,tkk,tqk->pq', two_body_leaves, two_core_prime, two_body_leaves
)

one_body_eigvals, one_body_eigvecs = qml.math.linalg.eigh(one_shift + one_body_extra)
one_body_cores = qml.math.expand_dims(qml.math.diag(one_body_eigvals), axis=0)
one_body_leaves = qml.math.expand_dims(one_body_eigvecs, axis=0)

print(f"One-body tensors' shape: {two_body_cores.shape, two_body_leaves.shape}")

######################################################################
# We can now express the entire Hamiltonian as sum of the products of core and leaf tensors
#
# .. math:: H_{\text{CDF}} = \mu + \sum_{\sigma} U^{(0)}_{\sigma} \left( \sum_{p} Z^{(0)}_{p} a^\dagger_{\sigma, p} a_{\sigma, p} \right) tU_{\sigma}^{(0)\ \dagger} + \sum_t^T \sum_{\sigma, \tau} U_{\sigma, \tau}^{(t)} \left( \sum_{pq} Z_{pq}^{(t)} \right) U_{\sigma, \tau}^{(t)\ \dagger},
#
# and specify each term in the above summation for a Hamiltonian in the double factorized
# form as ``nuc_core_cdf`` (:math:`\mu`), ``one_body_cdf`` (:math:`Z^{(0)}, U^{(0)}`) and
# ``two_body_cdf`` (:math:`Z^{(t)}, U^{(t)}`):
#

nuc_core_cdf = core_shift[0]
one_body_cdf = (one_body_cores, one_body_leaves)
two_body_cdf = (two_body_cores, two_body_leaves)

######################################################################
# The above representation allows obtaining the Hamiltonian in the qubit basis via
# Jordan-Wigner transformation, which uses :math:`a_p^\dagger a_p = n_p = 0.5 * (1 - z_p)`.
# One can obtain the coefficients and observables of such a Hamiltonian with
# :func:`~pennylane.qchem.basis_rotation`, which automatically accounts for the spin.
# However, to be efficient when simulating these Hamiltonians, we can account for the
# spin during the circuit construction itself, which we will see in the next section.
#

######################################################################
# Simulating double factorized Hamiltonians
# -----------------------------------------
#
# In general, the Suzuki-Trotter product formula provides a method to approximate the evolution
# operator :math:`e^{iHt}` for a time :math:`t` with symmetrized products :math:`S_m` defined
# for an order :math:`m \in [1, 2, 4, \ldots, 2k \in \mathbb{N}]` and repeated multiple times
# [#trotter]_. This can be easily implemented using the :class:`~.pennylane.TrotterProduct`
# operation that defines those products recursively for a given number of steps and therefore
# leads to an exponential scaling in its complexity with the number of terms in the Hamiltonian,
# making it inefficient for larger system sizes.
#
# Such a scaling behaviour could be managed to a great extent by working with the compressed double
# factorized form of the Hamiltonian to perform the above approximation of the evolution operator.
# While doing this is not directly supported in PennyLane in the form of a template, we can still
# implement the first-order Trotter approximation using the following :func:`CDFTrotterProduct`
# function that uses the compressed double factorized form of the Hamiltonian:
#

import itertools as it

def CDFTrotterProduct(nuc_core_cdf, one_body_cdf, two_body_cdf, time, num_steps=1):
    """Implements a first-order Trotter circuit for a CDF Hamiltonian.

    Args:
        nuc_core_cdf (float): The nuclear core energy.
        one_body_cdf (tuple): core and leaf tensors for the one-body terms.
        two_body_cdf (tuple): core and leaf tensors for the two-body terms.
        time (float): The total time for the evolution.
        num_steps (int): The number of Trotter steps. Default is 1.
    """
    norbs = qml.math.shape(one_body_cdf[0])[1]
    cores = qml.math.concatenate((one_body_cdf[0], two_body_cdf[0]), axis=0)
    leaves = qml.math.concatenate((one_body_cdf[1], two_body_cdf[1]), axis=0)
    btypes = qml.math.array([1] * len(one_body_cdf[0]) + [2] * len(two_body_cdf[0]))

    step = time / num_steps
    ranks = qml.math.argsort(qml.math.linalg.norm(cores, axis=(1, 2)))
    cores, leaves, btypes = cores[ranks], leaves[ranks], btypes[ranks]

    basis_mat = qml.math.eye(norbs)
    qml.BasisRotation(unitary_matrix=basis_mat, wires=range(0, 2 * norbs, 2))
    qml.BasisRotation(unitary_matrix=basis_mat, wires=range(1, 2 * norbs, 2))

    for _ in range(num_steps):
        for core, leaf, btype in zip(cores, leaves, btypes):
            # we undo the previous basis rotation and apply the new one
            qml.BasisRotation(unitary_matrix=basis_mat @ leaf, wires=range(0, 2 * norbs, 2))
            qml.BasisRotation(unitary_matrix=basis_mat @ leaf, wires=range(1, 2 * norbs, 2))
            basis_mat = leaf.T

            if btype == 1:  # gates for one-body term
                for wire, cval in enumerate(qml.math.diag(core)):
                    for sigma in [0, 1]:
                        qml.RZ(-step * cval, wires=2 * wire + sigma)
                qml.GlobalPhase(step * qml.math.sum(core), wires=range(2 * norbs))

            else:  # gates for two-body term
                for odx1, odx2 in it.product(range(norbs), repeat=2):
                    cval = core[odx1, odx2]
                    for sigma, tau in it.product(range(2), repeat=2):
                        if odx1 != odx2 or sigma != tau:
                            two_wires = [2 * odx1 + sigma, 2 * odx2 + tau]
                            qml.MultiRZ(step * cval / 4.0, wires=two_wires)
                qml.GlobalPhase(
                    -step / 2.0 * (qml.math.sum(core) - qml.math.trace(core) / 2),
                    wires=range(2 * norbs),
                )

    qml.GlobalPhase(nuc_core_cdf * time, wires=range(2 * norbs))

######################################################################
# We can use it to simulate the evolution of the Hamiltonian described in the double factorized form
# for a given number of steps ``num_steps`` and starting from the Hartree-Fock state ``hf_state``:
#

num_wires, time = 2 * mol.n_orbitals, 1.0
hf_state = qml.qchem.hf_state(electrons=mol.n_electrons, orbitals=num_wires)

@qml.qnode(qml.device("lightning.qubit", wires=num_wires))
def cdf_circuit(num_steps):
    qml.BasisState(hf_state, wires=range(num_wires))
    CDFTrotterProduct(nuc_core_cdf, one_body_cdf, two_body_cdf, time, num_steps=num_steps)
    return qml.state()

circuit_state = cdf_circuit(num_steps=20)

######################################################################
# We can test it by evolving the Hartree-Fock state analytically ourselves
# and testing the fidelity of the ``evolved_state`` with the ``circuit_state``:
#

from pennylane.math import fidelity_statevector
from scipy.linalg import expm

init_state = qml.math.array([1] + [0] * (2**num_wires - 1))
hf_state_vec = qml.matrix(qml.BasisState(hf_state, wires=range(num_wires))) @ init_state

H = qml.qchem.molecular_hamiltonian(mol)[0]
evolved_state = expm(1j * qml.matrix(H) * time) @ hf_state_vec

print(f"Fidelity of two states: {fidelity_statevector(circuit_state, evolved_state)}")

######################################################################
#
# As we can see, the fidelity of the evolved state from the circuit is close to
# :math:`1.0`, which indicates that the evolution of the CDF Hamiltonian sufficiently
# matches that of the original one.
#
# Conclusion
# -----------
#
# Compressed double-factorized representation for the Hamiltonians serves three key purposes.
# First, it allows for a more compact representation that can be stored and manipulated with
# greater ease. Second, it provides for more efficient simulations for approximating the
# Hamiltonian evolution as the number of terms is at least quadratically lesser. Third, the
# compact representation can be further manipulated to reduce the one-norm of the Hamiltonians
# which helps reduce the direct simulation when using block encoding or qubitization techniques.
# Therefore, employing CDF-based Hamiltonians for quantum chemistry problems provides a
# relatively promising path to reducing the complexity of fault-tolerant quantum algorithms.
#
# References
# -----------
#
# .. [#bliss]
#
#     Ignacio Loaiza, Artur F. Izmaylov,
#     "Block-Invariant Symmetry Shift: Preprocessing technique for second-quantized Hamiltonians to improve their decompositions to Linear Combination of Unitaries",
#     `arXiv:2304.13772 <https://arxiv.org/abs/2304.13772>`__, 2023.
#
# .. [#cdf]
#
#     Oumarou Oumarou, Maximilian Scheurer, Robert M. Parrish, Edward G. Hohenstein, Christian Gogolin,
#     "Accelerating Quantum Computations of Chemistry Through Regularized Compressed Double Factorization",
#     `Quantum 8, 1371 <https://doi.org/10.22331/q-2024-06-13-1371>`__, 2024.
#
# .. [#trotter]
#
#     Sergiy Zhuk, Niall Robertson, Sergey Bravyi,
#     "Trotter error bounds and dynamic multi-product formulas for Hamiltonian simulation",
#     `arXiv:2306.12569 <https://arxiv.org/abs/2306.12569>`__, 2023.
#
# About the author
# ----------------
#
