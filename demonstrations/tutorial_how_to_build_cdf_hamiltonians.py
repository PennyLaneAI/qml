r"""How to build compressed double factorized Hamiltonians
==========================================================

Two primary concerns for the quantum algorithms for quantum chemistry simulations of the
electronic Hamiltonians are the dependency of their runtime on its one-norm and their shot
requirements on the number of its terms. In this tutorial, we will learn how to tackle both
of these via a technique called compressed double factorization that involves approximately
representing the Hamiltonian in the form of tensor contractions that require a linear depth
circuit with Givens rotations for simulations and has a linear combination of unitaries (LCU)
representation suitable for error-corrected algorithms [#cdf]_.

Revisiting the electronic Hamiltonian
-------------------------------------

The Hamiltonian :math:`H` of the molecular systems with :math:`N` spatial orbitals in the
second-quantized form can be expressed as a sum of one-body and two-body terms as follows:

.. math::  H = \mu + \sum_{\sigma, pq} h_{pq} a^\dagger_{\sigma, p} a_{\sigma, q} + \frac{1}{2} \sum_{\sigma \tau, pqrs} h_{pqrs} a^\dagger_{\sigma, p} a^\dagger_{\tau, q} a_{\tau, r} a_{\sigma, s},

where tensor :math:`h_{pq}` (:math:`g_{pqrs}`) is the one-body (two-body) integrals,
:math:`a^\dagger` (:math:`a`) is the creation (annihilation) operator, :math:`\mu` is the nuclear
repulsion energy constant, :math:`\sigma \in {\uparrow, \downarrow}` represents the spin, and
:math:`p, q, r, s` are the orbital indices. In PennyLane, we can obtain the :math:`\mu`,
:math:`h_{pq}` and :math:`g_{pqrs}` using the :func:`~pennylane.qchem.electron_integrals` function:
"""

import pennylane as qml

symbols = ["H", "H", "H", "H"]
geometry = qml.math.array([[0., 0., -0.2], [0., 0., -0.1], [0., 0., 0.1], [0., 0., 0.2]])

mol = qml.qchem.Molecule(symbols, geometry)
nuc_core, one_body, two_body = qml.qchem.electron_integrals(mol)()

print(f"One-body and two-body tensor shapes: {one_body.shape}, {two_body.shape}")

######################################################################
# In the above expression, the two-body tensor (:math:`g_{pqrs}`)
# can be rearranged to define :math:`V_{pqrs}` in the `chemist notation
# <http://vergil.chemistry.gatech.edu/notes/permsymm/permsymm.pdf>`_,
# which leads to an one-body offset term :math:`\sum_{s} g_{pssq}`. This
# allows us to rewrite the above Hamiltonian as :math:`H_{\text{C}}` as:
#
# .. math::  H_{\text{C}} = \mu + \sum_{\sigma \in {\uparrow, \downarrow}} \sum_{pq} T_{pq} a^\dagger_{\sigma, p} a_{\sigma, q} + \sum_{\sigma, \tau \in {\uparrow, \downarrow}} \sum_{pqrs} V_{pqrs} a^\dagger_{\sigma, p} a_{\sigma, q} a^\dagger_{\tau, r} a_{\tau, s},
#
# where the transformed one-body terms :math:`T_{pq} = h_{pq} - 0.5 \sum_{s} g_{pssq}`.
# We can easily obtain these modified one-body and two-body tensors with the following:
#

two_chem = 0.5 * qml.math.swapaxes(two_body, 1, 3)  # V_pqrs
one_chem = one_body - qml.math.einsum("prrs", two_chem)  # T_pq

######################################################################
# A key feature of this representation is that the modified two-body terms can be factorized
# into sum of low-rank terms, which can be used to efficiently simulate the Hamiltonian. We
# will see how to do this with double factorization methods in the next section.
#
# Double factorizing the Hamiltonian
# -----------------------------------
#
# The double factorization of a Hamiltonian can be described as a Hamiltonian manipulation
# technique based on decomposing the :math:`V_{pqrs}` to symmetric tensors :math:`L^{(t)}`
# called *factors*, such that :math:`V_{pqrs} = \sum_t^T L_{pq}^{(t)} L_{rs}^{(t) {\dagger}}`
# and the rank :math:`T \leq N^2`. We can do this by performing an eigenvalue, or a pivoted
# Cholesky decomposition (``cholesky=True``), of the modified two-body tensor.
# Moreover, each of these tensors can be further eigendecomposed as
# :math:`L^{(t)}_{pq} = \sum_{i} U_{pi}^{(t)} W_i^{(t)} U_{qi}^{(t)}` to perform a second
# tensor factorization. This enables us to express the above double factorized two-body tensor
# in terms of orthonormal core tensors (:math:`Z^{(t)}`) and symmetric leaf tensors
# (:math:`U^{(t)}`) as:
#
# .. math::  V_{pqrs} \approx \sum_t^T \sum_{ij} U_{pi}^{(t)} U_{pj}^{(t)} Z_{ij}^{(t)} U_{qk}^{(t)} U_{ql}^{(t)},
#
# where :math:`Z_{ij}^{(t)} = W_i^{(t)} W_j^{(t)}` [#cdf2]_. This decomposition is referred
# to as the *explicit* double factorization (XDF) and allows decreasing the number of terms
# in the qubit basis to :math:`O(N^3)` from :math:`O(N^4)`, assuming the rank of second
# tensor factorization will be :math:`O(N)`. In PennyLane, this can be done using the
# :func:`~pennylane.qchem.factorize` function, where we can truncate the resulting factors
# by discarding the ones with individual contributions below a specified threshold and the
# ranks of their second factorization using the ``tol_factor`` and ``tol_eigval`` keyword
# arguments, respectively, as shown below:
#

factors, _, _ = qml.qchem.factorize(two_chem, cholesky=True, tol_factor=1e-5)
print("Shape of the factors: ", factors.shape)

approx_two_chem = qml.math.tensordot(factors, factors, axes=([0], [0]))
assert qml.math.allclose(approx_two_chem, two_chem, atol=1e-5)

######################################################################
# Performing block-invariant symmetry shift
# ------------------------------------------
#
# We can further improve the above factorization by employing the block-invariant symmetry
# shift (BLISS) [#bliss]_, which modifies the action by the Hamiltonian on the undesired
# electron-number subspace. This helps decrease the one-norm and the spectral range of the
# Hamiltonian and can be done via the :func:`~pennylane.qchem.symmetry_shift` function:
#

core_shift, one_shift, two_shift = qml.qchem.symmetry_shift(
    nuc_core, one_chem, two_chem, n_elec = mol.n_electrons
)

######################################################################
# We can compare the improvement in the one-norm of the shifted Hamiltonian over the original
# one by accessing the :class:`~.pennylane.resource.DoubleFactorization`'s ``lamb`` attribute:
#

from pennylane.resource import DoubleFactorization as DF

DF_chem_norm = DF(one_chem, two_chem, chemist_notation=True).lamb
DF_shift_norm =  DF(one_shift, two_shift, chemist_notation=True).lamb
print(f"Decrease in one-norm: {DF_chem_norm - DF_shift_norm}")

######################################################################
# Compressing the double factorization
# -------------------------------------
#
# In many practical scenarios, the above double factorization can be further optimized by
# obtaining a numerical tensor-fitting of the decomposed two-body terms to give :math:`V^\prime`,
# such that the approximation error :math:`||V - V^\prime||` remains below a desired threshold.
# This is referred to as the *compressed* double factorization (CDF) as it reduces the number of
# terms in the factorization of the two-body term to :math:`O(N)` from :math:`O(N^3)`, and it
# achieves lower approximation errors than the truncated XDF. This can be done by beginning with
# random :math:`N` orthornormal and symmetric tensors and optimizing them based on the following
# cost function :math:`\mathcal{L}` in a greedy layered-wise manner:
#
# .. math::  \mathcal{L}(U, Z) = \frac{1}{2} \bigg|V_{pqrs} - \sum_t^T \sum_{ij} U_{pi}^{(t)} U_{pj}^{(t)} Z_{ij}^{(t)} U_{qk}^{(t)} U_{ql}^{(t)}\bigg|_{\text{F}} + \rho \sum_t^T \sum_{ij} \bigg|Z_{ij}^{(t)}\bigg|^{\gamma},
#
# where :math:`|\cdot|_{\text{F}}` computes the Frobenius norm, :math:`\rho` is a constant
# scaling factor, and :math:`|\cdot|^\gamma` specifies the optional L1 and L2 regularization
# that improves the energy variance of the resulting representation [#cdf]_. In PennyLane,
# these can be achieved by using the ``compressed=True`` and ``regularization`` keyword
# arguments, respectively, in the :func:`~pennylane.qchem.factorize` method as shown below:
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
# Constructing the double factorized Hamiltonian
# -----------------------------------------------
#
# Using the above factorization of one-body and two-body terms, we can now express
# the entire Hamiltonian more compactly as sum of the products of core and leaf tensors:
#
# .. math:: H_{\text{CDF}} = \mu + \sum_{\sigma \in {\uparrow, \downarrow}} U^{(0)}_{\sigma} \left( \sum_{p} Z^{(0)}_{p} a^\dagger_{\sigma, p} a_{\sigma, p} \right) U_{\sigma}^{(0)\ \dagger} + \sum_t^T \sum_{\sigma, \tau \in {\uparrow, \downarrow}} U_{\sigma, \tau}^{(t)} \left( \sum_{pq} Z_{pq}^{(t)} a^\dagger_{\sigma, p} a_{\sigma, p} a^\dagger_{\tau, q} a_{\tau, q} \right) U_{\sigma, \tau}^{(t)\ \dagger},
#
# where we specify each term in the above summation for a Hamiltonian in the
# double factorized form as ``nuc_core_cdf`` (:math:`\mu`), ``one_body_cdf``
# (:math:`Z^{(0)}, U^{(0)}`) and ``two_body_cdf`` (:math:`Z^{(t)}, U^{(t)}`):
#

nuc_core_cdf = core_shift[0]
one_body_cdf = (one_body_cores, one_body_leaves)
two_body_cdf = (two_body_cores, two_body_leaves)

######################################################################
# The above representation enables obtaining the measurement grouping of the Hamiltonian
# for reducing the shot requirements for measurements in the qubit basis. This makes use of
# the Jordan-Wigner transformation (:math:`a_p^\dagger a_p = n_p = 0.5 * (1 - z_p)`), where
# each term within the basis transformation :math:`U^{(i)}` could be measured simultaneously.
# One can obtain the corresponding Pauli terms and basis transformation for these groupings
# using the :func:`~pennylane.qchem.basis_rotation` function, which automatically accounts
# for the spin. However, this is not the only advantage of the double factorized form. It
# also allows for a more efficient simulation of the Hamiltonian evolution, which we will
# see in the next section.
#

######################################################################
# Simulating the double factorized Hamiltonian
# ---------------------------------------------
#
# To simulate the Hamiltonian in the double factorized form,
# we will first need to learn how to apply the unitary operations represented by the
# exponentiated leaf and core tensors. The former can be done using the
# :class:`~.pennylane.BasisRotation` operation, which implements the unitary transformation
# :math:`\exp \left( \sum_{pq}[\log U]_{pq} (a^\dagger_p a_q - a^\dagger_q a_p) \right)`.
# The following ``leaf_unitary_rotation`` function does this for a leaf tensor:
#

def leaf_unitary_rotation(leaf, norbs):
    """Applies the basis rotation transformation corresponding to the leaf tensor."""
    basis_mat = qml.math.kron(leaf, qml.math.eye(2)) # account for spin
    qml.BasisRotation(unitary_matrix=basis_mat, wires=range(2 * norbs))

######################################################################
# The above can be decomposed in terms of the Givens rotation networks that can be efficiently
# implemented on the quantum hardware. Similarly, the unitary transformation for the core tensor
# can also be applied efficiently via the ``core_unitary_rotation`` function. It uses the
# :class:`~.pennylane.RZ` and :class:`~.pennylane.MultiRZ` operations for implementing the
# exponentiated Pauli-Z tensors for the one-body and two-body core tensors, respectively,
# and :class:`~.pennylane.GlobalPhase` for the corresponding global phases:
#

def core_unitary_rotation(core, norbs, body_type):
    """Applies the unitary transformation corresponding to the core tensor."""
    if body_type == "one_body":  # gates for one-body term
        for wire, cval in enumerate(qml.math.diag(core)):
            for sigma in [0, 1]:
                qml.RZ(-cval, wires=2 * wire + sigma)
        qml.GlobalPhase(qml.math.sum(core), wires=range(2 * norbs))

    else:  # gates for two-body term
        for odx1, odx2 in it.product(range(norbs), repeat=2):
            cval = core[odx1, odx2]
            for sigma, tau in it.product(range(2), repeat=2):
                if odx1 != odx2 or sigma != tau:
                    two_wires = [2 * odx1 + sigma, 2 * odx2 + tau]
                    qml.MultiRZ(cval / 4.0, wires=two_wires)
        gphase = 0.5 * qml.math.sum(core) + 0.25 * qml.math.trace(core)
        qml.GlobalPhase(-gphase, wires=range(2 * norbs))

######################################################################
# We can now use them to approximate the evolution operator :math:`e^{-iHt}` for a time
# :math:`t` with the Suzuki-Trotter product formula, which uses symmetrized products
# :math:`S_m` defined for an order :math:`m \in [1, 2, 4, \ldots, 2k \in \mathbb{N}]`
# and repeated multiple times [#trotter]_. In general, this can be easily implemented for
# standard non-factorized Hamiltonians using the :class:`~.pennylane.TrotterProduct` operation
# that defines those products recursively for a given number of steps and therefore leads
# to an exponential scaling in its complexity with the number of terms in the Hamiltonian,
# making it inefficient for larger system sizes.
#
# Such a scaling behaviour could be managed to a great extent by working with the compressed
# double factorized form of the Hamiltonian as it allows reducing the number of terms in the
# Hamiltonian from :math:`O(N^4)` to :math:`O(N)`. While doing this is not directly supported
# in PennyLane in the form of a template, we can still implement the first-order Trotter
# approximation using the following :func:`CDFTrotterProduct` function that uses the
# compressed double factorized form of the Hamiltonian with the ``leaf_unitary_rotation``
# and ``core_unitary_rotation`` functions defined earlier:
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
    for _ in range(num_steps):
        for core, leaf, btype in zip(cores, leaves, btypes):
            # apply the basis rotation for leaf tensor
            leaf_unitary_rotation(leaf, norbs)

            # apply the rotation for core tensor scaled by the step size
            body_type = "one_body" if btype == 1 else "two_body"
            core_unitary_rotation(step * core, norbs, body_type)

            # undo the previous basis rotation
            leaf_unitary_rotation(leaf.conjugate().T, norbs)

    # apply the globals phase based on the nuclear core energy
    qml.GlobalPhase(nuc_core_cdf * time, wires=range(2 * norbs))

######################################################################
# We can use it to simulate the evolution of the linear hydrogen chain Hamiltonian H\ :math:`_4`
# described in the compressed double factorized form for a given number of steps ``num_steps``
# and starting from the Hartree-Fock state ``hf_state``:
#

num_wires, time = 2 * mol.n_orbitals, 1.0
hf_state = qml.qchem.hf_state(electrons=mol.n_electrons, orbitals=num_wires)

@qml.qnode(qml.device("lightning.qubit", wires=num_wires))
def cdf_circuit(num_steps):
    qml.BasisState(hf_state, wires=range(num_wires))
    CDFTrotterProduct(nuc_core_cdf, one_body_cdf, two_body_cdf, time, num_steps=num_steps)
    return qml.state()

circuit_state = cdf_circuit(num_steps=10)

######################################################################
# We can test its performance by evolving the Hartree-Fock state analytically ourselves
# and testing the fidelity of the ``evolved_state`` with the ``circuit_state``:
#

from pennylane.math import fidelity_statevector
from scipy.linalg import expm

init_state = qml.math.array([1] + [0] * (2**num_wires - 1)) # |00...0>
hf_state_vec = qml.matrix(qml.BasisState(hf_state, wires=range(num_wires))) @ init_state

H = qml.qchem.molecular_hamiltonian(mol)[0] # original Hamiltonian
evolved_state = expm(-1j * qml.matrix(H) * time) @ hf_state_vec

print(f"Fidelity of two states: {fidelity_statevector(circuit_state, evolved_state)}")

######################################################################
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
# .. [#cdf]
#
#     Oumarou Oumarou, Maximilian Scheurer, Robert M. Parrish, Edward G. Hohenstein, and Christian Gogolin,
#     "Accelerating Quantum Computations of Chemistry Through Regularized Compressed Double Factorization",
#     `Quantum 8, 1371 <https://doi.org/10.22331/q-2024-06-13-1371>`__, 2024.
#
# .. [#cdf2]
#
#     Jeffrey Cohn, Mario Motta, and Robert M. Parrish,
#     "Quantum Filter Diagonalization with Compressed Double-Factorized Hamiltonians",
#     `PRX Quantum 2, 040352 <https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.2.040352>`__, 2021.
#
# .. [#bliss]
#
#     Ignacio Loaiza, and Artur F. Izmaylov,
#     "Block-Invariant Symmetry Shift: Preprocessing technique for second-quantized Hamiltonians to improve their decompositions to Linear Combination of Unitaries",
#     `arXiv:2304.13772 <https://arxiv.org/abs/2304.13772>`__, 2023.
#
# .. [#trotter]
#
#     Sergiy Zhuk, Niall Robertson, and Sergey Bravyi,
#     "Trotter error bounds and dynamic multi-product formulas for Hamiltonian simulation",
#     `arXiv:2306.12569 <https://arxiv.org/abs/2306.12569>`__, 2023.
#
# About the author
# ----------------
#
