r"""Quantum low-density parity-check (qLDPC) codes for quantum error correction
================================================================================

Quantum computers are envisioned to be incredibly powerful computational devices. While many of the
machines available today boast hundreds of qubits, their performance remains limited due to noise
present in these systems, which manifests itself as computational errors and impacts their utility.
Thus, for a fault-tolerant quantum computer to exist, where one can run these devices indefinitely
with minimal permissible errors, we need quantum error correction (QEC).

While designing error-correction codes for this purpose, we look for some desirable properties such
as (i) a high encoding rate :math:`R = k / n`, (ii) local and low-weight parity checks,
(iii) an over-complete universal gate set with as big a transversal gate set as possible,
and (iv) linear time classical decoding. Unfortunately, these requirements are not all
mutually compatible. For example, widely used topological codes such as surface codes
use local, nearest-neighbour connections, but have an inefficient encoding rate, i.e., protecting
one logical qubit (:math:`k=1`) requires a patch of thousands of physical qubits (:math:`n \gg k`).

However, it remains unclear which combination of these options would lead to the best long-term
solution. But as solving real-world problems requires scaling up to thousands of logical qubits,
removing the locality condition is crucial. Quantum low-density parity-check (qLDPC) codes
enable us to do exactly that. In this demo, we will cover the basics of the qLDPC codes,
including their construction and decoding. To better assimilate all the topics covered in this
tutorial, we recommend reading our tutorials on the :doc:`Surface Code
<demos/tutorial_game_of_surface_codes>` and :doc:`Stabilizer Codes <demos/tutorial_stabilizer_codes>`
for a quick recap of the QEC topics.

.. figure::    
    ../_static/demo_thumbnails/opengraph_demo_thumbnails/pennylane-demo-stabilizer-codes-open-graph.png
    :align: center
    :width: 50%
    :target: javascript:void(0)


Classical LDPC Codes
--------------------

To understand quantum LDPC codes, we begin by looking at their classical counterparts, which
have revolutionized modern telecommunications (powering Wi-Fi and 5G networks) by approaching
the absolute theoretical limits of data transmission, known as the `Shannon limit
<https://en.wikipedia.org/wiki/Shannon_limit>`_. Classical LDPC codes achieve this with
highly efficient, linear-time decoding that exploits their structure of the parity checks.

A classical LDPC code :math:`C[n,k,m]` protects :math:`k` logical bits by encoding them into
:math:`n` physical bits based on rules defined by an :math:`m\times n` parity-check matrix
(:math:`H`) [#qldpc1]_. The "low-density" part of their name comes from this matrix being
overwhelmingly sparse, i.e., filled mostly with zeros, and more specifically, the column/row
weights (number of 1s in each column/row) are strictly bounded constants independent of
:math:`n`.

Mathematically, these are visualized as `Tanner graphs
<https://en.wikipedia.org/wiki/Tanner_graph>`_, which are bipartite graphs with edges
representing connections between the variable nodes (:math:`n` physical bits)
and the check nodes (:math:`m` parity constraints). For example, the following
is a Tanner graph for a simple :math:`[7, 4]` LDPC code:
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Define the parity-check matrix H
H = np.array([[1, 0, 1, 0, 1], [0, 1, 1, 0, 0], [1, 1, 0, 1, 0]])

# Construct the Tanner Graph with variable nodes and check nodes.
G = nx.Graph()
num_checks, num_vars = H.shape
var_nodes = [f"v{i}" for i in range(num_vars)]
check_nodes = [f"c{j}" for j in range(num_checks)]
G.add_nodes_from(var_nodes, bipartite=0)
G.add_nodes_from(check_nodes, bipartite=1)

for j in range(num_checks):
    for i in range(num_vars):
        if H[j, i] == 1:
            G.add_edge(f"c{j}", f"v{i}")

# Plot the Bipartite Graph
plt.figure(figsize=(6, 4))
pos = nx.bipartite_layout(G, var_nodes)
colors = ["#70CEFF"] * num_vars + ["#C756B2"] * num_checks
nx.draw(G, pos, with_labels=True, node_color=colors, node_size=500)
text_options = {"verticalalignment": "center", "fontsize": 12}
plt.text(-0.75, 0.0, "Variable Nodes", rotation=90, color=colors[0], **text_options)
plt.text(1.1, 0.0, "Check Nodes", rotation=270, color=colors[-1], **text_options)
plt.show()

######################################################################
#
# When noise corrupts our bits, creating an error vector :math:`\vec{e}`, the system identifies the
# problem by computing the syndrome :math:`s = H\vec{e} \mod\ 2`. As each check node only connects
# to a handful of data nodes (and vice versa), these codes scale beautifully because errors can be
# decoded using local message-passing algorithms sharing probabilistic information along the edges
# of the Tanner graph until all parity constraints are satisfied. This will come in handy when we
# learn about decoding, but before that, let's return to first constructing qLDPC codes.
#
# Calderbank-Shor-Steane (CSS) construction
# ------------------------------------------
#
# The key goal of qLDPC codes is to replicate the sparsity (and unlock linear-time decoding) in the
# quantum realm. This is a non-trivial task, as due to no-go theorems, one no longer has direct
# access to the state to perform the standard parity checks, and in addition to bit flips, the
# qubits also suffer from phase flips, which don't have a classical analog. Therefore, to detect
# errors, we need to rely on measuring commuting multi-qubit Pauli operators that help detect
# phase and bit flips.
#
# The most elegant and widely used solution to build a series of such operators is the
# Calderbank-Shor-Steane (CSS) code construction [#CSS]_, where instead of trying to design one
# massive, complicated matrix that mixes the Pauli X and Z operations, the problem is sliced into
# two halves, defining a stabilizer code using generator sets containing only Pauli-Z (Pauli-X)
# operators to catch bit (phase) flips. We can then represent these sets (or stabilizers) using
# two separate classical parity-check matrices: :math:`H_X` and :math:`H_Z`, where the elements
# :math:`H_{ij}^{P={X/Z}} = 1` are only if the :math:`i^{th}` :math:`P`-type check has support
# on the :math:`j^{th}` qubit. For example, look at the following CSS code known as the
# `Steane code <https://errorcorrectionzoo.org/c/steane>`_ :math:`[[7,1,3]]` constructed from
# the two :math:`d=3` Hamming codes:
#

import pennylane as qp


def hamming_code(distance: int) -> np.ndarray:
    """Returns a Hamming code parity check matrix of a given rank."""
    bit_masks = np.arange(1, 2**distance)[:, None] & (1 << np.arange(distance)[::-1])
    return (bit_masks > 0).astype(np.uint8).T


h1, h2 = hamming_code(3), hamming_code(3)
(m1, n1), (m2, n2) = h1.shape, h2.shape

css_code = np.hstack((
        np.vstack([np.zeros((m1, n1), dtype=np.uint8), h1]),
        np.vstack([h2, np.zeros((m2, n2), dtype=np.uint8)])
))

hx, hz = css_code[m1:, :n1], css_code[:m2, n2:]
print(f"Does H_X * H_Z^T = 0? {np.allclose((hx @ hz.T) % 2, 0)}")
print(f"Does H_Z * H_X^T = 0? {np.allclose((hz @ hx.T) % 2, 0)}\n")

code_dim = hx.shape[1] - qp.math.binary_matrix_rank(hx) - qp.math.binary_matrix_rank(hz)
print(f"Code dimension (k): {code_dim}")

######################################################################
# As implicity shown above, for satisfying the commutation condition, these matrices must
# satisfy a symplectic orthogonality condition :math:`H^X(H^Z)^T = 0\mod\ 2`, i.e., each
# of the :math:`X - Z` stabilizer pairs must overlap on an even number of qubits.
#
# Hypergraph Product Codes
# ------------------------
#
# While finding just one classical matrix that is sparse is easy, finding two sparse matrices that
# randomly happen to share an even number of connections everywhere their checks intersect is
# mathematically laborious. It severely limits the design space compared to classical codes, which
# is why it took decades for researchers to discover families of good qLDPC codes. However, a
# foundational breakthrough happened with the Hypergraph Product (HGP) codes [#HGP]_,
# which takes two classical LDPC codes, :math:`C_1` (:math:`[n_1, k_1, d_1]`) and :math:`C_2`
# (:math:`[n_2, k_2, d_2]`), with parity-check matrices :math:`H_1` (:math:`m_1 \times n_1`) and
# :math:`H_2` (:math:`m_2 \times n_2`), respectively, and produces a CSS code with the following
# parity-check matrices:
#
# .. math::
#
#     H^X = (H_1 \otimes I_{n_2} | I_{m_1} \otimes H_2^T),\\
#     H^Z = (I_{n_1} \otimes H_2 | H_1^T \otimes I_{m_2}),
#
# where the algebraic properties of the tensor product ensure that :math:`H^X` and :math:`H^Z`
# satisfy the symplectic orthogonality condition, and :math:`H_i^T` defines the transpose code
# of :math:`H_i`, with parameters :math:`[m_i, k_i^T, d_i^T ]`. For example, look at the
# following HGP code constructed from two :math:`d_1=3` and :math:`d_2=3` repetition codes,
# which is equivalent to a Toric code :math:`[[13, 1, 3]]`:
#

from pennylane.qchem.tapering import _kernel as binary_matrix_kernel


def rep_code(distance: int) -> np.ndarray:
    """Construct repetition code parity check matrix for specified distance."""
    return np.eye(distance - 1, distance, k=0, dtype=np.uint8) + np.eye(
        distance - 1, distance, k=1, dtype=np.uint8
    )


def hgp_code(h1: np.ndarray, h2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Construct HGP code parity check matrices."""
    (m1, n1), (m2, n2) = h1.shape, h2.shape
    hx = np.hstack((np.kron(h1, np.identity(n2)), np.kron(np.identity(m1), h2.T)))
    hz = np.hstack((np.kron(np.identity(n1), h2), np.kron(h1.T, np.identity(m2))))
    return hx.astype(np.int8), hz.astype(np.int8)


h1, h2 = rep_code(3), rep_code(3)
hx, hz = hgp_code(h1, h2)
print(f"Does H_X * H_Z^T = 0? {np.allclose((hx @ hz.T) % 2, 0)}")

(m1, n1), (m2, n2) = h1.shape, h2.shape
r1, r2 = qp.math.binary_matrix_rank(h1), qp.math.binary_matrix_rank(h2)
k1, k2 = n1 - r1, n2 - r2
k1t, k2t = m1 - r1, m2 - r2
print(f"Code dimension (k) of the HGP code: {k1 * k2 + k1t * k2t}\n")


def compute_distance(parity_matrix: np.ndarray) -> int:
    """Compute the classical distance of the code based on the parity-check matrices."""
    kernel_matrix = binary_matrix_kernel(
        qp.math.binary_finite_reduced_row_echelon(parity_matrix)
    )  # compute the kernel of the parity-check matrix
    if (k := kernel_matrix.shape[0]) == 0:
        return np.inf  # the code distance is not defined

    # Compute every single codeword simultaneously and compute Hamming weight
    ints = np.arange(1 << k, dtype=np.uint32)[:, None]
    shifts = np.arange(k, dtype=np.uint32)
    coeffs = ((ints >> shifts) & 1).astype(np.uint8)
    codewords = (coeffs @ kernel_matrix) % 2
    weights = codewords[1:].sum(axis=1)
    return int(np.min(weights))


d1, d2 = compute_distance(h1), compute_distance(h2)
d1t, d2t = compute_distance(h1.T), compute_distance(h2.T)
print(f"Distance (d) of the HGP code: {(dist := min(d1, d2, d1t, d2t))}")
print(f"Physical qubits (n) of the HGP code: {n1*n2 + m1*m2} == {2*dist*(dist-1) + 1}")

######################################################################
# As shown above, the resulting quantum code can be represented by :math:`Q[[n,k,d]]`,
# which encodes logical qubits with distance :math:`d=min(d_1,d_2,d_1^T,d_2^T)`,
# with :math:`n = n_1n_2 + m_1m_2` and :math:`k = k_1k_2 + k_1^T k_2^T`. This means that
# the HGP codes achieve a constant encoding rate :math:`R=\Theta(1)`, but their distance
# grows only as :math:`d=\mathcal{O}(\sqrt{n})`, matching the surface code scaling. Note
# that, the distance computed here is the classical distance, which is not the same as
# the quantum distance. The latter is more complex to compute as it requires finding
# the minimum weight of an error that goes undetected by the checks but is
# not a stabilizer.
#
# Modern qLDPC Codes
# -------------------
#
# To build truly scalable quantum computation devices, we need to at least achieve a linear
# distance scaling, i.e., :math:`d=\Theta(n)`. In recent years, there has been some progress
# in achieving this goal, primarily through a series of breakthroughs, some of which are:
#
# 1. Lifted Product (LP) Codes: To overcome the :math:`O(\sqrt{n})` distance barrier of standard
#    HGP codes, these codes replace the binary scalar entries of a classical seed matrix with
#    elements of a group algebra, such as polynomials representing cyclic shifts [#LPCodes]_.
#    By taking the hypergraph product over this polynomial space and "lifting" the result back
#    into a massive, sparse binary matrix, this hidden group structure injects powerful
#    algebraic constraints. This drastically boosts the minimum distance to
#    :math:`d = \Theta(\sqrt{n} \log n)`, while maintaining a constant encoding rate
#    :math:`R = \Theta(1)` and the crucial sparsity required for fast message-passing decoding.
#
# 2. Quantum Tanner (QT) Codes: Seeking to maximize both storage density and error-correcting
#    power, these codes move away from flat grids and are constructed using Cayley graphs of
#    finite groups with high expansion properties [#QTCodes]_. By placing qubits on the faces of
#    a multidimensional square complex and rigidly enforcing local classical constraints at
#    every vertex, the expander graph geometry physically prevents small errors from forming
#    undetectable logical operators. Consequently, these codes achieve both a constant rate
#    :math:`R = \Theta(1)` and a strictly linear distance :math:`d = \Theta(n)`,
#    approaching the theoretical limit of the quantum Singleton bound, where doubling the
#    physical qubits strictly doubles the error-correcting power.
#
# 3. Bivariate Bicycle (BB) Codes: To bridge the gap between the abstract algebra of expander
#    graphs (which require highly non-local hardware wiring) and the reality of physical
#    quantum processors, these codes were developed [#BBCodes]_. They are built using simple
#    bivariate polynomials of two commuting variables (:math:`x` and :math:`y`) that directly
#    correspond to local spatial shifts on a periodic 2D grid, ensuring the resulting
#    parity-check matrices commute. By construction, the physical qubits can be laid
#    out in a quasi-2D architecture with strictly bounded, short-range connections,
#    while sacrificing infinite asymptotic linear scaling.
#
# Below, we take a look at a simplified QT codes construction and benchmark the
# improvements in the distance scaling compared to the HGP codes:
#


def tanner_code(h1: np.ndarray, h2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Construct Tanner code parity check matrices [arXiv:2309.11719]."""
    itr, idx = h1.shape[0] * h2.shape[0], h1.shape[1] * h2.shape[1]

    def eliminate_col(mat: np.ndarray) -> np.ndarray:
        col = mat[:, idx]
        if not col.any():
            return mat
        pivot = int(col.argmax())
        mask = col.astype(bool)
        mask[pivot] = False
        mat[mask] ^= mat[pivot]
        return np.delete(mat, pivot, axis=0).astype(np.int8)

    hx, hz = hgp_code(h1, h2)
    for ix in range(itr):
        hx = np.delete(eliminate_col(hx) if ix % 2 == 0 else hx, idx, axis=1)
        hz = np.delete(eliminate_col(hz) if ix % 2 == 1 else hz, idx, axis=1)

    return hx, hz


ns_hgp, ns_tan = [], []
for dist in (distances := range(2, 20)):
    h1, h2 = rep_code(dist), rep_code(dist)
    ns_hgp.append(h1.shape[0] * h2.shape[0] + h1.shape[1] * h2.shape[1])
    hx, hz = tanner_code(h1, h2)
    ns_tan.append(max(hx.shape[1], hz.shape[1]))

plt.plot(distances, ns_hgp, '-o', label="HGP Code")
plt.plot(distances, ns_tan, '-*',label="Quantum Tanner Code")
plt.grid(True, which="both", ls="--", c="lightgray", alpha=0.7)
plt.ylabel("# Physical qubits")
plt.xlabel("Rep. code distance")
plt.legend()
plt.tight_layout()
plt.show()

######################################################################
# Decoding qLDPC Codes
# ----------------------
#
# As mentioned earlier, Tanner graphs constructed using the parity-check matrix of the code
# can be used for decoding errors efficiently using a decoder that implements iterative
# message-passing algorithms like Belief Propogation (BP) [#BProp]_. It forms the backbone of the
# many standardly algorithms used for qLDPC codes, which iteratively passes the messages between
# the variable and check nodes. These messages consist of a log-likelihood ratio (LLR) summarizing
# the evidence from the channel output and all other check nodes, who then use them to compute a
# parity constraint and sends back an updated LLR. Generally, all the variable nodes need to
# reach a consensus in a fixed number of iterations, making the whole process executable in
# polynomial time.
#
# For classical LDPC codes, BP is near-optimal and runs in :math:`O(\log n)` iterations. However,
# for quantum LDPC codes, handling degeneracy (multiple error patterns producing the same syndrome)
# is a challenging task, which requires using a post-processing step of Ordered Statistics Decoding
# (OSD) with order-0 to maintain the performance [#OSD0]_. Let us define a decoder class that
# implements this, where the BP is implemented using the ``tanh`` product rule. When the BP
# fails to converge, the OSD-0 is used as a fallback, which ranks the qubits by their final
# LLR reliability and uses Gaussian elimination to mathematically force a valid parity solution.
#


class BPOSDDecoder:
    """A lightweight Belief Propagation + OSD-0 decoder.

    Args:
        H (np.ndarray): Parity-check matrix for the codeword (m x n).
        error_rate (float): Prior probability that any single bit is flipped.
        max_iter (int): Maximum BP iterations before falling back to OSD.
    """

    def __init__(self, H, error_rate=0.05, max_iter=50):
        self.H = np.asarray(H, dtype=int)
        self.m, self.n = self.H.shape
        self.max_iter = max_iter
        self.channel_llr = np.log((1 - error_rate) / error_rate)

    def decode(self, syndrome: np.ndarray) -> tuple[bool, np.ndarray, str]:
        """Decode a length-m syndrome vector and return the estimated error."""
        # Initialize messages from check to variable nodes and the total belief
        parity_matrix, target_syndrome = self.H, np.asarray(syndrome, dtype=int)
        c2v_messages = np.zeros((self.m, self.n))
        prior_llr = self.channel_llr # baseline likelihood - prior belief
        posterior_llr = np.full(self.n, prior_llr)
        for _ in range(self.max_iter): # BP loop
            # Variable-to-Check Update (Extrinsic Information)
            var_to_check_msgs = parity_matrix * (posterior_llr[None, :] - c2v_messages)
            # Check-to-Variable Update (Tanh Product Rule)
            c2v_messages = self.update_checks(var_to_check_msgs, target_syndrome)
            # Update Total Beliefs (Posterior LLR)
            posterior_llr = prior_llr + (c2v_messages * parity_matrix).sum(axis=0)
            # Make a Hard Decision
            estimated_error = (posterior_llr < 0).astype(int)  
            # Verify the Syndrome
            if np.all((parity_matrix @ estimated_error) % 2 == target_syndrome):
                return (True, estimated_error, "BP")

        # OSD Fallback and verify the syndrome
        estimated_error = self.osd0(target_syndrome, posterior_llr)  
        is_success = bool(np.all((parity_matrix @ estimated_error) % 2 == target_syndrome))
        return (is_success, estimated_error, "OSD-0")

    def update_checks(self, var_to_check_msgs: np.ndarray, syndrome: np.ndarray) -> np.ndarray:
        r"""Check-to-variable update via the tanh product rule.

        .. math::
            M_cv[i,j] = (-1)^{s_i} \cdot 2 \arctanh{\prod_{j'\neqj} \tanh(M_vc[i,j'] / 2)}
        """
        parity_matrix, delta = self.H, 1e-15

        # Compute the tanh (for incoming messages) and check for edge-existence
        tanh_msgs = np.tanh(var_to_check_msgs / 2.0)
        tanh_msgs = np.where((parity_matrix == 1) & (tanh_msgs == 0.0), delta, tanh_msgs)

        # Compute the "extrinsic" product for each edge and apply the syndrome
        masked_tanh_msgs = np.where(parity_matrix == 1, tanh_msgs, 1.0)
        check_node_prods = np.prod(masked_tanh_msgs, axis=1, keepdims=True)
        extrinsic_tanh = np.clip(check_node_prods / masked_tanh_msgs, -1 + delta, 1 - delta)

        # Apply the syndrome constraint and convert back to LLR space
        syndrome_sign = (1 - 2 * syndrome)[:, None] # (-1)^s_i
        check_to_var_msgs = parity_matrix * syndrome_sign * 2 * np.arctanh(extrinsic_tanh)
        return check_to_var_msgs

    def osd0(self, syndrome: np.ndarray, llr: np.ndarray) -> np.ndarray:
        """Implements the ordered statistics decoding with order-0 fallback"""
        # Permute using the LLR magnitude and augment it with the syndrome
        reliability_order = np.argsort(-np.abs(llr))
        H_permuted = self.H[:, reliability_order]
        augmented_matrix = np.hstack([H_permuted, syndrome.reshape(-1, 1)])

        # Perform Gaussian elimination over GF(2) and extract results
        rref_matrix = qp.math.binary_finite_reduced_row_echelon(augmented_matrix)
        H_reduced, updated_syndrome = rref_matrix[:, :self.n], rref_matrix[:, -1]

        # Set all non-pivot variables to 0 and permute the errors back
        has_pivot, pivot_cols = H_reduced.any(axis=1), H_reduced.argmax(axis=1)
        final_error, permuted_error = np.zeros((2, self.n), dtype=int)
        permuted_error[pivot_cols[has_pivot]] = updated_syndrome[has_pivot]
        final_error[reliability_order] = permuted_error
        return final_error

######################################################################
# Let us test our decoder on the Hypergraph Product (HGP) code constructed from the
# repetition codes with distance 3. We will intentionally inject a specific 2-qubit
# error, compute its syndrome, and ask the decoder to find a correction.
#

h1, h2 = rep_code(3), rep_code(3)
hx, hz = hgp_code(h1, h2)

# Inject a weight-2 X-error (bit-flip)
x_error = np.zeros(hx.shape[1], dtype=int)
x_error[1], x_error[5] = 1, 1
print(f"X-error: {x_error}")

# Z-stabilizers detect X-errors
z_syndrome = (hz @ x_error) % 2
print(f"Z-syndrome: {z_syndrome}")

dec_z = BPOSDDecoder(hz, error_rate=0.05, max_iter=50)
res = dec_z.decode(z_syndrome)[1]
print(f"Decoded error: {res}")

residual = (res + x_error) % 2
if np.allclose(residual, np.zeros(hx.shape[1], dtype=int)):
    print("Result: exact correction")
elif np.all((hz @ residual) % 2 == 0):
    print("Result: corrected up to stabilizer.")
else:
    print("Result: logical error.")

######################################################################
# As multiple physical error patterns map to the exact same syndrome, the decoder often
# finds an alternative, equally valid path. So, as shown in the above example, sometimes
# the decoder will find a correction that matches a valid stabilizer, rather than the exact
# correction, which means the logical codespace is preserved, i.e., quantum information
# remains protected.
#
# Transversal Gates for qLDPC Codes
# ----------------------------------
#
# Transversal gates are the preferred way to perform computation fault-tolerantly with
# a quantum code, as they are supposed to be relatively easy to implement and propagate
# minimal errors. However, the `Earnest-Knill theorem
# <https://en.wikipedia.org/wiki/Eastin%E2%80%93Knill_theorem>`_ restricts the set of the
# logical unitary product operators that can be applied transversally for any nontrivial
# local-error-detecting quantum code to be non-universal. While this limits the ways to implement
# fault-tolerant gates on quantum codes, there are still ways to use non-transversal methods to
# implement a logical gate. For example, we can implement non-Clifford gates such as
# :class:`~.pennylane.T` can be implemented through state injection [#Transversal]_.
#
# One of the key challenges in using quantum LDPC codes in practice is to construct the ones that
# support transversal gates that allow universality. Similar to the commonly used quantum error
# correcting codes, the Clifford gates can be applied transversally for qLDPC codes as well.
# However, there are ways to construct certain families of quantum LDPC codes that support
# applying certain non-Clifford gates like :class:`~.pennylane.CCZ` gates transversally.
#
# Below, we take a look at how to test if a given operation is transversal for a given qLDPC code
# by testing if it preserves its codespace.
#

from itertools import product
import stim

# 2-bit repetition code on a ring
h1, h2 = np.ones((2, 2)), np.ones((2, 2))
hx, hz = hgp_code(h1, h2)  # Toric code


def compute_stabilizer_group(hx: np.ndarray, hz: np.ndarray) -> tuple[list, set]:
    """Generates the independent Pauli checks and the full stabilizer group."""
    n_qubits = hx.shape[1]

    # Create PauliStrings for X-checks and Z-checks
    generators = [
        stim.PauliString(["".join(["I", "X"][bit]) for bit in row])
        for row in hx if np.any(row)
    ]
    generators += [
        stim.PauliString(["".join(["I", "Z"][bit]) for bit in row])
        for row in hz if np.any(row)
    ]

    full_group = set()
    for bits in product([0, 1], repeat=len(generators)):
        current_pauli = stim.PauliString(n_qubits)
        for bit, gen in zip(bits, generators):
            if bit:
                current_pauli *= gen
        full_group.add(str(current_pauli))
        full_group.add(str(-current_pauli))
    return generators, full_group


def verify_transversality(operations: str, gens: list, group: set):
    """Verify if the given operations are transversal for the given qLDPC code."""
    tableau = stim.Tableau(num_qubits=len(gens[0]))
    for op in operations:
        tableau.append(*op)

    is_transversal = True
    for gix, gen in enumerate(gens):
        if str(evolved := tableau(gen)) in group:
            print(f"{gen}  -->  {evolved}  (Valid!)")
        else:
            print(f"{gen}  -->  {evolved}  (Invalid!)")
            is_transversal = False
            break

    return is_transversal


swap = stim.Tableau.from_named_gate("SWAP")
ops = [[swap, (0, 1)], [swap, (2, 3)], [swap, (4, 5)], [swap, (6, 7)]]
gens, stabs = compute_stabilizer_group(hx, hz)
result = verify_transversality(ops, gens, stabs)

print(f"Result: The codespace is preserved: {result}")

######################################################################
# In addition to the gate operations being transversal, there's active work being done to develop
# efficient frameworks to perform logical Pauli measurements, as well [#LMHM]_, which is another
# critical requirement for the partical utility of these codes.
#
# Conclusion
# ----------
#
# The journey to fault-tolerant quantum computing hinges on managing errors without requiring an
# astronomical number of physical qubits. By relaxing the strict nearest-neighbor constraints of
# topological codes, Quantum Low-Density Parity-Check (qLDPC) codes offer a profound paradigm
# shift: they trade massive qubit overhead for a complex hardware connectivity challenge.
# Advancements in dynamically reconfigurable and modular architectures are turning these highly
# connected codes into a physical reality. While still many engineering hurdles remain,
# particularly in designing universal transversal gate sets and executing efficient logical
# measurements. However, supported by fast, linear-time decoding algorithms, qLDPC codes have
# evolved past elegant mathematical formalism and are progressing towards practicality.
#
# References
# ----------
#
# .. [#qldpc1]
#
#     N. P. Breuckmann, J. N. Eberhardt,
#     "Quantum Low-Density Parity-Check Codes",
#     `PRX Quantum 2, 040101 <https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.2.040101>`__, 2021.
#
# .. [#CSS]
#
#     T. Rakovszky, V. Khemani,
#     "The Physics of (good) LDPC Codes I. Gauging and dualities",
#     `arXiv:2310.16032 <https://arxiv.org/abs/2310.16032>`__, 2023.
#
# .. [#HGP]
#
#     J.-P. Tillich, G. Zémor,
#     "Quantum LDPC Codes With Positive Rate and Minimum Distance Proportional to the Square Root of the Blocklength",
#     `IEEE Transactions on Information Theory 60(1), 119–136 <https://ieeexplore.ieee.org/document/6671468>`__, 2014.
#
# .. [#LPCodes]
#
#     F. G. Jeronimo, T. Mittal, R. O'Donnell, P. Paredes, M. Tulsiani,
#     "Explicit Abelian Lifts and Quantum LDPC Codes",
#     `arXiv:2112.01647 <https://arxiv.org/abs/2112.01647>`__, 2021.
#
# .. [#QTCodes]
#
#     A. Leverrier, G. Zémor,
#     "Quantum Tanner codes",
#     `arXiv:2202.13641 <https://arxiv.org/abs/2202.13641>`__, 2022.
#
# .. [#BBCodes]
#
#     S. Bravyi, A. W. Cross, J. M. Gambetta, D. Maslov, P. Rall, T. J. Yoder,
#     "High-threshold and low-overhead fault-tolerant quantum memory",
#     `Nature <https://www.nature.com/articles/s41586-024-07107-7>`__, 2024.
#
# .. [#BProp]
#
#     J. Old, M. Rispler,
#     "Generalized Belief Propagation Algorithms for Decoding of Surface Codes",
#     `Quantum 7, 1037 <https://quantum-journal.org/papers/q-2023-06-07-1037/>`__, 2023.
#
# .. [#OSD0]
#
#     J. Valls, F. Garcia-Herrero, N. Raveendran, B. Vasic,
#     "Syndrome-Based Min-Sum vs OSD-0 Decoders: FPGA Implementation and Analysis for Quantum LDPC Codes",
#     `IEEE Access <https://ieeexplore.ieee.org/document/9562513>`__, 2021.
#
# .. [#Transversal]
#
#     H. Leitch, A. Kay,
#     "Transversal Gates for Highly Asymmetric qLDPC Codes",
#     `arXiv:2506.15905 <https://arxiv.org/abs/2506.15905>`__, 2025.
#
# .. [#LMHM]
#
#     B. Ide, M. G. Gowda, P. J. Nadkarni, G. Dauphinais,
#     "Fault-tolerant logical measurements via homological measurement",
#     `Phys. Rev. X 15, 021088 <https://arxiv.org/abs/2410.02753>`__, 2024.
#