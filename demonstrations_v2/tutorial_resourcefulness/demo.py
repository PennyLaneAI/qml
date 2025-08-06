r"""
Analysing quantum resourcefulness with the generalized Fourier transform
========================================================================

.. figure:: ../_static/demo_thumbnails/opengraph_demo_thumbnails/OGthumbnail_resourcefulness.png
    :align: center
    :width: 70%
    :alt: DESCRIPTION.
    :target: javascript:void(0)

Resource theories in quantum information theory ask how "complex" a given quantum state is with respect to a certain
measure of complexity. For example, using the resource of entanglement, we can ask how entangled a quantum state is. Other well-known
resources are Clifford stabilizerness, which measures how close a state is from being prepared by a circuit that only uses
classically simulatable Clifford gates, or Gaussianity, which measures how far away a state is from a so-called "Gaussian state"
that is relatively easy to prepare in quantum optics, and likewise classically simulatable. As the name "resourceful" suggests,
these measures of complexity often relate to how much "effort" states are, for example with respect to classical simulation or
preparation in the lab.

It turns out that the resourcefulness of quantum states can be investigated with tools from generalised Fourier analysis [#Bermejo_Braccia]_.
Fourier analysis here refers to the well-known technique of computing Fourier coefficients of a mathematical object, which in our case
is not a function over :mathbb:`R` or :mathbb:`Z`, but a quantum state. "Generalised" indicates that we don't use the
standard Fourier transform, but its group-theoretic generalisations [LINK TO RELATED DEMOS]. This is important, because
[#Bermejo_Braccia]_ link a resource to a group -- essentially, by defining the set of unitaries that maps resource-free
states to resource-free states as a "representation" of a group, which gets block-diagonalised to find a generalised Fourier basis.
The intuition, however, is exactly the same as in the standard Fourier transform: large higher-order Fourier
coefficients indicate a less "smooth" or "resourceful" function.

In this tutorial we will illustrate the idea of generalised Fourier analysis for resource theories with two simple examples.
First we will look at a standard Fourier decomposition of a function, but from the perspective of resources,in order to
introduce the basic idea in a familiar setting. Secondly, we will use the same concepts to analyse the entanglement
 resource of quantum states, reproducing Figure 2 in [#Bermejo_Braccia]_.

.. figure:: ../_static/demonstration_assets/resourcefulness/figure2_paper.png
   :align: center
   :width: 70%
   :alt: Fourier coefficients, or projections into "irreducible subspaces", of different states using 2-qubit entanglement as a resource.
         A Bell state, which is maximally entangled, has high contributions in higher-order Fourier coefficients, while
         a tensor product state with little entanglement has contributions in lower-order Fourier coefficients. The interpolation
         between the two extremes, exemplified by a Haar random state, has a Fourier spectrum in between.

Luckily, in the case of entanglement as a resource, the generalised Fourier coefficients can be computed using Pauli expectations.
This saves us from diving too deep into representation theory. In fact, the tutorial should be informative without knowing much
about groups at all! But of course, even in this simple case the numerical analysis scales exponentially with the number of qubits
in general. It is a fascinating question in which situations the Fourier coefficients of a physical state could be read out
on a quantum computer, which is known to sometimes perform the block-diagonalisation efficiently.


Standard Fourier analysis through the lense of resources
--------------------------------------------------------

Let's start recalling the standard Fourier transform, and for simplicity we'll work with the
discrete version. Given N real values :math:`x_0,...,x_{N-1}`, that we can interpret [LINK TO OTHER DEMO] as the values
of a function :math:`f(0), ..., f(N-1)` over the integers :math:`x \in {0,...,N-1}`, the Fourier transform
computes the Fourier coefficients

.. math::
        \hat{f}(k) = \frac{1}{\sqrt{N}\sum_{x=0}^{N-1} f(x) e^{\frac{2 \pi i}{N} k x}, k = 0,...,N-1

In words, the Fourier coefficients are projections of the function :math:`f` onto the "Fourier" basis functions
:math:`e^{\frac{2 \pi i}{N} k x}`. Note that we use a normalisation here that is consistent with a unitary transform that
we construct as a matrix below.

Let's see this equation in action.
"""

import matplotlib.pyplot as plt
import numpy as np

N = 12

def f(x):
    """Some function."""
    return 0.5*(x-4)**3

def f_hat(k):
    """Fourier coefficients of f."""
    projection = [ f(x) * np.exp(-2 * np.pi * 1j * k * x / N)/np.sqrt(N) for x in range(N)]
    return  np.sum(projection)

def plot(f, f_hat):

    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.bar(range(N), [np.real(f(x)) for x in range(N)], color='dimgray') # casting to real is needed in case we perform an inverse FT
    ax1.set_title(f"function f")

    ax2.bar(np.array(range(N))+0.05, [np.imag(f_hat(x)) for x in range(N)], color='lightpink', label="imaginary part")
    ax2.bar(range(N), [np.real(f_hat(k)) for k in range(N)], color='dimgray', label="real part")
    ax2.set_title("Fourier coefficients")
    plt.legend()
    plt.tight_layout()
    plt.show()

plot(f, f_hat)

######################################################################
# Now, what kind of resource are we dealing with here? In other words, what
# measure of complexity gets higher when a function has large higher-order Fourier coefficients?
#
# Well, let us look for the function with the least resource or complexity! For this we can just
# work backwards: define a Fourier spectrum that only has a contribution in the lowest order coefficient,
# and apply the inverse Fourier transform to look at the function it corresponds to!
#
#

def g_hat(k):
    """The least complex Fourier spectrum possible."""
    if k==0:
        return 1
    else:
        return 0

def g(x):
    """Function whose Fourier spectrum is `g_hat`."""
    projection = [g_hat(k) * np.exp(2 * np.pi * 1j * k * x / N)/np.sqrt(N) for k in range(N)]
    return np.sum(projection)

plot(g, g_hat)

######################################################################
# Well, the function is constant. This makes sense, because we know that the decay of the Fourier coefficient
# is related to the number of times a function is differentiable, which in turn is the technical definition of smoothness. A
# constant function is maximally smooth -- it does not wiggle at all. In other words,
# the resource of the standard Fourier transform is smoothness, and a resource-free function is constant!
#

######################################################################
# Linking resources to group representations
# ------------------------------------------
#
# Fourier transforms are intimately linked to groups (in fact, the x-domain :math:`0,...,N-1` is strictly speaking a group,
# which you can learn more about [here]). Without expecting you to know group jargon, we have to establish a few concepts
# that generalise the above example to quantum states and more generic resources. The crucial idea is to
# define a resource by fixing a set of vectors (later, quantum states) that are considered resource-free.
# We also need to define unitary matrices that map resource-free vectors to resource-free vectors,
# and these matrices need to form a *unitary representation* of a group :math:`G`,
# which is a matrix-valued function :math:`R(g), g \in G` on the group.
# This is all we need to guarantee that the matrices can be simultaneously block-diagonalised, or written as a
# direct sum of smaller matrix-valued functions over the group, :math:`r^{\alpha}(g)`. Note that some of the blocks may be identical.
# If you know group theory, then you will recognise that the :math:`r^{\alpha}(g)` are the *irreducible representations* of the group.
#
# [TODO: image]
#
#
# What is important for us is that these smaller matrix-valued functions :math:`r^{\alpha}(g)` define a
# subspace :math:`V_{\alpha, j}`, where the index :math:`j` accounts for the fact that there may be several copies of
# an :math:`r^{\alpha}(g)` on the diagonal of :math:`R(g)`, each of which corresponds to one subspace.
# [TODO: BASIS :math:`w^{(i)}_{\alpha, j}`].
#
# The Generalised Fourier Decomposition (GFD) purity is the length of a projection of a vector :math:`v \in V` onto one of these subspaces :math:`V_{\alpha, j}`,
#
# .. math::
#           \mathcal{P}(v) = \sum_i  | \langle w^{(i)}_{\alpha, j}, v \rangle |^2.
#
# In our standard
# example above, the subspaces are one-dimensional spaces spanned by the Fourier basis functions :math:`\chi_k(x) = e^{\frac{2 \pi i}{N} k x}`.
# The vector space is the space of functions (if this is confusing, think of the function values as arranged in
# N-dimensional vectors :math:`\vec{f}` and :math:`\vec{\chi}_k`).
# The projection is executed by the sum :math:`\sum_x f(x) \chi_k(x) = \langle f, \chi_k`. In other words, the GFD purity
# is the absolute square of the Fourier coefficient,
#
#.. math::
#           \mathcal{P}(\vec{f}) =  | \vec{\chi}_k, \vec{f} \rangle |^2 = |\hat{f}|^2.
#
# Generalising from the standard Fourier basis functions to irreducible
# representations allows us to generalise the Fourier transform to lots of other groups, and hence, resources.
#
# So far so good, but what is the representation :math:`R(g)` for the standard Fourier transform? Let's follow the recipe:
#
# 1. We first need to consider our function :math:`f(0), ..., f(N-1)` as a vector :math:`[f(0), ..., f(N-1)]^T \in V = \mathbb{R}^N`.
# 2. As argued above, the set of resource-free vectors correspond to constant functions, :math:`f(0) = ... = f(N-1)`.
# 3. We need a set of unitary matrices that does not change the "constantness" of the vector. This set is given by permutation matrices,
#    which swap the entries of the vector but do not change any of them.
# 4. The *circulant* permutation matrices can be shown to form a unitary representation :math:`R(g)` for :math:`g \in G = Z_N`, called the *regular representation*.
# 5. We are now guaranteed that there is a basis change that diagonalises all matrices :math:`R(g)` together.
#    (Note that the Fourier transform is sometimes defined as the basis change that block-diagonalises the regular representation!) As mentioned, this is
#    a block-diagonalisation where the blocks happen to be 1-dimensional, as is the rule for so-called "Abelian" groups.
# 6. The values on the diagonal of :math:`R(g)` under this basis change are exactly the Fourier basis functions `:math:`e^{\frac{2 \pi i}{N} k x}`.
#
# Let's verify this!
#
# First, let us write our function and Fourier spectrum above as vectors:
#

f_vec = np.array([f(x) for x in range(N)])
f_hat_vec = np.array([f_hat(k) for k in range(N)])

######################################################################
# The Fourier transform then becomes a matrix multiplication with the matrix:
#
# .. math::
#      F = \frac{1}{\sqrt{N}} \begin{bmatrix}
#            1&1&1&1&\cdots &1 \\
#            1&\omega&\omega^2&\omega^3&\cdots&\omega^{N-1} \\
#            1&\omega^2&\omega^4&\omega^6&\cdots&\omega^{2(N-1)}\\ 1&\omega^3&\omega^6&\omega^9&\cdots&\omega^{3(N-1)}\\
#            \vdots&\vdots&\vdots&\vdots&\ddots&\vdots\\
#            1&\omega^{N-1}&\omega^{2(N-1)}&\omega^{3(N-1)}&\cdots&\omega^{(N-1)(N-1)}
#           \end{bmatrix}
#
# Let's check that this is indeed the same as what we did above.
#

# get the Fourier matrix from scipy
from scipy.linalg import dft
F = dft(N)/np.sqrt(N)

# compare to what we previously computed
print(np.allclose(F @ f_vec, f_hat_vec))


######################################################################
# Above we claimed that this matrix F is the basis transform that diagonalises
# a permutation matrix (in other words, the unitary representation evaluated at some group element).
#

# create a circulant permutation matrix
i = np.random.randint(0, N)
first_row = np.zeros(N)
first_row[i] = 1

# initialize the matrix with the first row
P = np.zeros((N, N))
P[0, :] = first_row

# generate subsequent rows by cyclically shifting the previous row
for i in range(1, N):
    P[i, :] = np.roll(P[i-1, :], 1)


# change into the Fourier basis using F
np.set_printoptions(precision=2, suppress=True)
P_F =  F @ P @ np.linalg.inv(F)

# check if the resulting matrix is diagonal
# trick: remove diagonal and make sure the remainder is zero
print(np.allclose(P_F - np.diag(np.diagonal(P_F)), np.zeros((N,N)) ))


######################################################################
# To recap, we saw that the standard Fourier analysis can be generalised by
# interpreting "smoothness" or "constantness" as a resource, linking it to a vector space and a group
# representation and (block-)diagonalising the representation. The blocks, here one-dimensional,
# form a basis for subspaces. The GFD purities suggested in [#Bermejo_Braccia]_ as a resource fingerprint
# are just projections of some vector (here, a function) onto the basis vectors and taking the absolute square.
#
# Armed with this recipe, we can now try to analyse entanglement as a resource,
# and density matrices describing quantum states as vectors in a vector space :math:`L(H)`.
#

######################################################################
# Fourier analysis of entanglement
# --------------------------------
#
# Now that we have a grasp of how standard Fourier transforms can measure the resource of "smoothness" of a classical function,
# let's apply the same kind of reasoning to the most popular resource of quantum states: multipartite entanglement.
# We can think of multipartite entanglement as a resource of the state of a quantum system that measures how "wiggly"
# the correlations between its constituents are.
# While this is a general statement, we will restrict our attention to systems made of our favourite quantum objects, qubits.
# Just like smoother functions have simpler Fourier spectra (mostly low-momentum components),
# quantum states with simpler entanglement structures, or no entanglement at all like in the case of product states,
# have simpler generalized Fourier spectra, where most of their purity resides in the lower-order "irreps"
# (that we recall is short for irreducible representations, but you can think of them as generalized frequencies).
# On the flip side, highly entangled states spread their purity across more and higher-order (i.e., larger dimensional) irreps,
# similar to how more complex ("wiggly") functions have larger Fourier coefficients at higher frequencies.
# This means that analyzing a state's Fourier spectrum can tell us how "resourceful" or entangled it is.
#
# Let us walk the same steps we took when studying the resource of "smoothness".
# 1. **Vectorise**. Luckily for us, our objects of interest, the quantum states :math:`\psi \in \mathbb{C}^{2n}`, are already in the form of vectors.
# 2. **Identify free vectors**. It's easy to define the set of free states for multipartite entanglement: tensor products of single-qubit quantum states.
# 3. **Identify free linear transformations**. Now, what unitary transformation of a quantum state does not generate
#    entanglement? You guessed it right, "non-entangling" circuits that consist only of single-qubit gates :math:`U=\Bigotimes U_j` for  :math:`U_j \in SU(2)`.
# 4. **Ensure they form a representation**. It turns out that non-entangling unitaries are the standard representation
#    of the group :math:`G = SU(2) x SU(2) ... x SU(2)` for the vector space :math:`H`, the Hilbert space of the state vectors.
#    Again, this implies that we can find a basis of the Hilbert space, where any :math:`U` is simultaneously block-diagonal.
#
# Let's stop here for now, and try to block-diagonalise one of the non-entangling unitaries. We can use an
# eigenvalue decomposition for this.
#

from scipy.stats import unitary_group
from functools import reduce

n = 2

# create n haar random single-qubit unitaries
Us = [unitary_group.rvs(dim=2) for _ in range(n)]
# compose them into a non-entangling n-qubit unitary
U = reduce(lambda A, B: np.kron(A, B), Us)

# Block-diagonalise
vals, U_bdiag = np.linalg.eig(U)

print(U_bdiag)

######################################################################
# Wait, this is not a block-diagonal matrix, even if we'd shuffle the rows and columns. What is happening here?
# It turns out that the non-entangling unitary matrices only have a single block of size :math:`2^n \times 2^n`.
# Technically speaking, the representation :math:`R(g) = U(g)` is irreducible over :math:`H`.
# As a consequence, the invariant subspace is :math:`H` itself, and the GFD purity is simply the purity of the state :math:`psi`, which is :math:`1`.
# This, of course, is not a very informative measure!
#
#
# However, rather than a bug, this is a feature of the GFD framework. Indeed, nobody forces us to restrict our attention to :math:`H` and to
# the (standard) representation of non-entangling unitary matrices.
# After all, what's the first symptom of entanglement in a quantum state? You guessed right again, the reduced density matrix of some subsystem is mixed!
# Wouldn't it make more sense then to study the multipartite entanglement form the point of view of density matrices?
# It turns out that moving into the space :math:`B(H)` of bounded linear operators in which the density matrices live leads to
# a much more nuanced Fourier spectrum.
#
# Let us walk the steps above again
# 1. **Vectorise**. :math:`L(H)` is a vector space in the technical sense, but one of matrices. To use the linear algebra
#    tricks from before we have to "vectorize" density matrices :math:`rho=\sum_i,j c_i,j |i\rangle \langle j|`
#    into vectors :math:`|rho\rangle \rangle = \sum_i,j c_i,j |i\rangle |j\rangle \in H \otimes H^*`.
#    For example:
#

n = 2

# create a random quantum state
psi = np.random.rand(2**n)
# construct the corresponding density matrix
rho = np.outer(psi, psi.conj())
# vectorise it
rho_vec = rho.flatten(order='F')

######################################################################
# 2. **Identify free vectors**. The set of free states is again that of product states :math:`\rho = \bigotimes \rho_j`
#    where each :math:`\rho_j` is a single-qubit state. We only have to write them in flattened form.
# 3. **Identify free linear transformations**. The free operations are still given by non-entangling unitaries, but
#    of course, they act on density matrices via :math:`\rho' = U \rho U^{\dagger}`.
#    We can also vectorise this operation by defining the matrix :math:`U_{\mathrm{vec}}(g) = U \otimes U^*`.
#    We then have that :math:`|\rho'\rangle \rangle = U_{\mathrm{vec}} \rho`.
#    To demonstrate:
#


# Vectorise U
U_vec = np.kron(U.conj(), U)

# evolve the state above by U, using the vectorised objects
rho_out_vec = U_vec @ rho_vec
# reshape the result back into a density matrix
rho_out = np.reshape(rho_out_vec, newshape=(2**2, 2**2))
# this is the same as the usual adjoint application of U
print(np.allclose(rho_out, U @ rho @ U.conj().T ))

######################################################################
# 4. **Ensure they form a representation**. This "adjoint action" is indeed a valid representation of
# :math:`G = SU(2) x SU(2) ... x SU(2)`, called the *defining representation*. However, it is a different one from before,
# and this time there is a basis transformation that properly block-diagonalises all matrices in the representation.
# To find this transformation we compute the eigendecomposition of an arbitrary linear combination of a set of matrices in the representation.
#

U_vecs = []
for i in range(10):
    # create n haar random single-qubit unitaries
    Ujs = [unitary_group.rvs(dim=2) for _ in range(n)]
    # compose them into a non-entangling n-qubit unitary
    U = reduce(lambda A, B: np.kron(A, B), Ujs)
    # Vectorise U
    U_vec = np.kron(U.conj(), U)
    U_vecs.append(U_vec)

# Create a random linear combination of the matrices
alphas = np.random.randn(len(U_vecs)) + 1j * np.random.randn(len(U_vecs))
M_combo = sum(a * M for a, M in zip(alphas, U_vecs))

# Eigendecompose the linear combination
vals, Q = np.linalg.eig(M_combo)

######################################################################
# Let's test this basis change
#

Qinv = np.linalg.inv(Q)
# take one of the vectorised unitaries
U_vec = U_vecs[0]
U_vec_diag = Qinv @ U_vec @ Q

np.set_printoptions(
    formatter={'float': lambda x: f"{x:5.2g}"},
    linewidth=200,    # default is 75; increase to fit your array
    threshold=10000   # so it doesn’t summarize large arrays
)

print(U_vec_diag)

######################################################################
# But `U0_diag` does not look block diagonal. What happened here?
# Well, it is block-diagonal, but we have to reorder the columns and rows of the final matrix to make this visible.
# This takes a bit of pain, encapsulated in the following function:
#

from collections import OrderedDict

def group_rows_cols_by_sparsity(B, tol=0):
    """
    Given a matrix B, this function:
      1. Groups identical rows and columns.
      2. Orders these groups by sparsity (most zeros first).
      3. Returns the permuted matrix C2, and the row & column permutation
         matrices P_row, P_col such that C2 = P_row @ C @ P_col.

    Parameters
    ----------
    B : ndarray, shape (n, m)
        Input matrix.

    Returns
    -------
    P_row : ndarray, shape (n, n)
        Row permutation matrix.
    P_col : ndarray, shape (m, m)
        Column permutation matrix.
    """
    # Compute boolean mask where |B| >= tol
    mask = np.abs(B) >= 1e-8
    # Convert boolean mask to integer (False→0, True→1)
    C = mask.astype(int)
    n, m = C.shape

    # Helper to get a key tuple and zero count for a vector
    def key_and_zeros(vec):
        if tol > 0:
            bin_vec = (np.abs(vec) < tol).astype(int)
            key = tuple(bin_vec)
            zero_count = int(np.sum(bin_vec))
        else:
            key = tuple(vec.tolist())
            zero_count = int(np.sum(np.array(vec) == 0))
        return key, zero_count

    # Group rows by key
    row_groups = OrderedDict()
    row_zero_counts = {}
    for i in range(n):
        key, zc = key_and_zeros(C[i, :])
        row_groups.setdefault(key, []).append(i)
        row_zero_counts[key] = zc

    # Sort row groups by zero_count descending
    sorted_row_keys = sorted(row_groups.keys(),
                             key=lambda k: row_zero_counts[k],
                             reverse=True)
    # Flatten row permutation
    row_perm = [i for key in sorted_row_keys for i in row_groups[key]]

    # Group columns by key
    col_groups = OrderedDict()
    col_zero_counts = {}
    for j in range(m):
        key, zc = key_and_zeros(C[:, j])
        col_groups.setdefault(key, []).append(j)
        col_zero_counts[key] = zc

    # Sort column groups by zero_count descending
    sorted_col_keys = sorted(col_groups.keys(),
                             key=lambda k: col_zero_counts[k],
                             reverse=True)
    col_perm = [j for key in sorted_col_keys for j in col_groups[key]]

    # Build permutation matrices
    P_row = np.eye(n)[row_perm, :]
    P_col = np.eye(m)[:, col_perm]

    return P_row, P_col

P_row, P_col = group_rows_cols_by_sparsity(U_vec_diag)

Q = Q @ P_col
Qinv = P_row @ Qinv

U_vec_diag = Qinv @ U_vec @ Q

print(U_vec_diag)

######################################################################
# The reordering made the block structure visible. You can check that any vectorised non-entangling matrix `U_vec`
# has the same block structure if we change the basis via `Qinv @ U_vec @ Q`.
#
# But what basis have we actually changed into? It turns out that `Q` changes into the Pauli basis!
#

Qinv @ U_vec @ Q

# We cheat here a little: we already know the basis in which :math:`\tilde{U}` is block-diagonal: the Pauli basis.
# This allows us to construct the matrix implementing the basis change directly.
#

import functools
import itertools

# single‑qubit Paulis
_pauli_map = {
    'I': np.array([[1,0],[0,1]],   dtype=complex),
    'X': np.array([[0,1],[1,0]],   dtype=complex),
    'Y': np.array([[0,-1j],[1j,0]],dtype=complex),
    'Z': np.array([[1,0],[0,-1]],  dtype=complex),
}

def pauli_basis(n):
    """
    Generates the basis of Pauli operators, and orders it by appearence in the isotypical decomp of Times_i SU(2).
    """
    all_strs    = [''.join(s) for s in itertools.product('IXYZ', repeat=n)]
    sorted_strs = sorted(all_strs, key=lambda s: (n-s.count('I'), s))
    norm        = np.sqrt(2**n)
    mats        = []
    for s in sorted_strs:
        factors = [_pauli_map[ch] for ch in s]
        M       = functools.reduce(lambda A,B: np.kron(A,B), factors)
        mats.append(M.reshape(-1)/norm)
    B = np.column_stack(mats)
    return sorted_strs, B




# compute the basis change matrix
basis, B = pauli_basis(n)
# apply basis change
U_vec_rot = B.conj().T @ U_vec @ B

# make sure the imaginary part is zero, so we don't have to print it
# (this is a property of the Pauli basis)
assert np.isclose(np.sum(U_vec_rot.imag), 0)

# print the real part
print(np.round(U_vec_rot.real, 2))


# 5. **Identify basis for invariant subspaces**. Will we now find interesting subspaces by simultaneously block-diagonalizing :math:`\tilde{R}`?
#


#
# References
# ----------
#
# .. [#Bermejo_Braccia]
#
#     Bermejo, Pablo, Paolo Braccia, Antonio Anna Mele, Nahuel L. Diaz, Andrew E. Deneris, Martin Larocca, and M. Cerezo. "Characterizing quantum resourcefulness via group-Fourier decompositions." arXiv preprint arXiv:2506.19696 (2025).
#


######################################################################
#
# About the author
# ----------------
#
