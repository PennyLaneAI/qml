r"""
Analysing quantum resourcefulness with the generalized Fourier transform
========================================================================

.. figure:: ../_static/demo_thumbnails/opengraph_demo_thumbnails/OGthumbnail_resourcefulness.png
    :align: center
    :width: 70%
    :alt: DESCRIPTION.
    :target: javascript:void(0)

Resource theories in quantum information ask how "complex" a given quantum state is with respect to a certain
measure of complexity. For example, using the resource of *entanglement*, we can ask how entangled a quantum state is. Other well-known
resources are *Clifford stabilizerness*, which measures how close a state is from being preparable by a
classically simulatable "Clifford circuit", or *Gaussianity*, which measures how far away a state is from "Gaussian states"
considered simple in quantum optics. As the name "resourceful" suggests,
these measures of complexity often relate to how much "effort" states are, for example with respect to classical simulation or
preparation in the lab.

It turns out [#Bermejo_Braccia]_ that the resourcefulness of quantum states can be investigated with tools from *generalised Fourier analysis*.
"Fourier analysis" here refers to the well-known technique of computing Fourier coefficients of a mathematical object, which in our case
is not a function over :mathbb:`R` or :mathbb:`Z`, but a quantum state. "Generalised" indicates that we don't use the
standard Fourier transform, but a generalisation of its group theoretic definition.
[#Bermejo_Braccia]_ suggest to compute a quantity that they call the **Generalised Fourier Decomposition (GFD) Purity**,
and use it as a "footprint" of a state's resource profile. When using the standard Fourier transform,
the GFD purities are just the absolute squares of the normal Fourier coefficients, which is also known as the *power spectrum*.

The basic idea is to identify the set of unitaries that maps resource-free
states to resource-free states with a *linear representation* of a group.
The basis in which this representation, and hence the free unitaries, are (block-)diagonal, reveals so-called *irreducible subspaces*.
The GFD Purities are then the "weight" a state has in these subspaces. As in standard Fourier analysis,
higher-order Purities indicate a less resourceful function.
This intuition carries over to the generalised case, where more resourceful states have higher weights in higher-order subspaces.

In this tutorial we will illustrate with two simple examples how to compute the GFD Purities to analyse resource.
To introduce the basic recipe in a familiar setting, we will first look at the power spectrum of a discrete function.
We will then use the same concepts to analyse the *entanglement*
of quantum states as a resource, reproducing Figure 2 in [#Bermejo_Braccia]_.

.. figure:: ../_static/demonstration_assets/resourcefulness/figure2_paper.png
   :align: center
   :width: 70%
   :alt: GFD Purities of different states using 2-qubit entanglement as a resource.
         A GHZ state, which is maximally entangled, has high contributions in higher-order Purities, while
         a product state with no entanglement has contributions in lower-order Purities. The interpolation
         between the two extremes, exemplified by a Haar random state, has a spectrum in between.

While the theory rests in group theory, the tutorial is aimed at readers who don't know much about groups at all,
since everything can be understood by applying standard linear algebra.

.. note::
    Of course, even in this simple case the numerical analysis scales exponentially with the number of qubits, and
    everything we present here is only possible to compute for small system sizes. It is a fascinating question in
    which situations the GFD Purities of a physical state could be read out
    on a quantum computer, which is known to sometimes perform the block-diagonalisation efficiently.


Standard Fourier analysis through the lense of resources
--------------------------------------------------------

The power spectrum as GFD Purities
++++++++++++++++++++++++++++++++++

Let's start recalling the standard discrete Fourier transform.
Given N real values :math:`x_0,...,x_{N-1}`, that we can interpret [LINK TO OTHER DEMO] as the values
of a function :math:`f(0), ..., f(N-1)` over the integers :math:`x \in {0,...,N-1}`, the Fourier transform
computes the Fourier coefficients

.. math::
        \hat{f}(k) = \frac{1}{\sqrt{N}\sum_{x=0}^{N-1} f(x) e^{\frac{2 \pi i}{N} k x}, k = 0,...,N-1

In words, the Fourier coefficients are projections of the function :math:`f` onto the "Fourier" basis functions
:math:`e^{\frac{2 \pi i}{N} k x}`. Note that we use a normalisation here that is consistent with a unitary transform.

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




f_vals = [f(x) for x in range(N)]
f_hat_vals = [f_hat(k) for k in range(N)]

fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.bar(range(N), f_vals, color='dimgray')
ax1.set_title(f"function f")

ax2.bar(np.array(range(N))+0.05, np.imag(f_hat_vals), color='lightpink', label="imaginary part")
ax2.bar(range(N), np.real(f_hat_vals), color='dimgray', label="real part")
ax2.set_title("Fourier coefficients")
plt.legend()
plt.tight_layout()
plt.show()


######################################################################
# Now, we mentioned that the absolute square of the standard Fourier coefficients are the simplest example of
# GFD Purities, so for this case, we can easily compute our quantity of interest!
#

purities = [np.abs(f_hat(k))**2 for k in range(N)]
plt.plot(purities)
plt.ylabel("GFD purities")
plt.xlabel("k")
plt.tight_layout()
plt.show()


######################################################################
# Smoothness as a resource
# ++++++++++++++++++++++++
#
# But what kind of resource are we dealing with here? In other words, what
# measure of complexity gets higher when a function has large higher-order Fourier coefficients?
#
# Well, let us look for the function with the *least* resource or complexity! For this we can just
# work backwards: define a Fourier spectrum that only has a contribution in the lowest order coefficient,
# and apply the inverse Fourier transform to look at the function it corresponds to!
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

g_hat_vals = [g_hat(x) for x in range(N)]
g_vals = [g(k) for k in range(N)]

fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.bar(range(N), g_hat_vals, color='dimgray')
ax1.set_title("Fourier coefficients")

ax2.bar(np.array(range(N))+0.05, np.imag(g_vals), color='lightpink', label="imaginary part")
ax2.bar(range(N), np.real(g_vals), color='dimgray', label="real part")
ax2.set_title(f"function g")

plt.legend()
plt.tight_layout()
plt.show()

######################################################################
# The function is constant. This makes sense, because we know that the decay of the Fourier coefficient
# is related to the number of times a function is differentiable, which in turn is the technical definition of "smoothness". A
# constant function is maximally smooth -- it does not wiggle at all. In other words,
# the resource of the standard Fourier transform is smoothness, and a resource-free function is constant!
#

######################################################################
# The general recipe
# ++++++++++++++++++
#
# Computing the GFD Purities for the resource of smoothness was very simple. We will now make it more complicated,
# in order to set up the machinery that calculates the Purities for more general cases like quantum entanglement.
#
# 1. Find free vectors
# ********************
# The first step is to find "free vectors". So far we dealt with a discrete function :math:`f(0), ..., f(N-1)`,
# but we can easily write it as a vector :math:`[f(0), ..., f(N-1)]^T \in V = \mathbb{R}^N`. As argued above,
# the set of resource-free vectors corresponds to constant functions, :math:`f(0) = ... = f(N-1)`.
#

f_vec = np.array([f(x) for x in range(N)])

######################################################################
# 2. Find free unitary transformations
# ************************************
# Next, we need to identify a set of unitary matrices that does not change the
# "constantness" of the vector. In the case above this is given by the permutation matrices, which
# swap the entries of the vector but do not change any of them.
#
# 3. Identify a group representation
# **********************************
# This is the most difficult step, which sometimes requires a lot of group-theoretic knowledge. Essentially, we have to associate
# the unitary transformations, or a subset of them, with a *unitary group representation*. A representation is a
# matrix-valued function :math:`R(g), g \in G` on a group with the property that :math:`R(g) R(g') = R(gg')`,
# where :math:`gg'` is the composition of elements according to the group. Without dwelling further, we simply
# recognise that the *circulant* permutation matrices -- those that shift vector
# entries by :math:`s` positions -- can be shown to
# form a unitary representation for the group :math:`g \in G = Z_N`, called the *regular representation*. The group :math:`Z_N`
# are the integers from 0 to N-1 under addition modulo N, which can be seen as the x-domain of our function.
# Every circulant matrix hence gets associated with one x-value.
#

s = 2

# create a circulant permutation matrix
first_row = np.zeros(N)
first_row[s] = 1

# initialize the matrix with the first row
P = np.zeros((N, N))
P[0, :] = first_row

# generate subsequent rows by cyclically shifting the previous row
for i in range(1, N):
    P[i, :] = np.roll(P[i-1, :], 1)

print(P)


######################################################################
# 4. Find the basis that block-diagonalises the representation
# ************************************************************
# This was quite a bit of group jargon, but for a good reason: We are now guaranteed that there is
# a basis change that block-diagonalises *all* circulant permutation matrices. If you know group theory,
# then you will recognise that the blocks reveal the *irreducible representations* of the group.
# In more general linear algebra language, the blocks form different *irreducible subspaces* which we need to compute
# the GFD Purities in the next step.
#
# In our toy case, and for all *Abelian* groups, the blocks are 1-dimensional, and the basis
# change *diagonalises* every matrix of the representation. And the basis change is nothing other than the Fourier transform.
# In matrix notation, the Fourier transform looks as follows:
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

# make sure this is what we previously computed
assert np.allclose(F @ f_vec, f_hat_vals)

# change the circulant matrix into the Fourier basis using F
np.set_printoptions(precision=2, suppress=True)
P_F =  np.linalg.inv(F) @ P @ F

# check if the resulting matrix is diagonal
# trick: remove diagonal and make sure the remainder is zero
print("is diagonal:", np.allclose(P_F - np.diag(np.diagonal(P_F)), np.zeros((N,N)) ))

######################################################################
# In fact, the Fourier transform is sometimes *defined* as the basis change that diagonalises the regular representation!
# For other representations, however, we need other transforms.
#
# 5. Find a basis for each subspace
# *********************************
# The rows of the basis transformation matrix, here ``F``, with the row/column indices of a block in ``P_F`` form a basis
# for the subspace associated with each block. In [#Bermejo_Braccia]_ these basis vectors where
# called :math:`w^{(i)}_{\alpha, j}`, where :math:`\alpha` marks the subspace and :math:`j` the copy or multiplicity
# of a block on the diagonal (a technical detail we will not explore further here).
#
# 6. Compute the GFD Purities
# ***************************
# We can now compute the GFD Purities according to Eq. (5) in [#Bermejo_Braccia]_.
#
# .. math::
#           \mathcal{P}(v) = \sum_i  | \langle w^{(i)}_{\alpha, j}, v \rangle |^2.
#
#
# We now have a very different perspective on the power spectrum :math:`|\hat{f}|^2`: It is a projection of the function
# :math:`f` into irreducible subspaces revealed by moving into the Fourier basis. The power spectrum, as
# is widely known, is a footprint for the smoothness of the function. But, having chosen a rather general
# group-theoretic angle, can we do the same for other groups and vector spaces -- for example those
# relevant to investigate resources of quantum states?
#

######################################################################
# Fourier analysis of entanglement
# --------------------------------
#
# Let us try to apply the same kind of reasoning to the most popular resource of quantum states: multipartite entanglement.
# We can think of multipartite entanglement as a resource of the state of a quantum system that measures how "wiggly"
# the correlations between its constituents are.
# While this is a general statement, we will restrict our attention to systems made of our favourite quantum objects, qubits.
# Just like smoother functions have simpler Fourier spectra (mostly low-momentum components),
# quantum states with simpler entanglement structures, or no entanglement at all like in the case of product states,
# have simpler generalized Fourier spectra, where most of their purity resides in the lower-order "irreps"
# (that we recall is short for irreducible representations, but you can think of them as generalized frequencies).
# Here are a few examples that we will use below:
# * **Product state**:
# * **GHZ state**: A highly structured, maximally entangled state. It behaves like a quantum analog of a
# high-frequency oscillation, having purity concentrated mostly in the highest-order irreps, creating a clear quantum fingerprint.
# * **W state**: Moderately entangled, somewhat between the GHZ and product states, with a broader yet smoother distribution across the irreps.
# * **Random (Haar) state**: Highly complex but without structured entanglement patterns, its purity is distributed more evenly across the spectrum.# In the code examples we'll show shortly, you'll see precisely these patterns emerge.
#

import math
import functools

def haar_product_state(n: int):
    """A "smooth" state with no entanglement. It shows high purity only in the lowest-order GFD Purity coefficients,
    similar to a constant function in classical Fourier analysis.."""
    states = [haar_state(1) for _ in range(n)]
    return functools.reduce(lambda A, B: np.kron(A, B), states)


def haar_state(n: int):
    """A Haar random state vector for *n* qubits is likely to have an intermediate amount of entanglement."""
    N = 2 ** n
    # i.i.d. complex Gaussians
    X = (np.random.randn(N, 1) + 1j * np.random.randn(N, 1)) / np.sqrt(2)
    # QR on the N×1 “matrix” is just Gram–Schmidt → returns Q (N×1) and R (1×1)
    Q, R = np.linalg.qr(X, mode='reduced')
    # fix the overall phase so it’s uniformly distributed
    phase = np.exp(-1j * np.angle(R[0, 0]))
    return (Q[:, 0] * phase)


def w_state(n: int):
    """Moderately entangled, somewhat between the GHZ and product states,
    with a broader yet smoother distribution across the irreps"""
    psi = np.zeros(2 ** n)
    for q in range(n):
        psi[2 ** q] = 1 / math.sqrt(n)
    return psi


def ghz_state(n: int):
    """A highly structured, maximally entangled state. It behaves like a quantum analog of a
       high-frequency oscillation, having purity concentrated mostly in the highest-order coefficients."""
    psi = np.zeros(2 ** n)
    psi[0] = 1 / math.sqrt(2)
    psi[-1] = 1 / math.sqrt(2)
    return psi

n = 2
states = [haar_product_state(n),  haar_state(n), w_state(n), ghz_state(n)]
# move to density matrices
states = [np.outer(state, state.conj()) for state in states]
labels = [ "Product", "Haar", "W", "GHZ"]

#####################################################################
# On the flip side, highly entangled states spread their purity across more and higher-order (i.e., larger dimensional) irreps,
# similar to how more complex ("wiggly") functions have larger Fourier coefficients at higher frequencies.
# This means that analyzing a state's Fourier spectrum can tell us how "resourceful" or entangled it is.
#
# Let us walk the same steps we took when studying the resource of "smoothness".
# 1. **Find free vectors**. Luckily for us, our objects of interest, the quantum states :math:`\psi \in \mathbb{C}^{2n}`,
#    are already in the form of vectors. It's easy to define the set of free states for multipartite entanglement:
#    tensor products of single-qubit quantum states.
# 2. **Find free unitary transformations**. Now, what unitary transformation of a quantum state does not generate
#    entanglement? You guessed it right, "non-entangling" circuits that consist only of single-qubit
#    gates :math:`U=\Bigotimes U_j` for  :math:`U_j \in SU(2)`.
# 3. **Identify a group representation**. It turns out that non-entangling unitaries are the standard representation
#    of the group :math:`G = SU(2) x SU(2) ... x SU(2)` for the vector space :math:`H`, the Hilbert space of the state vectors.
#    Again, this implies that we can find a basis of the Hilbert space, where any :math:`U` is simultaneously block-diagonal.
# 4. **Find the basis that block-diagonalises the representation**.
#
# Let's stop here for now, and try to block-diagonalise one of the non-entangling unitaries. We can use an
# eigenvalue decomposition for this.
#

from scipy.stats import unitary_group
from functools import reduce


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
# Technically speaking, the representation is *irreducible* over :math:`H`.
# As a consequence, the invariant subspace is :math:`H` itself, and the single GFD purity is simply the purity of
# the state :math:`\psi`, which is 1.
# This, of course, is not a very informative footprint -- one that is much too coarse!
#
# However, nobody forces us to restrict our attention to :math:`H` and to
# the (standard) representation of non-entangling unitary matrices.
# After all, what's the first symptom of entanglement in a quantum state?
# You guessed right again, the reduced density matrix of some subsystem is mixed!
# Wouldn't it make more sense to study multipartite entanglement from the point of view of density matrices?
# It turns out that moving into the space :math:`B(H)` of bounded linear operators in which the density matrices live leads to
# a much more nuanced Fourier spectrum.
#
# Let us walk the steps above again
# 1. **Find free vectors**. :math:`L(H)` is a vector space in the technical sense, but one of matrices. To use the linear algebra
#    tricks from before we have to "vectorize" density matrices :math:`rho=\sum_i,j c_i,j |i\rangle \langle j|`
#    into vectors :math:`|rho\rangle \rangle = \sum_i,j c_i,j |i\rangle |j\rangle \in H \otimes H^*`.
#    For example:
#

# create a random quantum state
psi = np.random.rand(2**n) + 1j*np.random.rand(2**n)
psi /= np.linalg.norm(psi)
# construct the corresponding density matrix
rho = np.outer(psi, psi.conj())
# vectorise it
rho_vec = rho.flatten(order='F')

######################################################################
# The set of free states is again that of product states :math:`\rho = \bigotimes \rho_j`
# where each :math:`\rho_j` is a single-qubit state. We only have to write them in flattened form.
#
# 2. **Find free unitary transformations**. The free operations are still given by non-entangling unitaries, but
#    of course, they act on density matrices via :math:`\rho' = U \rho U^{\dagger}`.
#    We can also vectorise this operation by defining the matrix :math:`U_{\mathrm{vec}}(g) = U \otimes U^*`.
#    We then have that :math:`|\rho'\rangle \rangle = U_{\mathrm{vec}} \rho`.
#    To demonstrate:
#

# vectorise U
Uvec = np.kron(U.conj(), U)

# evolve the state above by U, using the vectorised objects
rho_out_vec = Uvec @ rho_vec
# reshape the result back into a density matrix
rho_out = np.reshape(rho_out_vec, shape=(2**n, 2**n), order='F')
# this is the same as the usual adjoint application of U
print(np.allclose(rho_out, U @ rho @ U.conj().T ))

######################################################################
# 3. **Identify a group representation**. This "adjoint action" is indeed a valid representation of
# :math:`G = SU(2) x SU(2) ... x SU(2)`, called the *defining representation*. However, it is a different one from before,
# and this time there is a basis transformation that properly block-diagonalises all matrices in the representation.
#
# 4. **Find the basis that block-diagonalises the representation**.
# To find this basis we compute the eigendecomposition of an arbitrary linear combination
# of a set of matrices in the representation.
#
rng = np.random.default_rng(42)

Uvecs = []
for i in range(10):
    # create n haar random single-qubit unitaries
    Ujs = [unitary_group.rvs(dim=2, random_state=rng) for _ in range(n)]
    # compose them into a non-entangling n-qubit unitary
    U = reduce(lambda A, B: np.kron(A, B), Ujs)
    # Vectorise U
    Uvec = np.kron(U.conj(), U)
    Uvecs.append(Uvec)

# Create a random linear combination of the matrices
alphas = rng.random(len(Uvecs)) + 1j * rng.random(len(Uvecs))
M_combo = sum(a * M for a, M in zip(alphas, Uvecs))

# Eigendecompose the linear combination
vals, Q = np.linalg.eig(M_combo)

######################################################################
# Let's test this basis change
#

Qinv = np.linalg.inv(Q)
# take one of the vectorised unitaries
Uvec = Uvecs[0]
Uvec_diag = Qinv @ Uvec @ Q

np.set_printoptions(
    formatter={'float': lambda x: f"{x:5.2g}"},
    linewidth=200,    # default is 75; increase to fit your array
    threshold=10000   # so it doesn’t summarize large arrays
)

print(Uvec_diag)

######################################################################
# But ``Uvec_diag`` does not look block diagonal. What happened here?
# Well, it *is* block-diagonal, but we have to reorder the columns and rows of the final matrix to make this visible.
# This takes a bit of pain, encapsulated in the following function:
#

from collections import OrderedDict

def group_rows_cols_by_sparsity(B, tol=0):
    """
    Given a binary or general matrix C, this function:
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
    # order by sparsity

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

P_row, P_col = group_rows_cols_by_sparsity(Uvec_diag)

# reorder the block-diagonalising matrices
Q = Q @ P_col
Qinv = P_row @ Qinv

Uvec_diag = Qinv @ Uvec @ Q

print(np.round(np.abs(Uvec_diag), 4))


######################################################################
# The reordering made the block structure visible. You can check that now any vectorised non-entangling matrix ``Uvec``
# has the same block structure if we change the basis and reorder via ``Qinv @ Uvec @ Q``.
#
# The next step is to
# 5. **Find a basis for each subspace**.
# We have everything ready to do this: the basis of a subspace are just the rows of ``Q`` that correspond to the row/column
# indices of a given block. Hence, the GFD Purity calculation can be performed by changing a new vector ``v`` into the
# basis we just identified by applying ``v_diag = Q v``, and taking the sum of absolute squares of those entries in ``v_diag``
# that correspond to one block or subspace.
#
# Notice that the similarity transform 'Qinv @ Uvec @ Q' needs not be implemented by a unitary change of basis! 
# Indeed in genear Q will not be unitary, as we can quickly check by checking if any two of its columns are orthogonal
#

print("The inner product between the second and third column of Q is: ", np.dot(Q[:1],Q[:2]).item())

######################################################################
# This is just becuase the eigenvecotrs of the random linear combiniation that we implemented before are only guaranteed to
# 'span' the blocks, not to be an orthonormal basis of them.
# However, rep-theory actually guarantees that we can indeed find such an orthonormal basis, or, in other words, that we can
# unitarize Q. One way is to polar decompose it

def unitary_polar(Q):
    U, s, Vh = np.linalg.svd(Q, full_matrices=False)
    W = U @ Vh 
    return W

W = unitary_polar(Q)
# It is indeed unitary
print(np.allclose(W @ W.conj().T, np.eye(len(W))))
print(np.allclose(W.conj().T @ W, np.eye(len(W))))

# And yields the same block-decomposition
print(np.round(np.abs(W.conj().T @ Uvec @ W), 4))

# We can now proceed without the risk of messing up the normalization of our states

# vectorise the states we defined earlier
states_vec = [state.flatten(order='F') for state in states]
states_vec_diag = [W.conj().T  @ state_vec for state_vec in states_vec]

purities = []
for v in states_vec_diag:
    v = np.abs(v)**2
    purity_spectrum = [np.sum(v[0:1]), np.sum(v[1:4]), np.sum(v[4:7]), np.sum(v[7:16])]
    purities.append(purity_spectrum)

for data, label in zip(purities, labels):
    print(f"{label} state purities: ")
    for k, val in enumerate(data):
        print(f" - Block {k+1}: ", val)

######################################################################
# Notice that, as it should given the normalization of the states considered, the purities are a probability distribution.
# Indeed they are positive and we can check that they add up to 1, reconstructing the states' total purity.
#

for data, label in zip(purities, labels):
    print(f"{label} state has total purity: ", np.sum(data).item())

######################################################################
# We can further visualize the purities as done in [#Bermejo_Braccia]_, by aggregating them
# by the corresponding block's size
#

agg_purities = [[p[0], p[1]+p[2], p[3]] for p in purities]


# Plot the purities and their comulative
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
for i, data in enumerate(agg_purities):
    ax1.plot(data, label=f'{labels[i]}')
    ax2.plot(np.cumsum(data), label=f'{labels[i]}')

ax1.set_ylabel('Purity')
ax2.set_ylabel('Cumulative Purity')
ax2.set_xlabel('Module weight')
ax2.set_xticks(list(range(n+1)))
ax1.legend(loc='upper left')
ax2.legend(loc='upper left')
plt.tight_layout()
plt.show()

###########################################################################
# The fascinating aspect here is that while multipartite entanglement can get complex very quickly as we
# add qubits (due to the exponential increase in possible correlations),
# our generalized Fourier analysis still provides a clear, intuitive fingerprint of a state's entanglement structure.
# Even more intriguingly, just as smoothness guides how we compress classical
# signals (by discarding higher-order frequencies), the entanglement fingerprint suggests ways we might compress quantum states,
# discarding information in higher-order purities to simplify quantum simulations and measurements.
# In short, generalized Fourier transforms allow us to understand quantum complexity, much like classical Fourier transforms give insight into smoothness.
# By reading a state's quantum Fourier fingerprint, we gain a clear window into the subtle quantum world of multipartite entanglement.
#

###########################################################################
# Bonus section
# --------------
#
# If your head is not spinning yet, let's ask a final, and rather instructive, question.
# What basis have we actually changed into in the above example of multi-partite entanglement?
# It turns out that `W` changes into the Pauli basis!
#

import itertools

_pauli_map = {
    'I': np.array([[1,0],[0,1]],   dtype=complex),
    'X': np.array([[0,1],[1,0]],   dtype=complex),
    'Y': np.array([[0,-1j],[1j,0]],dtype=complex),
    'Z': np.array([[1,0],[0,-1]],  dtype=complex),
}


def pauli_basis(n):
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


# let's create the vectorized basis of Pauli matrices
basis,B  = pauli_basis(n)

# now let's take the first basis vector, the one corresponding to the smallest block
# and project it onto paulis

v = W[:,0]
v_pauli_coeffs = [np.abs(np.dot(v, pv)) for pv in B.T]

print("The basis corresponding to the 1x1 block has nonzero overlap with the following Pauli basis elements:")
for c, ps in zip(v_pauli_coeffs, basis):
    if c > 1e-12:
        print(" - ", ps)

###########################################################################
# It's just the Identity! This is not a surprise, since any unitary maps the identity to itself.
# Let's keep going, what about a vector in the 3x3 blocks?
#

v_pauli_coeffs = np.zeros(len(B))
for idx in range(1, 4):
    v = W[:, idx]
    v_pauli_coeffs += [np.abs(np.dot(v, pv)) for pv in B.T]

print("The basis corresponding to the first 3x3 block has nonzero overlap with the following Pauli basis elements:")

for c, ps in zip(v_pauli_coeffs, basis):
    if c > 1e-12:
        print(" - ", ps)

v_pauli_coeffs = np.zeros(len(B))
for idx in range(4, 7):
    v = W[:, idx]
    v_pauli_coeffs += [np.abs(np.dot(v, pv)) for pv in B.T]


print("The basis corresponding to the second 3x3 block has nonzero overlap with the following Pauli basis elements:")

for c, ps in zip(v_pauli_coeffs, basis):
    if c > 1e-12:
        print(" - ", ps)

###########################################################################
# This block corresponds to operators acting non trivially on a single qubit!

# We can verify that the last block corresponds to those fully supported over our two qubits
v_pauli_coeffs = np.zeros(len(B))
for idx in range(7, 16):
    v = W[:, idx]
    v_pauli_coeffs += [np.abs(np.dot(v, pv)) for pv in B.T]

# we can ask over which paulis v is supported
print("The basis corresponding to the last block has nonzero overlap with the following Pauli basis elements:")
for c, ps in zip(v_pauli_coeffs, basis):
    if c > 1e-12:
        print(" - ", ps)

###########################################################################
# Now that we realized that the blocks correspond to all the possible supports of operators over the Hilbert spaces of the different qubits,
# we can also see why this is the case. Remember that the representation :math:`U = R(g)` conjugates an operator :math:`\rho`
# by a tensor product of single-qubit unitary operators. [TODO: finish this thought with a bit more rigour]
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
# About the authors
# -----------------
#
