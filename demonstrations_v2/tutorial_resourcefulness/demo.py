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
standard Fourier transform, but its group-theoretic generalisations [LINK TO RELATED DEMO].
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
# Well, the function is constant. This makes sense, because we know that the decay of the Fourier coefficient
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
# Find free vectors
# *****************
# The first step is to find "free vectors". So far we dealt with a discrete function :math:`f(0), ..., f(N-1)`,
# but we can easily write it as a vector :math:`[f(0), ..., f(N-1)]^T \in V = \mathbb{R}^N`. As argued above,
# the set of resource-free vectors corresponds to constant functions, :math:`f(0) = ... = f(N-1)`.
#

f_vec = np.array([f(x) for x in range(N)])

# Find free unitary transformations
# *********************************
# Next, we need to identify a set of unitary matrices that does not change the
# "constantness" of the vector. In the case above this is given by the permutation matrices, which
# swap the entries of the vector but do not change any of them.
#
# Identify a group representation
# *******************************
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
# Find the basis that block-diagonalises the representation
# *********************************************************
# This was quite a bit of group jargon, but for a good reason: We are now guaranteed that there is
# a basis change that block-diagonalises *all* circulant permutation matrices. If you know group theory,
# then you will recognise that the blocks reveal the *irreducible representations* of the group.
# What is important is that the blocks form different "irreducible" subspaces which we need to compute
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
P_F =  F @ P @ np.linalg.inv(F)

# check if the resulting matrix is diagonal
# trick: remove diagonal and make sure the remainder is zero
print("is diagonal:", np.allclose(P_F - np.diag(np.diagonal(P_F)), np.zeros((N,N)) ))

######################################################################
# In fact, the Fourier transform is sometimes *defined* as the basis change that diagonalises the regular representation!
# For other representations, however, we need other transforms.
#
# Project into the basis
# **********************
# We now have a very different perspective on the power spectrum. We wrote a function as a vector,
# and changed into the basis that diagonalises circulant matrices. For every 1-d block, we took the corresponding
# coordinate in the new vector (which we know are the Fourier coefficients) and computed its absolute square.
#
# .. math:: \mathcal{P}(\vec{f}) =  |\hat{f}|^2.
#
# More generally, if
# we have different blocks labeled by :math:`\alpha` and their copies by the multiplicity factor :math:`j`,
# then each block marks a subspace spanned by a basis :math:`\{w^{(i)}_{\alpha, j}\}
# The GFD purity is the length of a projection of a new  vector :math:`v \in V` onto one of these subspaces,
#
# .. math::
#           \mathcal{P}(v) = \sum_i  | \langle w^{(i)}_{\alpha, j}, v \rangle |^2.
#
#

######################################################################
# To recap, we saw that by mobilising group theory and linear algebra,
# the standard power spectrum of a function over integers can be interpreted as the "GFD Purities"
# for the resource of "smoothness". Armed with this recipe, we can now try to analyse entanglement as a resource.
# We will first try to work with Dirac vectors as our vectors, and unitary matrices as the representation,
# but see very quickly that this does not lead to rich enough "footprints".
# Instead, we switch to density matrices and their unitary evolution (which need to be vectorised in order to
# use numerical linear algebra tools). This will allow us to find "power spectra" or GFD Purities of states,
# indicating how entangled they are.
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
# Here are a few examples that we will use below.
#

import math
import functools

def ghz_state(n: int):
    """The |GHZ_n⟩ state vector for *n* qubits is famous for having maximal entanglement."""
    psi = np.zeros(2 ** n)
    psi[0] = 1 / math.sqrt(2)
    psi[-1] = 1 / math.sqrt(2)
    return psi


def w_state(n: int):
    """The |W_n⟩ state vector for *n* qubits ...."""
    psi = np.zeros(2 ** n)
    for q in range(n):
        psi[2 ** q] = 1 / math.sqrt(n)
    return psi


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


def haar_product_state(n: int):
    """A Haar random tensor product state of *n* qubits is maximally unentangled."""
    states = [haar_state(1) for _ in range(n)]
    return functools.reduce(lambda A, B: np.kron(A, B), states)


n = 2
states = [haar_product_state(n),  haar_state(n), ghz_state(n), w_state(n)]
# move to density matrices
states = [np.outer(state.conj(), state) for state in states]
labels = [ "Product", "Haar", "GHZ", "W"]

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
# Wouldn't it make more sense then to study the multipartite entanglement form the point of view of density matrices?
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
psi = np.random.rand(2**n)
# construct the corresponding density matrix
rho = np.outer(psi, psi.conj())
# vectorise it
rho_vec = rho.flatten(order='F')

######################################################################
#    The set of free states is again that of product states :math:`\rho = \bigotimes \rho_j`
#    where each :math:`\rho_j` is a single-qubit state. We only have to write them in flattened form.
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
rho_out = np.reshape(rho_out_vec, newshape=(2**2, 2**2))
# this is the same as the usual adjoint application of U
print(np.allclose(rho_out, U @ rho @ U.conj().T ))

######################################################################
# 3. **Identify a group representation**. This "adjoint action" is indeed a valid representation of
# :math:`G = SU(2) x SU(2) ... x SU(2)`, called the *defining representation*. However, it is a different one from before,
# and this time there is a basis transformation that properly block-diagonalises all matrices in the representation.
# To find this transformation we compute the eigendecomposition of an arbitrary linear combination of a set of matrices in the representation.
#

Uvecs = []
for i in range(10):
    # create n haar random single-qubit unitaries
    Ujs = [unitary_group.rvs(dim=2) for _ in range(n)]
    # compose them into a non-entangling n-qubit unitary
    U = reduce(lambda A, B: np.kron(A, B), Ujs)
    # Vectorise U
    Uvec = np.kron(U.conj(), U)
    Uvecs.append(Uvec)

# Create a random linear combination of the matrices
alphas = np.random.randn(len(Uvecs)) + 1j * np.random.randn(len(Uvecs))
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
# But `Uvec_diag` does not look block diagonal. What happened here?
# Well, it is block-diagonal, but we have to reorder the columns and rows of the final matrix to make this visible.
# This takes a bit of pain, encapsulated in the following function:
#

from collections import OrderedDict

def group_rows_cols_by_sparsity(mat, tol=0):
    """
    Given a matrix B, this function:
      1. Groups identical rows and columns.
      2. Orders these groups by sparsity (most zeros first).
      3. Returns the permuted matrix C2, and the row & column permutation
         matrices P_row, P_col such that C2 = P_row @ C @ P_col.

    Parameters
    ----------
    mat : ndarray, shape (n, m)
        Input matrix.

    Returns
    -------
    P_row : ndarray, shape (n, n)
        Row permutation matrix.
    P_col : ndarray, shape (m, m)
        Column permutation matrix.
    """
    # Compute boolean mask where |B| >= tol
    mask = np.abs(mat) >= tol
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

P_row, P_col = group_rows_cols_by_sparsity(Uvec_diag)

Uvec_diag = P_row @ Uvec_diag @ P_col

print("\n\n ---------------")
print(Uvec_diag)

######################################################################
# The reordering made the block structure visible. You can check that any vectorised non-entangling matrix `Uvec`
# has the same block structure if we change the basis and reorder via `P_row @ Qinv @ Uvec @ Q @ P_col`.
#
# The next step is to
# 5. **Find the basis that block-diagonalises the representation**.
#
# [TRY: Rotate a vector into this basis and only summarise the entries]
#
_pauli_map = {
    'I': np.array([[1,0],[0,1]],   dtype=complex),
    'X': np.array([[0,1],[1,0]],   dtype=complex),
    'Y': np.array([[0,-1j],[1j,0]],dtype=complex),
    'Z': np.array([[1,0],[0,-1]],  dtype=complex),
}
factors = [_pauli_map[ch] for ch in 'IX']
rho_P = functools.reduce(lambda A, B: np.kron(A, B), factors)
rho_P = rho_P.flatten(order="F")
rho_P = Q @ rho_P
print("HERE", rho_P)


# vectorise the states we defined earlier
states_vec = [state.flatten(order='F') for state in states]



# change into the block-diagonal basis
states_vec = [Q @ state_vec for state_vec in states_vec]
purities = [[np.vdot(state_vec[0], state_vec[0]),
             np.vdot(state_vec[1:4], state_vec[1:4]),
             np.vdot(state_vec[4:8], state_vec[4:8]),
             np.vdot(state_vec[8:16], state_vec[8:16]),
             ] for state_vec in states_vec ]



# Grab default color cycle
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Create two vertically aligned subplots sharing the x-axis
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

for i, data in enumerate(purities):
    color = colors[i % len(colors)]
    ax1.plot(data, label=f'{i}', color=color)
    ax2.plot(np.cumsum(data), label=f'{i}', color=color)

ax1.set_ylabel('Purity')
ax2.set_ylabel('Cumulative Purity')
ax2.set_xlabel('Module weight')

ax1.legend(loc='upper left')
ax2.legend(loc='upper left')

plt.tight_layout()

plt.show()


###########################################################################
# Bonus section
# --------------
#
# But what basis have we actually changed into? It turns out that `Q` changes into the Pauli basis. We could have constructed
# `Q` from first principles:
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
Uvec_rot = B.conj().T @ Uvec @ B

# make sure the imaginary part is zero, so we don't have to print it
# (this is a property of the Pauli basis)
assert np.isclose(np.sum(Uvec_rot.imag), 0)

# print the real part
print(np.round(Uvec_rot.real, 2))



######################################################################
# The different blocks correspond to subspaces spanned by Pauli words with different structures:
#
# * The first block of size 1x1 corresponds to a subspace spanned by the Pauli word operator :math:`\mathbm{1} \otimes \mathbm{1}`.
# * The second two blocks of size 4x4 corresponds to a subspace spanned by to Pauli word operators :math:`\mathbm{1} \otimes P_2` and :math:`P_1 \otimes \mathbm{1}`, where
#   :math:`P_1, P_2 \in \{X, Y, Z\}`.
# * The third block of size XxX corresponds to a subspace spanned by Pauli word operators of the form :math:`P_1 \otimes P_2`.
#
# In other words, we didn't need to vectorise everything and use linear algebra tools to compute the GFD purities. We could have
# simply computed the inner product with the rigth set of Pauli operators in :math:`B(H)`:
#

def generate_pauli_strings(n: int):
    """
    Generate all length‑n strings over the Pauli alphabet ['I','X','Y','Z'].
    Returns a list of 4**n strings, e.g. ['IIX', 'IXZ', …].
    """
    return [''.join(p) for p in itertools.product('IXYZ', repeat=n)]


def pauli_string_to_matrix(pauli_str: str):
    """
    Convert a Pauli string (e.g. 'XIY') to its full 2^n × 2^n matrix.
    """
    mats = [_pauli_map[s] for s in pauli_str]
    return functools.reduce(lambda A, B: np.kron(A, B), mats)


# function to project into the modules
def compute_me_purities(op):
    """
    Computes GFD purities of op (assumed to be np.matrix)
    by explicitly computing overlaps with Paulis
    """

    if op.ndim == 1:
        # state vector
        is_vector = True
    elif op.ndim == 2 and op.shape[0] == op.shape[1]:
        # density/operator
        is_vector = False
    else:
        raise ValueError("`op` must be either a 1D state vector or a square matrix")

    d = op.shape[0]
    n = int(np.log2(d))

    basis = generate_pauli_strings(n)
    purities = np.zeros(n + 1)
    for belem in basis:
        k = n - belem.count('I')
        P = pauli_string_to_matrix(belem)

        if is_vector:
            ovp = op.conj() @ (P @ op)
        else:
            ovp = np.trace(op @ P)

        #assert ovp.imag < 1e-10
        purities[k] += (ovp.real) ** 2

    return purities / (2 ** n)


purities = [compute_me_purities(op) for op in states]

# Grab default color cycle
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Create two vertically aligned subplots sharing the x-axis
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

for i, data in enumerate(purities):
    color = colors[i % len(colors)]
    ax1.plot(data, label=f'{labels[i]}', color=color)
    ax2.plot(np.cumsum(data), label=f'{labels[i]}', color=color)

ax1.set_ylabel('Purity')
ax2.set_ylabel('Cumulative Purity')
ax2.set_xlabel('Module weight')

ax1.legend(loc='upper left')
ax2.legend(loc='upper left')

plt.tight_layout()
plt.savefig("/home/maria/Desktop/purities2.png")
plt.show()

"""
# To illustrate this, we'll consider a few key examples:
# * **Product state**: A "smooth" state with no entanglement. It shows high purity only in the lowest-order irreps, similar to a constant function in classical Fourier analysis.
# * **GHZ state**: A highly structured, maximally entangled state. It behaves like a quantum analog of a high-frequency oscillation, having purity concentrated mostly in the highest-order irreps, creating a clear quantum fingerprint.
# * **W state**: Moderately entangled, somewhat between the GHZ and product states, with a broader yet smoother distribution across the irreps.
# * **Random (Haar) state**: Highly complex but without structured entanglement patterns, its purity is distributed more evenly across the spectrum.# In the code examples we'll show shortly, you'll see precisely these patterns emerge.
# For each state, we compute the purity distribution across generalized Fourier modes (irreps), illustrating exactly how entanglement complexity relates to spectral structure.

# # Here, show purity plots for these states as computed by the provided code

# The fascinating aspect here is that while multipartite entanglement can get complex very quickly as we add qubits (due to the exponential increase in possible correlations),
# our generalized Fourier analysis still provides a clear, intuitive fingerprint of a state's entanglement structure.
# Even more intriguingly, just as smoothness guides how we compress classical signals (by discarding higher-order frequencies), the entanglement fingerprint suggests ways we might compress quantum states,
# discarding information in higher-order irreps to simplify quantum simulations and measurements.
# In short, generalized Fourier transforms allow us to understand quantum complexity, much like classical Fourier transforms give insight into smoothness.
# By reading a state's quantum Fourier fingerprint, we gain a clear window into the subtle quantum world of multipartite entanglement.


"""






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
