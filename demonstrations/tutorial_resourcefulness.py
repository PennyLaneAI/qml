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
# Every subspace is spanned by a basis :math:`\{w^{(i)}_{\alpha, j}\}_{i=1}^{\mathrm{dim(V_{\alpha, j})}}`.
#
# The Generalised Fourier Decomposition (GFD) purity is the length of a projection of a vector :math:`v \in V` onto one of these subspaces :math:`V_{\alpha, j}`,
# computed via the inner product with all basis vectors:
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
# We now move to another resource, the entanglement between 2 qubits. We could simply use the Hilbert space :math:`H` of
# the two-qubit states as our vector space :math:`V`. However, the Fourier analysis is richer if we choose the space of
# density matrices :math:`\rho \in L(H)`, which is the space of bounded operators acting on quantum states in a Hilbert space :math:`H`,
# as :math:`V`. The density matrices get transformed by the adjoint action :math:`U \rho U^{\dagger}`.
#
# Performing numerics like block-diagonalisation on such a vector space is a little more complicated, and best done by moving from
# matrices :math:`\rho` to "flattened" vectors in :math:`\mathbb{C}^{2n}`, and from adjoint unitary evolution to a superoperator
# which is a :math:`2n x 2n`-dimensional matrix that can be applied to the flattened density matrices.
#

import functools

def haar_unitary(N):
    """
    Generates a Haar random NxN unitary matrix
    """
    X = (np.random.randn(N, N) + 1j*np.random.randn(N, N)) / np.sqrt(2)
    Q, R = np.linalg.qr(X)
    phases = np.exp(-1j * np.angle(np.diag(R)))
    return Q @ np.diag(phases)

n = 2
U = haar_unitary(2**n)
U_vec = np.kron(U.conj(), U)

psi = np.random.rand(shape=(2**2,))
rho = np.outer(psi, psi.conj())
rho_vec = rho.flatten(order='F')
rho_out_vec = U_vec @ rho_vec
rho_out = np.reshape(rho_out_vec, shape=(2**2, 2**2))

# show that flattening works
print(np.allclose(rho_out, U @ rho @ U.conj().T ))


######################################################################
# With the vector space fixed, we can now ask what unitaries keep entanglement free density matrices
# entanglement free? Of course, all unitaries that can be written as a tensor product of single-qubit unitaries!
# Such unitaries form a group called :math:`SU(2) x SU(2)`. We claim that they also form a representation of this group
# (the "defining" representation that just consists of the group elements themselves). If this claim is true,
# there must be a basis change into the "Fourier basis" that block-diagonalises all non-entangling unitaries into
# the same block structure. This can be easily checked using the superoperators constructed above!
#

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
    generates the basis of Pauli operators, and orders it by appearence in the isotypical decomp of Times_i SU(2)
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

def rotate_superoperator(U):
    """
    Rotates the superop of the adj of a unitary U to the irrep-sorted Pauli basis
    """
    S        = np.kron(U.conj(), U)
    n        = int(np.log2(U.shape[0]))
    basis,B  = pauli_basis(n)
    S_rot    = B.conj().T @ S @ B
    return basis, S_rot

n = 2
Us = [haar_unitary(2) for _ in range(n)]
# U is a tensor product of single-qubit unitaries
U = functools.reduce(lambda A, B: np.kron(A, B), Us)

basis, S_rot = rotate_superoperator(U)
np.set_printoptions(
    formatter={'float': lambda x: f"{x:5.2g}"},
    linewidth=200,    # default is 75;
    threshold=10000   # so it doesn’t summarize large arrays
)

# now round and print
## NOTICE THAT IN PAULI BASIS THE UNITARY ADJOINT ACTION IS ORTHOGONAL
print("Adjoint Superoperator of U in the Irrep basis")
S_real = S_rot.real
Sr_round = np.round(S_real, 2)
print("Rounded real part (the operator is real):")
print(Sr_round)

######################################################################
# Given a new state :math:`rho`, we can compute the GDF purity by transforming it with the same basis transform.
# [TODO: compute result for a few different rhos without knowing the basis]
#

######################################################################
# With a bit of knowledge on representation theory, one might recognise that the subspaces
# are spanned by a basis of Pauli words.
#
# [TODO: compute same result with Pauli basis]

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
