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
The intuition, however, is exactly the same as in the standard Fourier transform, where large higher-order Fourier
coefficients indicate a less "smooth" function.

In this tutorial we will illustrate the idea of generalised Fourier analysis for resource theories with two simple examples.
First we will look at a standard Fourier decomposition of a function from the perspective of resources, to introduce the
basic idea. Secondly, we will use these concepts to analyse the entanglement
 resource of quantum states, reproducing Figure 2 in [#Bermejo_Braccia]_.

.. figure:: ../_static/demonstration_assets/resourcefulness/figure2_paper.png
   :align: center
   :width: 70%
   :alt: Fourier coefficients, or projections into "irreducible subspaces", of different states using 2-qubit entanglement as a resource.
         A Bell state, which is maximally entangled, has high contributions in higher-order Fourier coefficients, while
         a tensor product state with little entanglement has contributions in lower-order Fourier coefficients. The interpolation
         between the two extremes, exemplified by a Haar random state, has a Fourier spectrum in between.

Luckily, in the case of entanglement as a resource, the bases for the subspaces are associated with Pauli operators,
and generalised Fourier analysis can be done by computing Pauli expectations.
This saves us from diving too deep into representation theory. In fact, the tutorial should be informative without knowing much
about groups at all!

.. note::
    Note that all methods discussed here are classical methods to analyse properties of quantum states,
    and of course, they will scale only as much as the mathematical objects involved can be efficiently described classically.
    It is a fascinating question in which situations the Fourier coefficients of a physical states could be read out on a quantum computer, which can
    sometimes perform the block-diagonalisation efficiently.


Standard Fourier analysis through the lense of resources
--------------------------------------------------------

Let's start recalling the standard Fourier transform, and for simplicity we'll work with the
discrete version. Given N real values :math:`x_0,...,x_{N-1}`, that we can interpret [LINK TO OTHER DEMO] as the values
of a function :math:`f(0), ..., f(N-1)` over the integers :math:`x \in {0,...,N-1}`, the Fourier transform
computes the Fourier coefficients

.. math::
        \hat{f}(k) = \frac{1}{\sqrt{N}\sum_{x=0}^{N-1} f(x) e^{\frac{2 \pi i}{N} k x}, k = 0,...,N-1

Here, :math:`e^{\frac{2 \pi i}{N} k x}` is a basis for the space of functions over the integers, the so-called *Fourier basis*.

For example:
"""

import matplotlib.pyplot as plt
import numpy as np

N = 12

def f(x):
    """Some function"""
    return 0.5*(x-4)**3

def f_hat(k):
    """Fourier coefficients of f"""
    projections = [f(x)*np.exp(2 * np.pi * 1j * k * x / N) for x in range(N)]
    return (1/np.sqrt(N)) * np.sum(projections)

def plot(f, f_hat):

    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.bar(range(N), [f(x) for x in range(N)], color='dimgray')
    ax1.set_title(f"function f")
    ax2.bar(range(N), [f_hat(k) for k in range(N)], color='dimgray')
    ax2.set_title("Fourier coefficients")
    plt.tight_layout()
    plt.show()

plot(f, f_hat)

######################################################################
# Now, what kind of resource are we dealing with here? In other words, what
# measure of complexity gets higher when a function has large higher-order Fourier coefficients?
#
# Well, let us look for the function with the least resource or complexity! For this we can just
# work backwards: define a Fourier spectrum that only has a contribution in the lowest order coefficient,
# and inverse Fourier transform to look at the function!
#
#

def g_hat(k):
    """The least complex Fourier spectrum possible"""
    if k==0:
        return 1
    else:
        return 0

def g(x):
    """Function whose Fourier spectrum is `g_hat`"""
    projections = [g_hat(k)*np.exp(-2 * np.pi * 1j * k * x / N) for k in range(N)]
    return (1/np.sqrt(N)) * np.sum(projections)

plot(g, g_hat)

######################################################################
# Well, the function is constant. This makes sense, because we know that the decay of the Fourier coefficient
# is related to the smoothness of a function (by defining how often it is differentiablle), and a
# constant function is maximally smooth -- it does not wiggle at all. In other words,
# the resource of the standard Fourier transform is smoothness!
#

######################################################################
# Linking resources to group representations
# ------------------------------------------
#
# Fourier transforms are intimately linked to groups (in fact, the x-domain :math:`0,...,N-1` is strictly speaking a group,
# which you can learn more about [here]. Without expecting you to know group jargon, we have to establish a few concepts
# that generalise the above example to quantum states and more generic resources. The crucial idea is to
# define a resource by fixing a set of vectors (later, quantum states) that are considered resource-free.
# We also need to define unitary matrices that map free vectors to free vectors, and these matrices need to form a "unitary representation" of a group :math:`G`,
# which is a matrix valued function :math:`R(g), g \in G` on the group.
# This is all we need to guarantee that the matrices can be simultaneously block-diagonalised, or written as a
# direct sum of smaller matrix-valued functions over the group, :math:`r(g)`. Some of the blocks may be identical.
#
# These blocks are so-called irreducible representations of the group :math:`G`, which are fundamental concepts in
# group theory [Refer to book]. What is important for us is that these smaller matrix valued functions define a
# subspace [TODO: clarify how this works, it always confused me].
#
# A Fourier coefficient is nothing but a projection of a vector in :math:`V` onto one of these subspaces. In our standard
# exmaple above, the subspaces are one-dimensional and spanned by the Fourier basis functions :math:`\chi_k(x) = e^{\frac{2 \pi i}{N} k x}`.
# The projection is executed by the sum :math:`\sum_x f(x) \chi_k(x)`. The concept of these functions as irreducible
# representations allows us to generalise the Fourier transform to lots of other groups, and hence, resources.
#
# But what is the representation :math:`R(g)` for the standard Fourier transform? Let's follow the recipe:
#
# 1. We first need to consider our function :math:`f(0), ..., f(N-1)` as a vector :math:`[f(0), ..., f(N-1)]^T \in V = \mathbb{R}^N`.
# 2. As argued above, the set of smoothness-free vectors correspond to constant functions, :math:`f(0) = ... = f(N-1)`.
# 3. We need a set of unitary matrices that does not change the constantness of the vector. This set is given by permutation matrices,
#    which swap the entries of the vector but do not change any of them.
# 4. These matrices actually form a representation :math:`R(g)` for :math:`g \in G = Z_N`, the *regular representation*.
# 5. We are now guaranteed that there is a basis change that diagonalises all matrices :math:`R(g)` together.
#    (Note that the Fourier transform is sometimes defined as the basis change that block-diagonalises the regular representation!) As mentioned, this is
#    a block-diagonalisation where the blocks happen to be 1-dimensional, as is the rule for all "Abelian" groups.
# 6. The values on the diagonal of :math:`R(g)` under this basis change are exactly the Fourier basis functions `:math:`e^{\frac{2 \pi i}{N} k x}`.
#
# Let's verify this!
#
# First, let us write our function and Fourier spectrum above as vectors:
#

f_vec = np.array([f(x) for x in range(N)])
f_hat_vec = np.array([f(k) for k in range(N)])

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

# TODO: NOT YET THE SAME
print(F.dot(f_vec), f_hat_vec)


######################################################################
# This matrix F is supposed to be the basis transform that diagonalises
# a permutation matrix (which was a unitary representation evaluated at some group element).
#

# create a permutation matrix
P = np.eye(N)
np.random.shuffle(P)

# change into the Fourier basis using F
np.set_printoptions(precision=2, suppress=True)
# TODO: Not yet working
print(F @ P @ F.conj().T)


######################################################################
# To recap, we saw that the standard Fourier analysis can be generalised by
# interpreting "smoothness" or "constantness" as a resource, linking it to a vector space and a group
# representation and (block-)diagonalising the representation. The blocks, here one-dimensional,
# form a basis for subspaces, and Fourier coefficients are just projections of some vector (here,
# a function) into these subspaces.
#
# Armed with this recipe, we can now try to analyse entanglement as a resource,
# and density matrices describing quantum states as vectors in a vector space :math:`L(H)`.
#

######################################################################
# Fourier analysis of entanglement
# --------------------------------
#
# [TODO]
#
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
