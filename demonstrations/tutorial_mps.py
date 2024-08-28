r"""What is a Matrix Product State (MPS)?
=========================================

Two parts: MPS basics and then application to quantum circuit simulation

.. figure:: ../_static/demo_thumbnails/opengraph_demo_thumbnails/OGthumbnail_shadow_hamiltonian_simulation.png
    :align: center
    :width: 70%


# Part 1

Matrix Product State basics
---------------------------

Matrix Product States (MPS) are an efficient representation of low entanglement states.
The amount of entanglement the MPS can represent is user-controlled via a hyper-parameter, the so-called `bond dimension` :math:`\chi`.
If we allow :math:`\chi` to be of :math:`\mathcal{O}(2^{\frac{n}{2}})` for a system of :math:`n` qubits, we can write `any` state as an `exact` MPS.
To avoid exponentially large resources, however, one typically sets a finite bond dimension :math:`\chi` at the cost of introducing an approximation error.
For some specific classes of states, this is provably sufficient to have a faithful representations (see note box below). 
But because MPS come with a lot of powerful computational features that we are going to discuss later (in particular canonical forms),
they are still used in much more complex systems where these requirements do not hold anymore, and still yield good results.
For example, state of the art quantum chemistry simulations were performed using MPS (citation needed).

It is known that there are more suitable tensor network states for more complex situations (see note box below).
However, they all suffer from a significantly larger algorithmic cost as a lot of the things that make MPS so attractive are not true anymore.
To put it plainly, it is often simply much easier to use MPS and throw a large amount of resources into the bond dimension than to develop
more advanced tensor network methods.


While more advanced tensor network methods are developed, optimized and democratized, MPS continue to be the workhorse of
many quantum simulation techniques.


.. note::

    Ground states of local and gapped Hamiltonians are known to satisfy the so-called area law of entanglement.
    This law states that the entanglement entropy of a sub-system grows with its area and not its volume.
    For one dimensional systems, the surface area of a non-disjoint sub-system is just a constant, and the entanglement
    between any such sub-system in an MPS with a finite bond dimension :math:`\chi` is naturally bounded by :math:`\log(\chi)=\text{const.}`,
    so MPS satisfy the area law of entanglement for one-dimensional systems.

    PEPS are the natural generalization to regular 2D or 3D grids as well as more general graph connectivities, and are known to fulfill the
    respective area laws of entanglement, making them the correct ansätze for local and gapped Hamiltonians in those cases. For example, for a 2D
    PEPS with a square subsystem of size :math:`L` and bond dimension :math:`\chi`, the entanglement entropy between the that square and the rest of the system is bounded by
    :math:`\log(\chi^L) = L \log(\chi)`, which is proportional to the circumference.



Compression using Singular Value Decomposition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To understand MPS and bond dimension, we first need to understand how one can generally use the singular value decomposition (SVD) to do compression.
Any matrix :math:`M = \mathbb{C}^{M\times N}` can be singular-value-decomposed as

.. math:: M = U \Lambda V^\dagger,

where :math:`\Lambda` is the diagonal matrix of the :math:`r=\min(m, n)` real and non-negative singular values, 
:math:`U \in \mathbb{C}^{m\times r}` is left-unitary :math:`U^\dagger U = \mathbb{I}_r`, and
:math:`V^\dagger \in \mathbb{C}^{r\times n}` is right-unitary :math:`V V^\dagger = \mathbb{I}_r`.
We say the columns and rows of :math:`U` and :math:`V^\dagger` are the left- and right-orthogonal
singular vectors, respectively. In the case of square and normal matrices, the singular values and singular vectors
are just the eigenvalues and eigenvectors.

Small singular values and their corresponding singular vectors carry little information of the matrix. When the singular values have
a tail of small values, we can compress the matrix by throwing away these numbers, and the corresponding singular vectors.
This is best seen by a small example, let us load the image we have shown in the header of this demo and compress it.

"""

import numpy as np
import matplotlib.pyplot as plt

img = plt.imread("../_static/demo_thumbnails/regular_demo_thumbnails/thumbnail_shadow_hamiltonian_simulation.png")
# only look at one color channel for demonstration purposes
img = img[:, :, 0]

U, Lambda, Vd = np.linalg.svd(img)

chi = 50
U_compressed = U[:, :chi]
Lambda_compressed = Lambda[:chi]
Vd_compressed = Vd[:chi]

compressed_img = U_compressed @ np.diag(Lambda_compressed) @ Vd_compressed

fig, axs = plt.subplots(ncols=2)
ax = axs[0]
ax.imshow(img, vmin=0, vmax=1)
ax.set_title("Uncompressed image")

ax = axs[1]
ax.imshow(compressed_img, vmin=0, vmax=1)
ax.set_title("compressed image")

plt.show()

size_original = np.prod(img.shape)
size_compressed = np.prod(U_compressed.shape) + np.prod(Lambda_compressed.shape) + np.prod(Vd_compressed.shape)

print(f"original image: {size_original}, compressed image: {size_compressed}, factor {size_original/size_compressed:.3f} saving")
##############################################################################
# 
#
# The original image is :math:`334 \times 542` pixels, that we compress in the :math:`334 \times 50`
# :math:`U`, :math:`50` :math:`\Lambda` and :math:`50 \times 542` :math:`V^\dagger`.
# This is possible because the information density in the image is low, as seen by the distribution of singular values


_, Lambda, _ = np.linalg.svd(img) # recompute full spectrum
plt.plot(Lambda)

##############################################################################
# 
# We are later going to do the same trick with state vectors.
# Note that the compressed information is encoded in :math:`U`, :math:`S` and :math:`Vd`.
# If we want to retrieve the actual image :math:`M` (or state vector), we still need to reconstruct the full :math:`334 \times 542` pixels.
# Luckily, as we will later see in the case of MPS, we can retrieve all relevant information efficiently from the compressed components without ever 
# having to reconstruct the full state vector.
#
# Turn any state into an MPS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# Formally, any quantum state :math:`|\psi\rangle \in \mathbb{C}^{2^n}` on :math:`n` qubits can be written as a Matrix Product State (MPS).
# The goal will be to write an arbitrary state :math:`|\psi\rangle = \sum_{\sigma_1, .., \sigma_n} \psi_{\sigma_1, .., \sigma_n} |\sigma_1 .. \sigma_n\rangle` in the form
#
# ..math:: |\psi\rangle = \sum_{\sigma_1, .., \sigma_n} U^{\sigma_1} .. U^{\sigma_n} |\sigma_1 .. \sigma_n\rangle,
#
# where we decomposed the rank :math:`n` tensor :math:`\psi_{\sigma_1, .., \sigma_n}` into a product of matrices :math:`U^{\sigma_j}`
# for each value of :math:`\sigma_j=0, 1` for qubits. This is why it is called a **matrix product** state, even though most of these object are, technically, rank-3 tensors.
#
# Graphically, this corresponds to splitting up the big rank-n tensor into :math:`n` smaller tensors, 
# similar to what we did above in the example of compressing an image.
#
# .. figure:: ../_static/demonstration_assets/mps/psi_to_mps_0.png
#     :align: center
#     :width: 70%
#
# To make things simpler, let us look at concrete state vector with :math:`n=3` sites, so :math:`\psi_{\sigma_1 \sigma_2 \sigma_3}`.

n = 3 # three sites = three legs
psi = np.random.rand(2**3)
psi = psi / np.linalg.norm(psi)  # random, normalized state vector
psi = np.reshape(psi, (2, 2, 2)) # rewrite psi as rank-n tensor

##############################################################################
#
#
# We rewrite the tensor as a matrix with indices of the first site :math:`\sigma_1` and the combined indices of all remaining sites, :math:`(\sigma_2 \sigma_3)`
# Now that we have a matrix, we can perform SVD to split off the first site. Mathematically, this is
#
# .. math:: \psi_{\sigma_1 \sigma_2 \sigma_3} = \psi_{\sigma_1, (\sigma_2 \sigma_3)} = \sum_{\mu_1} U_{\sigma_1 \mu_1} \Lambda_{\mu_1} V^\dagger_{\mu_1 (\sigma_2 \sigma_3)}.
#

# reshape vector to matrix
psi = np.reshape(psi, (2, 2**(n-1)))
# SVD to split off first site
U, Lambda, Vd = np.linalg.svd(psi, full_matrices=False)

##############################################################################
#
# For convenience, we separate so-called physical indices :math:`\sigma_j` as superscripts, and so-called virtual indices :math:`\mu_1` as subscripts.
# We also multiply the singular values onto :math:`V^\dagger` and call this the remainder state :math:`\psi'_{\mu_1, (\sigma_2 \sigma_3)}`,
# so overall we have
# 
# .. math:: \psi_{\sigma_1 \sigma_2 \sigma_3} = \sum_{\mu_1} U^{\sigma_1}_{\mu_1} \psi'_{\mu_1, (\sigma_2 \sigma_3)}.
#
# Graphically, this corresponds to
#
# .. figure:: ../_static/demonstration_assets/mps/psi_to_mps_1.png
#     :align: center
#     :width: 70%
#
# We keep the :math:`U` tensors. We want to maintain the convention that they are of shape ``(virtual_left, physical, virtual_right)``.
# Because there is not virtual index on the left for the first site, we introduce a dummy index.

Us = []
U = np.reshape(U, (1, 2, 2)) # mu1, s2, mu2
Us.append(U)

##############################################################################
#
# This procedure is repeated through all sites. The first step was special in that :math:`U^{\sigma_1}_{\mu_1}` is a vector for each value of :math:`\sigma_1`.
# When splitting up :math:`\psi'_{\mu_1, (\sigma_2 \sigma_3)}` we combine the virtual bond with the current site, and have all remaining sites be the other leg of the matrix we create for SVD.
# In particular, we do
# 
# .. math:: \psi'_{\mu_1, (\sigma_2 \sigma_3)} = \psi'_{(\mu_1 \sigma_2), (\sigma_3)} = \sum_{\mu_2} U^{\sigma_2}_{\mu_1 \mu_2} \Lambda_{\mu_2} V^\dagger^{\sigma_3}_{\mu_2}
#

psi_remainder = np.diag(Lambda) @ Vd                 # mu1 (s2 s3)
psi_remainder = np.reshape(psi_remainder, (2*2, 2))  # (mu1 s2), s3
U, Lambda, Vd = np.linalg.svd(psi_remainder, full_matrices=False)

U = np.reshape(U, (2, 2, 2)) # mu1, s2, mu2
Us.append(U)

U.shape, Lambda.shape, Vd.shape

##############################################################################
# We again multiply the singular values onto the new :math:`V^\dagger` and take that as the remainder state :math:`\psi''`.
# The state overall now reads
#
# .. math:: \psi_{\sigma_1 \sigma_2 \sigma_3} = \sum_{\mu_1 \mu_2} U^{\sigma_1}_{\mu_1} U^{\sigma_2}_{\mu_1 \mu_2} \psi''^{\sigma_3}_{\mu_2}.
#
# When the state is normalized, we are done. Else, we can do the procedure one more time again with a virtual dummy dimension on the right-most site.

psi_remainder = np.diag(Lambda) @ Vd                 # mu1 (s2 s3)
psi_remainder = np.reshape(psi_remainder, (2*2, 1))  # (mu1 s2), s3
U, Lambda, Vd = np.linalg.svd(psi_remainder, full_matrices=False)

U = np.reshape(U, (2, 2, 1)) # mu1, s2, mu2
Us.append(U)

U.shape, Lambda.shape, Vd.shape

##############################################################################
# Because our state vector was already normalized, the singular value in this last SVD is just ``1.``. Else it would yield the norm of ``psi``
# (a good exercise to confirm by skipping the normalization step in the definition of ``psi`` above).
#
# The collected tensors :math:`U^{\sigma_i}_{\mu_{i-1}, \mu_i}` now make up the matrix product state and describe the original state :math:`|\psi\rangle`
# by appropriately contracting the virtual indices :math:`\mu_i`. We can briefly confirm this by reverse engineering the original state. Due to the convention of
# the indices as ``(virtual_left, physical, virtual_right)``, the contraction is simple and we can use the standard 
# `np.tensordot <https://numpy.org/doc/stable/reference/generated/numpy.tensordot.html>`_ with ``axes=1`` indicating matrix-product-like contraction of the left-most and right-most index.
# This is one way of thinking of the obtained state as a **matrix product** state.

print(f"Shapes of Us: {[_.shape for _ in Us]}")

psi_reconstruct = Us[0]

for i in range(1, len(Us)):
    psi_reconstruct = np.tensordot(psi_reconstruct, Us[i], axes=1) # contract the rightmost with the left most index

print(f"Shape of reconstructed psi: {[_.shape for _ in Us]}")
# remove dummy dimensions
psi_reconstruct = np.reshape(psi_reconstruct, (2, 2, 2))
# original shape of original psi
psi = np.reshape(psi, (2, 2, 2))

np.allclose(psi, psi_reconstruct)


##############################################################################
# 
# Up to this point, the description of the original state in terms of the MPS, made up by the three matrices :math:`U^{\sigma_i}_{\mu_{i-1}, \mu_i}`, is exact.
# With this construction, the sizes of the virtual bonds grow exponentially from :math:`2` to :math:`2^{n/2}` until the middle of the chain.
#
# Just like in the example with images before, we can compress the state by only keeping 
# the :math:`\chi` largest singular values, and respective singular vectors.
# The hyper-parameter :math:`\chi` is called the bond dimension and it allows us 
# to control the amount of entanglement the state can represent between everything that 
# is left and right of the bond (more on that later).
#
# A full subroutine from :math:`|\psi\rangle` to its compressed MPS description is given by the following function ``dense_to_mps``.
# It is convenient to also keep the singular values for each bond, we will shortly see why.

def split(M, bond_dim):
    """Split a matrix M via SVD and keep only the ``bond_dim`` largest entries"""
    U, S, Vd = np.linalg.svd(M, full_matrices=False)
    bonds = len(S)
    Vd = Vd.reshape(bonds, 2, -1)
    U = U.reshape((-1, 2, bonds))
    
    # keep only chi bonds
    chi = np.min([bonds, bond_dim])
    U, S, Vd = U[:, :, :chi], S[:chi], Vd[:chi]
    return U, S, Vd

def dense_to_mps(psi, bond_dim):
    """Turn a state vector ``psi`` into an MPS with bond dimension ``bond_dim``"""
    Ms = []
    Ss = []

    psi = np.reshape(psi, (2, -1))   # split psi[2, 2, 2, 2..] = psi[2, (2x2x2...)]
    U, S, Vd = split(psi, bond_dim)  # psi[2, (2x2x..)] = U[2, mu] S[mu] Vd[mu, (2x2x2x..)]

    Ms.append(U)
    Ss.append(Ss)
    bondL = Vd.shape[0]
    psi = Vd

    for _ in range(n-2):
        psi = np.reshape(psi, (2*bondL, -1))   # reshape psi[2 * bondL, (2x2x2...)]
        U, S, Vd = split(psi, bond_dim)        # psi[2, (2x2x..)] = U[2, mu] S[mu] Vd[mu, (2x2x2x..)]
        Ms.append(U)
        Ss.append(Ss)

        psi = Vd
        bondL = Vd.shape[0]

    Ms.append(Vd)
    
    return Ms, Ss

##############################################################################
# Let us look at a larger state. First, let us observe how the dimension scales exponentially when we don't truncate the bonds.

n = 12
bond_dim = 10000

psi = np.random.rand(*[2]*n)
psi = psi/np.linalg.norm(psi)
Ms, Ss = dense_to_mps(psi, bond_dim)

[M.shape for M in Ms]

##############################################################################
# When setting a finite bond dimension :math:`\chi \leq 2^{n/2}`, 
# we see the virtual bonds grow at the boundaries until reaching 
# the maximal bond dimension and staying constant thereafter until 
# reaching the other side.

Ms, Ss = dense_to_mps(psi, 5)

[M.shape for M in Ms]

# This was all to conceptually understand the relationship between dense vectors and a compressed matrix product state.
# We want to use MPS for many sites, where it is often not possible to write down the exponentially large state vector in the first place.
# In that case we would simply start from an MPS description in terms of :math:`n` :math:`\chi \times \chi` matrices.
# Luckily, we can obtain all relevant information without ever reconstructing the full state vector.
# 
# Canonical forms
# ~~~~~~~~~~~~~~~
#
# In the above construction, we unknowingly already baked in a very useful feature of our MPS because all the :math:`U` matrices from the SVD
# are left-orthogonal. In particular, they satisfy
#
# .. math:: \sum_{\sigma_i} \left(U^{\sigma_i} \right)^\dagger U^{\sigma_i} = \mathbb{I}.
#
# Let us briefly confirm that:



#
#
# Expectation values
# ~~~~~~~~~~~~~~~~~~
# 
# Density Matrix Renormalization Group (DMRG) algorithm
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



# # Part 2
# 
# Quantum Simulation with MPS
# ---------------------------
# 
# How a gate is applied to an MPS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# Circuits that are well-suited to MPS methods and circuits that are not
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# Use of default.tensor as a tool for demonstration
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 


import pennylane as qml
import numpy as np
from pennylane import X, Y, Z, I

dev = qml.device("default.qubit")


##############################################################################
# 
#

##############################################################################
# 
#

##############################################################################
# 
#

##############################################################################
# 
#


##############################################################################
# 
# Conclusion
# ----------
#
# conclusion
#



##############################################################################
# 
# References
# ----------
#
# .. [#SommaShadow]
#
#     Rolando D. Somma, Robbie King, Robin Kothari, Thomas O'Brien, Ryan Babbush
#     "Shadow Hamiltonian Simulation"
#     `arXiv:2407.21775 <https://arxiv.org/abs/2407.21775>`__, 2024.
#
# .. [#Somma]
#
#     Rolando D. Somma
#     "Quantum Computation, Complexity, and Many-Body Physics"
#     `arXiv:quant-ph/0512209 <https://arxiv.org/abs/quant-ph/0512209>`__, 2005.
#
# .. [#Somma2]
#
#     Rolando Somma, Howard Barnum, Gerardo Ortiz, Emanuel Knill
#     "Efficient solvability of Hamiltonians and limits on the power of some quantum computational models"
#     `arXiv:quant-ph/0601030 <https://arxiv.org/abs/quant-ph/0601030>`__, 2006.
#
# .. [#Galitski]
#
#     Victor Galitski
#     "Quantum-to-Classical Correspondence and Hubbard-Stratonovich Dynamical Systems, a Lie-Algebraic Approach"
#     `arXiv:1012.2873 <https://arxiv.org/abs/1012.2873>`__, 2010.
#
# .. [#Goh]
#
#     Matthew L. Goh, Martin Larocca, Lukasz Cincio, M. Cerezo, Frédéric Sauvage
#     "Lie-algebraic classical simulations for variational quantum computing"
#     `arXiv:2308.01432 <https://arxiv.org/abs/2308.01432>`__, 2023.
#
# .. [#Wiersema]
#
#     Roeland Wiersema, Efekan Kökcü, Alexander F. Kemper, Bojko N. Bakalov
#     "Classification of dynamical Lie algebras for translation-invariant 2-local spin systems in one dimension"
#     `arXiv:2309.05690 <https://arxiv.org/abs/2309.05690>`__, 2023.
#
# .. [#Aguilar]
#
#     Gerard Aguilar, Simon Cichy, Jens Eisert, Lennart Bittel
#     "Full classification of Pauli Lie algebras"
#     `arXiv:2408.00081 <https://arxiv.org/abs/2408.00081>`__, 2024.
#
# .. [#Babbush]
#
#     Ryan Babbush, Dominic W. Berry, Robin Kothari, Rolando D. Somma, Nathan Wiebe
#     "Exponential quantum speedup in simulating coupled classical oscillators"
#     `arXiv:2303.13012 <https://arxiv.org/abs/2303.13012>`__, 2023.
#
# .. [#Barthe]
#
#     Alice Barthe, M. Cerezo, Andrew T. Sornborger, Martin Larocca, Diego García-Martín
#     "Gate-based quantum simulation of Gaussian bosonic circuits on exponentially many modes"
#     `arXiv:2407.06290 <https://arxiv.org/abs/2407.06290>`__, 2024.
#


##############################################################################
# About the author
# ----------------
#
