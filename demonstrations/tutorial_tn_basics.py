r"""Tutorial: ABC of tensor networks
=============================================================

Intro:
- introduce where are tensor networks used (physics, math, machine learning), why are they so widely spread, and mention how they are not just a data structure but also a diagrammatic language for formal reasoning and even proofs!
- Define the scope of this tutorial: basics of tensor networks, their relation to quantum circuits, and the importance of choosing the contraction paths.

From matrices to tensors
^^^^^^^^^^^^^^^^^^^^^^^^^^^

A common and intuitive way of thinking about tensors is as generalizations of vectors and matrices. That is, we can think of them as multi dimensional arrays - i.e., multi dimensional maps that are linear with respect to every parameter.
A tensor of dimensions :math:`d_1 \times d_2 \times \ldots \times d_r` can be expressed as

.. math::
    T_{i_1, i_2, \ldots, i_r} \in \mathbb{C}^{d_1 \times d_2 \times \ldots \times d_r}.
    
Where each :math:`i_n` is an *index* of dimension :math:`d_n` - it can take values ranging from :math:`1` to :math:`d_n`-, and the number of indices :math:`r` is known as the *rank* of the tensor. We say :math:`T` is a rank-r tensor.

.. tip::
    Some authors refer to the indices of the tensors as their dimensions. In this tutorial, these two concepts will have have different meanings, although related.

For example, a scalar :math:`s` is a rank-0 tensor, a vector :math:`v_i` is a rank-1 tensor and a matrix :math:`G_{i,j}` is a rank-2 tensor.
    
A beautiful and powerful tool accompanying tensors (networks) is their graphical language representation. The diagram of a tensor is simply a geometric shape with a leg sticking out of it for every index in the tensor. For example,
    
.. figure:: ../_static/demonstration_assets/tn_basics/01-tensor.png
    :align: center
    :width: 50%

We can apply this same idea to represent a scalar, a vector and a matrix:

.. figure:: ../_static/demonstration_assets/tn_basics/02-tensors.png
    :align: center
    :width: 50%

Does the last diagram seem familiar? It is because this is the representation of a single-qubit gate! We will see later in this tutorial the relation between quantum circuits and tensor networks. 

When working within the quantum computing notation, we adopt the convention that writing the leg of a quantum state (i.e., a vector) towards the right direction corresponds to a ket, i.e., a vector living in the Hilbert space, while drawing the legs to the left means they are a bra vector, i.e., living in the dual space.

.. figure:: ../_static/demonstration_assets/tn_basics/03-braket.png
    :align: center
    :width: 50%

.. tip::
    The diagrammatic representation of tensors is rooted in category theory, which equips the diagrams with all the relevant information so they can used in proofs and formal reasoning! [#Selinger2010]_

Creating a tensor in code is straightforward, and chances are you have already created one yourself. Using ``Numpy``, all we have to do is to create a ``np.array`` of the desired rank. For instance, we can start crearting a rank-1 tensor (a vector).
"""

import numpy as np

tensor_rank1 = np.array([1, 2, 3, 4])
print("rank: ", tensor_rank1.ndim)
print("dimensions: ", tensor_rank1.shape)

##############################################################################
# Then, we can use this to construct a rank-2 tensor (a matrix).

tensor_rank2 = np.array([tensor_rank1, tensor_rank1, tensor_rank1])
print("rank: ", tensor_rank2.ndim)
print("dimensions: ", tensor_rank2.shape)

##############################################################################
# As you might have guessed, we can repeat this procedure to create a rank-3 tensor.

tensor_rank3 = np.array([tensor_rank2, tensor_rank2])
print("rank: ", tensor_rank3.ndim)
print("dimensions: ", tensor_rank3.shape)
print("Rank-3 tensor: \n", tensor_rank3)

##############################################################################
# We can create a tensor of arbitrary rank following a similar procedure. This recursive approach is instructive to understand a rank-r tensor as made up of rank-(r-1) tensors, which translates to the code as adding an additional level in the nested bracket structure ``[tensor_rank_r-1]``.
#

##############################################################################
# From matrix multiplication to tensor contractions
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Matrix-matrix and matrix-vector multiplications are familiar operations within the context of quantum computing. We can now study these operations under the lens of the tensor notation introduced above. First, a matrix and a vector can be multiplied as
#
#
# .. math::
#     (w)_j = G \cdot v = \sum_i G_ij v_i
#
# .. figure:: ../_static/demonstration_assets/tn_basics/04-matrix-vector.png
#     :align: center
#     :width: 50%
# 
# .. note::
# 
#   Recall we are adopting the convention of writing a ket vector with its leg pointing right, as done in quantum circuits. In turn, this means the "input" index of a matrix - its column index - points towards the left, while the "output" index - its row index - points towards the right.
# 
# We see that summing over the shared index :math:`i` is equivalent to **contracting** the corresponding legs from the matrix and vector diagrams. As expected, the result of this multiplication is another rank-1 tensor with dangling leg :math:`j`. Similarly, we can look at the matrix-matrix multiplication:
#
#
# .. math::
#     (G^3)_{i,k} = G^2 \cdot G^1 = \sum_j G^{2}_{j,k} G^{1}_{i,j}
# 
# .. figure:: ../_static/demonstration_assets/tn_basics/05-matrix-matrix.png
#     :align: center
#     :width: 50%
#
# In this case, the resulting tensor has two dangling indices, :math:`i` and :math:`k`, which defines a matrix, as expected!
#
# We can now generalize this concept to tensors, and consequently, to more than a pair of legs being contracted. For example, let us look at three tensors :math:`A_{i,j,k}`, :math:`B_{j,l,m}` and :math:`C_{k,m,n}`. To contract them, all we need to do is to sum over repeated indices (:math:`j`, :math:`k`, :math:`m`), just as we would do in `Einstein convention <https://en.wikipedia.org/wiki/Einstein_notation>`_. Thus, the (i,l,n)-th element of the resulting tensor :math:`D` is
#
# .. math::
#     (D)_{i,l,n} = \sum_{j,k,m} A_{i,j,k} B_{j,l,m} C_{k,m,n} .
#
# Note how the resulting rank-3 tensor is made up of the remaining open legs from the initial tensors :math:`(i,l,n)`. The diagrammatic representation of this equation is obtained by sticking all the legs with the same indices together.
#
# .. figure:: ../_static/demonstration_assets/tn_basics/06-tensor-tensor.png
#     :align: center
#     :width: 50%
#
# With the above contraction we have formed a network of tensors, i.e., a **Tensor Network**!
#
# .. tip::
#   A common question arising when drawing a tensor is "what is the correct order to draw the indices". For instance, in the figure above we have adopted the convention that a tensor :math:`A_{i,j,k}` corresponds to a diagram with the the first leg (:math:`i`) pointing left, the second leg (:math:`j`) pointing upwards and the third leg (:math:`k`) pointing right, and similarly for the other two tensors. However, this need not be the case. We could have defined the first leg to be the one pointing upwards, for example. Based on the use case, and the user, some conventions might seem more natural than others. The only important thing to keep in mind is to be consistent. In other words, once we choose a convetion for the order, we should apply it to all the tensors to avoid contracting the wrong indices ❌.
#
# Remember we pointed out the similarity in the notation between the tensor network contractions and Einstein notation? Then, it doesn't come as a suprise that we can perform a contraction using the function ``np.einsum``. To do so, we can start creating the 3 tensors to be contracted by reshaping a 1D array (created using``np.arange``) into rank-3 tensors of the correct dimensions.

# Create the individual rank-3 tensors
A = np.arange(6).reshape(1, 2, 3)  # ijk
B = np.arange(6).reshape(2, 3, 1)  # jlm
C = np.arange(6).reshape(3, 1, 2)  # kmn

##############################################################################
# The ``np.einsum`` function takes as inputs the tensors to be contracted and a string showing the indices of the each tensor and (optionally) the indices of the output tensor.

D = np.einsum("ijk, jlm, kmn -> iln", A, B, C)
print(D.shape)

##############################################################################
# To end this section, we want to discuss a common example of a tensor network contraction arising in quantum computing, namely the **CNOT** gate. The CNOT gate can be expressed in the computational basis as
#
# .. math::
#   CNOT = |0\rangle \langle 0 | \otimes I + |1\rangle \langle 1 | \otimes X.
#
# That is, if the control qubit is in the :math:`|1\rangle` state, we apply the :math:`X` gate on the target qubit, otherwise, we leave them untouched. However, we can also rewrite this equation as a contraction. To do so, we define two tensors:
#
# .. math::
#   T^1 = \begin{pmatrix}
#           |0\rangle \langle 0\\
#           |1\rangle \langle 1 |
#         \end{pmatrix}
#
# and
#
# .. math::
#   T^2 = \begin{pmatrix}
#           I \\
#           X
#         \end{pmatrix}
#
# This means :math:`(T^1)_{i,j,k}` and :math:`(T^2)_{l,j,m}` are two rank-3 tensors, where the index :math:`j` "*picks*" the elements in the column vector while the first and last indices correspond to the indices of the internal tensors (matrices). For instance, the :math:`j = 0`-th element of :math:`T^1` and :math:`T^2` are :math:`|0\rangle \langle 0 |` and :math:`I`, respectively. And similarly for their :math:`j = 1`-st element. This means we can redefine the CNOT expression from above as
#
# .. math::
#   CNOT = \sum_j (T^1)_{i,j,k} \otimes (T^2)_{l,j,m}.
#
# It turns out :math:`T^1` and :math:`T^2` are special tensors known as the COPY and XOR tensors, respectively, and therefore have special diagrammatic representations.
#
# .. figure:: ../_static/demonstration_assets/tn_basics/07-t1-t2.png
#     :align: center
#     :width: 50%
#
# Then, their contraction results in the well known CNOT quantum circuit representation.
#
# .. figure:: ../_static/demonstration_assets/tn_basics/08-cnot.png
#     :align: center
#     :width: 50%
# 
# This demonstrates the relation between the CNOT acting as a rank-4 tensor with dimensions :math:`2 \times 2 \times 2 \times 2` and its decomposition in terms of two rank-3 local tensors (:math:`T^1` and :math:`T^2`) of dimensions :math:`2 \times 2 \times 2`.
# 
# .. note::
# 
#   More generally, we can find decompositions of multi-qubit gates into local tensors by means of the ubiquitous singular valued decomposition (SVD). This method is explained in detail in our :doc:`MPS tutorial </demos/tutorial_mps>`. 

##############################################################################
# The cost of contracting a network:
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ 
# 
# A common task when dealing with tensors is the contraction of large networks resulting in a single tensor (including scalars). To arrive to this final tensor, we can start with a single tensor and contract it with adjacent tensors one-at-a-time. The order in which this is carried out is known as the *contraction path* or *bubbling*.
# While the final tensor is independent of the order of the contraction, the number of operations performed can vary greatly with the order in which we contract the intermediate tensors. Moreover, in a general setup, finding the optimal order of indices to be contracted is not at all a trivial task.
# 
# .. note::
# 
#   Actually finding the optimal contraction path of a tensor network is a NP-complete problem [#Arad]_.
# 
# For this reason, it is useful to look at how to calculate the computational cost or the *complexity* of a tensor network contraction. First, we look at a simple matrix-matrix contraction. Given rank-2 tensors :math:`G^1_{i,j}` and :math:`G^2_{j,k}`, we have seen the :math:`(i,k)`-th element of the resulting contraction along the :math:`j`-th index is
# 
# .. math::
#   (G^3)_{i,k} = G^2 \cdot G^1 = \sum_{j=1}^{d_j} G^{2}_{j,k} G^{1}_{i,j}
# 
# where the indices :math:`i, j, k` have dimensions :math:`d_i, d_j, d_k`, respectively. Thus, obtaining the :math:`(i,k)`-th element requires :math:`\mathcal{O}(d_j)` operations. To construct the full tensor :math:`G^3`, we must repeat this procedure :math:`d_i \times d_k` times (once for every possible value of :math:`i` and :math:`k`). Therefore, the total complexity of the contration is
# 
# .. math::
#   \mathcal{O}(d_i \times d_j \times d_k)
# 
# To illustrate the importance of choosing an efficient contraction path, let us look at a similar contraction between 3 rank-3 tensors as shown in the previous section.
# 
# .. figure:: ../_static/demonstration_assets/tn_basics/09-contraction.png
#     :align: center
#     :width: 50%
# 
# In this case, the tensors are such that :math:`A_{i,j,k} \in \mathcal{C}^{d_i \times d_j \times d_j}`, :math:`B_{j,l,m} \in \mathcal{C}^{d_j \times 1 \times d_m}`, and :math:`C_{k,m,n} \in \mathcal{C}^{d_j \times d_m \times d_i}`. First, we look at the complexity of contracting :math:`(AB)` and then :math:`C`. Following the procedure explained above, the first contraction results in a complexity of
# 
# .. math::
#   \sum_{j} A_{i,j,k} B_{j,l,m} \implies \mathcal{O}(d_i \times d_m \times d_j^2 )
# 
# Then, contracting the resulting tensor :math:`(AB)_{i, k, l, m}` with :math:`C_{k,m,n}` requires
# 
# .. math::
#   \sum_{k, m} (AB)_{i, k, l, m} C_{k,m,n}  \implies \mathcal{O}(d_j \times d_m \times d_i^2)
# 
# operations. Assuming :math:`d_j \leq d_m \leq d_i`, assymptotially, the whole contraction will have a cost of :math:`\mathcal{O}(d_j \times d_m \times d_i^2)`. Alternatively, we could first contract :math:`B` and :math:`C`, incurring in the cost
# 
# .. math::
#   \sum_{m}  B_{j,l,m} C_{k,m,n} \implies \mathcal{O}(d_i \times d_m \times d_j^2 ) .
# 
# Then, contracting the result with :math:`A` results in
# 
# .. math::
#   \sum_{j, k} A_{i,j,k} (BC)_{j,l,k,n}  \implies \mathcal{O}(d_i^2 \times d_j^2) .
# 
# This means, the second contraction path results in an asymptotic cost of :math:`\mathcal{O}(d_i^2 \times d_j^2)` - lower than the first contraction path.
# To see this in practice, let us perform the above contraction using ``np.einsum``. First, we create 3 tensors with the correct dimensions.

import timeit

di = 1000
dm = 100
dj = 10

A = np.arange(di*dj*dj).reshape(di,dj,dj) # ijk
B = np.arange(dj*1*dm).reshape(dj,1,dm) # jlm
C = np.arange(dj*dm*di).reshape(dj,dm,di) # kmn

##############################################################################
# Then, we can perform the individual contractions between pairs of tensors and time it using ``timeit``. We repeat the contraction ``20`` times and average the computation time to account for smaller fluctuations. First, we start contracting :math:`A` and :math:`B`.

iterations = 20

contraction = "np.einsum('ijk, jlm -> iklm', A, B)"
execution_time = timeit.timeit(contraction, globals=globals(), number=iterations)

average_time_ms = execution_time * 1000 / iterations
print(f"Computation cost for AB contraction: {average_time_ms:.8f} ms")

##############################################################################
# Then, we contract the result with :math:`C`.

AB = np.einsum('ijk, jlm -> iklm', A, B)
contraction = "np.einsum('iklm, kmn -> iln', AB, C)"
execution_time = timeit.timeit(contraction, globals=globals(), number=iterations)

average_time_ms = execution_time * 1000 / iterations
print(f"Computation cost for (AB)C contraction: {average_time_ms:.8f} ms")

##############################################################################
# As expected, the last contraction is much more costly than the first one. We now repeat the procedure, contracting :math:`B` and :math:`C` first. 

contraction = "np.einsum('jlm, kmn -> jlkn', B, C)"
execution_time = timeit.timeit(contraction, globals=globals(), number=iterations)

average_time_ms = execution_time * 1000 / iterations
print(f"Computation cost for BC contraction: {average_time_ms:.8f} ms")

##############################################################################
# We see this contraction is in the same order of magnitude as the contraction between :math:`A` and :math:`B`, as expected by the complexity analysis since they both yield :math:`\mathcal{O}(d_i \times d_m \times d_j^2 )`. Then, we perform the contraction between the resulting tensor and :math:`A`.

BC = np.einsum('jlm, kmn -> jlkn', B, C)

contraction = "np.einsum('ijk, jlkn -> iln', A, BC)"
execution_time = timeit.timeit(contraction, globals=globals(), number=iterations)

average_time_ms = execution_time * 1000 / iterations
print(f"Computation cost for A(BC) contraction: {average_time_ms:.8f} ms")

##############################################################################
# From this we see that the second contraction path yields a much lower complexity compared to the first one, just as we expected!

##############################################################################
# Contraction paths:
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# 
# In the previous section, through a toy example, we explored how the choice of the contraction path affects the computational cost of the tensor network contraction. As shown in [#Lam1997]_ , finding an optimal contraction path is equivalent to solving the "multiplication problem" and thus, it is NP-hard. In this section, we provide a general description of wide-spread techniques used to tackle this ubiquitous task.
# 
# First, we set up the framework of the problem. While multiway contractions - contractions between more than 2 tensors at a time - are possible, we will consider only pairwise contractions since the former can always be decomposed in terms of the latter. In addition, contracting a tensor network need not result in a single tensor. However, here we consider only the single tensor case as it underlies the more general scenario [#Gray2021]_. 
# 
# The underlying idea behind finding a contraction path is based on the construction of the computational graph, i.e., a rooted binary tree - also known as the **contraction tree** - that specifies the sequence of pairwise contractions to be executed. In this tree structure, a leaf node corresponds to a tensor from the original network (blue tensors) and the pairwise contractions (red lines) give rise to the intermediate tensors (red tensors) corresponding to the rest of the nodes in the contraction tree. 
# 
# .. figure:: ../_static/demonstration_assets/tn_basics/10-contraction-tree.png
#     :align: center
#     :width: 50%
# 
# Transforming a tensor network with an arbitrary structure into this binary tree can be achieved by a *tree embedding* of the tensor network graph [#Bienstock1990]_. Thus, optimization of the contraction path is equivalent to a search over tree embeddings of the network.
# 
# .. note::
# 
#   Besides finding a contraction path that minimizes the computational cost, we can also attempt to find a path that optimizes the memory cost. That is, a contraction path in which all intermediate tensors are below a certain size.
# 
# Now, how do we traverse the space of trees to optimize over? The most straightforward idea is to perform an exhaustic search through all of them. As explained in [#Pfeifer2014]_, this can be done (with some additional improvements) using the following well known algorithms:
# 
# - Depth-first search
# - Breadth-first search
# - Dynamic programming
# 
# While the exhaustive approach scales like :math:`\mathcal{O}(N!)`, with :math:`N` the number of tensors in the network, it can handle a handful of tensors within seconds, providing a good benchmark. In addition, compared to the following algorithms, the exhaustive search guarantees to find the global minimum optimizing the desired metric - space and/or time.
# 
# .. note::
# 
#   A recursive implementation of the depth-first search is used by default in the well known package ``opt_einsum`` `(see docs) <https://optimized-einsum.readthedocs.io/en/stable/optimal_path.html>`_.
# 
# Further approaches introduced in [#Gray2021]_ are based on alternative common graph theoretic tasks, rather than searching over the contraction tree space, such as the `balanced bipartitioning <https://en.wikipedia.org/wiki/Balanced_number_partitioning>`_ and `community detection <https://en.wikipedia.org/wiki/Community_structure>`_ algorithms. And even though, these are only heuristics that do not guarantee an optimal contraction path, they can often achieve an arbitrarliy close to optimal performance. 
# 
# An extra level of optimization, known as *hyper-optimization* is introduced by the use of different algorithms to find the optimal contraction based on the specific tensor network, as some algorithms are better suited for certain network structures. For an in-depth exploration of these heuristics, please refer to [#Gray2021]_.
# 
# .. note::
# 
#   The size of (intermediate) tensors can grow exponential with the number of indices and dimensions, specially for large-scale tensor networks. Thus, we might run into memory problems when performing the contractions. A useful additional technique to split these tensors into more manageable pieces is known as *slicing*. The idea is to change space for computation time, by temporarily fixing the values of some indices in the tensors, performing independently the contraction for each fixed value and summing the results [#Gray2021]_.
# 
# As we will explore in the next section, we can use tensor networks to simulate quantum circuits. In particular, the calculation of an expectation value corresponds to the contraction of the tensor network into a single tensor (scalar) as discussed in this section. In ``Pennylane``, this simulation can be performed using the ``DefaultTensor`` device and the method used to find the contraction path can be chosen via the ``contraction_optimizer`` keyword argument.

import pennylane as qml

dev = qml.device("default.tensor", method="tn", contraction_optimizer="auto-hq")

##############################################################################
# The different types of values accepted for ``contraction_optimizer`` are determined by the ``optimize`` parameter in ``Quimb`` (see `docs <https://quimb.readthedocs.io/en/latest/tensor-circuit.html#finding-a-contraction-path-the-optimize-kwarg>`_). See `this tutorial <https://pennylane.ai/qml/demos/tutorial_How_to_simulate_quantum_circuits_with_tensor_networks/>`_ to learn more about the ``DefaultTensor`` device.

##############################################################################
# From tensor networks to quantum circuits:
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# 
# - Quantum circuits are a restricted subclass of tensor networks
# - show examples on https://arxiv.org/pdf/1912.10049 page 8 and 9 showing a quantum circuit for a bell state, defining each component as a tensor and show their contraction.
# - Include suggestions from Korbinian: https://github.com/PennyLaneAI/qml/pull/1193#discussion_r1719502008.

##############################################################################
# References
# ----------
# 
# .. [#Selinger2010]
#    P. Selinger.
#    "A Survey of Graphical Languages for Monoidal Categories",
#    `<http://dx.doi.org/10.1007/978-3-642-12821-9_4>`__, in *New Structures for Physics*, Springer Berlin Heidelberg, pp. 289–355, 2010.
# 
# .. [#Arad]
#    I. Arad and Z. Landau.
#    "Quantum computation and the evaluation of tensor networks",
#    `<https://arxiv.org/abs/0805.0040>`__, 2010.
# 
# .. [#Lam1997]
#    C.-C. Lam, P. Sadayappan, and R. Wenger.
#    "On Optimizing a Class of Multi-Dimensional Loops with Reduction for Parallel Execution",
#    `<https://doi.org/10.1142/S0129626497000176>`__, Parallel Processing Letters, Vol. 7, No. 2, pp. 157-168, 1997.
# 
# .. [#Gray2021]
#    J. Gray, S. Kourtis, and S. Choi.
#    "Hyper-optimized tensor network contraction",
#    `<https://quantum-journal.org/papers/q-2021-03-15-410/>`__, Quantum, Vol. 5, pp. 410, 2021.
# 
# .. [#Bienstock1990]
#    D. Bienstock.
#    "On embedding graphs in trees",
#    `<https://doi.org/10.1016/009
# 
# .. [#Pfeifer2014]
#    R. N. C. Pfeifer, J. Haegeman, and F. Verstraete.
#    "Faster identification of optimal contraction sequences for tensor networks",
#    `<http://dx.doi.org/10.1103/PhysRevE.90.033315>`__, Physical Review