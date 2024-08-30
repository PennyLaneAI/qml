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

TODO: add diagram here as the second part of the equation.

.. math::
    T_{i,j,k,l} = 
    
.. figure:: ../_static/demonstration_assets/tn_basics/tensor_one.png
    :align: center
    :width: 60%

We can apply this same idea to represent a scalar, a vector and a matrix:

TODO: add diagrams here.

.. math::
    s = 
    
.. math::
    v_i =
    
.. math::
    G_{i,j} =  

Does the last diagram seem familiar? It is because this is the representation of a single-qubit gate! We will see later in this tutorial the relation between quantum circuits and tensor networks. 

Some authors choose to give the shape of the tensor an additional meaning. For example, you might encounter square tensors representing symmetric matrices, to represent its invariance under transposition.

TODO: add diagram here showing: 1. Square matrix 2. Bending of wires as transpose 3. Final matrix

In addition, an isometry :math:`V \in \mathbb{C}^{n \times p}` - i.e., a rectangular matrix such that :math:`V V^\dagger = I_p` but :math:`V^\dagger V \neq I_n` - with :math:`n > p`  can be depicted using a triangle to emphasize the isometry property:

TODO: add diagram here showing the isometry definition (identity from the smaller space to the bigger space). See https://www.math3ma.com/blog/matrices-as-tensor-network-diagrams

For quantum states, it is often useful to adopt the convention that writing the legs towards one direction (e.g. the right) corresponds to a ket, i.e., a vector living in the Hilbert space, while drawing the legs to the opposite direction (e.g. left) means they are a bra vector, i.e., living in the dual space.

TODO: 1) drawing of a vector with legs towards right diretion (ket) 2) legs towards left (bra)

TODO: add reference here for category theory. P. Selinger. A survey of graphical languages for monoidal categories. Lecture Notes
in Physics, page 289–355, 2010.

.. tip::
    The diagrammatic representation of tensors is rooted in category theory, which equips the diagrams with all the relevant information so they can used in proofs and formal reasoning!

Creating a tensor in code is straightforward, and chances are you have already created one yourself. Using ``Numpy``, all we have to do is to create a ``np.array`` of the desired rank. For instance, we can start crearting a rank-1 tensor (a vector).
"""

import numpy as np

tensor_rank1 = np.array([1, 2, 3])
print("rank: ", tensor_rank1.ndim)
print("dimensions: ", tensor_rank1.shape)

##############################################################################
# Then, we can use this to construct a rank-2 tensor (a matrix).

tensor_rank2 = np.array([tensor_rank1, tensor_rank1])
print("rank: ", tensor_rank2.ndim)
print("dimensions: ", tensor_rank2.shape)

##############################################################################
# As you might have guessed, we can repeat this procedure to create a rank-3 tensor.

tensor_rank3 = np.array([tensor_rank2])
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
# TODO: add diagram at the end of the equation.
#
# .. math::
#     (w)_i = G \cdot v = \sum_j G_ij v_j =
#
# We see that summing over the shared index :math:`j` is equivalent to **contracting** the corresponding legs from the matrix and vector diagrams. As expected, the result of this multiplication is another rank-1 tensor with dangling leg :math:`i`. Similarly, we can look at the matrix-matrix multiplication:
#
# TODO: add diagram at the end of the equation.
#
# .. math::
#     (G^3)_{i,k} = G^1 \cdot G^2 = \sum_j G^{1}_{i,j} G^{2}_{j,k} =
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
# TODO: add figure here.
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
# TODO: add a diagram for each of these
#
# Then, their contraction results in the well known CNOT quantum circuit representation.
#
# TODO: add the CNOT diagram here.

##############################################################################
# The cost of contracting a network:
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ 
# 
# A common task when dealing with tensors is the contraction of large networks resulting in a single tensor (including scalars). To arrive to this final tensor, we can start with a single tensor and contract it with adjacent tensors one-at-a-time. The order in which this is carried out is known as the *contraction path* or *bubbling*.
# While the final tensor is independent of the order of the contraction, the number of operations performed can vary greatly with the order in which we contract the intermediate tensors. Moreover, in a general setup, finding the optimal order of indices to be contracted is not at all a trivial task - actually it is a NP-complete problem [#Arad]_.
# 
# For this reason, it is useful to look at how to calculate the computational cost or the *complexity* of a tensor network contraction. First, we look at a simple matrix-matrix contraction. Given rank-2 tensors :math:`G^1_{i,j}` and :math:`G^2_{j,k}`, we have seen the :math:`(i,k)`-th element of the resulting contraction along the :math:`j`-th index is
# 
# .. math::
#   (G^3)_{i,k} = G^1 \cdot G^2 = \sum_{j=1}^{d_j} G^{1}_{i,j} G^{2}_{j,k}
# 
# where the indices :math:`i, j, k` have dimensions :math:`d_i, d_j, d_k`, respectively. Thus, obtaining the :math:`(i,k)`-th element requires :math:`\mathcal{O}(d_j)` operations. To construct the full tensor :math:`G^3`, we must repeat this procedure :math:`d_i \times d_k` times (once for every possible value of :math:`i` and :math:`k`). Therefore, the total complexity of the contration is
# 
# .. math::
#   \mathcal{O}(d_i \times d_j \times d_k)
# 
# To illustrate the importance of choosing an efficient contraction path, let us look at a similar contraction between 3 rank-3 tensors as shown in the previous section.
# 
# TODO: add diagram here with 3 tensors. Tensor A has dimensions (i,j,k)->(d_i, d_j, d_j), tensor B has dimension (j,l,m)->(d_j, 1, d_m) and tensor C has dimensions (k,m,n)->(d_j, d_m, d_i). Positioned in a traingle-like structure similar to the previous example.
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
# To see this in practice, let us perform the above contraction using ``np.einsum``. First, we crate 3 tensors with the correct dimensions.

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
# From this we see tha the second contraction path yields a much lower complexity compared to the first one, just as we expected!
# 
# - For this reason there exist heuristics for optimizing contraction path complexity. NP problem -> no perfect solution but great heuristics (https://arxiv.org/pdf/2002.01935).
#     (optional) mention the idea behind some of them
#     Link to quimb examples.

##############################################################################
# From tensor networks to quantum circuits:
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# 
# - Quantum circuits are a restricted subclass of tensor networks
# - show examples on https://arxiv.org/pdf/1912.10049 page 8 and 9 showing a quantum circuit for a bell state, defining each component as a tensor and show their contraction.
# - What else?

##############################################################################
# References
# ----------
# .. [#Arad]
#    I. Arad and Z. Landau.
#    "Quantum computation and the evaluation of tensor networks",
#    `<https://arxiv.org/abs/0805.0040>`__, 2010.