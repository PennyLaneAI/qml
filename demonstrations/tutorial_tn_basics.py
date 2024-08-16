r"""Tutorial: ABC of tensor networks
=============================================================

Intro:
- introduce where are tensor networks used (physics, math, machine learning), why are they so widely spread, and mention how they are not just a data structure but also a diagrammatic language for formal reasoning and even proofs!
- Define the scope of this tutorial: basics of tensor networks, their relation to quantum circuits, and the importance of choosing the contraction paths.

From matrices to tensors
^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Definition of a tensor network as an n-dimensional array. Show notation with n indices, stating that it belongs to C^{d1,...,dn}. Define rank, index, dimension (mention how these terms are sometimes used (wrongly?) interchangeably in literature).
- Graphical notation. Mention that there exist certain representations in the literature that allow to represent properties of the tensors (e.g. symmetry, orthogonality). In our case, we can adhere to a general circle.
- Specific examples used in a day-to-day: scalars, vectors, matrices. Mention that for quantum states, we can adopt the convention that the legs in one direction mean that the state belongs to one Hilbert space, and the legs to the other side to the dual space.
- CODE: include code using numpy creating a >2 dimensional array.

From matrix multiplication to tensor contractions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Show the matrix multiplication in terms of summation over indices, then using the diagrammatic representation. This results in another rank 2 tensor (matrix)
- Analagously, we can represent matrix-vector multiplication resulting in a rank 1 tensor (vector). Just as we expected!
- We can generalize this concept to tensors. This is done by summing over repeated indices (just as in einstein convention - external link for it) resulting in another tensor made up of the open legs of all the tensors together.
    In diagrammatic notation, this is simply sticking together legs with same indices! (show nice diagram with >3 tensors). We have just formed a network of tensors, i.e. a Tensor Network!
- CODE: Talking about einstein convetion, we can perform this contraction of tensors using `np.einsum`.

The cost of contracting a network (bubbling):
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Mention that the resulting tensor network doesn't change but the way we arrive to the final tensor affects how expensive it is to get there.
- Show how can we can calculate the complexity of a contraction by means of a simple example using 2 matrices (rank 2 tensors): dimension_contracted x (dimensions_open).
    Intuition behind: we perform one operation (contraction) and repeat many times to "populate" the resulting tensor (dimension_open1 x dimension_open2). Show the equation with indices.
- Show an example with at least three tensors where they all have different dimensions. Walk through it showing that choosing to contract two indices (the ones with lower dimensions) results in a worst computational complexity than contracting other ones (the ones with higher dimensions).
- For this reason there exist heuristics for optimizing contraction path complexity. NP problem -> no perfect solution but great heuristics (https://arxiv.org/pdf/2002.01935).
    (optional) mention the idea behind some of them 
    Link to quimb examples.
- CODE: show this using np.einsum, timeit, and very large dimensions expecting to see a difference.

From tensor networks to quantum circuits:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Quantum circuits are a restricted subclass of tensor networks
- show examples on https://arxiv.org/pdf/1912.10049 page 8 and 9 showing a quantum circuit for a bell state, defining each component as a tensor and show their contraction.
- What else?
"""

