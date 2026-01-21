r"""
Resource Estimation of QSVT for Matrix Inversion
================================================

.. meta::
    :property="og:description": Learn how to estimate the cost of matrix inversion with QSVT for CFD applications
    :property="og:image": https://pennylane.ai/qml/_static/demonstration_assets/resource_estimation.jpeg

.. related::
    tutorial_apply_qsvt QSVT in Practice
    tutorial_lcu_blockencoding Linear combination of unitaries and block encodings

The Quantum Singular Value Transformation (QSVT) is a versatile algorithm, used to solve a range of computational
problems including unstructured search, phase estimation, and Hamiltonian simulation [#chuang2021]_. One such
problem of practical importance is **matrix inversion** because of its application to computational fluid dynamics
(CFD) simulations [#lapworth2022]_. For more information on how to use PennyLane's :func:`~.pennylane.qsvt`
functionality to run matrix inversion on a quantum computer see our demo on `QSVT in Practice
<https://pennylane.ai/qml/demos/tutorial_apply_qsvt>`_.

Simulations can only take us so far, and industrially relevant system sizes are often too large to
meaningfully simulate. Let's see how we can use PennyLane to gather insights for the problem in this regime.
In this demo, we will use the :mod:`~.pennylane.estimator` module to answer the question:

**What are the logical resource requirements for solving my *matrix inversion* problem on a quantum computer?**

Estimating the cost of QSVT
---------------------------
The logical cost of QSVT depends primarily on two factors, the **block encoding** operator and the **degree** of
the polynomial transformation. For example let's estimate the cost of performing a quintic (:code:`poly_deg = 5`)
polynomial transformation to the matrix :math:`A`:

.. math::

    A = \begin{bmatrix}
        a & 0 & 0 & b \\
        0 & -a & b & 0 \\
        0 & b & a & 0 \\
        b & 0 & 0 & -a \\
        \end{bmatrix},

for :math:`a = 0.25` and :math:`b = 0.75`. Note that this matrix can be expressed as a linear combination of
unitaries (LCU) :math:`A = 0.25 \cdot \hat{Z}_{1} + 0.75 \cdot \hat{X}_{0}\hat{X}_{1}`. We begin by building the
**block encoding** operator using the standard method of LCUs. For a recap on this technique, see our
`Linear combination of unitaries and block encodings <https://pennylane.ai/qml/demos/tutorial_lcu_blockencoding>`
demo.
"""
import pennylane.estimator as qre

lcu_A = qre.PauliHamiltonian(  
    num_qubits = 2,
    pauli_terms = {"Z": 1, "XX": 1},
)  # represents A = 0.25 * Z(1) + 0.75 * X(0)X(1)

def Standard_BE(prep, sel):
    return qre.ChangeOpBasis(prep, sel)

# Preparing a qubit in the state (√0.25)|0> + (√0.75)|1>
Prep = qre.RY(wires=["prep_1"])

Select = qre.SelectPauli(
    lcu_A,
    wires = ["prep_1", 0, 1],
)  # Select over the operators in the LCU

resources = qre.estimate(Standard_BE(Prep, Select))
print(resources)

##############################################################################
# Here the :class:`~.pennylane.estimator.PauliHamiltonian` class is used to represent the LCU in a 
# compact object for resource estimation, and the :class:`~.pennylane.estimator.ChangeOpBasis` class
# uses the compute-uncompute pattern to implement the 
# :math:`\hat{\text{Prep}}^{\dagger} \cdot \hat{\text{Select}} \cdot \hat{\text{Prep}}` operator.
#
# The resources for :class:`~.pennylane.estimator.QSVT` can then be obtained using this block encoding:

qsvt_op = qre.QSVT(
    block_encoding = Standard_BE(Prep, Select),
    encoding_dims = (4, 4),  # The shape of matrix A
    poly_deg = 5,
)

resources = qre.estimate(qsvt_op)
print(resources)

##############################################################################
# In the next sections we will explore how to optimize the *degree* and *block encoding*
# in order to minimize the resource requirements of QSVT for matrix inversion.
#
# Polynomial Approximations of the Inverse Function
# -------------------------------------------------
# The cost of QSVT is directly proportional to the degree of the polynomial transformation. More
# specifically, the number of calls to the block encoding operator scales linearly with the degree.
# For matrix inversion, we want to apply a polynomial transformation that :math:`f(x) \approx \frac{1}{x}`
# within some target error :math:`\epsilon`. In this case, the higher the degree, the better the
# approximation will be in general.
#
# This creates a delicate balance, we want to reduce the degree of the transformation as much as
# possible without compromising the accuracy of our matrix inversion. Luckily, this problem has 
# been studied in the literature and there is an expression for the *optimal* polynomial approximation
# of the inverse function that minimizes the target error given the maximum allowed degree[#sunderhauf2025]_.
#
# |
#
# .. figure:: ../_static/demonstration_assets/re_qsvt/optimal_polynomial_approx.png
#     :align: center
#     :width: 50%
#     :target: javascript:void(0)
#
# |
#
# The error in this polynomial approximation can be derived from the degree (:math:`d`) and the condition
# number of the matrix (:math:`\kappa`) [#sunderhauf2025]_:
#
# .. math:: \epsilon_{d}(a) = \frac{(1 - a^{n})}{a(1 + a)^{n-1}} .
# 
# Where :math:`a = \frac{1}{\kappa}` and :math:`n = 2(d + 1)`. We can use this expression to determine the
# optimal degree needed to approximate :math:`\frac{1}{x}` within the target error :math:`\epsilon`.

import numpy as np

def polynomial_approx_error(degree, kappa):
    """The error in the polynomial approximation of 1/x 
    as described in Theorem 1 of https://arxiv.org/pdf/2507.15537"""
    a = 1 / kappa
    n = (degree + 1) // 2 # the degree must be an odd number!

    numerator = (1 - a)**n
    denominator = a * (1 + a)**(n-1)

    return numerator / denominator


def optimal_degree_from_error(error_threshold, kappa):
    """Determine the optimal degree of the polynomial 
    approximation for 1/x required to be within the given threshold"""
    degree = 3
    eps = np.inf  # set initial value as upper bound
    while eps > error_threshold:
        degree += 2
        eps = polynomial_approx_error(degree, kappa)

    return degree

kappa = 10
epsilon = 1e-5
print("Minimum degree:", optimal_degree_from_error(epsilon, kappa))

##############################################################################
# Block Encodings for Structured Matrices
# ---------------------------------------
# The other parameter that impacts the cost of QSVT is the block encoding operator. While
# the strategy of decomposing :math:`A` into an LCU of Pauli operators works for any (square)
# matrix in general, it suffers from a few fatal flaws (try saying that five times fast).
# 
# The number of terms in the LCU scales like :math:`O(4^{n})` in the number of qubits, and thus
# the cost of the block encoding also scales exponentially. Even computing the LCU decomposition
# becomes a computational bottleneck. Furthermore, there is no way of knowing a priori how many
# terms there will be in the LCU. For these reasons, a general "one size fits all" block encoding
# scheme is usually too expensive for our systems of interest.
#
# Instead, we leverage the inherent patterns and *structure* of our matrix in order to implement
# an efficient block encoding operator. Let's focus on a specific test case from CFD simulations,
# the lid driven cavity[#lapworth2022]_. The pressure correction matrix that arises from simulating
# flow within this cavity has a very sparse, structured diagonal form. The figure below highlights
# the non-zero entries in this :math:`(64, 64)` matrix [#lapworth2022]_.
#
# |
#
# .. figure:: ../_static/demonstration_assets/re_qsvt/sparse_diagonal_matrix.png
#     :align: center
#     :width: 50%
#     :target: javascript:void(0)
#
# |
#
# This matrix (:math:`A`) can be block encoded using a *d-diagonal encoding* technique[#linaje2025]_ developed
# by my colleagues here at Xanadu. The method works by loading each diagonal in parallel and then
# shiftting them to their respective ranks in the matrix. The quantum circit that implements the
# d-diagonal block encoding is presented below, for more information checkout our paper
# `"Quantum compilation framework for data loading" <https://arxiv.org/abs/2512.05183>`_.
#
# |
#
# .. figure:: ../_static/demonstration_assets/re_qsvt/SparseBE.png
#     :align: center
#     :width: 50%
#     :target: javascript:void(0)
#
# |
#
# Estimating the resource cost for this circuit may seem like a daunting task, but we have
# the :mod:`~.pennylane.estimator` module to help us construct each piece!
#
# Diagonal Matrices & the Walsh-Hadamard Transform
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Let's start with the :math:`D_{k}` operators. These are a list of diagonal operators where
# each operator stores the normalized entries from one of the diagonals of our d-diagonal matrix :math:`A`.
# By multiplexing over the :math:`D_{k}` operators, we can load all of the diagonals in *parallel*.
# Typically, each diagonal operator is implemented using a product of Controlled PhaseShift gates, 
# in this case we will be leveraging the Walsh-Hadamard transformation ([#linaje2025]_, [#zylberman2025]_).
# 
# The Walsh-Hadamard transform allows us to naturally optimize the cost of our block encoding by tuning the
# :code:`num_walsh_coeffs` parameter. This parameter takes values from :math:`[1, N]` where :math:`N` is the
# the size of the matrix. If the entries in our diagonal are sparse in the Walsh basis, as is the case for
# the CFD example, then we can get away with much fewer :code:`num_walsh_coeffs`. This results in a much more
# efficient encoding. The circuit below prepares a single such Walsh-Hadamard diagonal operator, ultimately
# we need to prepare as many operators as non-zero diagonals in :math:`A`.
#
# |
#
# .. figure:: ../_static/demonstration_assets/re_qsvt/WH_a.png
#     :align: center
#     :width: 50%
#     :target: javascript:void(0)
#
# |
#

def WalshHadamard_Dk(num_diags, size_diagonal, num_walsh_coeffs):
    num_diag_wires = int(np.ceil(np.log2(size_diagonal)))
    list_of_diagonal_ops = []

    for _ in range(num_diags):
        compute_op = qre.Prod([qre.Hadamard(), qre.S()])

        zero_ctrl_wh = qre.Controlled(
            qre.Prod(((qre.MultiRZ(num_diag_wires//2), num_walsh_coeffs),)),
            num_ctrl_wires = 1,
            num_zero_ctrl = 1,
        )
        one_ctrl_wh = qre.Controlled(
            qre.Prod(((qre.MultiRZ(num_diag_wires//2), num_walsh_coeffs),)),
            num_ctrl_wires = 1,
            num_zero_ctrl = 0,
        )
        target_op = qre.Prod([zero_ctrl_wh, one_ctrl_wh])

        list_of_diagonal_ops.append(qre.ChangeOpBasis(compute_op, target_op))

    return list_of_diagonal_ops

##############################################################################
# Here the :class:`~.pennylane.estimator.Prod` class is used to represent a product of operators.
# 
# Shift Operator
# ^^^^^^^^^^^^^^
# Next we will implement the *Shift* operator. This is a product of three subroutines which
# shift the diagonal entries into the correct rank of the d-diagonal block encoding. Each rank
# has an associated shift value, the main diagonal has shift value 0. Each diagonal going up
# is labelled :math:`(+1, +2, \elipsis)` respectively and each diagonal going down is labelled
# with its negative counterpart. The Shift operator works in three steps: 
#
# 1. *Load* the binary representation of the shift value for each non-zero diagonal in :math:`A`.
#
# 2. *Shift* each of the diagonals in parallel using an :math:`Adder` subroutine.
#
# 3. *Unload* the shift values to restore the quantum register.
#

def ShiftOp(num_shifts, num_load_wires, wires):
    prep_wires, load_shift_wires, load_diag_wires = wires

    Load = qre.QROM(
        num_bitstrings = num_shifts,
        size_bitstring = num_load_wires,
        restored = False,
        select_swap_depth = 1,
        wires = prep_wires + load_shift_wires,
    )

    Adder = qre.SemiAdder(
        max_register_size = num_load_wires,
        wires = load_shift_wires + load_diag_wires,
    )
    
    Unload = qre.Adjoint(
        qre.QROM(
            num_bitstrings = num_shifts,
            size_bitstring = num_load_wires,
            restored = False,
            select_swap_depth = 1,
            wires = prep_wires + load_shift_wires,
        )
    )
    return qre.Prod([Load, Adder, Unload])

##############################################################################
# d-Diagonal Block Encoding
# ^^^^^^^^^^^^^^^^^^^^^^^^^
# Now that we have developed all of the pieces, lets bring them together to implement
# the full d-diagonal block encoding operator[#linaje2025]_.
#
# |
#
# .. figure:: ../_static/demonstration_assets/re_qsvt/SparseBE.png
#     :align: center
#     :width: 50%
#     :target: javascript:void(0)
#
# |
#

def WH_Diagonal_BE(num_diagonals, matrix_size, num_walsh_coeffs):
    # Initialize qubit registers:
    num_prep_wires = int(np.ceil(np.log2(num_diagonals)))
    num_diag_wires = int(np.ceil(np.log2(matrix_size)))

    prep_wires = [f"prep{i}" for i in range(num_prep_wires)]
    load_diag_wires = [f"load_d{i}" for i in range(num_diag_wires)]
    load_shift_wires = [f"load_s{i}" for i in range(num_diag_wires)]

    # Prep: 
    Prep = qre.QROMStatePreparation(num_prep_wires, wires = prep_wires)

    # Shift:
    Shift = ShiftOp(
        num_shifts = num_diagonals,
        num_load_wires = num_diag_wires,
        wires = (prep_wires, load_diag_wires, load_shift_wires)
    )
    
    # Multiplexed - Dk:
    diagonal_ops = WalshHadamard_Dk(num_diagonals, matrix_size, num_walsh_coeffs)
    Dk = qre.Select(diagonal_ops, wires = load_diag_wires + prep_wires + ["target"])

    # Prep^t:
    Prep_dagger = qre.Adjoint(qre.QROMStatePreparation(num_prep_wires, wires = prep_wires))

    return qre.Prod([Prep, Shift, Dk, Prep_dagger])

##############################################################################
# Application: Computational Fluid Dynamics
# -----------------------------------------
# In this section we use all of the tools we have developed to **estimate the resource requirements
# for solving a practical matrix inversion problem**. Following the CFD simulations example
# [linaje2025]_, we will estimate the resources required to invert a tri-diagonal matrix of size
# :math:(2^{20}, 2^{20}). This system required a :math:`10^{8}`-degree polynomial approximation of the
# inverse function.
# 
# We will also restrict the gateset to Clifford plus T-gates. This will allows to generate results that
# are faithful to the ones presented in [linaje2025]_.

degree = 10**8
matrix_size = 2**20  # 2^20 x 2^20
num_diagonals = 3
num_walsh_coeffs = 512  # imperically determined

def matrix_inversion(degree, matrix_size, num_diagonals):
    num_state_wires = int(np.ceil(np.log2(matrix_size)))  # 20 qubits

    # Prepare |b> vector: 
    qre.MPSPrep(num_state_wires, max_bond_dim = 2**5)  # bond dimension = 2**5

    # Apply A^-1:
    qre.QSVT(
        block_encoding = WH_Diagonal_BE(num_diagonals, matrix_size, num_walsh_coeffs),
        encoding_dims = (matrix_size, matrix_size),
        poly_deg = degree,
    )
    return

gate_set = {'X', 'Y', 'Z', 'Hadamard', 'T', 'S', 'CNOT'}
res = qre.estimate(matrix_inversion, gate_set)(degree, matrix_size, num_diagonals)
print(res)

##############################################################################
# The T-cost of this matrix inversion workflow matche the reported results :math:`3 \cdot 10^{13}` from
# the reference.
#
# Challenge (Next Steps)
# ----------------------
# In this demo, you learned how to use PennyLane's :mod:`~.pennylane.estimator` module to determine the 
# resource requirements for *matrix inversion* by QSVT. Along the way we explored the two major factors 
# that impact the cost of QSVT (block encoding and degree of the approximation), as well as how they can
# be optimized to reduce the overall cost of the algorithm. We showcased how complex circuits can be
# built from our library of primatives, making resource estimation as simple as putting together Legos.
# Finally, we applied all of these techniques to estimate the resources for a CDF matrix inversion workflow;
# validating the results from the literature.
#
# Now that you are armed with these tools for resource estimation, I challenge you to find another problem
# of interest and answer the question:
#
# **What are the logical resource requirements for solving my *matrix inversion* problem on a quantum computer?**
#
# References
# ----------
#
# .. [#chuang2021]
#
#     John M. Martyn, Zane M. Rossi, Andrew K. Tan, and Isaac L. Chuang,
#     "A Grand Unification of Quantum Algorithms"
#     `arxiv.2105.02859 <https://arxiv.org/pdf/2105.02859>`__, 2021.
#
# .. [#sunderhauf2025]
#
#     Christoph Sunderhauf, Zalan Nemeth, Adnaan Walayat, Andrew Patterson, and Bjorn K. Berntson,
#     "Matrix inversion polynomials for the quantum singular value transformation"
#     `arxiv.2507.15537 <https://arxiv.org/pdf/2507.15537>`__, 2025.
#
# .. [#lapworth2022]
#
#     Leigh Lapworth
#     "A HYBRID QUANTUM-CLASSICAL CFD METHODOLOGY WITH BENCHMARK HHL SOLUTIONS"
#     `arxiv.2206.00419 <https://arxiv.org/pdf/2206.00419>`__, 2022.
#
# .. [#linaje2025]
#
#     Guillermo Alonso-Linaje, Utkarsh Azad, Jay Soni, Jarrett Smalley,
#     Leigh Lapworth, and Juan Miguel Arrazola,
#     "Quantum compilation framework for data loading"
#     `arxiv.2512.05183 <https://arxiv.org/pdf/2512.05183>`__, 2025.
#
# .. [#zylberman2025]
#
#     Julien Zylberman, Ugo Nzongani, Andrea Simonetto, and Fabrice Debbasch,
#     "Efficient Quantum Circuits for Non-Unitary and Unitary Diagonal Operators with Space-Time-Accuracy trade-offs"
#     `arxiv.2404.02819 <https://arxiv.org/pdf/2404.02819>`__, 2025.
#
