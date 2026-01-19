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

**What is the logical resource cost of solving my *matrix inversion* problem on a quantum computer?**

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
# (Add something here!)

##############################################################################
# Polynomial Approximations of the Inverse Function
# -------------------------------------------------
# The cost of QSVT is directly proportional to the degree of the polynomial transformation
# we wish to app

algo = qml.estimator.DoubleFactorization(one, two)

##############################################################################
# and obtain the estimated number of non-Clifford gates and logical qubits.

print(f'Estimated gates : {algo.gates:.2e} \nEstimated qubits: {algo.qubits}')

##############################################################################
# This estimation is for a target error that is set to the chemical accuracy, 0.0016
# :math:`\text{Ha},` by default. We can change the target error to a larger value which leads to a
# smaller number of non-Clifford gates and logical qubits.

chemical_accuracy = 0.0016
error = chemical_accuracy * 10
algo = qml.estimator.DoubleFactorization(one, two, error=error)
print(f'Estimated gates : {algo.gates:.2e} \nEstimated qubits: {algo.qubits}')

##############################################################################
# We can also estimate the number of non-Clifford gates with respect to the threshold error values
# for discarding the negligible factors in the factorized Hamiltonian [#vonburg2021]_ and plot the
# estimated numbers.

threshold = [10**-n for n in range(10)]
n_gates = []
n_qubits = []

for tol in threshold:
    algo_ = qml.estimator.DoubleFactorization(one, two, tol_factor=tol, tol_eigval=tol)
    n_gates.append(algo_.gates)
    n_qubits.append(algo_.qubits)

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(threshold, n_gates, ':o', markerfacecolor='none', color='teal')

ax.set_ylabel('n gates')
ax.set_xlabel('threshold')
ax.set_xscale('log')
fig.tight_layout()

##############################################################################
# QPE cost for simulating periodic materials
# ******************************************
# For periodic materials, we estimate the cost of implementing the QPE algorithm of [#zini2023]_
# using Hamiltonians represented in first quantization and in a plane wave basis. We first need to
# define the number of plane waves, the number of electrons, and the lattice vectors that construct
# the unit cell of the periodic material. Let's use dilithium iron silicate
# :math:`\text{Li}_2\text{FeSiO}_4` as an example taken from [#delgado2022]_. For this material, the
# unit cell contains 156 electrons and has dimensions :math:`9.49 \times 10.20 \times 11.83` in
# `atomic units <https://en.wikipedia.org/wiki/Bohr_radius>`_. We also use :math:`10^5` plane waves.

planewaves = 100000
electrons = 156
vectors = np.array([[9.49,  0.00,  0.00],
                    [0.00, 10.20,  0.00],
                    [0.00,  0.00, 11.83]])

##############################################################################
# We now create an instance of the :class:`~.pennylane.estimator.qpe_resources.FirstQuantization` class
algo = qml.estimator.FirstQuantization(planewaves, electrons, vectors=vectors)

##############################################################################
# and obtain the estimated number of non-Clifford gates and logical qubits.
print(f'Estimated gates : {algo.gates:.2e} \nEstimated qubits: {algo.qubits}')

##############################################################################
# We can also plot the estimated numbers as a function of the number of plane waves for different
# target errors

error = [0.1, 0.01, 0.001]  # in atomic units
planewaves = [10 ** n for n in range(1, 10)]
n_gates = []
n_qubits = []

for er in error:
    n_gates_ = []
    n_qubits_ = []

    for pw in planewaves:
        algo_ = qml.estimator.FirstQuantization(pw, electrons, vectors=vectors, error=er)
        n_gates_.append(algo_.gates)
        n_qubits_.append(algo_.qubits)
    n_gates.append(n_gates_)
    n_qubits.append(n_qubits_)

fig, ax = plt.subplots(2, 1)

for i in range(len(n_gates)):
    ax[0].plot(planewaves, n_gates[i], ':o', markerfacecolor='none', label=error[i])
ax[1].plot(planewaves, n_qubits[i], ':o', markerfacecolor='none', label=error[-1])

ax[0].set_ylabel('n gates')
ax[1].set_ylabel('n qubits')

for i in [0, 1]:
    ax[i].set_xlabel('n planewaves')
    ax[i].tick_params(axis='x')
    ax[0].set_yscale('log')
    ax[i].set_xscale('log')
    ax[i].legend(title='error [Ha]')

fig.tight_layout()

##############################################################################
# The algorithm uses a decomposition of the Hamiltonian as a linear combination of unitaries,
#
# .. math:: H=\sum_{i} c_i U_i.
#
# The parameter :math:`\lambda=\sum_i c_i,` which can be interpreted as the 1-norm of the
# Hamiltonian, plays an important role in determining the cost of implementing the QPE
# algorithm [#delgado2022]_. In PennyLane, the 1-norm of the Hamiltonian can be obtained with

print(f'1-norm of the Hamiltonian: {algo.lamb}')

##############################################################################
# PennyLane allows you to get more detailed information about the cost of the algorithms as
# explained in the documentation of :class:`~.pennylane.estimator.qpe_resources.FirstQuantization`
# and :class:`~.pennylane.estimator.qpe_resources.DoubleFactorization` classes.
#
# Variational quantum eigensolver
# ------------------------------------------
# In variational quantum algorithms such as VQE, the expectation value of an observable is
# typically computed by decomposing the observable into a linear combination of Pauli words,
# which are tensor products of Pauli and Identity operators. The expectation values are calculated
# through linearity by measuring the expectation value for each of these terms and combining the
# results. The number of qubits required for the measurement is trivially determined by
# the number of qubits the observable acts on. The number of gates required to implement the
# variational algorithm is determined by a circuit ansatz that is also known a priori. However,
# estimating the number of circuit evaluations, i.e. the number of shots, required to achieve a
# certain error in computing the expectation value is not as straightforward. Let's now use
# PennyLane to estimate the number of shots needed to compute the expectation value of the water
# Hamiltonian.
#
# First, we construct the molecular Hamiltonian.

molecule = qml.qchem.Molecule(symbols, geometry)
H = qml.qchem.molecular_hamiltonian(molecule)[0]
H_coeffs, H_ops = H.terms()

##############################################################################
# The number of measurements needed to compute :math:`\left \langle H \right \rangle` can be
# obtained with the :func:`~.pennylane.estimator.measurement.estimate_shots` function, which requires the
# Hamiltonian coefficients as input. The number of measurements required to compute
# :math:`\left \langle H \right \rangle` with a target error set to the chemical accuracy, 0.0016
# :math:`\text{Ha},` is obtained as follows.

m = qml.estimator.estimate_shots(H_coeffs)
print(f'Shots : {m:.2e}')

##############################################################################
# This number corresponds to the measurement process where each term in the Hamiltonian is measured
# independently. The number can be reduced by using
# :func:`~.pennylane.pauli.group_observables()`, which partitions the Pauli words into
# groups of commuting terms that can be measured simultaneously.

ops, coeffs = qml.pauli.group_observables(H_ops, H_coeffs)
coeffs = [np.array(c) for c in coeffs] # cast as numpy array

m = qml.estimator.estimate_shots(coeffs)
print(f'Shots : {m:.2e}')

##############################################################################
# It is also interesting to illustrate how the number of shots depends on the target error.

error = np.array([0.02, 0.015, 0.01, 0.005, 0.001])
m = [qml.estimator.estimate_shots(H_coeffs, error=er) for er in error]

e_ = np.linspace(error[0], error[-1], num=50)
m_ = 1.4e4 / np.linspace(error[0], error[-1], num=50)**2

fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(error, m, 'o', markerfacecolor='none', color='teal', label='estimated')
ax.plot(e_, m_, ':', markerfacecolor='none', color='olive', label='$ 1.4e4 * 1/\epsilon^2 $')

ax.set_ylabel('shots')
ax.set_xlabel('error [Ha]')
ax.set_yscale('log')
ax.tick_params(axis='x', labelrotation = 90)
ax.legend()
fig.tight_layout()

##############################################################################
# We have added a line showing the dependency of the shots to the error, as
# :math:`\text{shots} = 1.4\text{e}4 \times 1/\epsilon^2,` for comparison. Can you draw any
# interesting information form the plot?
#
# Conclusions
# -----------
# This tutorial shows how to use the resource estimation functionality in PennyLane to compute the
# total number of non-Clifford gates and logical qubits required to simulate a Hamiltonian with
# quantum phase estimation algorithms. The estimation can be performed for second-quantized
# molecular Hamiltonians obtained with a double low-rank factorization algorithm,
# and first-quantized Hamiltonians of periodic materials in the plane wave basis. We also discussed
# the estimation of the total number of shots required to obtain the expectation value of an
# observable using the variational quantum eigensolver algorithm. The functionality allows one to
# obtain interesting results about the cost of implementing important quantum algorithms. For
# instance, we estimated the costs with respect to factors such as the target error in obtaining
# energies and the number of basis functions used to simulate a system. Can you think of other
# interesting information that can be obtained using this PennyLane functionality?
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
# .. [#linaje2022]
#
#     Guillermo Alonso-Linaje, Utkarsh Azad, Jay Soni, Jarrett Smalley,
#     Leigh Lapworth, and Juan Miguel Arrazola,
#     "Quantum compilation framework for data loading"
#     `arxiv.2512.05183 <https://arxiv.org/pdf/2512.05183>`__, 2025.
#
