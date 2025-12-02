r"""Exploring Qubit and T-gate Trade-offs in Qubitized Quantum Phase Estimation
===============================================================================

The description of any chemical system begins with its Hamiltonian, :math:`\hat{H}`. While Quantum Phase Estimation (QPE)
is the primary algorithm for resolving the energy spectra of such systems, it faces a fundamental constraint: quantum processors
execute Unitary operations, not Hermitian ones. To extract the energy spectrum, whether the ground state or excited states,
we must bridge this mathematical divide.

Qubitization provides the solution to this problem via Block Encoding. Conceptually, this technique fits the Hamiltonian
into a subspace of a larger Unitary matrix, the "Quantum Walk" operator :math:`W`. This operator encodes the spectrum of :math:`H`
into its own eigenphases via the rigorous map :math:`e^{\pm i \arccos(E_k/\lambda)}`. This transformation is exact,
allowing us to query the system's energy levels without the approximation errors inherent in simulating time evolution.
For a detailed theoretical background on constructing :math:`W`, refer to our `Qubitization demo <https://pennylane.ai/qml/demos/tutorial_qubitization>`__.

However, constructing the operator is only half the battle; we also need to know what it costs to run. In this demo, we move from
abstract scaling to concrete costs using logical resource estimation tools in PennyLane.

.. figure:: ../_static/demo_thumbnails/opengraph_demo_thumbnails/OGthumbnail_how_to_build_spin_hamiltonians.png
    :align: center
    :width: 70%
    :target: javascript:void(0)
"""

######################################################################
# The resources required for this algorithm are dictated by the particular Block Encoding technique, i.e., the process
# of decomposing the operator into a Linear Combination of Unitaries (LCU) that the quantum hardware can execute. In this demo,
# we perform logical resource estimation for Qubitized QPE using the Tensor Hypercontracted (THC) representation. THC is the
# state of the art LCU representation for quantum chemistry that approximates the interaction tensor via a low-rank factorization:
#
# .. math::
#    V_{pqrs} \approx \sum_{\mu, \nu}^{M} \chi_{p\mu} \chi_{q\mu} \zeta_{\mu\nu} \chi_{r\nu} \chi_{s\nu},
#
# where :math:`M` is the factorization rank, and :math:`\chi` and :math:`\zeta` are the factorized tensors.
# To implement this decomposition on a quantum computer, the Walk operator is constructed from two primary subroutines:
# the ``Prepare`` oracle, which encodes the coefficients into an ancillary register, and the ``Select`` oracle,
# which applies the Hamiltonian terms controlled by that register.
#
#
# Standard resource estimates often treat these oracles as fixed "black boxes,"
# yielding a single cost number. However, this demo is more than just a static cost report. We demonstrate how PennyLane exposes the
# tunable knobs of the circuit implementation, allowing us to actively navigate the circuit design and trade off between T-gates and
# logical qubits to suit different constraints.
#
#
# Resource Estimation for FeMoco
# ------------------------------
# Estimating resources for large-scale chemical systems is often bottlenecked by the challenge of constructing and storing the full Hamiltonian tensor.
# But why carry the entire building when a blueprint will do? The resource estimator allows us to sidestep this bottleneck for a quick estimation
# or for Hamiltonians available in the literature. By using a `compact representation <https://docs.pennylane.ai/en/latest/code/qml_estimator.compact_hamiltonian>`__,
# we capture the essential structure of the Hamiltonian needed for resource estimation, without instantiating the full Hamiltonian tensor.
#
# Specifically, the resource estimation for THC requires three structural parameters of the Hamiltonian:
#
# 1. Number of spatial orbitals (:math:`N`).
# 2. Tensor rank of the THC factorization (:math:`M`).
# 3. One-norm of the Hamiltonian (:math:`\lambda`).
#
# As a concrete example, we perform resource estimation for the FeMoco molecule, which plays a crucial role in biological nitrogen fixation.
# We utilize the THC representation of its Hamiltonian, with parameters obtained from the literature [#lee2020]_. Let's start by creating the
# compact Hamiltonian representations for the two active space sizes considered in the reference:

from pennylane import estimator as qre

femoco_54 = qre.THCHamiltonian(num_orbitals=54, tensor_rank=350, one_norm=306.3)
femoco_76 = qre.THCHamiltonian(num_orbitals=76, tensor_rank=450, one_norm=1201.5)

#######################################################################
# Next let's determine the precision with which we want to simulate these systems, and how it translates to the circuit parameters.
#
# Defining the error budget
# ^^^^^^^^^^^^^^^^^^^^^^^^^
# Based on the desired accuracy of the energy estimation, we can derive the circuit parameters needed for Qubitized QPE.
# The total error budget is distributed among several components as follows:
#
# 1. :math:`\epsilon_{QPE}`: Error from the Quantum Phase Estimation algorithm itself.
# 2. :math:`\epsilon_{THC}`: The approximation of tensor hypercontraction.
# 3. :math:`\epsilon_{coeff}`: Error from approximating the Hamiltonian coefficients as part of the Prepare routine.
# 4. :math:`\epsilon_{angle}`: Error from approximating the individual Givens rotations needed for basis rotation in the Select routine.
#
# Since :math:`\epsilon_{THC}` is determined at the time of Hamiltonian construction, we focus on the other three errors here.
# We will take :math:`\epsilon_{QPE} \leq 0.001` Ha, same as the reference [#lee2020]_. And using the bounds derived in
# `Lee et al. (2021) <https://journals.aps.org/prxquantum/pdf/10.1103/PRXQuantum.2.030305>`__ (Appendix C),
# we calculate the number of bits required for coefficient loading (:math:`n_{coeff}`) and rotation angles (:math:`n_{angle}`) as:
#
# .. math::
#    n_{coeff} = \left\lceil 2.5 + \log_2\left(\frac{10 \lambda}{\epsilon_{QPE}}\right) \right\rceil
#
# .. math::
#    n_{angle} = \left\lceil 5.652 + \log_2\left(\frac{20 \lambda N}{\epsilon_{QPE}}\right) \right\rceil
#
# where :math:`\lambda` is the one-norm of the Hamiltonian and :math:`N` is the number of spatial orbitals.
#
# Estimating Walk Operator Cost
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# With these parameters in hand, we can instantiate the Walk operators for both FeMoco Hamiltonians using the
# :class:`~.pennylane.templates.QubitizeTHC` template:

import numpy as np

epsilon_qpe = 0.001  # Ha
walk_operators = []

for hamiltonian in [femoco_54, femoco_76]:

    # Calculate number of bits for coefficient approximation
    n_coeff = int(np.ceil(2.5 + np.log2(10 * hamiltonian.one_norm / epsilon_qpe)))

    # Calculate number of bits for rotation angle approximation
    n_angle = int(
        np.ceil(5.652 + np.log2(20 * hamiltonian.one_norm * hamiltonian.num_orbitals / epsilon_qpe))
    )

    wo_femoco = qre.QubitizeTHC(hamiltonian, coeff_precision=n_coeff, rotation_precision=n_angle)

    walk_operators.append(wo_femoco)

######################################################################
# We can estimate the resources required to implement the walk operator itself as follows:

for wo in walk_operators:
    walk_cost = qre.estimate(wo)
    print(
        f"Resources for implementing the walk operator for FeMoco({wo.thc_ham.num_orbitals}): \n {walk_cost}\n"
    )

######################################################################
# Estimating Qubitized QPE Cost
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# To estimate the resources required for QPE, we can use this walk operator
# as the unitary input to the :class:`~.pennylane.estimator.QPE` class. Along with the
# unitary, we need estimation wires, the number of which is determined by the target QPE
# precision and the Hamiltonian's 1-norm:
#
# .. math::
#     n_{est} = \lceil \log_2\left(\frac{2\pi \lambda}{\epsilon_{QPE}}\right) \rceil
#
# The circuit for QPE will thus look like this:


def qpe_circuit(unitary, estimation_wires):
    qre.QPE(unitary, num_estimation_wires=estimation_wires)


######################################################################
# Note that the QubitizeTHC template doesn't include the cost of preparation of
# the phase gradient state. This is an auxiliary
# resource state used to implement the rotation gates in the `Select` oracle. The
# cost of this state depends on the rotation precision (:math:`n_{angle}`) and can be
# estimated using the :class:`~.pennylane.estimator.PhaseGradientState` resource operator.
# We explicitly estimate this overhead and add it to the final cost of QPE circuit.
#
# Let's estimate the total resources for Qubitized QPE for both FeMoco Hamiltonians:

for wo in walk_operators:

    # Calculate number of estimation wires
    n_est = int(np.ceil(np.log2(2 * np.pi * wo.thc_ham.one_norm / epsilon_qpe)))

    # Estimate Phase Gradient State cost
    phase_grad_cost = qre.estimate(qre.PhaseGradient(wo.rotation_precision))

    # Estimate QPE cost
    qpe_cost = qre.estimate(qpe_circuit)(wo, n_est)

    # Qubitized QPE total cost
    total_cost = qpe_cost.add_parallel(phase_grad_cost)

    print(f"Resources for Qubitized QPE for FeMoco({wo.thc_ham.num_orbitals}): \n {total_cost}\n")

######################################################################
# Analyzing the Results
# ^^^^^^^^^^^^^^^^^^^^^
# Let's look at the results we just generated. For FeMoco (76), the resource estimator predicts a requirement
# of 2000 qubits and over 40 trillion (:math:`4 \times 10^{13}`) total gates.
#
# In the fault-tolerant era, logical qubits will be a precious resource. What if our hardware only supports
# 500 logical qubits? Are we unable to simulate this system? Not necessarily. We can actively trade **Space**
# (Qubits) for **Time** (T-gates) by modifying the circuit architecture.
#
# To see how this works in practice, let's switch gears and investigate these trade-offs on a different system:
# **Cytochrome P450**. This molecule is a standard benchmark in the literature and provides a fresh context
# for exploring our architectural knobs.
#
# First, we create the compact Hamiltonian for the :math:`N=58` active space of Cytochrome P450 using the data provided
# in the literature [Goings2022]_. We work with the largest thc rank that the paper uses for the resource estimates i.e 320.

p450 = qre.THCHamiltonian(num_orbitals=58, tensor_rank=320, one_norm=388.9)

######################################################################
# With the target system defined, we can now turn our attention to the specific architectural
# choices that allow us to balance our resource budget.
#
# Exploring Qubit Vs T-gate Trade-offs
# ------------------------------------
#
# In the THC algorithm, this trade-off between qubits and T-gates in the walk operator is governed by several architectural
# "knobs" distributed across the ``Prepare`` and ``Select`` subroutines:
#
# Knob 1: Batched Givens Rotations
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# In the ``Select`` operator, we need to implement a series of Givens rotations to change the basis.
# Naively, this requires a quantum register of size :math:`N \times n_{rotations}` to store all angles simultaneously.
# Here, we can choose to load fewer angles at a time instead of loading all the rotation angles at once. This leads to reduction
# in the register size, thus saving qubits, but necessitates repetition of the QROM subroutine for each batch and hence costs T-gates.
#
# We can illustrate this trade-off by varying the number of angles loaded in each batch. This particular knob
# argument is accessible through the :class:`~.pennylane.estimator.SelectTHC` operator as ``batched_rotations``.
# Let's see how the resources change for P450 as we vary this parameter:

batch_sizes = [1, 10, 20, 30, 40, 50]
qubit_counts = []
tgate_counts = []

n_coeff = int(
    np.ceil(2.5 + np.log2(10 * p450.one_norm / epsilon_qpe))
)  # coefficient precision bits
n_angle = int(
    np.ceil(5.652 + np.log2(20 * p450.one_norm * p450.num_orbitals / epsilon_qpe))
)  # rotation precision bits

for i in batch_sizes:

    select_thc = qre.SelectTHC(p450, rotation_precision=n_angle, batched_rotations=i)
    wo_batched = qre.QubitizeTHC(
        p450,
        select_op=select_thc,
        coeff_precision=n_coeff,
    )
    n_est = int(np.ceil(np.log2(2 * np.pi * p450.one_norm / epsilon_qpe)))

    phase_grad_cost = qre.estimate(qre.PhaseGradient(num_wires=n_angle))
    qpe_cost = qre.estimate(qpe_circuit)(wo_batched, n_est)
    total_cost = qpe_cost.add_parallel(phase_grad_cost)
    qubit_counts.append(
        total_cost.algo_wires + total_cost.zeroed_wires + total_cost.any_state_wires
    )
    tgate_counts.append(total_cost.gate_counts["Toffoli"])

# Plotting the trade-off curve
import matplotlib.pyplot as plt

fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.set_xlabel("Batch Size (Givens Rotations per Step)")
ax1.set_ylabel("Logical Qubits (Space)")
ax1.plot(batch_sizes, qubit_counts, marker="o", linestyle="-", linewidth=1.5)
ax1.tick_params(axis="y")
ax1.set_xscale("log")
ax1.grid(True, which="both", linestyle="--", alpha=0.5)

ax2 = ax1.twinx()
ax2.set_ylabel("T-Count (Time)")
ax2.plot(batch_sizes, tgate_counts, marker="s", linestyle="--", linewidth=1.5)
ax2.tick_params(axis="y")
ax2.set_yscale("log")

plt.title("Cytochrome P450: Batching Trade-off", fontsize=14)
plt.tight_layout()
plt.legend()
plt.show()

#################################################################################
# Knob-2: QROM SelectSwap:
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The second major optimization strategy is through QROM variants.
# In both :class:`~pennylane.estimator.PrepareTHC` and :class:`~.pennylane.estimator.SelectTHC` oracles,
# we use the `Select-Swap variant of QROM <https://pennylane.ai/qml/demos/tutorial_intro_qrom>`__
# to load the Hamiltonian coefficients. When we don't provide any
# specific optimization parameters, the default strategy is used, which minimizes T-gates at the cost of more qubits.
# Let's see how the cost changes for the most qubit efficient circuit found above, when we use the plain QROM instead of Select-Swap QROM.
# This can be achieved by setting the ``select_swap_depth`` argument to ``1`` in both the ``PrepareTHC`` and ``SelectTHC`` operators.

select_thc_qrom = qre.SelectTHC(
    p450, rotation_precision=n_angle, batched_rotations=57, select_swap_depth=1
)
prepare_thc_qrom = qre.PrepTHC(p450, coeff_precision=n_coeff, select_swap_depth=1)
wo_qrom = qre.QubitizeTHC(
    p450,
    select_op=select_thc_qrom,
    prepare_op=prepare_thc_qrom,
)
n_est = int(np.ceil(np.log2(2 * np.pi * p450.one_norm / epsilon_qpe)))
qpe_cost = qre.estimate(qpe_circuit)(wo_qrom, n_est)
total_cost = qpe_cost.add_parallel(phase_grad_cost)

print(f"Resources for Qubitized QPE for P450: \n {total_cost}\n")

######################################################################
# Conclusion
# ----------
#
# In this demo, we tackled the logical resource estimation for two of the most important systems in chemistry:
# **FeMoco** and **Cytochrome P450**. Our initial baseline for FeMoco revealed a requirement of ~2000 logical qubits, which
# underscores the magnitude of the challenge facing early fault-tolerant hardware.
#
# However, naive calculations tell only half the story. As we demonstrated with Cytochrome P450, these resource counts are not
# immutable constants. By actively navigating the architectural trade-offs between logical qubits
# and T-gates we can significantly reshape the cost profile of the algorithm.
#
# This is where the flexibility of PennyLane's resource estimation framework becomes crucial. Rather than treating subroutines
# like ``Prepare`` and ``Select`` as black boxes, PennyLane allows us to tune the internal circuit
# configurations. This transforms resource estimation from a passive reporting tool into an active design process, enabling
# researchers to optimize their algorithm implementation even before the hardware is available.
#
#
# References
# ----------
#
# .. [#lee2020]
#
#     Joonho Lee et al.
#     "Even More Efficient Quantum Computations of Chemistry Through Tensor Hypercontraction."
#     `PRX Quantum 2, 030305 (2021). <https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.2.030305>`__
#
# .. [#Goings2022]
#
#     Joshua J Goings et al.
#     "Reliably assessing the electronic structure of cytochrome P450 on today’s classical computers and tomorrow’s quantum computers"
#     `Proceedings of the National Academy of Sciences 119.38 (2022). <https://www.pnas.org/doi/abs/10.1073/pnas.2203533119>`__
#
