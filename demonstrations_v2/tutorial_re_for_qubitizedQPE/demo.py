r"""Qubit and T-gate Trade-offs in Qubitized Quantum Phase Estimation
===============================================================================

`Quantum Phase Estimation (QPE) <https://pennylane.ai/qml/demos/tutorial_qpe>`_ relies on unitary evolution,
yet chemical Hamiltonians :math:`\hat{H}` are Hermitian.
`Qubitization <https://pennylane.ai/qml/demos/tutorial_qubitization>`_ bridges this gap via
`Block Encoding <https://pennylane.ai/qml/demos/tutorial_lcu_blockencoding>`_, embedding :math:`\hat{H}` into a larger
"Quantum Walk" unitary :math:`W`. To construct this block encoding, we decompose the Hamiltonian into a Linear Combination of Unitaries (LCU).
In this demo, we use the **Tensor Hypercontraction (THC)** representation, a state of the art LCU decomposition for quantum chemistry
that approximates the interaction tensor via a low-rank factorization.

In order to get an understanding if this algorithm will run on first generation FTQC with X qubits,
we need to know what the resources look like.
In this demo, we show how to get concrete costs using logical resource estimation tools in PennyLane and find a version that will fit on FTQC hardware.

.. figure:: ../_static/demo_thumbnails/opengraph_demo_thumbnails/OGthumbnail_how_to_build_spin_hamiltonians.png
    :align: center

    :width: 70%
    :target: javascript:void(0)
"""

######################################################################
# To implement this decomposition on a quantum computer, the Walk operator is constructed from two primary subroutines:
# the ``Prepare`` oracle, which prepares the state, whose amplitudes encode the Hamiltonian coefficients, and the
# ``Select`` oracle, which applies the Hamiltonian terms controlled by that state. The implementation of these subroutines
# offers some architectural flexibility to go with higher qubits or T-gates. These architectural knobs can be defined as:
#
# Knob 1: Batched Givens Rotations
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# In the ``Select`` operator, we need to implement a series of Givens rotations to change the basis.
# Naively, this requires a quantum register of size :math:`N \times \beth` to store all angles simultaneously, where
# ``N`` is the number of rotations, and :math:`\beth` is the rotation precision.
# Here, we can choose to load these angles in batches instead of loading all the rotation angles at once. This leads to reduction
# in the register size, thus saving qubits, but necessitates repetition of the QROM(Quantum Read-Only Memory) subroutine for each batch
# and hence costs T-gates.
#
# .. figure:: ../_static/demonstration_assets/qubitization/batching.jpeg
#    :align: center
#    :width: 70%
#    :target: javascript:void(0)
#
# Knob-2: QROM SelectSwap:
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The second major optimization strategy is through `QROM <https://pennylane.ai/qml/demos/tutorial_intro_qrom>`_. Crucially, both ``Prepare`` and ``Select``
# rely on QROM to access Hamiltonian coefficients and rotation angles respectively. We can use
# the select-swap variant of the QROM, which allows us to trade the depth of the circuit for width as shown in the circuit diagrams.
#
# .. list-table::
#    :widths: 50 50
#    :header-rows: 0
#    :class: borderless
#
#    * - .. image:: ../_static/demonstration_assets/qrom/select_swap.jpeg
#           :width: 100%
#      - .. image:: ../_static/demonstration_assets/qrom/select_swap_4.jpeg
#           :width: 100%
#
# Standard resource estimates often treat these oracles as fixed "black boxes", yielding a single cost number.
# However, this demo is more than just a static cost report. We demonstrate how PennyLane exposes these
# tunable knobs of the circuit implementation, allowing us to actively navigate the circuit design and trade off between T-gates and
# logical qubits to suit different constraints. As a concrete example, let's perform resource estimation for the FeMoco molecule,
# which plays a crucial role in biological nitrogen fixation.
#
# Resource Estimation for FeMoco
# ------------------------------
# Estimating resources for large-scale chemical systems is often bottlenecked by the challenge of constructing and storing the full Hamiltonian tensor.
# But why carry the entire building when a blueprint will do? The resource estimator allows us to sidestep this bottleneck for a quick estimation.
# By using a `compact representation <https://docs.pennylane.ai/en/stable/code/api/pennylane.estimator.compact_hamiltonian.THCHamiltonian.html>`__ of the THC
# Hamiltonian, we capture only the essential structural parameters for the Hamiltonian: the number of spatial orbitals (:math:`N`),
# the THC factorization rank (:math:`M`), and the Hamiltonian one-norm (:math:`\lambda`).
#
# Let's initialize the THC representation of the FeMoco Hamiltonian with a 76 orbital active space, with parameters obtained from the literature [#lee2020]_:

from pennylane import estimator as qre

femoco = qre.THCHamiltonian(num_orbitals=76, tensor_rank=450, one_norm=1201.5)

#######################################################################
# Next we need to determine the precision with which we want to simulate these systems, and how it translates to the circuit parameters.
#
# Defining the error budget
# ^^^^^^^^^^^^^^^^^^^^^^^^^
# We begin by fixing the target accuracy for the Quantum Phase Estimation (QPE) routine to :math:`\epsilon_{QPE} = 0.001` Hartree,
# which dictates the total number of QPE iterations required:
#
# .. math::
#     n_{iter} = \lceil \log_2\left(\frac{2\pi \lambda}{\epsilon_{QPE}}\right) \rceil
#
# This choice also dictates the required bit-precision for the circuit's subroutines. Specifically, to maintain this
# overall accuracy, we must quantize the Hamiltonian coefficients in ``Prepare`` and the rotation angles in ``Select``
# with sufficient precision.
#
# Using the error bounds derived in `Lee et al. (2021) <https://journals.aps.org/prxquantum/pdf/10.1103/PRXQuantum.2.030305>`__ (Appendix C),
# we calculate the required number of bits for loading coefficients (:math:`n_{coeff}`) and rotation angles (:math:`n_{angle}`) as:
#
# .. math::
#    n_{coeff} = \left\lceil 2.5 + \log_2\left(\frac{10 \lambda}{\epsilon_{QPE}}\right) \right\rceil, \quad
#    n_{angle} = \left\lceil 5.652 + \log_2\left(\frac{20 \lambda N}{\epsilon_{QPE}}\right) \right\rceil
#
# Estimating Qubitized QPE Cost
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# With these parameters in hand, we can esimate the total resources. The full algorithm consists of the Walk Operator,
# constructed via :class:`~.pennylane.templates.templates.QubitizeTHC` running within a QPE routine.
#
# We must note that :class:`~.pennylane.estimator.templates.SelectTHC` oracle implementation is based on the description in
# von Burg et al. [vonburg]_. This work uses the phase gradient technique to implement Givens rotations, and thus requires an
# auxiliary resource state for addition of phase. The ``SelectTHC`` template doesn't include the cost of preparation of this
# phase gradient state, so we must explicitly estimate this overhead and add it to the final cost of QPE circuit.
#
# Let's estimate the total resources for Qubitized QPE for FeMoco:

import numpy as np

epsilon_qpe = 0.001  # Ha
n_iter = int(np.ceil(np.log2(2 * np.pi * femoco.one_norm / epsilon_qpe))) # QPE iterations
n_coeff = int(np.ceil(2.5 + np.log2(10 * femoco.one_norm / epsilon_qpe)))
n_angle = int(np.ceil(5.652 + np.log2(20 * femoco.one_norm * femoco.num_orbitals / epsilon_qpe)))

wo_femoco = qre.QubitizeTHC(femoco, coeff_precision=n_coeff, rotation_precision=n_angle)

phase_grad_cost = qre.estimate(qre.PhaseGradient(n_angle))

gate_set = {"T", "CNOT", "X", "Y", "Z", "S", "Hadamard"}
qpe_cost = qre.estimate(qre.QPE(wo_femoco, num_estimation_wires=n_iter), gate_set=gate_set)

total_cost = qpe_cost.add_parallel(phase_grad_cost)
print(f"Resources for Qubitized QPE for FeMoco(76): \n {total_cost}\n")

######################################################################
# Analyzing the Results
# ^^^^^^^^^^^^^^^^^^^^^
# Let's look at the results we just generated. For FeMoco (76), the resource estimator predicts a requirement
# of 2000 qubits and over 40 trillion (:math:`4 \times 10^{13}`) total gates.
#
# In the fault-tolerant era, logical qubits will be a precious resource. What if our hardware only supports
# 500 logical qubits? Are we unable to simulate this system? Not necessarily. We can actively trade **Space**
# (Qubits) for **Time** (T-gates) by modifying the circuit architecture. Let's apply the "tunable knobs" we discussed
# earlier to fit FeMoco onto this constrained device.
#
# With the target system defined, we can now turn our attention to the specific architectural
# choices that allow us to balance our resource budget.
#
# Exploring Trade-offs
# --------------------
#
# Let's first explore the impact of **Batched Givens Rotations**, by varying the number of rotation angles loaded
# simultaneously. This particular argument is accessible through the :class:`~.pennylane.estimator.SelectTHC` operator as
# ``batched_rotations``. Let's see how the resources change for FeMoco as we vary this parameter:

batch_sizes = [1, 10, 20, 30, 40, 50]
qubit_counts = []
tgate_counts = []

for i in batch_sizes:

    select_thc = qre.SelectTHC(femoco, rotation_precision=n_angle, batched_rotations=i)
    wo_batched = qre.QubitizeTHC(
        femoco,
        select_op=select_thc,
        coeff_precision=n_coeff,
    )

    qpe_cost = qre.estimate(qre.QPE(wo_batched, n_iter), gate_set=gate_set)
    total_cost = qpe_cost.add_parallel(phase_grad_cost)
    qubit_counts.append(
        total_cost.algo_wires + total_cost.zeroed_wires + total_cost.any_state_wires
    )
    tgate_counts.append(total_cost.gate_counts["T"])

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

plt.title("FeMoco(76): Batching Trade-off", fontsize=14)
plt.tight_layout()
plt.legend()
plt.show()

#################################################################################
# Now, if we need to control the resources even further, we can combine batching with our second strategy: **Select-Swap QROM**.
# Let's see how the resources change if we set the select_swap_depth to 1, for both :class:`~.pennylane.estimator.PrepTHC`,
# and :class:`~.pennylane.estimator.SelectTHC` operators:

select_thc_qrom = qre.SelectTHC(
    femoco, rotation_precision=n_angle, batched_rotations=38, select_swap_depth=1
)
prepare_thc_qrom = qre.PrepTHC(femoco, coeff_precision=n_coeff, select_swap_depth=1)
wo_qrom = qre.QubitizeTHC(
    femoco,
    select_op=select_thc_qrom,
    prepare_op=prepare_thc_qrom,
)
qpe_cost = qre.estimate(qre.QPE(wo_qrom, n_iter), gate_set)
total_cost = qpe_cost.add_parallel(phase_grad_cost)

print(f"Resources for Qubitized QPE for FeMoco(76): \n {total_cost}\n")

######################################################################
# Conclusion
# ----------
#
# In this demo, we tackled the logical resource estimation for FeMoco, a complex molecule central to understanding
# biological nitrogen fixation. Our initial baseline for FeMoco revealed a requirement of ~2000 logical qubits, which
# underscores the magnitude of the challenge facing early fault-tolerant hardware.
#
# However, naive calculations tell only half the story. As we demonstrated later, these resource counts are not
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
