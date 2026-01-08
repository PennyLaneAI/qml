r"""Qubit and T-gate Trade-offs in Qubitized Quantum Phase Estimation
======================================================================

Unlocking the full potential of quantum computing for chemistry, from designing better battery materials to discovering
new drugs, requires simulating the quantum dynamics of molecular systems.
To achieve this, we rely on `Quantum Phase Estimation (QPE) <https://pennylane.ai/qml/demos/tutorial_qpe>`_,
which allows us to estimate the energy eigenstates of a Hamiltonian with high precision.
QPE, however, relies on unitary evolution, and chemical Hamiltonians :math:`\hat{H}` are Hermitian.
`Qubitization <https://pennylane.ai/qml/demos/tutorial_qubitization>`_ bridges this gap via
`Block Encoding <https://pennylane.ai/qml/demos/tutorial_lcu_blockencoding>`_, embedding :math:`\hat{H}` into a larger
"Quantum Walk" unitary :math:`W`. To construct this block encoding, we decompose the Hamiltonian into a Linear Combination of Unitaries (LCU).
In this demo, we use the **Tensor Hypercontraction (THC)** representation, a state of the art LCU decomposition for quantum chemistry
that approximates the interaction tensor via a low-rank factorization.

**But is this feasible on early fault-tolerant hardware?**
To answer this, we must move beyond asymptotic scaling and determine the concrete resource requirements.
In this demo, we use PennyLane's logical resource estimation tools to calculate the precise costs and
demonstrate how to optimize the algorithm to fit on constrained devices.

.. figure:: ../_static/demo_thumbnails/opengraph_demo_thumbnails/OGthumbnail_how_to_build_spin_hamiltonians.png
    :align: center
    :width: 70%
    :target: javascript:void(0)
"""

######################################################################
# The key to this optimization lies in the specific circuit architecture used to build the Quantum Walk.
# To implement this decomposition on a quantum computer, the Walk operator is constructed from two primary subroutines:
# the ``Prepare`` oracle, which prepares the state, whose amplitudes encode the Hamiltonian coefficients, and the
# ``Select`` oracle, which applies the Hamiltonian terms controlled by that state. The implementation of these subroutines
# offers some architectural flexibility to go with higher qubits or T-gates. Specifically, we can tune two architectural knobs:
#
# Knob 1: Batched Givens Rotations
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# In the ``Select`` operator, we need to implement a series of Givens rotations to change the basis.
# Naively, this requires a quantum register of size :math:`N \times \beth` to store all angles simultaneously, where
# ``N`` is the number of rotations, and :math:`\beth` is the rotation precision.
# Here, we can choose to load these angles in batches instead of loading all the rotation angles at once.
# The tunable knob here is the **number of batches** in which the rotation angles are loaded. By increasing the number of batches,
# we save the qubits by reducing the register size, but necessitate repetition of the QROM(Quantum Read-Only Memory) subroutine for each batch
# and hence increase T-gates.
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
#    :class: plain
#
#    * - .. image:: ../_static/demonstration_assets/qrom/select_swap.jpeg
#           :width: 90%
#      - .. image:: ../_static/demonstration_assets/qrom/select_swap_4.jpeg
#           :width: 90%
#
# Standard resource estimates often treat these oracles as fixed "black boxes", yielding a single cost number.
# However, this demo is more than just a static cost report. We demonstrate how PennyLane exposes these
# tunable knobs of the circuit implementation, allowing us to actively navigate the circuit design and trade off between T-gates and
# logical qubits to suit different constraints. As a concrete example, let's perform resource estimation for the FeMoco molecule,
# which plays a crucial role in biological nitrogen fixation.
#
# Resource Estimation for FeMoco
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
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
# -------------------------
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
# -----------------------------
# With these parameters in hand, we can esimate the total resources. The full algorithm consists of the Walk Operator,
# constructed via :class:`~.pennylane.templates.templates.QubitizeTHC` running within a QPE routine.
#
# We must note that :class:`~.pennylane.estimator.templates.SelectTHC` oracle implementation is based on the description in
# von Burg et al. [#vonburg]_. This work uses the phase gradient technique to implement Givens rotations, and thus requires an
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
# ---------------------
# Let's look at the results we just generated. For FeMoco (76), the resource estimator predicts a requirement
# of over 3000 qubits and 1 trillion (:math:`1.06 \times 10^{12}`) total gates.
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
# ^^^^^^^^^^^^^^^^^^^^
#
# Step 1: Reducing Qubits with Batching
# -------------------------------------
# Let's first explore the impact of **Batched Givens Rotations**, by varying the number of batches in which rotation angles
# loaded. This particular argument is accessible through the :class:`~.pennylane.estimator.SelectTHC` operator as
# ``num_batches``. Let's see how the resources change for FeMoco as we vary this parameter:

batch_sizes = [1, 2, 3, 5, 10, 75]
qubit_counts = []
tgate_counts = []

for i in batch_sizes:
    prep_thc = qre.PrepTHC(femoco, coeff_precision=n_coeff, select_swap_depth=1)
    select_thc = qre.SelectTHC(femoco, rotation_precision=n_angle, num_batches=i)
    wo_batched = qre.QubitizeTHC(
        femoco,
        prep_op=prep_thc,
        select_op=select_thc,
        coeff_precision=n_coeff,
        rotation_precision=n_angle,
    )

    qpe_cost = qre.estimate(qre.QPE(wo_batched, n_iter), gate_set=gate_set)
    total_cost = qpe_cost.add_parallel(phase_grad_cost)
    qubit_counts.append(
        total_cost.algo_wires + total_cost.zeroed_wires + total_cost.any_state_wires
    )
    tgate_counts.append(total_cost.gate_counts["T"])


######################################################################
# Let's visualize the results by plotting the qubit and T-gate counts against the batch size:
#
# .. figure:: ../_static/demonstration_assets/qubitization_re/batching_tradeoff.jpeg
#    :align: center
#    :width: 85%
#    :target: javascript:void(0)
#
# The plot illustrates a clear crossover in resource requirements. At the left extreme (a single batch),
# we minimize T-gates but pay a massive penalty in qubits, requiring over 3000 logical qubits, which far exceeds
# our hypothetical 500-qubit limit.
# As we increase the number of batches, the qubit count plummets, eventually dipping below
# our 500-qubit limit. However, there is no free lunch: the T-gate count rises steadily because we must
# repeat the QROM readout for every additional batch.
# We have successfully brought the qubit count down using batching. Now, can we optimize the gate count
# without incurring extra qubit costs?
#
# Step 2: Finding a "Free Lunch" with Select-Swap
# -----------------------------------------------
# We have successfully brought the qubit count down using batching. Now, can we optimize the gate count
# without incurring extra qubit costs?
# To do this, we use the **Select-Swap QROM** strategy. Normally, this involves trading qubits for T-gates.
# But here is the trick: the register used to store rotation angles in the :class:`~.pennylane.resource.SelectTHC`
# operator is idle during the Prepare step. We can reuse these idle qubits to implement the
# ``QROM`` for the :class:`~.pennylane.resource.PrepareTHC` operator.
# This should allow us to decrease the T-gates without increasing the logical
# qubit count, effectively providing a "free lunch" of optimization, at least until we run out of reusable space.
#
# Let's verify this by sweeping through different ``select_swap_depth`` values:

swap_depths = [1, 2, 4, 8, 16]
qubit_counts = []
t_counts = []

for depth in swap_depths:
    select_thc_qrom = qre.SelectTHC(
        femoco, rotation_precision=n_angle, num_batches=10, select_swap_depth=1
    )
    prepare_thc_qrom = qre.PrepTHC(femoco, coeff_precision=n_coeff, select_swap_depth=depth)

    wo_qrom = qre.QubitizeTHC(
        femoco,
        select_op=select_thc_qrom,
        prep_op=prepare_thc_qrom,
    )

    qpe_cost = qre.estimate(qre.QPE(wo_qrom, n_iter), gate_set=gate_set)
    total_cost = qpe_cost.add_parallel(phase_grad_cost)

    qubit_counts.append(total_cost.algo_wires + total_cost.zeroed_wires + total_cost.any_state_wires)
    t_counts.append(total_cost.gate_counts["T"])

######################################################################
#
# .. figure:: ../_static/demonstration_assets/qubitization_re/qrom_selswap.jpeg
#    :align: center
#    :width: 85%
#    :target: javascript:void(0)
#
# The plot confirms our intuition. For depths 1, 2, and 4, the logical qubit count stays exactly the same, while the
# T-count decreases. However, moving to depth 8, the qubit count jumps as the swap network becomes too large to fit
# entirely within the reused register, forcing the allocation of additional qubits. This marks
# the point where the "free" optimization ends and the standard trade-off resumes.
#
# Conclusion
# ^^^^^^^^^^
#
# In this demo, we tackled the logical resource estimation for FeMoco, a complex molecule central to understanding
# biological nitrogen fixation. Our initial baseline for FeMoco revealed a requirement of ~3000 logical qubits, which
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
# .. [#vonburg]
#
#     Vera von Burg et al.
#     "Quantum computing enhanced computational catalysis"
#     `arXiv:2007.14460 (2020). <https://arxiv.org/abs/2007.14460>`__
#
