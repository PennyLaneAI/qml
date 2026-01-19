r"""Qubit and gate trade-offs in Qubitized Quantum Phase Estimation
======================================================================


`Quantum Phase Estimation (QPE) <https://pennylane.ai/qml/demos/tutorial_qpe>`_ is a powerful quantum algorithm
that allows us to estimate the eigenvalues of a Hamiltonian with high precision.
The most advanced versions of QPE rely on
`Qubitization <https://pennylane.ai/qml/demos/tutorial_qubitization>`_  to encode chemical Hamiltonians :math:`H` as unitary operators. This leverages a `Linear Combination of Unitaries (LCU) <https://pennylane.ai/qml/demos/tutorial_lcu_blockencoding>`_ decomposition to create a
`Block Encoding <https://pennylane.ai/qml/demos/tutorial_lcu_blockencoding>`_ of :math:`H` , which is then used to construct a "Quantum Walk" operator :math:`W` that is used as input to QPE.

In this demo, we use the **Tensor Hypercontraction (THC)** representation, a state-of-the-art LCU decomposition for quantum chemistry
that approximates the interaction tensor via a low-rank factorization.

**But is implementing this quantum algorithm feasible on early fault-tolerant hardware?**
To answer this, we must move beyond asymptotic scaling and determine the concrete resource requirements.
In this demo, we use PennyLane's logical resource :mod:`estimator <pennylane.estimator>`
to calculate the precise costs and demonstrate how to optimize the algorithm to fit on constrained devices with
a few hundred logical qubits.

.. figure:: ../_static/demo_thumbnails/opengraph_demo_thumbnails/OGthumbnail_how_to_build_spin_hamiltonians.png
    :align: center
    :width: 70%
    :target: javascript:void(0)
"""

######################################################################
# The key to this optimization lies in the specific method to build the Walk operator,
# which is constructed from two primary subroutines:
# the ``Prepare`` oracle, which prepares a state encoding the Hamiltonian coefficients, and the
# ``Select`` oracle, which applies the Hamiltonian terms controlled by that state. The implementation of these subroutines
# offers the flexibility to trade off qubits for gates, and vice versa. Specifically, we can tune two algorithmic knobs to perform this trade-off:
#
# Knob-1: Batched Givens Rotations
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# In the ``Select`` operator, we need to implement a series of Givens rotations to change the basis.
# Naively, to store all angles simultaneously, we require a register size defined by the number of rotations times the bits of precision per angle.
# Here, we can choose to load these angles in batches instead of loading all of them at once.
# The tunable knob here is the **number of batches** in which the rotation angles are loaded. By increasing the number of batches,
# we save the qubits by reducing the register size, but need a longer repetition of the `Quantum Read-Only Memory (QROM) <https://pennylane.ai/qml/demos/tutorial_intro_qrom>`_
# subroutine for each batch, which increases the Toffoli count.
#
# .. figure:: ../_static/demonstration_assets/qubitization/batching.jpeg
#    :align: center
#    :width: 70%
#    :target: javascript:void(0)
#
# Knob-2: QROM SelectSwap:
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The second major optimization strategy is through `QROM <https://pennylane.ai/qml/demos/tutorial_intro_qrom>`_ itself. Crucially, both ``Prepare`` and ``Select``
# rely on QROM to access Hamiltonian coefficients and rotation angles respectively. We can use
# the select-swap variant of the QROM, which allows us to trade the depth of the circuit for width, as shown in the diagrams below:
#
# .. list-table::
#    :widths: 50 50
#    :header-rows: 0
#    :class: plain
#
#    * - .. image:: ../_static/demonstration_assets/qrom/select_swap.jpeg
#           :width: 100%
#      - .. image:: ../_static/demonstration_assets/qrom/select_swap_4.jpeg
#           :width: 100%
#
# The configuration on the right achieves lower gate complexity by employing auxiliary work wires to enable block-wise data loading.
# This approach replaces expensive multi-controlled operations with simpler controlled-swap gates, significantly reducing the Toffoli
# count while requiring additional qubits.
#
# Standard resource estimates often treat these oracles as fixed "black boxes", yielding a single cost value.
# However, our quantum resource :mod:`estimator <pennylane.estimator>` provides much more than a static cost report.
# We demonstrate how PennyLane exposes these
# tunable knobs of the circuit implementation, allowing us to actively navigate the circuit design and trade-off between gates and
# logical qubits to suit different constraints. As a concrete example, let's perform resource estimation for the FeMoco molecule,
# which plays a crucial role in biological nitrogen fixation.
#
# Resource Estimation for FeMoco
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Estimating resources for large-scale chemical systems is often bottlenecked by the challenge of constructing and storing the full Hamiltonian tensor.
# PennyLane's resource estimator allows us to sidestep this bottleneck for a quick estimation.
# By using a `compact representation <https://docs.pennylane.ai/en/stable/code/api/pennylane.estimator.compact_hamiltonian.THCHamiltonian.html>`__ of the THC
# Hamiltonian, we capture only the essential structural parameters for the Hamiltonian: the number of spatial orbitals (:math:`N`),
# the THC factorization rank (:math:`M`), and the Hamiltonian one-norm (:math:`\lambda`).
#
# While calculating the exact one-norm typically requires constructing the Hamiltonian, this compact form is particularly
# useful for well-known benchmarks where these values are already reported in the literature. Furthermore, it allows
# us to rapidly generate quick estimates for different ranges of one-norms, enabling sensitivity analysis without
# needing to build the full operator for every case.
#
# Let's initialize the THC representation of the FeMoco Hamiltonian with a 76-orbital active space, with parameters obtained from the literature [#lee2021]_:

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
#     n_{iter} = \lceil \log_2\left(\frac{2\pi \lambda}{\epsilon_{QPE}}\right) \rceil.
#
# This choice also dictates the required bit-precision for the circuit's subroutines. Specifically, to maintain this
# overall accuracy, we must fix the numerical precision for expressing the Hamiltonian coefficients in ``Prepare``
# and the rotation angles in ``Select``.
#
# Using the error bounds derived in `Lee et al. (2021) <https://journals.aps.org/prxquantum/pdf/10.1103/PRXQuantum.2.030305>`__ (Appendix C),
# you can calculate the required number of bits for loading coefficients (:math:`n_{coeff}`) and rotation angles (:math:`n_{angle}`) as:
#
# .. math::
#    n_{coeff} = \left\lceil 2.5 + \log_2\left(\frac{10 \lambda}{\epsilon_{QPE}}\right) \right\rceil, \quad
#    n_{angle} = \left\lceil 5.652 + \log_2\left(\frac{20 \lambda N}{\epsilon_{QPE}}\right) \right\rceil.
#
# Since we are following the analysis in Lee et al. (2021), we use the same constants as the reference

import numpy as np

epsilon_qpe = 0.0016  # Ha
n_iter = int(np.ceil(2 * np.pi * femoco.one_norm / epsilon_qpe))  # QPE iterations
n_coeff = 10
n_angle = 20

########################################################################
# Estimating Qubitized QPE Cost
# -----------------------------
# With these parameters in hand, we can esimate the total resources. The full algorithm consists of the Walk Operator,
# constructed via :class:`~.pennylane.templates.templates.QubitizeTHC`, running within a QPE routine.
#
# We  note that :class:`~.pennylane.estimator.templates.SelectTHC` oracle implementation is based on the description in
# von Burg et al. [#vonburg]_. This work uses the phase gradient technique to implement Givens rotations, and thus requires an
# auxiliary resource state for addition of phase. The ``SelectTHC`` template doesn't include the cost of preparation of this
# phase gradient state, so we must explicitly estimate this overhead and add it to the final cost of the QPE circuit.
#
# Let's estimate the total resources for Qubitized QPE for FeMoco:

wo_femoco = qre.QubitizeTHC(femoco, coeff_precision=n_coeff, rotation_precision=n_angle)

phase_grad_cost = qre.estimate(qre.PhaseGradient(n_angle))

qpe_cost = qre.estimate(qre.UnaryIterationQPE(wo_femoco, num_iterations=n_iter))

total_cost = qpe_cost.add_parallel(phase_grad_cost)
print(f"Resources for Qubitized QPE for FeMoco(76): \n {total_cost}\n")

######################################################################
# Analyzing the Results
# ---------------------
# Let's look at the results we just generated. For FeMoco (76), the resource estimator predicts a requirement
# of over 2000 qubits and 11 trillion (:math:`11.5 \times 10^{12}`) total gates.
#
# In the fault-tolerant era, logical qubits will be a precious resource. What if our hardware only supports
# 500 logical qubits? Are we unable to simulate this system? Not necessarily. We can actively trade **Space**
# (Qubits) for **Time** (gates) by modifying the circuit architecture. Let's apply the "tunable knobs" we discussed
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
# are loaded.
#
# .. note::
#    To strictly isolate the effect of batching, we fix the ``select_swap_depth`` to 1 here.
#    While this does not represent the optimal gate count, it allows us to
#    observe the pure trade-off between batch size and qubit count without confounding factors.
#
# This particular argument is accessible through the :class:`~.pennylane.estimator.SelectTHC` operator as
# ``num_batches``. Let's see how the resources change for FeMoco as we vary this parameter:

batch_sizes = [1, 2, 3, 5, 10, 75]
qubit_counts = []
toffoli_counts = []

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

    qpe_cost = qre.estimate(qre.UnaryIterationQPE(wo_batched, n_iter))
    total_cost = qpe_cost.add_parallel(phase_grad_cost)
    qubit_counts.append(total_cost.total_wires)
    toffoli_counts.append(total_cost.gate_counts["Toffoli"])


######################################################################
# Let's visualize the results by plotting the qubit and toffoli counts against the batch size:
#
# .. figure:: ../_static/demonstration_assets/qubitization_re/batching_tradeoff.jpeg
#    :align: center
#    :width: 85%
#    :target: javascript:void(0)
#
# The plot illustrates a clear crossover in resource requirements. At the left extreme (a single batch),
# we minimize toffolis but pay a massive penalty in qubits, requiring over 2000 logical qubits, which far exceeds
# our hypothetical 500-qubit limit.
# As we increase the number of batches, the qubit count plummets, eventually dipping below
# our 500-qubit limit. However, there is no free lunch: the toffoli count rises steadily because we must
# repeat the QROM readout for every additional batch. To verify the feasibility, let's print the
# concrete numbers for the two extremes:
#
print("Resource counts with batch size: 1")
print(f"  Qubits: {qubit_counts[0]}")
print(f"  Toffolis: {toffoli_counts[0]:.3e}\n")
print("Resource counts with batch size: 75")
print(f"  Qubits: {qubit_counts[-1]}")
print(f"  Toffolis: {toffoli_counts[-1]:.3e}\n")

######################################################################
#
# Crucially, while the qubit requirements drop by nearly a factor of 5, the Toffoli count stays in the same order of magnitude.
# This favorable trade-off allows us to fit the algorithm on constrained hardware without
# making the runtime prohibitively long.
#
# We have successfully brought the qubit count down using batching. Now, can we optimize the gate count
# without incurring extra qubit costs?
#
# Step 2: Circuit optimization with Select-Swap
# -----------------------------------------------
# We have successfully brought the qubit count down using batching. Now, can we optimize the gate count
# without incurring extra qubit costs?
# To do this, we use the **Select-Swap QROM** strategy. Normally, this involves trading qubits for toffolis.
# But here is the trick: the register used to store rotation angles in the :class:`~.pennylane.resource.SelectTHC`
# operator is idle during the Prepare step. We can reuse these idle qubits to implement the
# ``QROM`` for the :class:`~.pennylane.resource.PrepareTHC` operator.
# This should allow us to decrease the toffolis without increasing the logical
# qubit count, at least until we run out of reusable space.
#
# Let's verify this by sweeping through different ``select_swap_depth`` values:

swap_depths = [1, 2, 4, 8, 16]
qubit_counts = []
toffoli_counts = []

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

    qpe_cost = qre.estimate(qre.UnaryIterationQPE(wo_qrom, n_iter))
    total_cost = qpe_cost.add_parallel(phase_grad_cost)

    qubit_counts.append(total_cost.total_wires)
    toffoli_counts.append(total_cost.gate_counts["Toffoli"])

######################################################################
#
# .. figure:: ../_static/demonstration_assets/qubitization_re/qrom_selswap.jpeg
#    :align: center
#    :width: 90%
#    :target: javascript:void(0)
#
# The plot confirms our intuition. For depths 1, 2, and 4, the logical qubit count stays exactly the same, while the
# Toffoli count decreases. However, moving to depth 8, the qubit count jumps as the swap network becomes too large to fit
# entirely within the reused register, forcing the allocation of additional qubits. This marks
# the point where the "free" optimization ends and the standard trade-off resumes.
# Let's look at the exact resources for the optimized configuration (Batch Size = 10, Select-Swap Depth = 4):

print(f"Optimized Configuration (Batch=10, Depth=4):")
print(f"Logical Qubits: {qubit_counts[2]}")
print(f"Toffoli Gates:  {toffoli_counts[2]:.2e}")

#######################################################################
# By applying these optimizations, we have successfully reduced the qubit requirements by a factor of 4,
# bringing the count down from ~2200 to 466. Crucially, this massive spatial saving comes with a relatively
# manageable cost: the Toffoli gate count increases from ~8.8e10 to ~3.5e11, which is also roughly a factor of 4.
#
# Conclusion
# ^^^^^^^^^^
#
# In this demo, we tackled the logical resource estimation for FeMoco, a complex molecule central to understanding
# biological nitrogen fixation. Our initial baseline for FeMoco revealed a requirement of ~2000 logical qubits, which
# underscores the magnitude of the challenge facing early fault-tolerant hardware.
#
# However, naive calculations tell only half the story. As we demonstrated later, these resource counts are not
# immutable constants. By actively navigating the architectural trade-offs between logical qubits
# and Toffolis we can significantly reshape the cost profile of the algorithm.
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
# .. [#lee2021]
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
