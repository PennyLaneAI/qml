r"""Resource Estimation for Spectroscopy Applications
=====================================================
Spectroscopy is a cornerstone of chemistry and physics, providing fundamental insights into the
structure and dynamics of matter. But accurate simulations of excited states
are notoriously expensive, often pushing classical supercomputers to their breaking point.

Can quantum computers do better?

In this demo, we'll find out. We will analyze the scalability of such key spectroscopy algorithms
through PennyLane's resource estimator and calculate their actual resource requirements. By benchmarking
these algorithms now, we can ensure they are ready for the fault-tolerant hardware of the future.

.. figure:: ../_static/demo_thumbnails/opengraph_demo_thumbnails/OGthumbnail_how_to_build_spin_hamiltonians.png
    :align: center
    :width: 70%
    :target: javascript:void(0)

Simulating Spectroscopy on a Quantum Computer
---------------------------------------------

How do we translate a physical spectroscopy experiment into a quantum circuit?

Regardless of the specific technique—whether it is X-ray Absorption (XAS) [#Fomichev2025]_,
Vibrational Spectroscopy [#Laoiza2025]_, or Electron Energy Loss Spectroscopy [#Kunitsa2025]_—the
core goal is the same: calculating the time-domain correlation function, :math:`\tilde{G}(t)`.

To measure this property on a quantum computer, we rely on a standard algorithmic template called the
**Hadamard Test**.

.. figure:: ../_static/demonstration_assets/xas/global_circuit.png
  :alt: Illustration of full Hadamard test circuit with state prep, time evolution and measurement.
  :width: 70%
  :align: center

  Figure 1: *Hadamard Test Circuit*.

The dominant cost for this circuit comes from the controlled time evolution block, the efficiency of which
thus dictates the resource requirements for our algorithms. This cost is dictated by the intersection of
the spectroscopic domain and our implementation strategy.

While the spectroscopic technique of interest determines the type of Hamiltonian (e.g., electronic, vibrational etc.),
we have significant freedom to optimize the implementation. We can select specific
Hamiltonian representations and pair them with the time evolution algorithm of choice, i.e. Trotterization
or Qubitization.

We begin with the example of X-ray Absorption Spectroscopy (XAS) to demonstrate how to use the
PennyLane resource estimator to generate logical resource counts.

X-Ray Absorption Spectroscopy
-----------------------------
`XAS <https://pennylane.ai/qml/demos/tutorial_xas>`_, is a critical spectroscopic method used to study the electronic
and local structural environment of specific elements within a material, achieved by probing core-level electron transitions.
For this simulation, we follow the algorithm established in Fomichev et al. (2025) [#Fomichev2025]_,
which utilizes the `Compressed Double Factorization (CDF) <https://pennylane.ai/qml/demos/tutorial_how_to_build_compressed_double_factorized_hamiltonians>`_
representation of the electronic Hamiltonian combined with Trotterization.

The first step to determining the resources for this simulation is to define the Hamiltonian.
Here, we must note that the resource requirements for the simulation are independent of the specific integral values,
and depend rather on the structural parameters of the Hamiltonian, specifically
the number of orbitals and the number of fragments. We leverage this by utilizing the specialized
`compact Hamiltonian <https://docs.pennylane.ai/en/stable/code/api/pennylane.estimator.compact_hamiltonian.CDFHamiltonian.html>`__ representation feature
offered by PennyLane, skipping the expensive Hamiltonian construction while retaining the exact cost topology required for analysis.
"""

import pennylane.estimator as qre

active_spaces = [11, 14, 16, 18]
limno_ham = [qre.CDFHamiltonian(num_orbitals=i, num_fragments=i) for i in active_spaces]


######################################################################
# Trotterization
# ^^^^^^^^^^^^^^
# With the Hamiltonian defined, we move to the time evolution. This requires determining the total simulation time
# and the number of Trotter steps needed to keep the error within bounds.
#
# Following the analysis in Fomichev et al. (2025) [#Fomichev2025]_, we adopt a :math:`2^{nd}` order Trotter-Suzuki product formula.
# The total simulation time is determined by the desired spectral resolution, while the time step :math:`\Delta t`
# is derived from the error budget.

import numpy as np

eta = 0.05  # Lorentzian width (experimental resolution) in Hartree
jmax = 100  # Number of time points (determines resolution limit)
Hnorm = 2.0  # Maximum final state eigenvalue used to determine tau.

tau = np.pi / (2 * Hnorm)  # Sampling interval
trotter_error = 1  # Hartree
delta = np.sqrt(eta / trotter_error)  # Trotter step size
num_trotter_steps = int(np.ceil(2 * jmax + 1) * tau / delta)  # Number of Trotter steps

######################################################################
# XAS Circuit Construction
# ^^^^^^^^^^^^^^^^^^^^^^^^
# Having established the Hamiltonian structure and the Trotter step count, we are now
# ready to estimate resources for the complete algorithm.
#
# We now assemble the complete XAS algorithm as shown in the Hadamard test circuit,
# integrating state preparation, the Hadamard test structure, and controlled time evolution.
# This function can then be passed to the PennyLane resource estimator to obtain logical resource counts.
#
# Note that we model the initial state using the **Sum of Slaters** method [#SOSStatePrep2024]_, which
# approximates the wavefunction by discarding determinants below a coefficient tolerance.
# For this demo, we assume a truncation level that yields 1e4 surviving determinants (`num_slaters=1e4`).
# The cost of this approach depends on two major subroutines: :class:`~.pennylane.estimator.templates.QROM` to load the determinants,
# and :class:`~.pennylane.estimator.templates.QROMStatePreparation` to prepare the
# superposition.
#


def xas_circuit(hamiltonian, num_trotter_steps, measure_imaginary=False, num_slaters=1e4):

    # State preparation
    num_qubits = int(np.ceil(np.log2(num_slaters)))
    qre.QROMStatePreparation(
        num_state_qubits=num_qubits,
        positive_and_real=False,
        select_swap_depths=1,
    )
    qre.QROM(num_bitstrings=2**num_qubits, size_bitstring=num_qubits, select_swap_depth=1)

    # Hadamard and S gates
    qre.Hadamard()

    if measure_imaginary:
        qre.Adjoint(qre.S())

    # Controlled time evolution
    qre.Controlled(
        qre.TrotterCDF(hamiltonian, num_trotter_steps, order=2), num_ctrl_wires=1, num_zero_ctrl=0
    )

    qre.Hadamard()

    # Uncompute state preparation
    qre.Adjoint(
        qre.QROMStatePreparation(
            num_state_qubits=num_qubits,
            positive_and_real=False,
            select_swap_depths=1,
        )
    )
    qre.Adjoint(
        qre.QROM(num_bitstrings=2**num_qubits, size_bitstring=num_qubits, select_swap_depth=1)
    )


######################################################################
# Resource Estimation
# ^^^^^^^^^^^^^^^^^^^
# We can now estimate resources for the full circuit. To ensure a valid comparison with
# the reference, we must first align our gate synthesis strategy.
#
# A major difference lies in how rotation gates are synthesized, PennyLane approximates rotation gates using the Repeat-Until-Success
# circuits [#Alex2014]_ which decomposes rotations into **T-gates**, while the reference
# implementation utilizes the **phase gradient trick** proposed by Gidney (2018) [#Gidney2018]_ which implements rotations
# using adder arithmetic, decomposing to **Toffoli gates**.
#
# Since the literature reports costs in Toffolis, we must configure our estimator to use the
# adder-based synthesis. We resolve this by leveraging :class:`~.pennylane.resource.ResourceConfig` to register
# a custom decomposition that models single-qubit rotations using the required phase gradient arithmetic.
#


def single_qubit_rotation(precision=None):
    """Gidney-Adder based decomposition for single qubit rotations"""
    num_bits = int(np.ceil(np.log2(1 / precision)))
    return [qre.GateCount(qre.resource_rep(qre.SemiAdder, {"max_register_size": num_bits + 1}))]


######################################################
# We can now set up the resource estimation with this custom decomposition using the ``set_decomp`` function and set the precision
# for single-qubit rotations to :math:`10^{-3}` as used in the reference.

cfg = qre.ResourceConfig()
cfg.set_decomp(qre.RX, single_qubit_rotation)
cfg.set_decomp(qre.RY, single_qubit_rotation)
cfg.set_decomp(qre.RZ, single_qubit_rotation)
cfg.set_single_qubit_rot_precision(1e-3)

##################################################################
# With the configuration set, we run the resource estimation for each Hamiltonian.

xas_resources = []
toffolis = []
for ham in limno_ham:
    resource_counts = qre.estimate(xas_circuit, config=cfg)(
        hamiltonian=ham, num_trotter_steps=num_trotter_steps, measure_imaginary=False
    )
    xas_resources.append(resource_counts)
    toffolis.append(resource_counts.gate_counts["Toffoli"])

######################################################################
# Let's visualize how these estimates compare to the results reported in the literature.

import matplotlib.pyplot as plt

toffolis_lit = [7.12e7, 1.46e8, 2.18e8, 3.11e8]  # From Fomichev et al. (2025) [#Fomichev2025]_
plt.plot(active_spaces, toffolis, "o-", label="Estimated Resources", color="fuchsia")
plt.plot(active_spaces, toffolis_lit, "s--", label="Literature Resources", color="gold")

plt.xlabel("Number of Orbitals")
plt.ylabel("Toffoli Count")
plt.title("XAS Resource Estimation Comparison")
plt.legend()
plt.show()

########################################################################
# Optimizing the Estimates
# ^^^^^^^^^^^^^^^^^^^^^^^^
# From the above plot, we observe that our resource estimates are significantly higher
# than those reported in literature. This discrepancy is expected. While we aligned the rotation synthesis, the reference algorithm
# employs several other specialized optimizations specific to this application and some of the decompositions used
# in PennyLane are more general and hence have higher costs.
#
# Fortunately, we can bridge this gap by further customizing the :class:`~.pennylane.estimator.ResourceConfig`.
# We can teach the estimator the specific cost models used in the paper by using custom decompositions for key
# operations, similar to how we changed the rotation gate synthesis decomposition above.
#
# Let's start by customizing the basis rotation decomposition to use the one from `Kivlichan et al. (2018)
# <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.110501>`_.


def basis_rotation_cost(dim):
    """Custom basis rotation decomposition from Kivlichan et al. (2018)"""

    counts = dim * (dim - 1) / 2

    ops_data = [
        (qre.RZ, dim, None),
        (qre.RX, counts, None),
        (qre.RY, counts, None),
        (qre.Hadamard, 4 * counts, None),
        (qre.S, counts, None),
        (qre.Adjoint, counts, {"base_cmpr_op": qre.resource_rep(qre.S)}),
        (qre.CNOT, 2 * counts + dim, None),
    ]

    return [
        qre.GateCount(qre.resource_rep(op, params or {}), count) for op, count, params in ops_data
    ]


cfg.set_decomp(qre.BasisRotation, basis_rotation_cost)

##################################################################
# Next, we use the double phase trick for CRZ decomposition as described in Section III A of Fomichev et al. (2025) [#Fomichev2025]_.
# This optimization reduces the cost of the controlled rotations inside the Trotter steps.


def custom_CRZ_decomposition(precision):
    """Decomposition of CRZ gate using double phase trick"""
    rz = qre.resource_rep(qre.RZ, {"precision": precision})  # resource representation of RZ
    cnot = qre.resource_rep(qre.CNOT)

    return [qre.GateCount(cnot, 2), qre.GateCount(rz, 1)]


cfg.set_decomp(qre.CRZ, custom_CRZ_decomposition)

##################################################################
# Finally, we can re-run and visualize the resource estimation with these customizations to see how the resources compare to literature now.

xas_resources_custom = []
for ham in limno_ham:
    resource_counts = qre.estimate(xas_circuit, config=cfg)(
        hamiltonian=ham, num_trotter_steps=num_trotter_steps, measure_imaginary=False
    )
    xas_resources_custom.append(resource_counts)

toffolis_custom = [
    resource_counts.gate_counts["Toffoli"] for resource_counts in xas_resources_custom
]
plt.plot(active_spaces, toffolis_custom, "o-", label="Estimated Resources", color="fuchsia")
plt.plot(active_spaces, toffolis_lit, "s--", label="Literature Resources", color="gold")

plt.xlabel("Number of Orbitals")
plt.ylabel("Toffoli Gate Count")
plt.title("XAS Resource Estimation Comparison")
plt.legend()
plt.show()

######################################################################
# As shown in the final plot, our customized resource estimates now align closely with
# the estimates in the literature.
#
# Photodynamic Therapy Applications
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Having validated our resource estimation workflow against the XAS benchmarks, let's see how we can
# apply these tools to a different application, i.e. **Photodynamic Therapy (PDT)**, where we need to
# estimate the cumulative absorption rates for a transition-metal photosensitizer.[#Zhou2025]_
#
# This application requires a fundamental shift in algorithmic strategy. While XAS simulates the system's time evolution to
# observe dynamic changes (Trotterization), PDT employs a spectral filtering approach. Here, rather than resolving individual
# eigenstates, we use **Generalized Quantum Signal Processing (GQSP)** to isolate and measure the total signal within a specific
# therapeutic energy window (typically 700–850 nm).
#
# To achieve this efficiently, the algorithm utilizes a **Walk Operator** constructed from the **Tensor Hypercontraction (THC)**
# Hamiltonian, which allows for a highly compact block encoding. Similar to the XAS example, we can use the
# `compact Hamiltonian <https://docs.pennylane.ai/en/stable/code/api/pennylane.estimator.compact_hamiltonian.THCHamiltonian.html>`__
# representation to skip the expensive Hamiltonian construction. Let's verify our model using the 11-orbital BODIPY system from the reference.

bodipy_ham = qre.THCHamiltonian(num_orbitals=11, tensor_rank=22, one_norm=6.48)

##################################################################
# We now construct the `Walk Operator from the Hamiltonian <https://pennylane.ai/qml/demos/tutorial_re_for_qubitizedQPE>`_
# using the `~.pennylane.estimator.templates.QubitizeTHC` template. For comprehensive details on how to construct and configure this operator,
# we recommend the `Resource Estimation for Qubitization <https://pennylane.ai/qml/demos/tutorial_re_for_qubitizedQPE>`_ demo.
# Let's define the precision parameters based on the error budget from reference and construct the walk operator accordingly:

error = 0.0016  # Error budget from Zhou et al. (2025)
n_coeff = int(np.ceil(2.5 + np.log2(10 * bodipy_ham.one_norm / error)))  # Coeff precision
n_angle = int(
    np.ceil(5.652 + np.log2(10 * bodipy_ham.one_norm * 2 * bodipy_ham.num_orbitals / error))
)  # Rotation angle precision

prep_op = qre.PrepTHC(bodipy_ham, coeff_precision=n_coeff, select_swap_depth=4)
select_op = qre.SelectTHC(bodipy_ham, rotation_precision=n_angle, num_batches=5)
walk_op = qre.QubitizeTHC(bodipy_ham, prep_op=prep_op, select_op=select_op)

##################################################################
# Next, we need to set up the GQSP parameters to construct the spectral filter. The filter is defined by a polynomial
# whose degree determines the sharpness of the filter. Following Zhou et al. (2025) [#Zhou2025]_, we set the polynomial
# degree using Figure 7 from the paper, which relates the degree to the desired spectral resolution.


def polynomial_degree(one_norm):
    """Calculate polynomial degree parameters from Zhou et al. (2025)"""
    e_hi = 0.0701  # Ha
    e_lo = 0.0507  # Ha
    d_hi = 4.7571 * (one_norm + e_hi) / 0.01 + 321.2051
    d_lo = 4.7571 * (one_norm + e_lo) / 0.01 + 321.2051
    poly_degree = int(np.ceil(d_hi + d_lo))
    return poly_degree


##################################################################
# We can now set up the resource estimation for the PDT algorithm by defining the circuit as shown in Figure 4
# of our reference.


def pdt_circuit(walk_op, poly_degree, num_slaters=1e4):
    num_qubits = int(np.ceil(np.log2(num_slaters)))
    qre.QROMStatePreparation(
        num_state_qubits=num_qubits,
        positive_and_real=False,
    )
    qre.QROM(num_bitstrings=2**num_qubits, size_bitstring=num_qubits, select_swap_depth=1)
    # GQSP Spectral Filter
    qre.GQSP(walk_op, d_plus=poly_degree, d_minus=0)
    # Multi-Controlled-X
    qre.MultiControlledX(
        num_ctrl_wires=2,
        num_zero_ctrl=1,
    )
    # Uncompute state preparation
    qre.Adjoint(
        qre.QROMStatePreparation(
            num_state_qubits=num_qubits,
            positive_and_real=False,
        )
    )
    qre.Adjoint(
        qre.QROM(num_bitstrings=2**num_qubits, size_bitstring=num_qubits, select_swap_depth=1)
    )


##################################################################
# Now, that we have all the components set up, we can run the resource estimation for the PDT algorithm
# to calculate cumulative absorption:

qubits_pdt = []
toffolis_pdt = []

poly_deg = polynomial_degree(bodipy_ham.one_norm)  # Calculate polynomial degree
resource_counts = qre.estimate(pdt_circuit, config=cfg)(walk_op, poly_deg, num_slaters=1e4)
print(resource_counts)

######################################################################
# The estimated resources of **174 qubits** and **:math:`1.96 \times 10^7` Toffoli gates**.
# align with the reference values of **177 qubits** and **:math:`2.72 \times 10^7` Toffoli gates**.
# The small differences can be attributed to different tunable parameters being used.
# We encourage users to explore further by testing other systems from the reference or analyzing how the resources scale
# with different error budgets, using the parameter tuning techniques detailed in our `Qubitization demo <https://pennylane.ai/qml/demos/tutorial_re_for_qubitizedQPE>`_.
#
# Conclusion
# ----------
# In this demo, we successfully validated our resource estimation workflow by reproducing results from
# cutting-edge spectroscopy literature. We showed that PennyLane provides logical resource counts that
# align closely with theoretical benchmarks across distinct algorithmic paradigms, ranging from standard
# time-evolution to advanced spectral filtering.
#
# Beyond verification, a key takeaway is the flexibility of the estimation framework. We demonstrated how
# differences in gate counts—often arising from differing compilation assumptions—can be resolved by
# customizing the :class:`~.pennylane.resource.ResourceConfig`. This allows researchers to seamlessly
# swap out decomposition rules to match specific
# hardware constraints or theoretical models without needing to rewrite the high-level circuit logic.
#
# References
# ----------
#
# .. [#Fomichev2025]
#
#    Stepan Fomichev et al., “Fast simulations of X-ray absorption spectroscopy
#    for battery materials on a quantum computer”. `arXiv preprint arXiv:2506.15784
#    (2025) <https://arxiv.org/abs/2506.15784>`__.
#
# .. [#Kunitsa2025]
#
#    Alexander Kunitsa et al., "Quantum Simulation of Electron Energy Loss Spectroscopy for Battery Materials".
#    `arXiv preprint arXiv:2508.15935 (2025) <https://arxiv.org/abs/2508.15935>`__.
#
# .. [#Laoiza2025]
#
#    Ignacio Laoiza et al., "Simulating near-infrared spectroscopy on a quantum computer for enhanced chemical detection".
#    `arXiv preprint arXiv:2504.10602 (2025) <https://arxiv.org/abs/2504.10602>`__.
#
# .. [#SOSStatePrep2024]
#
#    Stepan Fomichev et al., "Initial state preparation for quantum chemistry on quantum computers".
#    `arXiv preprint arXiv:2310.18410v2 (2024) <https://arxiv.org/abs/2310.18410v2>`__.
#
# .. [#Alex2014]
#    Alex Bocharov et al., "Efficient synthesis of universal Repeat-Until-Success circuits".
#    `arXiv preprint arXiv:1404.5320 (2014) <https://arxiv.org/abs/1404.5320>`__.
#
# .. [#Gidney2018]
#    Craig Gidney. "Halving the cost of quantum addition".
#    `arXiv preprint arXiv:1709.06648 (2018) <https://arxiv.org/abs/1709.06648>`__.
#
# .. [#Zhou2025]
#    Yanbing Zhou et al., "Quantum Algorithms for Photoreactivity in Cancer-Targeted Photosensitizers".
#    `arXiv preprint arXiv:2512.15889 (2025) <https://arxiv.org/abs/2512.15889>`__.
#
