r"""
Quantifying Resource Requirements for Vibronic Dynamics Simulation
==================================================================

Standard quantum chemistry often relies on the adiabatic Born-Oppenheimer approximation, assuming nuclei move on a single potential energy surface (PES)
well-separated from others. But in the world of photochemistry, where molecules absorb light, and break bonds, this approximation
breaks down. The timescales of electronic and nuclear motion become comparable, giving rise to vibronic coupling. This coupling, drives
critical processes like photosynthesis, and solar cell efficiency and accurately simulating this requires a Hamiltonian that treats the dynamics
of electrons and nuclei simultaneously.

However, modeling such dynamics is computationally demanding. Before we attempt to run such complex algorithms on future fault-tolerant hardware,
we need to answer a fundamental question: Is this algorithm actually feasible?

In this demo, we assume the role of an algorithm architect. We will construct a simulation workflow
for the vibronic Hamiltonian from the ground up, utilizing the modular building blocks within PennyLane's :mod:`~.pennylane.estimator`.
By constructing this pipeline ourselves, we can perform a feasibility analysis, determining exactly what resources are required to simulate
these complex non-adiabatic processes on future hardware.

.. figure:: ../_static/demonstration_assets/mapping/long_image.png
    :align: center
    :width: 80%
    :target: javascript:void(0)


The Vibronic Hamiltonian
------------------------
To perform this feasibility check, we must first define the system. We use the vibronic coupling Hamiltonian,
which describes a set of :math:`N` electronic states interacting with :math:`M` vibrational modes.

Unlike standard electronic structure problems, this model mixes discrete electronic levels with continuous
vibrational motion, and the Hamiltonian takes the form:

.. math::

    H = T + V,

where :math:`T` represents the kinetic energy of the nuclei, and is diagonal while :math:`V` contains off-diagonal
electronic couplings.

To simulate the time evolution :math:`U(t) = e^{-iHt}`, we employ a product formula (Trotterization) approach as outlined in
D. Motlagh et al. [#Motlagh2025]_. A key challenge in Trotterization is decomposing the Hamiltonian efficiently.
Here, we utilize a term-based fragmentation scheme, where terms are grouped by their vibrational monomial
(e.g., collecting all electronic terms coupled to :math:\hat{q}_1^2). For each fragment, the electronic component is
merely a small Hermitian matrix; by rotating this component into its eigenbasis, the complex potential simplifies to
a single vibrational monomial. This structure is central to the algorithm's feasibility, as simulating isolated monomials
is significantly more efficient than simulating the complex polynomial sums found in naive decompositions.

Defining the Hamiltonian
^^^^^^^^^^^^^^^^^^^^^^^^
With the algorithmic approach defined, the next step is to instantiate the system we wish to benchmark.

While access to the full Hamiltonian coefficients would allow us to further optimize costs by leveraging the commutativity of electronic parts,
we can still derive a reliable baseline using only structural parameters. By defining the number of modes, electronic states, and grid size,
we can map out the cost topology of a real system without needing full integral data.

As a case study, we select (NO):math:`_4`-Anth, a molecule created by introducing four N-oxyl radical fragments to
anthracene. Proposed for its theoretically record-breaking singlet fission speed [#Motlagh2025]_, we use this system to define our simulation parameters:
"""

num_modes = 19  # Number of vibrational modes
num_states = 5  # Number of electronic states
grid_size = 4  # Number of qubits per mode (discretization)
taylor_degree = (
    2  # Truncate to Quadratic Vibronic Coupling (Linear + Quadratic terms)
)

#################################################################################
# In our model, we truncate the interaction terms for potential energy fragment to linear and quadratic terms only.
# This approximation, known as the Quadratic Vibronic Coupling (QVC) model, captures the dominant physical effects while simplifying the potential
# energy circuits significantly.
#
# Constructing Circuits for Single Time-Step
# ------------------------------------------
# Next step is to define what the circuit will look like for evolving a single time step.
# Based on the term-based fragmentation scheme, the single Trotter step is composed of two distinct types of
# quantum circuits interleaved together:
#
# 1.  **Kinetic Energy Fragment:** This implements the nuclear kinetic energy. Since the kinetic operator depends
#     on momentum :math:`P`, this circuit uses the `Quantum Fourier Transform (QFT) <https://pennylane.ai/qml/demos/tutorial_qft>`_
#     to switch to the momentum basis, applies a phase rotation, and then switches back.
# 2.  **Potential Energy Fragments:** These implement the interaction terms. For each fragment, the algorithm
#     loads coefficients using a `QROM (Quantum Read-Only Memory) <https://pennylane.ai/qml/demos/tutorial_intro_qrom>`_,
#     computes the vibrational monomial product using quantum arithmetic, and applies a phase gradient.
#
# Let's first see what the circuit will look for the kinetic energy fragment, it implements the time evolution by diagonalizing the
# momentum operator:

import pennylane.estimator as qre


def kinetic_circuit(mode_wires, phase_wires, scratch_wires, coeff_wires):
    ops = []
    num_phase_wires = len(phase_wires)
    grid_size = len(mode_wires[0])
    for mode in range(num_modes):
        ops.append(qre.AQFT(order=1, num_wires=grid_size, wires=mode_wires[mode]))

    for i in range(num_modes):
        ops.append(
            qre.OutOfPlaceSquare(
                register_size=grid_size, wires=scratch_wires + mode_wires[i]
            )
        )

        for j in range(2 * grid_size):
            ctrl_wire = [scratch_wires[j]]
            target_wires = (
                coeff_wires[: len(phase_wires) - j]
                + phase_wires[: len(phase_wires) - j]
            )
            ops.append(
                qre.Controlled(
                    qre.SemiAdder(max_register_size=num_phase_wires - j),
                    num_ctrl_wires=1,
                    num_zero_ctrl=0,
                    wires=target_wires + ctrl_wire,
                )
            )

        ops.append(
            qre.Adjoint(
                qre.OutOfPlaceSquare(
                    register_size=grid_size, wires=scratch_wires + mode_wires[i]
                )
            )
        )

    for mode in range(num_modes):
        ops.append(
            qre.Adjoint(
                qre.AQFT(order=1, num_wires=grid_size, wires=mode_wires[mode])
            )
        )

    return qre.Prod(ops)


######################################################################
# Similarly, we can define the structure for the Potential Energy Fragments. For a QVC model truncated to quadratic terms,
# each fragment will implement a monomial of degree up to 2
#
# .. math::
#
#     V_{ji}^{m} = \sum_{r}c_r Q_r + \sum_{r} \tilde{c}_{r} Q_r^2
#
# where :math:`c_r` and :math:`\tilde{c}_{r}` are the linear and quadratic coupling coefficients respectively. The exponential
# of each term is implemented by:
#
# * Loading the coefficients from QROM controlled by the electronic state.
# * Computing the monomial product by using a square operation for quadratic terms.
# * Multiplying with the coefficients and adding to the resource register to accumulate the phase.
# * Uncomputing the intermediate steps to clean up ancilla wires.
#
# We can define this circuit using two different segments, one for linear terms and one for quadratic terms:


def linear_circuit(num_states, elec_wires, phase_wires, coeff_wires, scratch_wires):
    ops = []
    ops.append(
        qre.QROM(
            num_bitstrings=num_states,
            size_bitstring=len(phase_wires),
            restored=False,
            wires=elec_wires + coeff_wires,
        )
    )

    for i in range(grid_size):
        ctrl_wire = [scratch_wires[i]]
        target_wires = (
            coeff_wires[: len(phase_wires) - i] + phase_wires[: len(phase_wires) - i]
        )
        ops.append(
            qre.Controlled(
                qre.SemiAdder(max_register_size=len(phase_wires) - i),
                num_ctrl_wires=1,
                num_zero_ctrl=0,
                wires=target_wires + ctrl_wire,
            )
        )

    ops.append(
        qre.Adjoint(
            qre.QROM(
                num_bitstrings=num_states,
                size_bitstring=len(phase_wires),
                restored=False,
                wires=elec_wires + coeff_wires,
            )
        )
    )
    return qre.Prod(ops)


def quadratic_circuit(
    num_states, elec_wires, phase_wires, coeff_wires, mode_wires, scratch_wires
):
    ops = []
    grid_size = len(mode_wires)
    ops.append(
        qre.QROM(
            num_bitstrings=num_states,
            size_bitstring=len(phase_wires),
            restored=False,
            wires=elec_wires + coeff_wires,
        )
    )

    qre.OutOfPlaceSquare(register_size=grid_size, wires=mode_wires + scratch_wires)
    for i in range(2 * grid_size):
        ctrl_wire = [scratch_wires[i]]
        target_wires = (
            coeff_wires[: len(phase_wires) - i] + phase_wires[: len(phase_wires) - i]
        )
        ops.append(
            qre.Controlled(
                qre.SemiAdder(max_register_size=len(phase_wires) - i),
                num_ctrl_wires=1,
                num_zero_ctrl=0,
                wires=target_wires + ctrl_wire,
            )
        )

    ops.append(
        qre.Adjoint(
            qre.OutOfPlaceSquare(
                register_size=grid_size, wires=mode_wires + scratch_wires
            )
        )
    )
    ops.append(
        qre.Adjoint(
            qre.QROM(
                num_bitstrings=num_states,
                size_bitstring=len(phase_wires),
                restored=False,
                wires=elec_wires + coeff_wires,
            )
        )
    )
    return qre.Prod(ops)


#################################################################################
# Estimating the Number of Trotter Steps
# --------------------------------------
# With the Hamiltonian fragments defined, the next step is to combine them into a full time-evolution
# simulation. However, the feasibility of this simulation depends heavily on the total number of
# Trotter steps required to reach the target time :math:`T`.
#
# Since we employ a second-order Trotter-Suzuki product formula, the algorithmic error scales
# quadratically with the time step (:math:`\text{Error} \propto \Delta t^2`). This predictable scaling allows us
# to determine the required step count using benchmark values from the reference literature.
#
# If a reference step size :math:`\Delta t_{\text{ref}}` is known to produce an error :math:`\epsilon_{\text{ref}}`,
# the step size :math:`\Delta t_{\text{req}}` needed to meet a strict target error :math:`\epsilon_{\text{req}}` is:
#
# .. math::
#
#     \Delta t_{\text{req}} = \Delta t_{\text{ref}} \sqrt{\frac{\epsilon_{\text{req}}}{\epsilon_{\text{ref}}}}
#
# The total number of steps is then simply :math:`N = T / \Delta t_{\text{req}}`.

import math


def calculate_trotter_steps(target_error, ref_error, ref_dt, total_time):
    """Calculates total steps required using second-order error scaling."""
    required_dt = ref_dt * math.sqrt(target_error / ref_error)
    return math.ceil(total_time / required_dt)


# Benchmarks for Anthracene Dimer from Motlagh et al. [#Motlagh2025]
ref_error = 0.00263  # Reference error at dt=0.1 fs
ref_dt = 0.1  # Reference time step (fs)
total_time = 100.0  # Total simulation time (fs)
target_error = 0.01  # Target: 1% error

num_steps = calculate_trotter_steps(target_error, ref_error, ref_dt, total_time)

#################################################################################
# Finally, to ensure precise resource tracking, we explicitly label our wire registers. This avoids ambiguity
# about which qubits are active and ensures the resource estimator captures the full width of the circuit.


def get_wire_labels(num_modes, num_states, grid_size, phase_prec):
    """Generates the wire map for the full system."""
    num_elec_qubits = (num_states - 1).bit_length()
    elec_wires = [
        f"e_{i}" for i in range(num_elec_qubits)
    ]  # Electronic State Register

    phase_wires = [
        f"pg_{i}" for i in range(phase_prec)
    ]  # Resource State For Phase Gradients
    coeff_wires = [f"c_{i}" for i in range(phase_prec)]  # Coefficient Register

    mode_wires = []
    for m in range(num_modes):
        mode_wires.append([f"m{m}_{w}" for w in range(grid_size)])  # Mode m Register

    scratch_wires = [
        f"s_{i}" for i in range(2 * grid_size)
    ]  # Scratch Space for Arithmetic

    return elec_wires, phase_wires, coeff_wires, mode_wires, scratch_wires


#################################################################################
# We now combine these fragments to define the full Second-Order Trotter circuit using PennyLane's
# `TrotterProduct <https://docs.pennylane.ai/en/stable/code/api/pennylane.estimator.templates.TrotterProduct.html>`_ class.
#


def circuit(num_modes, num_states, grid_size, taylor_degree, phase_grad_wires=20):
    fragments = []
    elec_wires, phase_wires, coeff_wires, mode_wires, scratch_wires = (
        get_wire_labels(num_modes, num_states, grid_size, phase_grad_wires)
    )
    kinetic_fragment = kinetic_circuit(
        mode_wires, phase_wires, scratch_wires, coeff_wires
    )
    fragments.append(kinetic_fragment)

    for mode in range(num_modes):
        if taylor_degree >= 1:
            fragments.append(
                linear_circuit(
                    num_states, elec_wires, phase_wires, coeff_wires, scratch_wires
                )
            )
        if taylor_degree >= 2:
            fragments.append(
                quadratic_circuit(
                    num_states,
                    elec_wires,
                    phase_wires,
                    coeff_wires,
                    mode_wires[mode],
                    scratch_wires,
                )
            )

    qre.TrotterProduct(first_order_expansion=fragments, num_steps=num_steps, order=2)


#################################################################################
# Finally, we can estimate the resource requirements for this full circuit using the estimate function:

print(qre.estimate(circuit)(num_modes, num_states, grid_size, taylor_degree))

#################################################################################
# These numbers align closely with the findings in Motlagh et al., confirming that our resource estimation pipeline
# correctly captures the resource requirements of the vibronic simulation algorithm.
#
# Conclusions
# -----------
# By constructing the full simulation workflow for vibronic dynamics from the ground up, we have gained critical insights into
# the resource requirements for simulating these complex processes on future quantum hardware.
# This approach allows us to break down complex algorithms into manageable building blocks, providing a transparent view of how
# algorithmic choices directly impact gate counts and qubit overhead.
# This demonstration highlights the power of modular quantum algorithm design, enabling researchers to build, analyze, and optimize
# complex workflows for next-generation quantum simulations.
# Ultimately, this framework is designed to be an open sandbox for researchers. We encourage you to use the building blocks
# provided here to explore alternative algorithms tailored to your specific research interests.
#
# References
# ----------
#
# .. [#Motlagh2025]
#
#      Danial Motlagh et al., "Quantum Algorithm for Vibronic Dynamics: Case Study on Singlet Fission Solar Cell Design".
#      `arXiv preprint arXiv:2411.13669 (2014) <https://arxiv.org/abs/2411.13669>`__.
#
#
