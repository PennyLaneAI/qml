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

To simulate the time evolution :math:`U(t) = e^{-iHt}`, we employ a product formula (Trotterization) based approach as outlined
in D. Motlagh et al. [#Motlagh2025]_. A general challenge in Trotter-based approaches is to identify an optimal
decomposition of the Hamiltonian into fragments. Here, we utilize a term-based fragmentation scheme, where the Hamiltonian terms
are grouped based on their vibrational monomial, e.g., collecting all electronic terms coupled to :math:`\hat{q}_1 \hat{q}_2`
in one fragment.

For each resulting fragment, the electronic component is a small, arbitrary Hermitian matrix. We apply a
unitary transformation to rotate this component into its eigenbasis. With this diagonalization, the associated
vibrational operator simplifies to a **single monomial**. This structure is key to the algorithm's feasibility,
as simulating a single monomial is significantly more efficient than simulating the complex polynomial sums
found in naive decompositions.

Defining the Hamiltonian
^^^^^^^^^^^^^^^^^^^^^^^^
With the algorithmic approach defined, the next step is to instantiate the system we wish to benchmark.

While access to the full Hamiltonian coefficients would allow us to further optimize costs by leveraging the commutativity of electronic parts,
we can still derive a reliable baseline using only structural parameters. By defining the number of modes, electronic states, and grid size,
we can map out the cost topology of a real system without needing full integral data.

Let's take the example of Anthracene dimer, a system critical for understanding singlet fission in organic solar cells [#Motlagh2025]_ and
define these key parameters:
"""

num_modes = 19       # Number of vibrational modes
num_states = 5       # Number of electronic states
grid_size = 4        # Number of qubits per mode (discretization)
taylor_degree = 2    # Truncate to Quadratic Vibronic Coupling (Linear + Quadratic terms)

#################################################################################
# In our model, we truncate the interaction terms for potential energy fragment to linear and quadratic terms only.
# This approximation, known as the QVC model, captures the dominant physical effects while simplifying the potential
# energy circuits significantly.
#
# Constructing Circuits for Single Time-Step
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Next step is to define what the circuit will look like for evolving a single time step.
# Based on the term-based fragmentation scheme, the single Trotter step is composed of two distinct types of
# quantum circuits interleaved together:
#
# 1.  **Potential Energy Fragments:** These implement the interaction terms. For each fragment, the algorithm
#     loads coefficients using a `QROM (Quantum Read-Only Memory) <https://pennylane.ai/qml/demos/tutorial_intro_qrom>`_,
#     computes the vibrational monomial product using quantum arithmetic, and applies a phase gradient.
# 2.  **Kinetic Energy Fragment:** This implements the nuclear kinetic energy. Since the kinetic operator depends
#     on momentum :math:`P`, this circuit uses the `Quantum Fourier Transform (QFT) <https://pennylane.ai/qml/demos/tutorial_qft>`_
#     to switch to the momentum basis, applies a phase rotation, and then switches back.
#
# Let's first see what the circuit will look for the kinetic energy fragment, it implements the time evolution by diagonalizing the
# momentum operator:

import pennylane.estimator as qre
def kinetic_circuit(mode_wires, phase_wires, scratch_wires, coeff_wires):

    num_phase_wires = len(phase_wires)
    grid_size = len(mode_wires[0])
    for mode in range(num_modes):
        qre.AQFT(order=1, num_wires= grid_size, wires=mode_wires[mode])

    for i in range(num_modes):
        qre.OutOfPlaceSquare(register_size=grid_size, wires=scratch_wires + mode_wires[i])

        for j in range(2*grid_size):
            ctrl_wire = [scratch_wires[j]]
            target_wires = coeff_wires[:len(phase_wires)-j] + phase_wires[:len(phase_wires)-j]
            qre.Controlled(qre.SemiAdder(max_register_size=num_phase_wires - j), num_ctrl_wires=1, num_zero_ctrl=0, wires=target_wires + ctrl_wire)

        qre.Adjoint(qre.OutOfPlaceSquare(register_size=grid_size, wires=scratch_wires + mode_wires[i]))

    for mode in range(num_modes):
        qre.Adjoint(qre.AQFT(order=1, num_wires=grid_size, wires = mode_wires[mode]))

######################################################################
# Similarly, we can define the structure for the Potential Energy Fragments. For a QVC model truncated to quadratic terms,
# each fragment will implement a monomial of degree up to 2
#
# ..math::
#
#     V_{ji}^{m} = \sum_{r}c_r Q_r + \sum_{r} \tilde{c}_{r} Q_r^2
#
# where :math:`c_r` and :math:`\tilde{c}_{r}` are the linear and quadratic coupling coefficients respectively. The exponential
# of each term is implemented by:
# * Loading the coefficients from QROM controlled by the electronic state.
# * Computing the monomial product by using a square operation for quadratic terms.
# * Multiplying with the coefficients and adding to the resource register to accumulate the phase.
# * Uncomputing the intermediate steps to clean up ancilla wires.
#
# We can define this circuit using two different segments, one for linear terms and one for quadratic terms:
#
def linear_circuit(num_states, elec_wires, phase_wires, coeff_wires, mode_wires, scratch_wires):
    qre.QROM(num_bitstrings= len(elec_wires), size_bitstring=len(phase_wires), restored=False, wires=elec_wires + coeff_wires)

    for i in range(grid_size):
        ctrl_wire = [scratch_wires[i]]
        target_wires = coeff_wires[:len(phase_wires)-i] + phase_wires[:len(phase_wires)-i]
        qre.Controlled(qre.SemiAdder(max_register_size=len(phase_wires) - i), num_ctrl_wires=1, num_zero_ctrl=0, wires=target_wires + ctrl_wire)

    qre.Adjoint(qre.QROM(num_bitstrings= len(elec_wires), size_bitstring=len(phase_wires), restored=False, wires=elec_wires + coeff_wires))

def quadratic_circuit(num_states, elec_wires, phase_wires, coeff_wires, mode_wires, scratch_wires):
    grid_size = len(mode_wires)
    qre.QROM(num_bitstrings=len(elec_wires), size_bitstring=len(phase_wires), restored=False, wires=elec_wires + coeff_wires)

    qre.OutOfPlaceSquare(register_size=grid_size, wires=mode_wires + scratch_wires)
    for i in range(2*grid_size):
        ctrl_wire = [scratch_wires[i]]
        target_wires = coeff_wires[:len(phase_wires)-i] + phase_wires[:len(phase_wires)-i]
        qre.Controlled(qre.SemiAdder(max_register_size=len(phase_wires) - i), num_ctrl_wires=1, num_zero_ctrl=0, wires=target_wires + ctrl_wire)

    qre.Adjoint(qre.OutOfPlaceSquare(register_size=grid_size, wires=mode_wires + scratch_wires))
    qre.Adjoint(qre.QROM(num_bitstrings=len(elec_wires), size_bitstring=len(phase_wires), restored=False, wires=elec_wires + coeff_wires))

######################################################################
# Finally, to ensure precise resource tracking, we explicitly label our wire registers. This avoids ambiguity
# about which qubits are active and ensures the resource estimator captures the full width of the circuit.

def get_wire_labels(num_modes, num_states, grid_size, phase_prec):
    """Generates the wire map for the full system."""
    num_elec_qubits = (num_states - 1).bit_length()
    elec_wires = [f"e_{i}" for i in range(num_elec_qubits)] # Electronic State Register

    phase_wires = [f"pg_{i}" for i in range(phase_prec)] # Resource State For Phase Gradients
    coeff_wires = [f"c_{i}" for i in range(phase_prec)] # Coefficient Register

    print("wire labels grid size: ", grid_size)
    mode_wires = []
    for m in range(num_modes):
        mode_wires.append([f"m{m}_{w}" for w in range(grid_size)]) # Mode m Register

    scratch_wires = [f"s_{i}" for i in range(2 * grid_size)] # Scratch Space for Arithmetic

    return elec_wires, phase_wires, coeff_wires, mode_wires, scratch_wires

###########################################################################
# We now combine these fragments to define the full Second-Order Trotter Step:
# :math:`U(\Delta t) \approx e^{-iV \Delta t/2} e^{-iT \Delta t} e^{-iV \Delta t/2}`.
# For each second order Trotter step, all potential energy fragments
# are applied twice, each for half the time, and the kinetic energy fragments
# is applied once.


def apply_potential(num_states, elec_wires, phase_wires, coeff_wires, mode_wires, scratch_wires, taylor_degree):
    for mode in range(num_modes):
        if taylor_degree >= 1:
            linear_circuit(num_states, elec_wires, phase_wires, coeff_wires, mode_wires[mode], scratch_wires)
        if taylor_degree >= 2:
            quadratic_circuit(num_states, elec_wires, phase_wires, coeff_wires, mode_wires[mode], scratch_wires)

def trotter_step_circuit(num_modes, num_states, grid_size, phase_grad_wires, taylor_degree):
    elec_wires, phase_wires, coeff_wires, mode_wires, scratch_wires = get_wire_labels(num_modes, num_states, grid_size, phase_grad_wires)
    # Potential Energy Fragments
    apply_potential(num_states, elec_wires, phase_wires, coeff_wires, mode_wires, scratch_wires, taylor_degree)

    # Kinetic Energy Fragment
    kinetic_circuit(mode_wires, phase_wires, scratch_wires, coeff_wires)

    # Potential Energy Fragments
    apply_potential(num_states, elec_wires, phase_wires, coeff_wires, mode_wires, scratch_wires, taylor_degree)

#################################################################################
# With the single Trotter step defined, we can move to defining the rest of the parameters needed for Trotterization, that is the number of Trotter
# steps and order of the Suzuki-Trotter expansion. For the sake of brevity, we take these numbers directly from the reference. [#Motlagh2025]_.

print(qre.estimate(trotter_step_circuit)(num_modes, num_states, grid_size, phase_grad_wires=20, taylor_degree=taylor_degree))

#################################################################################
# Conclusions
# ^^^^^^^^^^^
# By constructing the full simulation workflow for vibronic dynamics from the ground up, we have gained critical insights into
# the resource requirements for simulating these complex processes on future quantum hardware. Through careful circuit design and leveraging efficient algorithmic techniques,
# we have outlined a feasible path forward for simulating non-adiabatic dynamics, paving the way for future explorations in quantum chemistry beyond the Born-Oppenheimer approximation.
# This demonstration highlights the power of modular quantum algorithm design, enabling researchers to build, analyze, and optimize complex workflows for next-generation quantum simulations.
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