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
in D. Motlagh et al. [#Motlagh2025]. A general challenge in Trotter-based approaches is to identify an optimal
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

num_modes = 21       # Number of vibrational modes
num_states = 6       # Number of electronic states
grid_size = 4        # Number of qubits per mode (discretization)
taylor_degree = 2    # Truncate to Quadratic Vibronic Coupling (Linear + Quadratic terms)

#################################################################################
# In our model, we truncate the interaction terms for potential energy fragment to linear and quadratic terms only.
# This approximation, known as the QVC model, captures the dominant physical effects while simplifying the potential
# energy circuits significantly.

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

def kinetic_circuit(num_modes, grid_size, phase_grad_wires):

    qre.Pow(qre.AQFT(num_wires= grid_size), pow_z= num_modes)

    for i in range(num_modes):
        qre.OutOfPlaceSquare(register_size=grid_size)

        for j in range(2*grid_size):
            qre.Controlled(qre.SemiAdder(max_register_size=phase_grad_wires - j), num_ctrl_wires=1, num_zero_ctrl=0)

        qre.Adjoint(qre.OutOfPlaceSquare(register_size=grid_size))

    qre.Pow(qre.Adjoint(qre.AQFT(num_wires=grid_size)), num_modes)

######################################################################
# Similarly, we can define the structure for the Potential Energy Fragments. For a QVC model truncated to quadratic terms,
# each fragment will implement a monomial of degree up to 2
#
# ..math::
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
def linear_circuit(num_states, grid_size, phase_grad_wires):
    qre.QROM(num_bitstrings=num_states, size_bitstring=phase_grad_wires, restored=False)

    for i in range(grid_size):
        qre.Controlled(qre.SemiAdder(max_register_size=phase_grad_wires - i), num_ctrl_wires=1, num_zero_ctrl=0)

    qre.Adjoint(qre.QROM(num_bitstrings=num_states, size_bitstring=phase_grad_wires, restored=False))

def quadratic_circuit(num_states, grid_size, phase_grad_wires):
    qre.QROM(num_bitstrings=num_states, size_bitstring=phase_grad_wires, restored=False)

    qre.OutOfPlaceSquare(register_size=grid_size)
    for i in range(2*grid_size):
        qre.Controlled(qre.SemiAdder(max_register_size=phase_grad_wires - i), num_ctrl_wires=1, num_zero_ctrl=0)

    qre.Adjoint(qre.OutOfPlaceSquare(register_size=grid_size))
    qre.Adjoint(qre.QROM(num_bitstrings=num_states, size_bitstring=phase_grad_wires, restored=False))

######################################################################
# Finally, we combine these fragments to define the full Second-Order Trotter Step:
# :math:`U(\Delta t) \approx e^{-iV \Delta t/2} e^{-iT \Delta t} e^{-iV \Delta t/2}`.

def trotter_step_circuit(num_modes, num_states, grid_size, phase_grad_wires, taylor_degree):
    # Potential Energy Fragments
    for mode in range(num_modes):
        if taylor_degree >= 1:
            linear_circuit(num_states, grid_size, phase_grad_wires)
        if taylor_degree >= 2:
            quadratic_circuit(num_states, grid_size, phase_grad_wires)

    # Kinetic Energy Fragment
    kinetic_circuit(num_modes, grid_size, phase_grad_wires)

#################################################################################
# With the single Trotter step defined, we can move to defining the rest of the parameters needed for Trotterization, that is the number of Trotter
# steps and order of the Suzuki-Trotter expansion. For the sake of brevity, we take these numbers directly from the reference. [#Motlagh2025]_
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