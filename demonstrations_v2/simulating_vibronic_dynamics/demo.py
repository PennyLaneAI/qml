r"""
Simulating Vibronic Dynamics
############################

Simulating static properties will not get us very far.

When it comes to simulating complex quantum effects, classical simulation 
is well poised to handle systems that, for the most part,
stay still. Ground state energies and other static quantities are good heuristics,
but the simulations we need to continue progressing areas
such as advanced materials discovery are dynamical in nature. This is something 
classical computers just can't seem to crack. 

In the paper `"Quantum Algorithm for Vibronic Dynamics"
<https://arxiv.org/abs/2411.13669>`_, Motlagh et al. propose a novel quantum
algorithm for vibronic simulations. This algorithm leverages several
tools including :doc:`phase gradient states <demos/efficient_rotations_with_phase_gradient_states>`, 
:doc:`QROM
coefficient loading <demos/tutorial_intro_qrom>`, and, foundationally, :doc:`Trotterization <demos/exploring_trotterization>` to reach
beyond Born-Oppenheimer and toward more realistic, capable dynamic simulations.

Why don't we add this to our toolkits! The following demo will show how the Motlagh et al. 
vibronics algorithm can be implemented in PennyLane for a toy example. We will aim
to simulate state evolution for a simple electronic system to demonstrate the functionality
and utility of this algorithm.

A note on Born-Oppenheimer
--------------------------
In theoretical chemistry, classical simulations tend to make the 
`Born-Oppenheimer approximation
<https://en.wikipedia.org/wiki/Born%E2%80%93Oppenheimer_approximation>`_, which 
assumes electrons and nuclei can be treated as completely separate entities. 
This works very well for ground state energies and simplifies molecular systems
adequately for classical systems to handle. 

When we go beyond Born-Oppenheimer to model dynamic systems, though,
it is no longer possible to isolate
electronic degrees of freedom (DOFs) from nuclear DOFs, meaning their paths
cannot be simulated separately. Long story short, if we want to capture the
realistic dynamics of a molecular system, we lose access to approximations that
allow for a reduction in the number of simulation variables and, as a result,
the size of the system will very quickly exceed standard computational
limits. This is especially true when we set out to simulate `vibronic systems <https://en.wikipedia.org/wiki/Vibronic_coupling>`_, 
which are concerned with vibrational
interactions between electrons and nuclei.

Motlagh et al. aim to enable beyond Born-Oppenheimer simulation
with the goal of allowing the limits of classical computation in dynamic simulations
to be overcome.

The Köppel-Domcke-Cederbaum Hamiltonian
---------------------------------------
The Köppel-Domcke-Cederbaum (KDC) Hamiltonian is a well known, straightforward
representation of a vibronic system in which the electronic and nuclear
vibrational modes are coupled. It takes the general
form
 
.. math:: 
   H = \mathcal{I}_{el} \otimes (T_{nuc}+V_0)+\textbf{W},

where :math:`\mathcal{I}_{el}` is the electronic identity operator,
:math:`T_{nuc}` is the vibrational
kinetic operator, :math:`V_0` is the reference 
potential energy operator, and
:math:`\textbf{W}` is a coupling matrix representing the diabatic
potential [#Motlagh2025]_. The source paper defines this potential
in a truncated quadratic vibronic coupling (QVC) model as

.. math:: 
   \mathbf{W}'_{ij}(\vec{Q})=\lambda^{(i,j)}+\sum_r a_r^{(i,j)}Q_r+\sum_{r,r'} b_{r,r'}^{(i,j)}Q_rQ_{r^\prime}.

We can interpret this expression by understanding :math:`\lambda^{(i,j)}`,
:math:`a_r^{(i,j)}`, and :math:`b_{rr'}^{(i,j)}` as coupling coefficients and
:math:`Q_r` as mode-dependent position operators (with :math:`r` being a specific mode) [#Motlagh2025]_. 
In this truncation, we deal with only the linear and quadratic coordinate terms, which is 
adequate to both reduce error and ensure implementability.

We can take the kinetic and potential operators as

.. math:: 
   T=\mathcal{I}_{el} \otimes \sum_{r=0}^{M-1}\frac{\omega_r}{2}P_r^2

and

.. math:: 
   V =\sum_{i,j=0}^{N-1}|j\rangle\langle i| \otimes V_{ji}

respectively, where
:math:`P` is the momentum operator and
:math:`V_{ji}` represents the sum of the expansion of
:math:`\textbf{W'}` in coefficient form and the ground-state 
potential :math:`V_0`.

Carrying out a time evolution of the KDC Hamiltonian is useful for modelling applications such as 
spectroscopy and energy transfer dynamics. For this
demonstration, we will aim to simulate electronic state evolution for a
small, simple system. 

Grid encoding
-------------
To keep our operations efficient, it is important to select a space
representation that allows for easy basis transformation. In classical 
computation, it is difficult to use real space for simulations due to the exponential
memory requirements of storing a grid-encoded wave packet. However, using real space provides optimal conditions 
for implementing our Hamiltonian, since we can represent both position and momentum operators diagonally,
pending a simple change of basis. This is a clear advantage of using quantum computers
to carry out vibronic simulations. Since memory is not a main bottleneck in the quantum case,
we are free to use real space as we please!

To do this, we need to define the grid on which our system will exist. 
Letting the total number of required
grid points for a given mode be :math:`K` (from which we can determine the number of required
qubits in the system :math:`k=\log_2(K)`) and taking the computational basis states to 
represent the position basis, we can define the 
:math:`Q` operator as

.. math::
   Q|x\rangle = \Delta(x-K/2)|x\rangle,

where :math:`\Delta=\sqrt{2\pi/K}` is the grid spacing term and :math:`x` is a
position-basis grid point index. It is more convenient to work in the signed integer representation

.. math::
   Q|x\rangle = \Delta \cdot x |x\rangle

for our implementation, letting :math:`x \in \{-\frac{K}{2}, -\frac{K}{2}+1, \dots, \frac{K}{2}-1\}`. This discretization dictates the 
size of our space and the number of qubits required for our system. We will consistently integrate this representation and the defined
grid convention throughout our variable and circuit definitions.

Fragmentation and diagonalization
---------------------------------
The Motlagh et al. vibronics algorithm is, at its core, a :doc:`Trotterization algorithm <demos/exploring_trotterization>`.
To review, Trotterization is a Hamiltonian
simulation method that addresses the issue of exponentiating non-commuting terms in
a Hamiltonian. In this method, a target Hamiltonian is separated into groups of commuting operators called *fragments* to be
individually exponentiated and interleaved in partial time steps to simulate time evolution.

To implement these fragments in a circuit using gates, they must be diagonal. Off-diagonal terms
run rampant in vibronic systems due to the high degree of coupling, so a complete Trotter step
must include a diagonalization procedure. To address this, Motlagh et al. lay out a Clifford gate-based
scheme for `block diagonalization
<https://pennylane.ai/compilation/diagonal-unitary-decomp/details>`_. 

We can take each Hamiltonian fragment to be expressed as the operator

.. math:: 
   H_m = \sum_{j=0}^{N-1}|j\rangle \langle m \oplus j|\otimes V_{j,m\oplus j},

where :math:`m` and :math:`j` are electronic state indices. 
Since :math:`|j\rangle \langle m \oplus j|\otimes` constructs the matrix geometry of
the fragment, the difference between :math:`j` and :math:`m\oplus j` (representing the `Hamming
weight <https://en.wikipedia.org/wiki/Hamming_weight>`_ in this case) will describe a fragment's proximity
to a diagonal configuration and 
dictate how the block should be treated. The logic is as follows:

1. IF :math:`m=0`, we are dealing with a diagonal fragment. Our work here is
   done!
2. IF :math:`j` and :math:`m\oplus j` differ by 1, we can achieve
   diagonalization by sandwiching the fragment between Hadamard gates.
3. ELSE, pick one of the differing (off-diagonal) bit positions to act as a control for CNOT operations 
   targeting all other differing bits. This reduces the Hamming weight to 1,
   enabling diagonalization via a Hadamard gate acting on the control bit.

.. figure::
   ../demonstrations_v2/simulating_vibronic_dynamics/pennylane-demo-simulating-vibronic-dynamics-Diagonalization.png
   :align: center 
   :width: 700px
   :alt: A visual representation of the diagonalization scheme.
    
    *Clifford gate diagonalization scheme* [#Motlagh2025]_.

This can be simply implemented according to the outlined logic.
"""
import pennylane as qp
import numpy as np

def DiagonalizationScheme(fragment, electron_wires):
    bits = []
    weight = 0

    for j in range(len(electron_wires)):
        if (fragment >> j) & 1:
            weight += 1
            bits.append(j)

    # Apply the diagonalization scheme according to Hamming weight
    if weight == 1:
        qp.Hadamard(wires = electron_wires[bits[0]])
    elif weight > 1:
        ctrl_wire = electron_wires[bits[0]]
        for bit in bits[1:]:
            qp.CNOT(wires=[ctrl_wire, electron_wires[bit]])
        qp.Hadamard(wires=ctrl_wire)
###############################################################################
# In the KDC Hamiltonian, it is intuitive to fragment our expression into kinetic
# terms and potential terms. As defined, each has its own specifications, so we will
# treat them individually.
#
# The kinetic step
# ----------------
# The kinetic energy fragment is, comparatively, simple to establish
# and evolve in time, so we will deal with it first. 
# Cutting to the chase, the kinetic step should, for each mode:
#
# 1. Perform a change of basis on the input state to momentum space,
# 2. Square the corresponding mode's register and store the result in an ancillary register,
# 3. Apply a rotation to the state register via a :doc:`phase gradient register <demos/efficient_rotations_with_phase_gradient_states>`,
# 4. Uncompute the ancillary register. 
#
# We also know that the kinetic energy coefficients will always be proportional to
# :math:`P^2` and will be independent of the electronic state. This makes our lives easy! Rather than loading
# state-dependent coefficients into the system, we can simply encode calculated
# coefficients into the initial state. 
# 
# We can intuitively understand these coefficients as describing the motion of the nuclear states. If
# the kinetic term coefficient is :math:`0`, the nucleus is frozen and there is nothing vibronic to simulate.
#
# Motlagh et al. specify the basis transformation should take place via the sequence
#
# .. math::
#   P = QFT^\dagger X_{k-1} Q X_{k-1} QFT,
#
# in which the X-gates are applied as a means of bit-ordering. 
#
# Once the
# basis has been switched, an
# :class:`~pennylane.OutPoly` operation targeting :math:`f(x)=(x-K/2)^2` (which is
# equivalent to :math:`x^2` in the signed-integer picture) can be used to carry
# out the squaring operation on the state register.
#
# To apply the rotation step, a :doc:`phase gradient <demos/efficient_rotations_with_phase_gradient_states>`
# approach will be implemented.
# Phase gradient rotations work by storing a pre-computed, catalytic state that
# holds position-dependent rotation angles. Via quantum addition, these angles can be
# applied to corresponding qubits to carry out cheap rotations. It takes the
# general form
#
# .. math::
#    |R\rangle = \frac{1}{2^{b/2}}\sum e^{i2\pi y/2^b}|y\rangle,
#
# where :math:`b` is the number of wires in the gradient register and determines the
# precision of the rotation. A major benefit of using phase gradients is they only need to be prepared
# once, since the addition step does not have any impact on the state. 
#
# To carry out the addition
# without error, we will perform a wire offset step that selects a 'subregister', using only a 
# portion of the available wires and shifting the binary representation. This virtual indexing
# reduces the wire requirements of this addition step, making our classical demonstration
# more feasible. This is mathematically
# equivalent to carrying out a `logical left shift
# <https://en.wikipedia.org/wiki/Logical_shift>`_ in classical computing. 
#
# ``AddPhaseGradient`` handles
# this and the logic that carries out the addition of the phase gradient register into the target
# register. We will reuse this function in each of our kinetic and potential steps, so 
# we must account for the requirements of each. The ``signed`` argument tells the function 
# if the integer being passed is signed (i.e., 
# positive or negative, as it will be in the potential step) or unsigned (as it will be in the kinetic step).

def AddPhaseGradient(k, cache_wires, coeff_wires, gradient_wires, scratch_wires, 
        signed=False):
    
    for point in range(2 * k):
    # Control on the spatial state register
        ctrl_wire = [cache_wires[point]]

         # Index to the current position in the register
        weight = 2*k - 1 - point
        target_length = len(gradient_wires) - weight

        if target_length <= 0:
            continue

        # Index the coefficient wires to the required size
        x_wire_current = coeff_wires
        if len(x_wire_current) > target_length:
            x_wire_current = coeff_wires[(len(coeff_wires) - target_length):]

        y_wire_current = gradient_wires[:target_length]

        # Apply addition operator based on the size of the numbers being added
        if target_length == 1:
            qp.ctrl(qp.CNOT, control = ctrl_wire
                )(wires=[x_wire_current[-1], y_wire_current[0]])
        elif target_length >= 2:
            # If we are dealing with a signed integer 
            # and the most significant bit, use an adjoint semiadder.
            if (signed and point == 0):
                adder = qp.adjoint(qp.SemiAdder)  
            else:
                adder = qp.SemiAdder 
            qp.ctrl(adder, control = ctrl_wire)(x_wires = x_wire_current, 
                    y_wires = y_wire_current, work_wires = scratch_wires)
################################################################################
# The result of this procedure is a set of target states that have
# accumulated a phase equivalent to the change incurred during a given
# kinetic time step.
#
# .. figure:: ../demonstrations_v2/simulating_vibronic_dynamics/pennylane-demo-simulating-vibronic-dynamics-KineticStepCircuit.png
#   :align: center
#   :width: 700px
#   :alt: A circuit diagram representing the kinetic energy step logic.
#
#   *Kinetic energy step circuit diagram*
#
# ``KineticStep()`` executes a single step in totality, first performing the basis transformation,
# followed by adding the coefficients to the register, then carrying out the required
# quantum arithmetic before uncomputing.

def KineticStep(time_step, kinetic_coeffs, num_modes, state_wires, gradient_wires, 
        coeff_wires, scratch_wires, cache_wires):
    
    k = len(state_wires[0])
    K = 2 ** k
    b = len(gradient_wires)

    # Set function to be executed by OutPoly()
    def f(x):
        return (x - K//2) ** 2 # Signed Integer Representation

    # Perform basis transformation
    for mode in range(num_modes):
        qp.QFT(wires=state_wires[mode])
        qp.X(wires=state_wires[mode][0])

    # Loop full computational procedure over all modes
    for i in range(num_modes):

        # Compute coefficients
        kin_coeff_raw = (kinetic_coeffs[i] * time_step * (2 ** b) / (2 * K))
        C = int(np.floor(kin_coeff_raw + 0.5))

        C_binary = format(C, f'0{len(coeff_wires)}b')

        # Encode coefficients
        for j, bit in enumerate(C_binary):
            if bit == '1':
                qp.X(wires=coeff_wires[j])

        # Square state
        qp.OutPoly(f, input_registers = [state_wires[i]], output_wires = cache_wires)
        
        AddPhaseGradient(k, cache_wires, coeff_wires, gradient_wires, scratch_wires)

        # Uncompute
        qp.adjoint(qp.OutPoly)(
            f, 
            input_registers=[state_wires[i]], 
            output_wires=cache_wires
            )

        for j, bit in enumerate(C_binary):
            if bit == '1':
                qp.X(wires=coeff_wires[j])

    for mode in range(num_modes):
        qp.X(wires = state_wires[mode][0])
        qp.adjoint(qp.QFT)(wires = state_wires[mode])
###############################################################################
# The potential step
# ------------------
# The goal of the potential energy step is to construct the full potential
# energy operator for each electron state and coupled
# vibrational state. To do this, we
# must consider the state-dependent potential coefficients and the vibrational
# modes of the system. The operations that will need to be carried out by this
# function are:
#
# 1. Load the electron state-dependent coefficient terms into the coefficient
#    register,
# 2. If there are multiple vibrational mode states involved in this step,
#    multiply them together,
# 3. Multiply the full mode state (either a single mode or the product of
#    multiple modes, depending on step 2) with the coefficient register,
# 4. Add the product of the mode state and coefficient register to the phase
#    gradient register,
# 5. Uncompute.
#
# To facilitate loading, the state-dependent coefficients must be determined and stored
# in a bit-position-dependent fashion prior to Trotterization. For now, we
# will assume that this has been handled and simply passed into our
# potential step function for use.
# Motlagh et al. specify that the coefficients
# can be easily loaded via a QROM. Using PennyLane's built in :class:`~pennylane.QROM`
# function, we can simply take computed coefficients and pass them in to the
# potential energy step of our choosing.

def LoadCoeffsKDC(fragment, output, electron_wires, coeff_wires, scratch_wires):
    fragment_coeffs = output[fragment]
    
    qp.QROM(fragment_coeffs, control_wires=electron_wires, 
        target_wires=coeff_wires, work_wires=scratch_wires)

###############################################################################
# As shown in our QVC representation of :math:`\textbf{W'}`, we are only
# concerned with scenarios with one or two mode states. Thus, in the case our system 
# is quadratic, we need to apply an :class:`~pennylane.OutPoly`
# operator that multiplies the two mode registers together, just like we did in
# the kinetic step. Otherwise, no arithmetic required. The outcome of either of these
# cases is added to the phase gradient register via quantum arithmetic multiplier-adder
# gates, inducing the corresponding rotation.
#
# .. figure::
#    ../demonstrations_v2/simulating_vibronic_dynamics/pennylane-demo-simulating-vibronic-dynamics-PotentialEnergyStep.png
#    :align: center 
#    :width: 700px
#    :alt: Circuit diagram representing the logic of the potential energy step.
#    
#    *Potential energy step circuit diagram*
#
# In ``PotentialStepLinear()``, the outlined procedure is implemented for the linear scenario, 
# in which no modes need to be multiplied during the execution. Note that here we are using signed
# integers to properly interface with the QROM, so the ``AddPhaseGradient()`` function must be
# configured as such.

def PotentialStepLinear(fragment, load_coeffs, mode, time_coeffs, state_wires, 
    electron_wires, gradient_wires, coeff_wires, cache_wires, scratch_wires):

    k = len(state_wires[mode])
    K = 2 ** k

    # Load pre-determined, electron state dependent coefficients
    load_coeffs(fragment, time_coeffs, electron_wires, coeff_wires, scratch_wires)

    qp.OutPoly(
        lambda x: (x - K // 2), 
        input_registers=[state_wires[mode]], 
        output_wires=cache_wires
        )

    AddPhaseGradient(k, cache_wires, coeff_wires, gradient_wires, 
        scratch_wires, signed=True)

    qp.adjoint(qp.OutPoly)(
        lambda x: (x - K // 2), 
        input_registers=[state_wires[mode]], 
        output_wires=cache_wires
        )

    qp.adjoint(load_coeffs)(fragment, time_coeffs, electron_wires, coeff_wires, 
        scratch_wires)
###############################################################################
# ``PotentialStepQuadratic()`` handles the other scenario, in which there are two modes to be multiplied and used in the arithmetic steps.

def PotentialStepQuadratic(fragment, load_coeffs, mode1, mode2, time_coeffs,
        state_wires, electron_wires, gradient_wires, coeff_wires, cache_wires, scratch_wires):

    k = len(state_wires[mode1])
    K = 2 ** k

    load_coeffs(fragment, time_coeffs, electron_wires, coeff_wires, scratch_wires)

    qp.OutPoly(
        lambda x0,x1: (x0 - K // 2)*(x1 - K // 2), 
        input_registers=[state_wires[mode1], state_wires[mode2]], 
        output_wires = cache_wires
        )

    AddPhaseGradient(k, cache_wires, coeff_wires, gradient_wires, scratch_wires, 
        signed = True)

    #Uncompute
    qp.adjoint(qp.OutPoly)(
        lambda x0,x1: (x0 - K // 2) * (x1 - K // 2), 
        input_registers=[state_wires[mode1], state_wires[mode2]], 
        output_wires=cache_wires
        )

    qp.adjoint(load_coeffs)(fragment, time_coeffs, electron_wires, coeff_wires, 
        scratch_wires)

###############################################################################
# Given an input of mode states, we can evaluate if an entry is a
# singular integer value, a list containing a single entry, or a list containing
# two entries, each of which are possible valid inputs. The first two scenarios
# will be taken as equivalent to a linear case while the third requires the
# multiplication step of the quadratic function. ``KDCFrag()`` handles this
# using simple evaluation logic.

def KDCFrag(fragment, load_coeffs, mode_list, coeff_array, state_wires, electron_wires, 
        gradient_wires, coeff_wires, cache_wires, scratch_wires):
    
    for entry in mode_list:
        if isinstance(entry, int):
            PotentialStepLinear(fragment, load_coeffs, entry, coeff_array, state_wires, 
                electron_wires, gradient_wires, coeff_wires,cache_wires, scratch_wires)

        if isinstance(entry, tuple) and len(entry) == 1:
            PotentialStepLinear(fragment, load_coeffs, entry[0], coeff_array, state_wires, 
                electron_wires, gradient_wires, coeff_wires, cache_wires, scratch_wires)

        if isinstance(entry, tuple) and len(entry) == 2:
            mode1 = entry[0]
            mode2 = entry[1]
            PotentialStepQuadratic(fragment, load_coeffs,mode1, mode2, coeff_array, state_wires, 
                electron_wires, gradient_wires, coeff_wires, cache_wires, scratch_wires)
###############################################################################
# Now that we have the tools we need to carry out a system-wide time evolution,
# how exactly do we execute?
#
# Assembling the Trotter step
# ---------------------------
# It is up to us which Trotterization order we implement. While first order
# Trotterization is less resource intensive, second order allows for reduced
# `Trotter error
# <https://arxiv.org/html/2606.30738v1>`_
# and sets us up for a useful uncompute trick later on. In general, a second
# order Trotterization is given by
#
# .. math:: U_2(\theta)=\prod_{m=0}^N e^{i\theta H_m} \prod_{m=N}^0 e^{i\theta
#    H_m},
#
# in which :math:`H_m` is a Hamiltonian fragment and :math:`N` is the total number of fragments. 
# Note that this second-order approach iterates the fragments both forward and
# backward. This serves to both reduce error and reduce gate requirements since
# splitting the potential step into these
# mirrored steps lands a :math:`QFT` next to a :math:`QFT^\dagger`, meaning we can
# easily maintain the proper basis without adding additional transformations.
# Phew!
#
# So, our Trotter process should:
#
# 1. Perform a half-step evolution on the outermost fragment,
# 2. For each intermediate fragment,
#    a. Diagonalize,
#    b. Perform a half-step evolution
#    c. Un-diagonalize
# 3. Repeat step 2 in reverse order,
# 4. Perform a half-step evolution on the outermost fragment.


def TrotterStepKDC(dt, frag_list, coupler, PotentialStep, KineticStep, kinetic_args, 
        coupler_args, potential_args):

    half_dt = dt / 2

    KineticStep(half_dt, *kinetic_args)

    for fragment in frag_list:
        # Diagonalization function
        coupler(fragment, *coupler_args)
        # Pass a function that can handle the potential step in the linear or quadratic case
        PotentialStep(fragment, *potential_args)
        qp.adjoint(coupler)(fragment, *coupler_args)

    # Second-order, reversed potential step
    for fragment in reversed(frag_list):
        # Diagonalization function
        coupler(fragment, *coupler_args)
        PotentialStep(fragment, *potential_args)
        qp.adjoint(coupler)(fragment, *coupler_args)

    KineticStep(half_dt, *kinetic_args)
###############################################################################
# We're almost there!
#
# Registers
# ---------
# The registers we will use for implementation
# can be defined according to the requirements of the system using
# :func:`~pennylane.registers`. As mentioned,
# the phase gradient steps require a register of size :math:`b`, which can be computed
# in terms of the desired rotation precision :math:`\delta`. This is the main
# driver of resolution for our task.
#
# It is worth noting that the
# flooring step in the kinetic energy coefficient calculation requires a minimum
# number of precision bits to be present in the system to avoid flooring to 0
# and, therefore, suppressing all coupling effects. As such, ``delta`` should be
# small enough to achieve :math:`b\geq 5`, though minimizing :math:`b` will always 
# lead to a loss of precision. For the purposes of this demo, we will take :math:`b=5`.

import math

qp.decomposition.enable_graph() # Enable graph-based decomposition for performance

time_steps = 10
k = 2
n = 1
num_modes = 1
delta = 0.04 

b = int(math.ceil(np.log2(1 / delta)))

regs = qp.registers({
    "electrons": n,
    "states": {f"mode_{i}": k for i in range(num_modes)}, 
    "gradient": b,
    "coefficients": b,
    "scratch": b + 1,
    "cache": 2 * k,
})

# Unpack state wires for the number of modes present
state_wires = [regs[f"mode_{i}"] for i in range(num_modes)]

# Calculate total wires
total_wires = n + (num_modes * k) + (3 * b + 1 + (2 * k))
###############################################################################
# Initial state definition
# ------------------------
# In "Quantum Algorithm for Vibronic Dynamics", the initial state of the KDC
# system is taken to be a simple vertical excitation represented in product form
# in relation to electronic state :math:`j` as 
#
# .. math:: |\psi_0\rangle = |j\rangle_{el}
#    \bigotimes_{r=0}^{M-1}|\chi_0\rangle.
#
# Here,
#
# .. math:: |\chi_0\rangle = \frac{1}{Z}\sum_{x=0}^{K-1}
#    \exp\left(\frac{-\pi \cdot (x-\frac{K}{2})^2}{K}\right) |x\rangle
#
# is the Hermite-Gauss function representation of the harmonic oscillator ground
# state, where :math:`Z` is a normalization constant and :math:`x` is, again, a
# position-basis grid point index. It is stated that one can choose to begin
# with a superposition state to enable functionalities such as spectroscopy. In
# this demo, since we will be targeting a simple electronic state time
# evolution, an initial superposition is not necessary. This state can be
# generated using ``KDCStatePrep()``.
import matplotlib.pyplot as plt

def KDCStatePrep(k):
    K = 2**k
    x = np.arange(K)

    amplitudes = np.exp((-np.pi * ((x - (K / 2)) ** 2)) / (K))
    norm_factor = np.linalg.norm(amplitudes)

    chi0 = amplitudes / norm_factor

    return chi0

plt.plot(KDCStatePrep(6))
plt.title("Initial State Distribution")
plt.xlabel("Index", fontsize=12)
plt.ylabel("Amplitude", fontsize=12)
###############################################################################
# Implementation
# --------------
# Finally, we have adequately built up the skeleton of our
# simulation!
#
# To keep things simple (and computationally feasible), we will begin by
# defining a small system with 2 electron states and 1 vibrational mode. 
# The potential coefficients will be taken to be a simple array of values
# that will soon be scaled by the required factors. 
#
# In the source paper, the
# full coefficient representation is given as
#
# .. math:: \Delta^{\alpha}c^{(j,j)}_\alpha,
#
# where :math:`\alpha` is the expansion degree of the polynomial term. Taking the bit-wise representation and
# considering the time dependence, the full representation of the coefficients
# that should be passed into the QROM is
#
# .. math:: c_{time}=[c_{\alpha} \Delta^\alpha \frac{dt}{2} \frac{2^b}{2\pi}] \text{mod}
#    2^b.
#
# This form allows for easy computation and conversion to the list-of-bit format
# required by the QROM. For this demo, :math:`\alpha` will be fixed to 1 since
# we are only dealing with linear mode behaviour. This can be easily changed
# depending on the needs and conditions of the system.

mode_list = [0]
omega = [1]
coeff_array = np.array([
    #  |0>    |1>
    [ 1.0,   0.0 ],   # Fragment 0: Diagonal potential energy terms
    [ -1.3,   1.3 ]    # Fragment 1: Off-diagonal electronic coupling terms
])
dt = 0.4

width = len(regs["coefficients"])
max_binary = 2 ** width
Delta = np.sqrt(2 * np.pi / (2 ** k))
alpha = 1 # Limiting to linear degree here


# Scale coefficients and introduce time dependence
multiplier = (dt / 2) * max_binary * (Delta ** alpha) / (2 * np.pi)
v_array = np.round(coeff_array * multiplier).astype(int) % max_binary
time_coeffs = [[f"{v:0{width}b}" for v in row] for row in v_array]   
################################################################################
# Now, at long last, we can carry out our time evolution. 

# Define argument lists
kinetic_args = [omega, num_modes, state_wires, regs["gradient"], regs["coefficients"], 
    regs["scratch"], regs["cache"]]

potential_args = [LoadCoeffsKDC, mode_list, time_coeffs, state_wires, 
    regs["electrons"], regs["gradient"], regs["coefficients"], regs["cache"], regs["scratch"]]

coupler_args = [regs["electrons"]]

dev = qp.device("lightning.qubit", wires=total_wires)

@qp.qnode(dev)
def ElectronStateVibronicsSimulation(steps, gradient_wires, StatePrepFunc, CouplerFunc, PotentialFunc, 
    KineticFunc, kinetic_args, potential_args, coupler_args):

    # Prepare the phase gradient state in the appropriate register
    for wire in gradient_wires:
        qp.X(wires=wire)

    qp.QFT(wires=gradient_wires)

    # Prepare the initial state
    initial_state = StatePrepFunc(k)
    for wire in state_wires:
        qp.StatePrep(state=initial_state, wires=wire)

    # Trotterize
    for t in range(steps):
        TrotterStepKDC(dt = dt, frag_list = range(2 ** n),coupler = CouplerFunc, PotentialStep = PotentialFunc,
            KineticStep = KineticFunc, kinetic_args = kinetic_args, coupler_args = coupler_args,
            potential_args = potential_args)
        
    return qp.probs(wires=regs["electrons"])
################################################################################
# This function can then be called for plotting, with each time step appended
# to a list that will output a tracked evolution of the electron state.

time_grid = list(range(time_steps + 1))
actual_time_axis = [step * dt for step in time_grid]

vibronic_state_0 = []
vibronic_state_1 = []

for t in time_grid:
    probs = (ElectronStateVibronicsSimulation)(steps = t, gradient_wires = regs["gradient"], 
    StatePrepFunc = KDCStatePrep, CouplerFunc = DiagonalizationScheme, PotentialFunc = KDCFrag, 
    KineticFunc = KineticStep, kinetic_args = kinetic_args, potential_args = potential_args, 
    coupler_args = coupler_args)

    vibronic_state_0.append(probs[0])
    vibronic_state_1.append(probs[1])
    
plt.figure(figsize=(9, 6))

# Plot Full Vibronic Simulation
plt.plot(actual_time_axis, vibronic_state_0, color='blue', label='Vibronic - State 0', 
    linewidth=2.5)
plt.plot(actual_time_axis, vibronic_state_1, color='red', label='Vibronic - State 1', 
    linewidth=2.5)

# Graph Formatting
plt.title("Electronic State Evolution", fontsize=14, fontweight='bold')
plt.xlabel("Time (a.u.)", fontsize=12)
plt.ylabel("State Probability", fontsize=12)
plt.ylim(-0.05, 1.05) #True Max: 1, True Min: 0
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='best', fontsize=10);
################################################################################
# We did it! The achieved plot depicts a physically expected outcome, in which
# an incomplete transfer between the ground and excited state is observed. 
# The anti-symmetry between the two energy levels shows that no unexpected leakage has occurred, 
# while the oscillation adheres to expected state transitions for an electron.
# This toy model is much smaller and simpler than a real, useful vibronic
# system would be, but the results are sufficient to show the utility of 
# Motlagh et al.'s quantum algorithm for vibronic simulation. 
#
# Conclusion
# ----------
# Preparing to make the most out of quantum technologies is crucial as we
# continue to move toward accessible, useful quantum devices. Identifying areas
# of interest that are known to be important to researchers, companies, and
# individuals is an important first step. Vibronic simulation has the potential
# to expand our capacity for material discovery, renewable energy expansion, and
# drug exploration. Quantum computing has the potential to make this possible.
#
# PennyLane is continually building out `resources related to quantum chemistry simulations <https://docs.pennylane.ai/en/stable/code/qp_qchem.html>`_,
# such as tools for :doc:`resource estimation of vibronic dynamic simulation <tutorial_resource_estimation_vibronic_dynamics>`. 
# In `"Quantum algorithm for simulating non-adiabatic dynamics
# at metallic surfaces" <https://arxiv.org/abs/2601.16264>`_, the authors
# employ similar techniques to simulate vibronic dynamics using a GAN Hamiltonian. Take advantage of these
# resources to explore the ways quantum computers could revolutionize the way we approach
# theoretical chemistry. 
#
# .. _references:
#
# References
# ----------
# .. [#Motlagh2025] D.\ Motlagh, R. A. Lang, P. Jain, J. A.
#    Campos-Gonzalez-Angulo, W. Maxwell, T. Zeng, A. Aspuru-Guzik, and J. M.
#    Arrazola, "Quantum Algorithm for Vibronic Dynamics: Case Study on Singlet
#    Fission Solar Cell Design," 2025, `doi: 10.48550/arXiv.2411.13669
#    <https://doi.org/10.48550/arXiv.2411.13669>`_.
#
# .. [#Lang2026] R.\ A. Lang, P. Jain, J. M. Arrazola, and D. Motlagh, "Quantum
#    Algorithm for Simulating Non-Adiabatic Dynamics at Metallic Surfaces," 2026,
#    `doi: 10.48550/arXiv.2601.16264 <https://doi.org/10.48550/arXiv.2601.16264>`_.