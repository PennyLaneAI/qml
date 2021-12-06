r""".. _superconducting_qubits:

Superconducting qubits
=============================

.. meta::
    :property="og:description": Description and assessment of superconducting quantum technology
    :property="og:image": https://pennylane.ai/qml/_images/sc_qubits.png

.. related::
   tutorial_trapped_ions Quantum computation with trapped ions

*Author: PennyLane dev team. Posted: XX November 2021. Last updated: XX November 2021.*

**Superconducting qubits** are among the most promising approaches to building quantum computers. 
It is no surprise that this technology is being used by companies such as Google and IBM in 
their quest to pioneer the quantum era. Google's Sycamore claimed quantum advantage back in 
2019, and IBM recently built its Eagle quantum computer with 127 qubits, outmatching its 
competitors by tenths of qubits! The central insight that allows for these quantum computers 
is that superconductivity is a quantum phenomenon, so we can use superconducting circuits 
as quantum systems that we can control at will. They are nothing but a modification of 
current microchip technology adapted to work with superconductors, so we have most of the
infrastructure in place! However, the large size of the qubit makes it prone to decoherence, 
making it more short-lived than other types of qubits. Nevertheless, we can get around this, 
and the results speak for themselves.

By the end of this demo, you will learn how superconductors are used to create, prepare, 
control, and measure the state of a qubit. Moreover, you will identify the strengths and
weaknesses of this technology in terms of Di Vincenzo's criteria, as introduced in the 
box below. You will be armed with the basic concepts to understand the main scientific 
papers on the topics and to keep up-to-date with the new developments that are bound 
to come soon.

.. container:: alert alert-block alert-info
    
    **Di Vincenzo's criteria**: In the year 2000, David DiVincenzo proposed a
    wishlist for the experimental characteristics of a quantum computer [#DiVincenzo2000]_.
    DiVincenzo's criteria have since become the main guideline for
    physicists and engineers building quantum computers:

    1. **Well-characterized and scalable qubits**. Many of the quantum systems that 
    we find in nature are not qubits, so we must find a way to make them behave as such.
    Moreover, we need to put many of these systems together.

    2. **Qubit initialization**. We must be able to prepare the same state repeatedly within
    an acceptable margin of error.

    3. **Long coherence times**. Qubits will lose their quantum properties after
    interacting with their environment for a while. We would like them to last long
    enough so that we can perform quantum operations.

    4. **Universal set of gates**. We need to perform arbitrary operations on the
    qubits. To do this, we require both single-qubit gates and two-qubit gates.

    5. **Measurement of individual qubits**. To read the result of a quantum algorithm,
    we must accurately measure the final state of a pre-chosen set of qubits. 

"""

##############################################################################
#
# Superconductivity
# ~~~~~~~~~~~~~~~~~~
#
# To introduce superconductivity, we need to understand why some materials resist
# the passage of electrons at the microscopic level. When an electric current travels
# through a material, there are two types of electrons: those flowing freely,
# known as *conduction electrons*; and those attached to an atom, known as *valence electrons*.
# In atoms, electrons populate some discrete energy levels, which have a population limit.
# Levels with lower energies are preferred, but they reject any new electrons to higher
# levels as they fill up. A very similar phenomenon occurs for the conduction electrons:
# if too many conduction electrons have a low energy, they will reject any new conduction
# electrons to a higher energy range.
#
# The energies of the valence electrons are usually lower than those of the conduction electrons.
# A material is a good conductor of electricity if a valence electron does not need too
# much energy to become a conduction electron. In other words, the *energy bandgap* of the
# material is small. In conductors, this gap is, in fact, zero: electrons in atoms can easily
# unbind themselves and become free to move in the conductor. However, even in conductors, the
# electrons rejected by the filled conduction levels may scatter and collide with the atoms,
# dissipating energy. The higher the temperature, the more likely these collisions are since
# the atoms move around more. These collisions are the origin of electric resistance.
#
# For some materials, at extremely low temperatures, something somewhat counterintuitive occurs.
# Conduction electrons start attracting each other and form pairs. This phenomenon is strange
# since electrons are supposed to repel each other. However, as the motional energy of the
# atoms in the conductor decreases, the conduction electrons can attract the positive nuclei,
# which in turn attract other electrons. The net effect is a coupling of electrons in *Cooper pairs*,
# which behave very differently to individual electrons: **any number of them can have the
# same energy**. Therefore, they can all be in the lowest conduction energy state.
# They will not reject any other Cooper pairs, and consequently, they will not scatter
# into the atoms and dissipate energy. Cooper pairs flow through the material without any
# resistance. They are the reason that we have superconductors!
#
# Quantum Circuits
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# What do we need for a system to be quantum? Contrary to common lore,
# it is neither necessary nor sufficient for it to be small. We have
# observed quantum behaviour in superconducting circuits, which are
# larger than most bacteria. To showcase quantum effects, we have
# observed that the superconducting circuit must be **isolated from
# an environment we cannot control**. To achieve this, we need to
# satisfy two conditions.
#
# First, we have to ensure that **we are not putting too much energy into
# the environment**, compared to the energy they store. An energy leak
# carries information about the system and counts as a partial measurement,
# thus destroying the quantum properties. Superconductors are great in this
# regard: since there is no resistance, there will be no energy output.
# But this is not enough, as we can observe by turning on our microwave.
# This appliance stores a lot of energy and does not dissipate it into our
# kitchen. So what makes a microwave non-quantum?
#
# Here is where the second condition kicks in. When a system is at a high temperature,
# **particles in the environment are constantly interacting with it, causing decoherence**.
# For a superconducting circuit to preserve its quantum properties for a long time,
# we need to **cool it to about 10 mK**, well below the temperature required by
# superconductivity. Such low temperatures are reached inside a dilution refrigerator.
#
# Let us study a simple circuit known as a *resonator*. This circuit has two components.
# The first one is a capacitor :math:`C`, which consists of two parallel metallic plates that
# can store charge. The second one is an inductor :math:`L` connected to both capacitor plates.
# When a varying current goes through the inductor, it exerts a force on the charges opposite
# to their motion. But the attraction between the plates is proportional to the charge contained
# in each of them.  The net effect of both components is that the more charge a capacitor has,
# the more the flow of charges is resisted by the inductor. This behaviour is the same as a spring's:
# the more one stretches a spring, the more it resists being extended and, once released,
# it starts to oscillate. Since its behaviour is similar to that of a prototypical harmonic
# oscillator, this circuit is also a harmonic oscillator.
#
# We can now make the resonator superconducting by bringing the temperature down. At these
# low temperatures, we experimentally observe that the possible energy values are given by
#
# .. math:: E = n\hbar\omega.
#
# where :math:`n`is a natural number, :math:`\hbar` is Planck's constant, and :math:`\omega` is the resonant
# frequency of the circuit, which depends on the the inductance and the capacitance.
# The circuit can only absorb or emit energy in multiples of :math:`\hbar\omega` in packets
# known as *photons*. This separation of energy levels is an inherently quantum effect,
# so superconducting circuits are quantum systems. In particular, the superconducting
# resonator is a quantum harmonic oscillator: its energy levels are evenly spaced.
#
# Building an artificial atom
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Can we use the superconducting resonator as a qubit? By definition, a qubit is a system
# with two energy levels. The harmonic oscillator has infinitely many, but this does not
# matter in many applications: we simply restrict to the two lowest energy levels.
# This cannot be done with the harmonic oscillator. The energy levels are equally spaced,
# which means that if we feed the system with photons of energy :math:`\hbar\omega`, there
# is the possibility of reaching the higher energy states. This is rather inconvenient
# since it makes it impossible to focus only on two energy states. If we want to get
# around this issue, we should find a way to change the energy level differences in our circuit.
#
# There is, in fact, a circuit element that does this: a *Josephson junction*. It
# consists of a very thin piece of an insulating placed between two superconducting metals.
# At first thought, the Josephson junction should play the role of a resistor, and
# since we have superconducting metals on either side of it, no current should go through it.
# Here is where *tunnelling*, another famous quantum effect, comes into play.
# Because of the spread-out nature of their wave function, Cooper pairs can
# actually go through the Josephson junction with some probability.
#
# By replacing the inductor in the superconducting circuit with a Josephson junction,
# we introduce anharmonicity in the energy levels. Like in atoms, energy levels
# become unevenly spaced, so we call such a circuit an *artificial atom*.
# They consist of the junction :math:`J` and the capacitor :math:`C`, but we have to add one
# element for them to be useful as a qubit. To interact with the environment in
# a controlled way, we need a *gate capacitor* :math:`C_g` in the artificial atom that
# receives electromagnetic inputs from outside. The amount of charge :math`Q_g` in this
# capacitor can be chosen by the experimenter, and it determines how strongly the
# circuit interacts with the environment. Moreover, the separation in energy levels
# depends on :math`Q_g` as shown below.
#
# There is a problem with this dependence, however. A small change in the gate charge :math:`Q_g`
# can change the difference in energy levels significantly. A solution to this issue is to
# work around the value :math:`Q_g/2e = 1/2`, where the levels are not too sensitive to changes in
# the gate charge. But there is a more straightforward solution: the difference in energy level
# also depends on the in-circuit capacitance :math:`C` and the physical characteristics of the junction,
# which we take as fixed. As we make the capacitance larger, the energy level differences become
# less sensitive to :math:`Q_g`.
#
# As we see from the graph above, there is a price to be paid: making :math:`C` larger does reduce
# the sensitivity to the gate charge, but it also makes the differences in energy levels more equal.
# However, the latter effect turns out to be smaller than the former, so we can adjust the capacitance
# value and preserve some anharmonicity. The regime that has been proven ideal is known as the
# **transmon regime**, and artificial atoms in the regime are called **transmons**.
# They have proven to be highly effective as qubits, and they are used in almost all
# applications nowadays. We can thus work with the first two energy levels of the transmon,
# which we take to be separated by an energy gap
#
# .. math:: E_a = \hbar\omega_a
#
# Here, :math:`\omega_a` is the resonant frequency of the qubit: a photon with that frequency
# will tend to make the transmon go from the ground to the excited state.
#
# We have now partially satisfied Di Vincenzo's first criterion of a well-defined qubit, and
# we will discuss scalability later. A great feature of superconducting qubits is that the second
# criterion is satisfied effortlessly. Since the excited states in an artificial atom are
# short-lived and prefer to be on the ground state, all we have to do is wait for a short period.
# If the circuit is well isolated, it is guaranteed that all the qubits will be in the ground
# state with a high probability after this short interval.
#
# Measuring the circuit's state
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Now that we have fabricated our qubit to our liking, we need to understand how to manipulate it.
# The way to do this is to put the qubit inside an *optical cavity*, a metal box where we can
# contain electromagnetic waves. Our focus will be on the so-called *Fabry-Perot* cavities.
# They consist of two mirrors facing each other and whose rears are coated with an
# anti-reflecting material. Something surprising happens when we shine a beam of light
# on a  Fabry-Perot cavity of length :math:`L`: electromagnetic waves will only be transmitted
# when they have a wavelength :math `\lambda` such that
#
# .. math:: L = L=n\lambda/2,
#
# where :math:`n` is an arbitrary positive integer. If this condition is not met, most photons
# will be reflected away. Therefore, we will have an electromagnetic field inside if we
# carefully tune our light source to one of these wavelengths. In this case, we say that
# we are *driving* the cavity. For superconducting qubits, it is most common to use
# wavelengths in the microwave range.
#
# Following Di Vincenzo's fifth criterion, let us see how we can measure the state of the qubit
# placed inside the cavity. To transmit information, we need to shine light at a frequency :math:`\omega_r`
# that the cavity can transmit (recall that the frequency and the wavelength are inversely proportional).
# We may also choose the frequency value to be far from the frequency gap :math:`\omega_a` of the
# superconducting circuit, so the photons are not absorbed by it. Namely, the *detuning*
# :math:`\Delta` needs to be large:
#
# .. math:: \Delta \equiv \left\lvert \omega_r - \omega_a \right\rvert \gg 1.
#
# What happens to the photons of this frequency that meet paths with the qubit? They are scattered
# by the circuit, so this chosen value for the frequency is known as the *dispersive regime*.
# Scattering counts as an interaction, albeit a weak one, so the scattered photons contain
# some information about the qubit's state. Indeed, the collision causes an exchange in
# momentum and energy. For the photon, this means that its frequency will change slightly.
# The qubit will simply acquire a negligible amount of kinetic energy. If we carefully measure
# the frequency of the scattered photons, we will distill the information about the state of
# the qubit and measure its state.
#
# Let us recall that Hamiltonians tell us how quantum states change in time due to external influences.
# For example, assuming that the cavity starts in the vacuum state :math:`\left\lvert 0 \right\rangle`,
# according to Schrodinger's equation,  the state evolves to :math:`\left\lvert \psi(t)\right\rangle=
# \exp(-i\hat{H}/\hbar)\left\lvert 0 \right\rangle` after a time :math:`t`. Suppose we shine an
# electromagnetic wave of amplitude :math:`\epsilon` and frequency within the dispersive regime on the cavity.
# The Hamiltonian that describes the field-qubit system inside it is
#
# .. math:: \hat{H}=\hbar(\omega_r I+\chi\hat{\sigma}_z)\otimes\hat{N} + \hbar\epsilon I\otimes \hat{P},
#
# where :math:`\hat{N}` counts the cavity number of photons, :math:`\hat{P}` is the photon momentum operator, and
# :math:`\epsilon` is the amplitude of the electromagnetic wave incident on the cavity. The shift :math:`\chi` is
# a quantity that depends on the circuit and gate capacitances and the detuning :math:`\Delta`.
#
# The effect of this evolution can be calculated explicitly. Driving the cavity with microwaves gives us a *coherent
# state* of light contained in it, which is the state of light that lasers give out.
# Coherent states are completely determined by their average position :math:`\bar{x}` and average momentum :math:`\bar{p}`,
# so we will denote them via :math:`\left\lvert \bar{x}, \bar{p}\right\rangle`. For the state of the qubit
# and cavity system, we write the ket in the form :math:`\left\lvert g \right\rangle \left\lvert \bar{x}, \bar{p}\right\rangle`.
# The Hamiltonian above has (approximately) the following effect:
#
# .. math:: \left\lvert g \right\rangle \left\lvert 0 \right\rangle \rightarrow \left\lvert g \right\rangle \left\lvert \epsilon t, (\omega_r+\chi)t \right\rangle
#
# .. math:: \left\lvert e \right\rangle \left\lvert 0 \right\rangle \rightarrow \left\lvert e \right\rangle \left\lvert \epsilon t, (\omega_r-\chi)t \right\rangle
#
# Consequently, if the state of the qubit were the superposition
# :math:`\alpha \left\lvert g \right\rangle +\beta \left\lvert e \right\rangle`, then the qubit-cavity system would evolve to the state
#
# .. math:: \left\lvert\psi(t)\right\rangle=\alpha \left\lvert g \right\rangle \left\lvert \epsilon t, (\omega_r+\chi)t \right\rangle +\beta \left\lvert e \right\rangle \left\lvert \epsilon t, (\omega_r-\chi)t \right\rangle.
#
# In general, this represents an entangled state between the qubit and the cavity. So if we measure the state of the light
# transmitted by the cavity, we are measuring the qubit's state as well.
#
# Let us see how this works in practice using PennyLane. The default.gaussian device allows us to work with an initial
# vacuum state of light. The PennyLane function qml.Displacement(x,0) applies a *displacement operator*, which creates
# a coherent state :math:`\left\lvert \bar{x}, 0\right\rangle`. The rotation operator qml.Rotation(phi) rotates the state
# :math:`\left\lvert \bar{x}, 0\right\rangle` in :math:`(\bar{x}, \bar{p})` space. When applied after a large displacement,
# it changes the value of :math:`\bar{x}` only slightly, but noticeably changes the value of :math:`\bar{p}` by shifting it
# off from zero, as shown in the figure:
#
# This sequence of operations implements the evolution of the cavity state exactly. Note that here we are
# taking :math:`\omega_r=0`, which simply corresponds to taking it as a reference frequency, so a rotation by
# angle :math:`\phi` actually means a rotation by :math:`\omega_r+\phi`. In PennyLane, the operations read:

import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

dev = qml.device("default.gaussian", wires=1, shots=50)
epsilon, chi = 1.0, 0.1


@qml.qnode(dev)
def measure_P_shots(time, state):
    qml.Displacement(epsilon * time, 0, wires=0)
    qml.Rotation((-1) ** state * chi * time, wires=0)
    return qml.sample(qml.P(0))


##############################################################################
#
# We measure the photon's momentum (its frequency) at the end since it allows us to distinguish qubit states
# as long as we can resolve them. If we plot for different durations of the microwave drive, we find, for
# 50 photons measuring qubits either in the ground or excited state:

N_meas = np.arange(1, 51)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Momentum measurement")
ax1.scatter(N_meas, measure_P_shots(1, 0))
ax1.scatter(N_meas, measure_P_shots(1, 1))
ax2.scatter(N_meas, measure_P_shots(3, 0))
ax2.scatter(N_meas, measure_P_shots(3, 1))
ax3.scatter(N_meas, measure_P_shots(5, 0))
ax3.scatter(N_meas, measure_P_shots(5, 1))
plt.show()

##############################################################################
#
# We see that the longer we shine microwaves on the cavity, the greater our ability
# to resolve the change of frequency, making for an equally good measurement of the qubit's
# state. However, this poses a problem: being a relatively large object, the state of the
# qubit is rather short-lived due to decoherence. Therefore, taking a long time to make
# the measurement introduces additional inaccuracies: the qubits may lose quantum
# properties due to decoherence before we even finish measuring! An outstanding
# challenge for superconducting qubit technologies is to perform measurements as fast
# as possible, in times well below the decoherence time.
#
# Superconducting single-qubit gates
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We have seen that shining light with frequency in the dispersive regime is used to perform
# measurements. It is precisely the opposite choice that allows us to manipulate the state
# of the qubit. But how can we make the photons get transmitted by the Fabry-Perot cavity
# if they are not at the frequency that the cavity can transmit? We must emphasize that
# the cavity reflects only the vast majority of photons, but not all of them. If we
# compensate by increasing the radiation intensity, some photons will still be transmitted
# into the cavity and be absorbed by the superconducting qubit.
#
# If we shine a coherent state light with frequency :math:`\omega_a` on the cavity, and with phase
# :math:`\phi` at the position of the qubit, then the Hamiltonian for the artificial atom is
#
# .. math:: \hat{H}=\hbar\Omega_R(\hat{\sigma}_{x}\cos\phi + \hat{\sigma}_{y}\sin\phi)
#
# Here, :math:`\Omega_R` is the Rabi frequency, which depends on the average electric field in the
# cavity and the size of the superconducting qubit. With this Hamiltonian, we can implement
# a universal set of single-qubit gates since :math: `\phi_d=0` implements an :math:`X`-rotation and :math:`\phi_d=\pi/2`
# applies a :math:`Y`-rotation. Let us check this using PennyLane. For qubits, we can define
# Hamiltonians using qml.Hamiltonian and evolve an initial state using ApproxTimeEvolution:

from pennylane.templates import ApproxTimeEvolution

dev2 = qml.device("default.qubit", wires=1)


@qml.qnode(dev2)
def H_evolve(state, phi, time):

    if state == 1:
        qml.PauliX(wires=0)

    coeffs = [np.cos(phi), np.sin(phi)]
    ops = [qml.PauliX(0), qml.PauliY(0)]
    Ham = qml.Hamiltonian(coeffs, ops)
    ApproxTimeEvolution(Ham, time, 1)
    return qml.state()


@qml.qnode(dev2)
def Sc_X_rot(state, phi):

    if state == 1:
        qml.PauliX(wires=0)

    qml.RX(phi, wires=0)
    return qml.state()


@qml.qnode(dev2)
def Sc_Y_rot(state, phi):

    if state == 1:
        qml.PauliX(wires=0)

    qml.RY(phi, wires=0)
    return qml.state()


print(np.isclose(Sc_X_rot(1, np.pi / 3), H_evolve(1, 0, np.pi / 6)))
print(np.isclose(Sc_X_rot(1, np.pi / 3), H_evolve(1, 0, np.pi / 6)))
print(np.isclose(Sc_Y_rot(0, np.pi / 3), H_evolve(0, np.pi / 2, np.pi / 6)))
print(np.isclose(Sc_Y_rot(1, np.pi / 3), H_evolve(1, np.pi / 2, np.pi / 6)))

##############################################################################
#
# Thus, for a particular choice of angle, we have verified that this Hamiltonian implements rotations around the X and Y axes.
# We can do this for any choice of angle, where we see that the time interval needed for a rotation by an angle
# :math: `\theta` is :math`t=2\theta/\Omega_R`. This time can be controlled simply by turning the source of microwaves on and off.
#
# The typical times in which a single-qubit gate is executed are in the order of the nanoseconds, making superconducting
# quantum computers the fastest ones out there. Why do they have this intrinsic advantage? The reason is that the
# Rabi frequency :math:`\Omega_R`, which grows with the size of the qubit and the magnitude of the electric field,
# is extremely high.  We have the technology to make very small cavities, which means we can pack strong
# electric fields in a small region. Moreover, superconducting qubits are billions of times larger than
# other qubits, such as atoms. The drawback is that superconducting qubit technology would be
# impossible without these extremely high speeds: large qubits are extremely short-lived, so we are
# constrained to find the quickest ways to perform all quantum operations.
#
# Two-qubit gates
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# One of the main challenges in all realizations of quantum computers is building two-qubit gates,
# needed to satisfy Di Vincenzo's fourth criterion in full. An advantage of superconducting technology
# is the variety of options to connect qubits with each other. For now, we will focus on *capacitative coupling*,
# which involves connecting the two superconducting qubits through a capacitor. In this case, the Hamiltonian
# for the system of two qubits reads
#
# .. math:: H=\hbar J (\sigma^{+}_1\sigma^{-}_2+\sigma^{-}_1\sigma^{+}_2),
#
# where the coupling :math:`J` depends on the coupling capacitance and the characteristics of both circuits. In the derivation
# of this Hamiltonian, we assumed that both qubits have the same energy difference between the ground and excited levels.
# The Hamiltonian allows us to implement the two-qubit :math:`iSWAP` gate
#
# .. math:: iSWAP = \left( \begin{array}{cccc} 1 & 0 & 0 & 0 \\ 0 & 0 & i & 0 \\ 0 & i & 0 & 0 \\ 0 & 0 & 0 & 1 \end{array} \right)
#
# when applied for a time :math`t=\pi/2J`, as shown with the following PennyLane code:
#
dev3 = qml.device("default.qubit", wires=2)

coeffs = [0.5, 0.5]
ops = [qml.PauliX(0) @ qml.PauliX(1), qml.PauliY(0) @ qml.PauliY(1)]

Two_qubit_H = qml.Hamiltonian(coeffs, ops)


@qml.qnode(dev3)
def Sc_ISWAP(basis_state, time):
    qml.templates.BasisStatePreparation(basis_state, wires=range(2))
    ApproxTimeEvolution(Two_qubit_H, time, 1)
    return qml.state()


@qml.qnode(dev3)
def iswap(basis_state):
    qml.templates.BasisStatePreparation(basis_state, wires=range(2))
    qml.ISWAP(wires=[0, 1])
    return qml.state()


print(np.isclose(Sc_ISWAP([0, 0], 3 * np.pi / 2), iswap([0, 0])))
print(np.isclose(Sc_ISWAP([0, 1], 3 * np.pi / 2), iswap([0, 1])))
print(np.isclose(Sc_ISWAP([1, 0], 3 * np.pi / 2), iswap([1, 0])))
print(np.isclose(Sc_ISWAP([1, 1], 3 * np.pi / 2), iswap([1, 1])))

##############################################################################
#
# To allow for universal computation, the two-qubit gate has to be equivalent to the well-known CNOT gate,
# up to single-qubit gates. The following quantum circuit illustrates how the two are related up to a
# global phase, and we can verify this using PennyLane:
#
@qml.qnode(dev3)
def cnot_with_iswap2(basis_state):
    qml.templates.BasisStatePreparation(basis_state, wires=range(2))
    qml.RZ(-np.pi/2,wires=0)
    qml.RX(np.pi/2,wires=1)
    qml.RZ(np.pi/2,wires=1)
    qml.ISWAP(wires=[0,1])
    qml.RX(np.pi/2,wires=0)
    qml.ISWAP(wires=[0,1])
    qml.RZ(np.pi/2,wires=1)
    
    return qml.state()
##############################################################################
#
# Note that capacitative coupling between qubits does not involve driving the cavity with a microwave.
# Instead, the Hamiltonian acts at all times between the connected circuits. We may wonder how to
# switch the interaction on and off since we need to tune its duration. We can switch off by
# changing the characteristics of one of the qubits since the Hamiltonian for capacitative coupling
# is only valid when both qubits have the same energy gap. For example, if we change the inductance
# in one of the circuits to be much different from the other, the interaction strength :math:`J` will go to zero.
#
# How can we change the characteristics of the circuit elements on-demand? One possibility is to use
# **flux-tunable qubits**, also known as superconducting quantum interference devices (SQUID).
# They use two parallel Josephson junctions in each circuit, a setup that allows us to alter
# the effective inductance of the qubit by controlling an external magnetic field. This architecture,
# although faster, requires further interaction with the qubit and decreases coherence times.
#
# Another option is to use **all-microwave gates**. In this scenario, two qubits placed in a single
# cavity are both driven at the frequency of the second qubit. The first qubit will scatter the
# photons, and the other will absorb them, causing a similar effect to that of the qubit-cavity
# system in the case of dispersive measurement. As a consequence, we can entangle the two qubits.
# Although this method increases the coherence time of the qubit, these gates turn out to be
# slower than those built using flux-tuning since the Rabi frequency for this interaction
# turns out to be smaller.
#
# The state of the art
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Superconducting qubits are a force to be reckoned with as practical implementations of quantum computers.
# Nonetheless, there are still some technological challenges preventing them from scaling further.
# Since artificial atoms are pretty large compared to other physical qubits, it is not surprising
# that the short coherence time of the qubit is the main hurdle. Therefore, the main obstacle is
# Di Vincenzo's third criterion: the quantum operations are still too slow for the coherence time of the qubit.
#
# The coherence times of superconducting qubits are in the order of microseconds. Single-qubit gates are
# acceptable since they can be applied in a matter of nanoseconds, thanks to the large Rabi frequencies.
# Problems arise, however, when we want to perform precise measurements of the qubit. As discussed above,
# the precision with which we can distinguish the ground and excited states is proportional to the duration
# of the cavity driving. Moreover, the dispersive Hamiltonian is only valid when the number of photons
# in the cavity does not exceed some critical number :math:`\bar{n}_{crit}` that depends on the characteristics
# of the cavity and the circuit. This upper bound on the duration sets additional constraints:
# we will need longer driving times to resolve the state since we have few photons. These times need
# to be shorter than the average lifetime of a qubit excited state.
#
# We see that there is a very delicate interplay between the duration of the measurement and its accuracy.
# Therefore, speeding up measurements without losing precision is a hot research topic in superconductor
# quantum computing. The primary approach has been to make the most out of the limited number of
# photons we have by reducing all possible environmental noise. Currently, we can perform
# measurements in about 50 nanoseconds with about 1% error. However, since frequent measurements
# are needed for error correction, better precision and shorter times are required
# to scale our architectures further.
#
# An additional challenge is the execution of the multi-qubit gates, which are currently ten times
# slower than single-qubit gates. One contender to the fastest two-qubit gate is used by Google,
# where the qubits are coupled with a capacitor and a SQUID. The net effect is a quick flux-tunable
# qubit, where the coupling :math:`J` is changed by using a magnetic flux that goes through the coupling SQUID.
# While slower, IBM prefers to use all-microwave gates to increase coherence time. Both approaches have
# the problem of *frequency crowding*: if we interconnect too many qubits together inside a cavity, there
# may start to have similar resonance frequencies. In that case, we may manipulate qubits that we did
# not intend to. The problem can be somewhat addressed by changing the geometry in which the qubits
# are placed are connected. For example, IBM preferred a hexagonal topology in their most recent
# Eagle quantum computer, leading to ground-breaking results. However, much more work needs to
# be done to address this scalability issue.
#
# Conclusing remarks
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#
# Superconducting quantum computing has gained momentum in the last decade as a leading competitor
# in the race for building a functional quantum computer. It is but an adaptation of microchip
# technology made to work in the superconducting regime, which allows for versatility and control.
# They are also easy to scale: although we currently need to be faster, multi-qubit gates are
# straightforward to implement. Therefore, increasing the qubit coherence time and the quantum
# operations are essential to scaling this technology for quantum computing. There are so many approaches
# to perform measurements and multi-qubit gates that we could not possibly cover them all in one demo!
# And this is an excellent point in favour of superconducting qubits: they are custom-made, so we can
# adapt them to our own advances in engineering. It is not surprising that they are in the news often,
# so keep an eye out for new and exciting developments!
#
#
