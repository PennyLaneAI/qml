r""".. _trapped_ions:

Superconducting qubits
=============================

.. meta::
    :property="og:description": Description and assessment of trapped ion quantum computers
    :property="og:image": https://pennylane.ai/qml/_images/sc_qubits.png

.. related::
   tutorial_pasqal Quantum computation with neutral atoms

*Author: PennyLane dev team. Posted: XX November 2021. Last updated: XX November 2021.*

Quantum computers are developing at a much faster rate than many physicists could have predicted.
However, the claim that near-term quantum computers will be used for everyday industrial 
applications is debatable. Realizing this vision will take a bit more time and a continuous 
joint effort between the scientific and engineering communities across the world. The main 
issue is scalability: the ability to control and measure many qubits without destroying
their quantum properties. The good news is that we have a plethora of approaches 
to solving this problem. We can try making quantum computers by trapping ions, 
creating artificial atoms using superconductors, using groups of photons to encode 
photons, among many other options. Each of the techniques seems promising on its own, 
but they also present unique engineering challenges.

This article is an introduction to **superconducting qubits** as an approach to 
building quantum computers. This technology is being used by renowned companies 
such as Google and IBM in their quest to pioneer the quantum era. Indeed, 
superconducting qubits are pretty promising. Google's Sycamore claimed quantum advantage 
back in 2019, and IBM recently built its Eagle quantum computer with 127 qubits, 
twice as many as its main competitors! The central insight that allows for these 
quantum computers is that superconductivity is a quantum phenomenon, so we can use 
superconducting circuits as quantum systems that we can control at will. 
They are nothing but a modification of current microchip technology adapted 
to work with superconductors, so we have most of the infrastructure in place! 
However, the large size of the qubit makes it prone to decoherence, making 
it more short-lived than other types of qubits. Nevertheless, we can get 
around this, and the results speak for themselves.

By the end of this demo, you will learn how superconductors are used to create, 
prepare, control, and measure the state of a qubit. Moreover, you will identify 
the strengths and weaknesses of this technology in terms of Di Vincenzo's criteria, as 
introduced in the box below. You will be armed with the basic concepts to understand 
the main scientific papers on the topics and to keep up-to-date with the new developments 
that are bound to come soon.

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
# To introduce superconductivity, we need to understand why some materials resist the passage 
# of electrons at the microscopic level. The electrons that flow through a conductor are 
# known as *conduction electrons*, and they do not form part of any atoms. The range of 
# energies that conduction electrons can have is called the *conduction band*. 
# These energy values are also discrete in analogy to atomic orbitals: the conduction
# band has energy levels. Electrons are *fermions*, which means that there is a
# limited number of electrons that can occupy the same energy level. Therefore,
# to conduct more electrons, we need to provide more energy. This is, roughly, where
# Ohm's Law comes from. The more voltage one feeds a circuit, the larger the current
# flow will be.  
#
# The electrons that are bound to an atomic nucleus are called *valence electrons*. 
# Their energy takes values in the *valence band*, which is lower than the conduction band. 
# A material is a good conductor of electricity if a valence electron does not need too 
# much energy to become a conduction electron. In other words, the *energy bandgap* 
# between the valence and conduction band is small. In conductors, this gap is, 
# in fact, zero. However, even in conductors, the electrons rejected by the 
# filled conduction band levels may scatter and collide with the atoms, dissipating energy. 
# The higher the temperature, the more likely these collisions are since the atoms move 
# around more. Indeed, we observe that conductivity decreases at high temperatures.
#
# For some materials, at extremely low temperatures, something somewhat counterintuitive occurs. 
# Conduction electrons start attracting each other and form pairs. This phenomenon is 
# strange since we know that electrons are supposed to repel each other. However, 
# as the motional energy of the atoms in the conductor decreases, the conduction 
# electrons can attract the positive nuclei, which in turn attract other electrons. 
# The net effect is a coupling of electrons in *Cooper pairs*, which behave very differently 
# to electrons. They are not fermions; instead, they are *bosons*, which means that any 
# number of them can have the same energy. Therefore, they can all be in the lowest 
# energy state of the conduction band. They will not reject any other Cooper pairs, 
# and consequently, they will not scatter into the atoms and dissipate energy. 
# Cooper pairs flow through the material without any resistance. They are the reason 
# that we have superconductors!
#
# Quantum Circuits
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The common lore is that quantum mechanics describes the tiniest of objects, 
# such as atoms and particles. When we look at these small scales, unusual 
# properties such as superposition and entanglement are the norm. What distinguishes
# these small objects from the larger ones? Is there some length scale below 
# which things behave in a quantum manner? Or can we observe quantum properties 
# in large objects provided we isolate them enough? The latter seems to be the case. 
# Granted, many large objects, such as lasers, or nuclear reactors, would not work 
# if not for the quantum properties of the matter they contain. But can we directly 
# observe quantum phenomena in large objects? This is indeed possible. One of the 
# most well-known examples is the Bose-Einstein condensate, a low-temperature phase
# of matter that exhibits quantum properties. Another example is, you guessed it, 
# superconductivity, which is also a phase at low temperatures! 
# The quantum character of superconductivity allows us to manufacture relatively 
# large qubits tailored to our needs.
#
# More precisely, for a system to be quantum, it is neither necessary nor 
# sufficient for it to be small. It is still an open question in physics to 
# come up with sufficient conditions for quantum behaviour. In the case of a 
# superconducting circuit, we have observed that it must be **isolated from 
# an environment we cannot control** for it to exhibit quantum properties. 
# To achieve this, we need to satisfy two conditions. First, we have to 
# make sure that we are not putting too much energy into the environment, 
# compared to the energy they store. An energy leak carries information 
# about the system and counts as a partial measurement, thus destroying
# the quantum properties. Superconductors are great in this regard: since 
# there is no resistance, there will be no energy output. But this is 
# not enough, as we can observe by turning on our microwave. This appliance 
# stores a lot of energy and does not dissipate it into our kitchen. What 
# makes a microwave non-quantum? Here is where the second condition kicks in.
# When a system is at a high temperature, particles in the environment are 
# constantly interacting with it, causing decoherence. For a superconducting 
# circuit to preserve its quantum properties for a long time, we need to 
# cool it to about 10 mK, well below the temperature where the 
# superconducting phase is achieved. 
#
# Let us study the quantum version of a simple circuit, known as an LC circuit 
# or *superconducting resonator*, which turns out to be a physical realization 
# of the quantum harmonic oscillator. The LC circuit has two components. 
# The first one is a capacitor :math:`C`, which consists of two parallel metallic plates
# with equal and opposite charges. The second one is an inductor :math:`L`,
# which is connected in between the capacitor plates. When a varying
# current goes through the inductor, it exerts a force on the charges opposite
# to their motion. But the attraction between the plates is proportional
# to the charge contained in each of them, so the net effect of both components
# is that the more charge a capacitor has, the more the flow of charges is
# resisted by the inductor. This is the same as a spring: the more one stretches
# a spring, the more it resists to being stretched. Since its behaviour is similar
# to that of a prototypical harmonic oscillator, this circuit is
# indeed a harmonic oscillator as well. 
# 
# Since the superconducting circuit is quantum, it will behave as a quantum harmonic oscillator, 
# which means that the possible energy values are given by
#
# .. math:: E = n\hbar\omega.
# 
# where :math:`n`is a natural number, :math:`\hbar` is Planck's constant, and :math:`\omega` is the resonant 
# frequency of the circuit, which depends on the physical characteristics of the inductor 
# and the capacitor. When a circuit has energy :math:`\hbar\omega`, we denote its state 
# as $\left\lvert n \right\rangle$, known as a *Fock state*. These discrete energy values 
# in the circuit can be observed in the simple LC superconducting circuit. 
#
# Josephson junctions and artificial atoms
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Therefore, unlike atoms, where the energy levels are unevenly spaced, superconducting 
# resonators cannot serve as a qubit. If we want to get around this issue, we should 
# find a way to change the energy level differences in our circuit. The reason that the 
# energy levels are evenly spaced boils down to the fact that the resistive force exerted
# by an inductor is proportional to the change in the current going through it. If this 
# relation between current and force weren't that simple, the energy levels could be 
# spaced unevenly. Therefore what we need is a different type of inductor with a more 
# complicated *non-linear behaviour*.
#
# There is in fact a ciruit element that does precisely this: a *Josephson junction*.
# This element consists of a very thin piece of an insulating placed between two 
# superconducting metals. At first thought, the Josephson junction should play the 
# role of a resistor, and since we have superconducting metals on either side of it, 
# no current should go through it. Here is where *tunneling*, another famous quantum 
# effect, comes into play. Because of the spread-out nature of their wave function,
# Cooper pairs can actually go through the Josephson junction with some probability. 
# The net effect is that the junction acts as a nonlinear inductor, introducing 
# anharmonicity in the energy levels.
#
# Let us study the variation in energy levels in more detail. In quantum mechanics, 
# systems are described by Hamiltonians, which play a double role. Firstly, they allow 
# us to calculate the allowed energies. Secondly, they tell us how an initial quantum 
# state will change in time. Deriving the Hamiltonian is a crucial step when we want to 
# study the properties of quantum systems, and it is generally hard to do. For the quantum 
# harmonic oscillator, it turns out to be of the form (this will be put in a box)
#
# .. math:: \hat{H}= a\hat{X}^2 + b\hat{P}^2
#
# where :math:`\hat{X}` and :math:`\hat{P}`` are (infinite dimensional) matrices that satisfy 
# :math:`\hat{X}\hat{P}-\hat{P}\hat{X}=i\hbar I`. The physical significance of these
# matrices depends on the system in question. For waves in a solid, :math:`\hat{X}` 
# represent the position of a particle, while :math:`\hat{P}` represents its momentum. 
# In the case of a superconducting resonator, :math:`\hat{X}` stands for the magnetic 
# flux through the inductor and is denoted by :math:`\hat{\phi}`, and $:math:`\hat{P}` corresponds 
# to the charge in the capacitor, and is denoted by :math:`\hat{Q}` . The Hamiltonian for 
# the resonator reads
#
# .. math:: \hat{H}_{res}=4 E_C (\hat{Q}-Q_g I)^2+4E_J \cos(\hat{\phi}).
#
# where blah blah. It is the last term that describes the energy contained in the 
# Josephson junction. Finding the eigenvalues of :math:`\hat{H_{sc}}`, we can find 
# what the allowed energy level are. They turn out to depend on the value of :math:`Q_g`. 
# The dependence of the energy levels with :math:`Q_g` looks as follows: **(draw plot)**. 
# for different values of :math:`E_J/E_C`. We see that if this quotient is too large, 
# then the anharmonicity is lost, but if it's too low, then small variations in :math:`Q_g` 
# can affect the energy levels. THe idea is to find a goldilocks zone, where 
# these variations are do not affect the energy differences too much, so the 
# value of :math:`Q_g` doesn't matter much, but we can also still resolve the energy differences. 
# With our current experimental bandwidths available, this occurs for :math:`E_J/E_C \sim 80`, 
# the **transmon regime**. 
#
# Homodyne measurement
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Here I still need to write this works. The idea is that the cavity+superconductor 
# circuit is driven to create a coherent state, which is then rotated in phase space.
# This drive is done away from the resonance frequency of the cavity. The angle of 
# this rotation depends on the state of the qubit, so the coherent state is entangled 
# with it. Here is the circuit that implements this:
# 
import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt 

dev = qml.device("default.gaussian", wires=1, shots=50)

@qml.qnode(dev)
def measure_X_shots(alpha, phi):
    qml.Displacement(alpha, 0, wires=0)
    qml.Rotation(phi, wires=0)
    return qml.sample(qml.X(0))

@qml.qnode(dev)
def measure_P_shots(alpha, phi):
    qml.Displacement(alpha, 0, wires=0)
    qml.Rotation(phi, wires=0)
    return qml.sample(qml.P(0))

##############################################################################
# The larger alpha, the better we can resolve our quantum states when measuring. 
# However, alpha is proportional to time, so there is decoherence badness. 
# phi is the angle of rotation in phase space, positive for ground qubit state, 
# and negative for excited qubit state. 

X1=measure_X_shots(3,0.1)
X2=measure_X_shots(3,-0.1)
P1=measure_P_shots(3,0.1)
P2=measure_P_shots(3,-0.1)

plt.scatter(X1,P1)
plt.scatter(X2,P2)

##############################################################################

X1=measure_X_shots(6,0.1)
X2=measure_X_shots(6,-0.1)
P1=measure_P_shots(6,0.1)
P2=measure_P_shots(6,-0.1)

plt.scatter(X1,P1)
plt.scatter(X2,P2)

##############################################################################

X1=measure_X_shots(12,0.1)
X2=measure_X_shots(12,-0.1)
P1=measure_P_shots(12,0.1)
P2=measure_P_shots(12,-0.1)

plt.scatter(X1,P1)
plt.scatter(X2,P2)

##############################################################################
# 
# Vertical axis is momentum. So if we measure momentum, 
# we can resolve given that we drive for a long enough time. 





