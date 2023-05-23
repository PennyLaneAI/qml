r""".. _neutral:

Neutral-atom quantum computers
=============================

.. meta::
    :property="og:description": Learn how neutral atom quantum devices work using code
    :property="og:image": https://pennylane.ai/qml/_images/thumbnail_neutral_atoms.png

.. related::
   tutorial_pasqal Quantum computation with neutral atoms
   tutorial_trapped_ions Trapped ion quantum computing
   tutorial_sc_qubits Quantum computing with superconducting qubits
   tutorial_photonics Photonic quantum computing

*Author: Alvaro Ballon — Posted: 30 May 2023.*

In the last few years, a new quantum technology has gained the attention of the quantum computing
community. Thanks to recent developments in optical-tweezer technology,
neutral atoms can be used as robust and versatile qubits. In 2020, a collaboration between
various academic institutions produced a neutral-atom device with a whooping 256 qubits 😲! It is 
no surprise that this family of devices has gained traction in the private sector, with startups such 
as Pasqal, QuEra, and Atom Computing suddenly finding themselves in the headlines. 

In this tutorial, we will explore the inner workings of neutral-atom quantum devices. We will also 
discuss their strengths and weaknesses in terms of Di Vincenzo's criteria, introduced in the blue box below.
By the end of this tutorial, you will have obtained a high-level understanding of neutral atom technologies
and be able to follow the new exciting developments that are bound to come.

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

We will start by explaining how neutral atoms can be manipulated and isolated enough to be used as qubits.
Then, we will use PennyLane's pulse programming capabilities to understand how to apply single and multi-qubit gates.
Afterwards, we will learn how to perform measurements on the atom's states. Finally, we will explore
the work that still needs to be done to scale this technology even further.

"""

##############################################################################
#
# Trapping individual atoms
# -------------------------
#
# In our cousin demo about trapped-ion technologies, we learned that we can trap individual charged
# atoms by carefully controlled electric fields. But neutral atoms, by definition, have no charge, 
# so they can't be affected by electric fields. How can we even hope to manipulate them individually?
# It turns out that the technology to do this has been around for decades. 
# **Optical tweezers**—highly focused laser beams—can grab small objects and hold them in place, no 
# need to charge them! Let's see how they are able to do this.
#
# Laser beams are nothing but electromagnetic waves, that is, oscillating electric and magnetic
# fields. It would seem that a neutral atom could not be affected by them—but it can! To understand how, we need
# to keep in mind two facts. First, in a laser beam, light is more intense at the center of the beam 
# and it dims progressively as we go toward the edges. This means that the average
# strength of the electric fields is higher closer to the center of the beam. 
# Secondly, as small as neutral atoms are, they're not just
# points. They do carry charges that can move around relative to each other when we expose them to electric fields.
#
# The consequence of these two observations is that, if an atom inside a laser beam tries to escape toward the edge
# of the beam, the negative charges will be pulled toward the center of the beam, while the positive charges are pushed
# away. But, since the electric fields are stronger toward the center, the
# negative charges are pulled harder, so more will accumulate in the center. 
# These negative charges will bring the positive charge that's trying to pull away back to the middle. You can 
# look at the figure below to gain a bit more intuition.
#
# .. figure:: ../demonstrations/neutral_atoms/force_gradient.png
#    :align: center
#    :width: 25%
#
#    ..
# 
# In the last decade, optical tweezer technology has evolved to the point where we can move atoms around
# into customizable arrays (check out :doc:`this tutorial </demos/tutorial_pascal>` and have some fun doing this!).
# This means that we have a lot of freedom in how and when our atom-encoded qubits interact with each other. Sounds 
# like a dream come true! However, there *are* some big challenges to address—we'll learn about these later.
# To get started, let's understand how neutral atoms can be used as qubits.
#
# Encoding a qubit in an atom
# ---------------------------
#
# To encode a qubit in a neutral atom, we need to have access to two distinct atomic quantum states. The most 
# easily accessible quantum states in an atom are the electronic energy states.  We would like to **switch
# one electron between two different energy states**, which means that we must make sure not to affect other
# electrons when we manipulate the atom. For this reason, the ideal atoms to work with are those with 
# one valence electron, i.e. one "loose" electron that is not too tightly bound to the nucleus. 
# 
# .. note::
#
#    In some cases, such as the devices built by Atom Computing,
#    qubits are not encoded in atomic energy levels, but in so-called nuclear-spin
#    energy levels instead. Such qubits are known as **nuclear spin qubits**. In this demo, we will not focus
#    on the physics of these qubits. However, similar principles to those we'll outline in this demo
#    for qubit preparation, control, and measurement will apply to this type of qubit.
# 
# 
# A common choice is the Rubidium-85
# atom, given that it's commonly used in atomic physics and we have the appropriate technology to change its
# energy state using lasers. If you need a refresher on how we change the electronic energy levels of atoms, do take
# a look at the blue box below!
#
# .. container:: alert alert-block alert-info
#
#    **Atomic Physics Primer:** Atoms consist of a positively charged nucleus
#    and negative electrons around them. The electrons inhabit energy
#    levels, which have a population limit. As the levels fill up, the
#    electrons occupy higher and higher electronic energy states, or
#    energy levels. But as long as
#    there is space, electrons can change energy levels, with a preference
#    for the lower ones. This can happen spontaneously or due to external
#    influences.
#
#    When the lower energy levels are not occupied, the higher energy levels
#    are unstable: electrons will prefer to minimize their energy and jump to
#    a lower level on their own. What happens when an electron jumps from
#    a high energy level to a lower one? Conservation of energy tells us
#    that the energy must go somewhere. Indeed, a photon with an energy
#    equal to the energy lost by the electron is emitted. This energy is
#    proportional to the frequency (colour) of the photon.
#
#    Conversely, we can use laser light to induce the opposite process.
#    When an electron is in a stable or ground state, we can use lasers
#    with their frequency set roughly to the difference
#    in energy levels, or energy gap, between the ground state and an
#    excited state. If a photon hits an electron, it will go to that
#    higher energy state. When the light stimulus is removed, the excited
#    electrons will return to stable states. The time it takes them to do
#    so depends on the particular excited state they are in since,
#    sometimes, the laws of physics will make it harder for electrons to
#    jump back on their own.
#
#    .. figure:: ../demonstrations/neutral_atoms/atomic.png
#       :align: center
#       :width: 60%
#
#       ..
#
# But even if we've chosen one electron in the atom, we need to make sure that we are effectively
# working with only two energy levels in that atom. This ensures that we have a qubit!
# One of the energy levels will be a ground state
# for the valence electron, which we call the *fiducial state* and denote by :math:`\lvert 0 \rangle.` 
# The other energy level will be an long-lived excited state, known as a hyperfine state, denoted by :math:`\lvert 1 \rangle.`  
# We'll induce transitions between these two states using light whose energy matches the energy difference between
# these atomic levels.
#
#
# Initializing the qubits
# -----------------------
#
# We have chosen our atom and its energy levels, so the easy part is over! But there are still some difficult tasks
# ahead of us. In particular, we need to isolate individual atoms inside our optical 
# tweezers *and* make sure that they are all in the **fiducial
# ground state,** as required by di Vincenzo's second criterion. This fiducial state
# is stable, since minimal-energy states will not spontaneously emit any energy.
# 
# The first step to initialize the qubits is to cool down a cloud of atoms in a way that 
# all of their electrons end up in the same state. There are many states of minimum energy,
# so we need to be careful that all electrons are in the same one! For Rubidium atoms, we
# use a technique known as **laser cooling**. It involves putting the atoms in a magnetic trap within a
# vacuum chamber and then
# using lasers both to freeze them in place and make sure all the electrons are in the same stable state. 
#
# To understand how neutral atoms can be used to build a quantum device, 
# let's figure out how all the electrons end up in the same energy state. It turns out that Rubidium-85 is 
# the ideal atom not only because it has one valence electron, but also because it has a **closed optical loop.** 
#
# .. figure:: ../demonstrations/neutral_atoms/closed_loop.png
#    :align: center
#    :width: 60%
#
#    ..
#
# Rubidium-85 has two ground states :math:`\vert 0\rangle` and :math:`\vert \bar{0}\rangle`, which are excited 
# using the laser to two excited states :math:`\vert 1\rangle` and :math:`\vert \bar{1}\rangle` respectively. However, 
# both of these excited states will decay to :math:`\vert 0\rangle` with high probability. This means that no
# matter what ground state the electrons occupied initially, they will most likely be driven to the same ground state
# through the laser cooling method. 
#
# Great! We have our cloud of atoms all frozen and in the same ground state. But now we need to pick out
# single atoms and arrange them in nice ways. Here's where we use our optical tweezers. The width of the laser
# can be focused enough so that we are sure that at most one atom gets trapped. Moreover, one laser beam
# can be split into a variety of arrays of beams through a spatial light modulator, allowing us to rearrange
# the positions of the atoms in many ways. With our atoms in position and in the fiducial ground state, we're
# ready to do some quantum operations on them!  
#
# Measuring an electronic state
# ------------------------------
# 
# Now that our fiducial state is prepared, let's focus on another essential part of a quantum computation: 
# measuring the state of our system. But wait... isn't measurement the last step in a quantum circuit?
# Aren't we skipping ahead a little bit? Not really! Once we have our initial state, we should measure it to 
# verify that we have indeed prepared the correct state. After all, some of the steps we carried out to
# prepare the atoms aren't really foolproof; there are two issues that we need to address.
# 
# The first problem is that traps are designed to trap *at most* one atom. This means that some traps might contain
# **no** atoms! Indeed, in the lab, it's usually the case that half of the traps aren't filled. The second issue is
# that laser cooling is not deterministic, which means that some atoms may not be in the ground state. We would like
# to exclude those from our initial state. Happily, there is a simple solution that addresses these two problems.
#
# To verify that a neutral atom is in the fiducial state :math:`\lvert 0 \rangle`, we shine a photon on it that stimulates 
# the transition between this state to some short-lived excited state :math:`\lvert h \rangle`. Electrons excited in this way will 
# promptly decay to the state :math:`\lvert 0 \rangle` again, emitting light. The electrons that were in some state different than
# :math:`\lvert 0 \rangle`, never get excited, since the photon does not have the right energy. And, of course, nothing will happen
# in traps where there is no atom. The net result is that atoms in the ground state will shine, while others won't. This
# phenomenon, known as fluorescence, is also used in trapped ion technologies. The same method can be used at the end of 
# a quantum computation to measure the final state of the atoms in the computational basis.
#
# .. figure:: ../demonstrations/neutral_atoms/fluorescence.png
#    :align: center
#    :width: 60%
#
#    ..
#
# Neutral atoms and light
# -----------------------
#
# We want to carry out computations using the electronic energy levels of the neutral atoms, which means that we
# need to be able to control their quantum state. To do so, we need to act on it with a *light pulse*—a 
# short burst of light whose amplitude and phase are carefully controlled over time. To predict exactly how
# pulses affect the quantum states, we need to write down the *Hamiltonian* of the system.
# 
# .. note::
#
#    Recall that the Hamiltonian :math:`H` is the observable for the energy of the system, but it also describes 
#    how the system's quantum state evolves in time. If a system's initial state is :math:`\vert \psi(0)\rangle,`
#    then after a time interval :math:`t,` the state is 
#    
#    .. math::
#       
#       \vert \psi(t)\rangle = \mathcal{T}\left\{ exp\left(-i\int_{0}^{t}H(t)dt\right) \right\}\vert \psi(0)\rangle.
#    
#    where :math:`\mathcal{T}` represents time ordering and we generally allow the Hamiltonian to be time-dependent.
#    In general, this is not easy to calculate. But `qml.evolve` comes to our rescue, since it will calculate 
#    :math:`\vert \psi(t)\rangle` for us using some very clever approximations. 
# 
# 
# When a pulse of light of frequency :math:`\nu(t),` amplitude :math:`\Omega(t)/2\pi` and phase :math:`\phi`  is shone 
# upon *all* the atoms in our array, the *Hamiltonian* describing this interaction turns out to be
#
# .. math::
#    
#    \mathcal{H}_d = \Omega(t)\sum_{q\in\text{wires}}(\cos(\phi)\sigma_{q}^x-\sin(\phi)\sigma_{q}^y) - \frac{1}{2}\delta(t)\sum_{q\in\text{wires}}(\mathbb{I}_q -\sigma_{q}^z).
# 
# Here, The **detuning** :math:`\delta(t)` is defined as the difference between the photon's energy and the energy 
# needed to transition between the ground state :math:`\lvert 0 \rangle` and the excited state 
# :math:`\lvert 1 \rangle.` 
#  
# We will call :math:`\mathcal{H}_d` the **drive Hamiltonian**, since the electronic states of the atoms are being 
# "driven" by the light pulse. This Hamiltonian is time-dependent, and it may also depend 
# on other parameters that describe the pulse. PennyLane's
# `ParametrizedHamiltonian` class will help us deal with such a mathematical object. 
# You can learn more about Parametrized Hamiltonians in our documentation and in this Pulse Programming demo.
# 
# Driving excitations with pulses
# -------------------------------   
# 
# The mathematical expression of the Hamiltonian tells us that the time evolution depends on
# the shape of the pulse, which we can control pretty much arbitrarily as long as it's finite in duration. 
# A sinusoidal pulse would be ideal, since it's a pure frequency. But in real life, pulses do need to start and
# die off, which introduces foreign frequencies into the pulse's spectrum.
# It turns out that one of the best choices is the *Blackman window* pulse, since its effect is 
# pretty close to that of a pure frequency. The amplitude of a Blackman pulse of duration :math:`T` is given by
#
# .. math::
#
#    \frac{\Omega(t)}{2\pi} = \left\{\begin{array}{lr} \left(\frac{1-\alpha}{2}\right)A-\frac{A}{2}\cos\left(\frac{2\pi t}{T}\right)+\frac{\alpha A}{2}\cos\left(\frac{4\pi t}{T}\right), & \text{if } 0 \leq t \leq T \\ 0  & \text{otherwise.} \end{array}\right.    
#
# Here, :math:`A` is the peak amplitude, which we will treat as an adjustable parameter. A
# standard choice is to fix :math:`\alpha = 0.16;` which we will use in this demo. We will also set :math:`T=0.2,` although
# this can be easily changed in programmable devices.
# Let's plot this function to get an idea of what the pulse looks like. First, let's import all the relevant libraries.

import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
import jax
from jax import numpy as jnp # Needed for pulse programming
jax.config.update('jax_platform_name', 'cpu') # Tell jax to use CPU by default

##############################################################################
#
# Now, let's define the `blackman_window` function and plot it.

duration = 0.2 # We'll set all of our pulses' duration to 0.2

def blackman_window(peak, time):
    
    blackman = peak*0.42 - 1/2*peak*jnp.cos(2*jnp.pi*time/duration) + peak*0.08*jnp.cos(4*jnp.pi*time/duration)

    return blackman

t_points = np.linspace(0,0.2,100)
y_points = [blackman_window(1,t) for t in t_points]

plt.xlabel('Time', fontsize=10)
plt.ylabel('Amplitude', fontsize=10)

plt.title('Blackman Window Pulse (duration = 0.2)')
plt.plot(t_points, y_points, c = '#66c4ed')
plt.show()
##############################################################################
#
# We will stick to using Blackman window pulses for the rest of this tutorial. 
# 
# Let us explore how an interaction with this pulse changes the quantum state. 
# The drive Hamiltonian is already coded for us in PennyLane. For conciseness
# let's just import it and call it `H_d.` Then, we can use `qml.evolve` 
# to calculate how an initial state interacting with 
# a pulse evolves in time. First, let's assume that the detuning :math:`\delta` is zero.

from pennylane.pulse import rydberg_drive as H_d

# Choose some arbitrary parameters
peak = 2
phase = np.pi/2
detuning = 0 

# For now, let's act on only one neutral atom
single_qubit_dev = qml.device('default.qubit.jax', wires = 1)

@qml.qnode(single_qubit_dev)
def state_evolution():
    
    # Use qml.evolve to find the final state of the atom after interacting with the pulse  
    qml.evolve(H_d(blackman_window,phase,detuning, wires=[0]))([peak], t = [0,duration])
    
    return qml.state()

print("The final state is {}".format(state_evolution()))
##############################################################################
#
# We see that the electronic state changes indeed. As a sanity-check, let's see what happens when the detuning is non-zero.

# Choose some arbitrary parameters

peak = 2
phase = np.pi/2
detuning = 100 # Some large detuning to prove the point

@qml.qnode(single_qubit_dev)
def state_evolution_detuned():
    
    # Use qml.evolve to find the final state of the atom after interacting with the pulse  
    qml.evolve(H_d(blackman_window,phase,detuning, wires=[0]))([peak], t = [0,duration])
    
    return qml.state()

print("The final state is {}, which is the initial state!".format(state_evolution_detuned().round(2)))
##############################################################################
#
# All works as expected!
#
# Single-qubit gates 
# ------------------
# 
# Note that, so far, we have paid no mind to the values for the peak amplitude
# nor the phase—we just chose some arbitrary values. But we can actually adjust these values 
# to create some well-known quantum gates. That's the magic of pulse programming!
# Let's see how to properly choose these values.
#
# When the detuning is zero and the pulse acts only on one qubit, Schrodinger's equation tells us we can 
# write the evolved state for one qubit as  
#
# .. math::
#       
#    \vert \psi(T)\rangle = exp\left(-i\int_{0}^{T}\Omega(t)(\cos(\phi)\sigma^x-\sin(\phi)\sigma^y)dt\right)\vert \psi(0)\rangle.
#
# For a fixed value of the phase :math:`\phi,` the evolution depends only on the integral of :math:`\Omega(t)` over the
# duration of the pulse :math:`T.` The integral can be calculated exactly for our Blackman window, in terms of the peak amplitude:
#
# .. math::
#       
#    \frac{1}{2\pi}\int_{0}^{T}\Omega(t)dt = \left(\frac{1-\alpha}{2}\right)A\times T = 0.42*0.2*A.
#  
# For example, for :math:`\phi = 0`, the evolved state is of the form :math:`\vert\psi(t)\rangle = e^{-i\theta \sigma^x },`
# with :math:`\theta = \int_{0}^{T}\Omega(t).` This is none other than the rotation gate :math:`RX(\theta).` Therefore, if 
# we want to implement a rotation by an angle :math:`\theta,` it suffices to use a Blackman pulse with peak amplitude 
#
# .. math::
#       
#    A = \frac{\theta}{2\pi\times 0.42 \times 0.2}. 
#
# We can program the pulse easily using PennyLane, and verify that it gives us the correct result. 

def neutral_atom_RX(theta):
    
    peak = theta/duration/0.42/(2*jnp.pi) # Recall that duration is 0.2
    
    # Set phase and detuning equal to zero for RX gate
    qml.evolve(H_d(blackman_window,0,0, wires=[0]))([peak], t = [0,duration])

print("For theta = pi/2, the matrix for the pulse-based RX gate is \n {} \n".format(qml.matrix(neutral_atom_RX)(jnp.pi/2).round(2)))
print("The matrix for the exact RX(pi/2) gate is \n {}".format(qml.matrix(qml.RX)(jnp.pi/2,wires = 0).round(2)))
##############################################################################
#
# A similar argument can be made for :math:`RY` rotations, with the only difference being that :math:`\phi = -\pi/2.`

def neutral_atom_RY(theta):
    
    peak = theta/duration/0.42/(2*jnp.pi) # Recall that duration is 0.2
    
    # Set phase equal to pi/2 and detuning equal to zero for RY gate
    qml.evolve(H_d(blackman_window,-jnp.pi/2,0, wires=[0]))([peak], t = [0,duration])

print("For theta = pi/2, the matrix for the pulse-based RY gate is \n {} \n".format(qml.matrix(neutral_atom_RY)(jnp.pi/2).round(2)))
print("The matrix for the exact RY(pi/2) gate is \n {}".format(qml.matrix(qml.RY)(jnp.pi/2,wires = 0).round(2)))
##############################################################################
#
# We have implemented two orthogonal rotations in our neutral-atom device. This means that we have a universal set of single-qubit gates:
# all one-qubit gates can be implemented using some combination of :math:`RX` and :math:`RY`! The easy part is over—now we need to
# figure out how to apply two-qubit gates.
#
# The Rydberg blockade
# --------------------
#
# In the case of a neutral-atom device, implementing a two-qubit gate amounts to more than just applying pulses. We
# must make sure that the two atoms (i.e. our qubits) in question interact in a controlled way. The atoms are neutral, though,
# so do they even interact? They do, through various electromagnetic forces that arise due to the distributions of charges 
# in the atoms, which are all accounted for in the so-called *Van der Waals* interaction. 
#
# The Van der Waals interaction is usually pretty weak and short-ranged, but its effect will noticeably grow if the atoms we work
# with are large and can be excited to a highly energetic state—one more reason yet to use Rubidium-85.
# Such atoms are known as **Rydberg atoms**, and the states of high energy are 
# known as **Rydberg states**. We will choose one such Rydberg state, which we denote by :math:`\vert r\rangle,` to serve as an auxiliary
# state in the implementation of two-qubit gates. Focusing only on the
# ground state :math:`\vert 0\rangle` and the Rydberg state :math:`\vert r\rangle` as accessible states, the Ryberg 
# interaction is described by the *interaction Hamiltonian.*
#
# .. math::
#       
#    \mathcal{H}_i = \sum_{i<j}^{N}\frac{C_6}{R_{ij}^6}\hat{n}_{i}\hat{n}_j
#
# for a system of :math:`N` atoms. Here, :math:`\hat{n}_{i}=(\mathbb{I}+\sigma^{z}_{i})/2,` :math:`C_6` is a coupling constant that
# describes the interaction strength between the atoms, and :math:`R_{ij}` is the distance between atom :math:`i` and atom :math:`j.`
# If we add a pulse that addresses the transition between :math:`\vert 0\rangle` and :math:`\vert r\rangle,` the full Hamiltonian is
# given by 
# 
# .. math::
#       
#    \mathcal{H} = \sum_{q\in\text{wires}}(\cos(\phi)\sigma_{q}^x-\sin(\phi)\sigma_{q}^y) - \frac{1}{2}\delta(t)\sum_{q\in\text{wires}}(\mathbb{I}_q -\sigma_{q}^z)+ \sum_{i<j}^{N}\frac{C_6}{R_{ij}^6}\hat{n}_{i}\hat{n}_j.
#
# Note that the first two terms are the same as :math:`\mathcal{H}_d`, but bear in mind that the two-level system we're working with in 
# this case is the one spanned by the states :math:`\vert 0\rangle` and :math:`\vert r\rangle,` as opposed to :math:`\vert 0\rangle` 
# and :math:`\vert 1\rangle.` Let us **focus only on two atoms** and create the interaction Hamiltonian in terms of the distance 
# between the two atoms and the coupling strength :math:`C_6`. 
# This Hamiltonian is also built into `qml.pulse` in the `rydberg_interaction` method. 

def H_i(distance, coupling):
    
    # Only two atoms, placed in the coordinates (0,0) and (0,r)
    atomic_coordinates = [[0,0],[0,distance]]
    
    # Return the interaction term for two atoms in terms of the distance
    return qml.pulse.rydberg_interaction(atomic_coordinates, interaction_coeff = coupling, wires = [0,1])
##############################################################################
#
# One way to assess how these extra interaction terms affect the physics of the system is to see how the energy levels are changed. 
# These correspond to the eigenvalues of the full Hamiltonian. Let's plot them for different values of the distance, fixed zero
# detuning, and other fixed values of other parameters for easy visualization

peak = 6
phase = np.pi/2
detuning = 0
coupling = 1
time = 0.1

def energy_gap(distance):

    """Calculates the energy eigenvalues for the full Hamiltonian, as a function of the distance
     between the atoms."""
    
    # Evaluate the drive term for peak and time parameters
    drive_term = H_d(blackman_window, phase, detuning, wires = [0,1])([peak,peak],time)
    interaction_term = H_i(distance,coupling)([],[])

    # Calculate the eigenvalues for the full Hamiltonian
    eigenvalues = jnp.linalg.eigvals(qml.matrix(drive_term + interaction_term))
        
    # We sort the eigenvalues by magnitude; numpy doesn't give them in any particular order.
    return jnp.sort(eigenvalues - eigenvalues[0])

x_points = np.linspace(0.55,1.3,30)
y_points = [np.real(energy_gap(elem)) for elem in x_points]

plot_colors = ['#e565e5','#66c4ed','#ffd86d', '#9e9e9e']

for i in range(4):
    y = [y_points[_][i] for _ in range(len(y_points))]
    plt.plot(x_points, y, c = plot_colors[i])   

plt.xlabel("Distance between atoms")
plt.ylabel("Energy levels")

plt.text(1.25,85, '|rr>', c = '#9e9e9e' )
plt.text(1.25,50, '|0r>', c = '#ffd86d' )
plt.text(1.25,25, '|r0>', c = '#66c4ed' )
plt.text(1.25,6, '|00>', c = '#e565e5' )
plt.show()
##############################################################################
#
# Let's analyze what we see above. When the atoms are far away, the energy levels are evenly spaced. This means that if a pulse
# excites the system from :math:`\vert 00 \rangle` (both atoms in the ground state) :math:`\vert 0r \rangle` (one atom in the ground state,
# one in the Rydberg state), then a similar second pulse could excite the system into :math:`\vert rr \rangle` (both atoms in the Ryberg state).
# However, as the atoms move close to each other, this is no longer true. When the distance becomes small,
# as soon as one of the atoms reaches the Rydberg state, the other one cannot reach that state with a similar pulse. 
#
# .. note::
#
#    We are not using any realistic values for either the amplitude or the coupling strength. These have been chosen in arbitrary
#    units for visualization purposes. If you would like to know more about the specifications for real quantum hardware, check out
#    this demo (citation needed). 
# 
# This phenomenon is called the **Rydberg blockade.** When the distance between two atoms is below a certain distance known as 
# the **blockade radius,** one atom being in the Rydberg state "blocks" the other one from reaching its Rydberg state. Let's see
# how the Ryberg blockade helps us build two-qubit gates.
#
# The Ctrl-Z gate
# ---------------
#
# The native two-qubit gate for neutral atoms devices turns out to be the `CZ` gate, which can be implemented with a sequence
# of `RX` rotations (in the space spanned by :math:`\vert 0 \rangle` and :math:`\vert r \rangle`). In particular, three pulses 
# be needed: a :math:`\pi`-**pulse** (inducing a rotation by an angle :math:`pi`) on the first atom, a :math:`2\pi`-**pulse** 
# (inducing a rotation by an angle :math:`2\pi`) on the second atom, and another :math:`\pi`-**pulse** on the first atom, in that
# order. Combined with the effects of the Rydberg blockade, this pulse combination will implement the desired gate. To see this,
# let's code the pulses needed first. 

def two_pi_pulse(distance, coupling, wires = [0]):


    # Build full Hamiltonian
    full_hamiltonian = H_d(blackman_window,0,0,wires)+H_i(distance,coupling)
    
    # Return the 2 pi pulse
    qml.evolve(full_hamiltonian)([2*jnp.pi/0.42/0.2/(2*jnp.pi)], t=[0, 0.2])
    

def pi_pulse(distance, coupling, wires = [0]):
    
    full_hamiltonian = H_d(blackman_window,0,0,wires)+H_i(distance,coupling)
    
    # Return the pi pulse
    qml.evolve(full_hamiltonian)([jnp.pi/0.42/0.2/(2*jnp.pi)], t=[0, 0.2])
##############################################################################
#
# Then, let's see the effect the sequence of pulses has in the :math:`\vert 00 \rangle` state when the atoms are close enough.
#
dev_two_qubits = qml.device('default.qubit.jax', wires = 2)

@qml.qnode(dev_two_qubits)
def neutral_atom_CZ(distance, coupling):
        
    pi_pulse(distance, coupling, wires = [0])
    
    two_pi_pulse(distance, coupling, wires = [1])
    
    pi_pulse(distance, coupling, wires = [0])
    
    return qml.state()

print("The final state after the set of pulses is {} when atoms are close.".format(neutral_atom_CZ(0.2,1).round(2)))
print("The final state after the set of pulses is {} when atoms are far.".format(neutral_atom_CZ(2,1).round(2)))
##############################################################################
#
# The effect is to add a phase of -1 to the state, which doesn't happen without the Rydberg blockade! In fact,
# this is the only case in which the Rydberg blockade has any effect on the state of the atoms. 
# 
# .. figure:: ../demonstrations/neutral_atoms/control_z00.png
#    :align: center
#    :width: 60%
#
#    ..
# 
# Indeed, if one of the atoms were to be in the state :math:`\vert 1 \rangle,` then the pulse wouldn't affect such an atom
# since it's not tuned to the :math:`\vert r \rangle \rightarrow \vert 1 \rangle` transition.
# 
# .. figure:: ../demonstrations/neutral_atoms/control_z01.png
#    :align: center
#    :width: 60%
#
#    .. 
#
# The net effect of the sequence of pulses is summarized in the following table.
#
# .. raw:: html
#
#     <style>
#         .docstable {
#             max-width: 300px;
#         }
#         .docstable tr.row-even th, .docstable tr.row-even td {
#             text-align: center;
#         }
#         .docstable tr.row-odd th, .docstable tr.row-odd td {
#             text-align: center;
#         }
#     </style>
#     <div class="d-flex justify-content-center">
#
# .. rst-class:: docstable
#
#     +-------------------------+-------------------------------+
#     |  Initial state          | Final state                   |
#     +=========================+===============================+
#     | :math:`\vert 00\rangle` | :math:`-\vert 00\rangle`      |
#     +-------------------------+-------------------------------+
#     | :math:`\vert 01\rangle` | :math:`-\vert 01\rangle`      |
#     +-------------------------+-------------------------------+
#     | :math:`\vert 10\rangle` | :math:`-\vert 10\rangle`      |
#     +-------------------------+-------------------------------+
#     | :math:`\vert 11\rangle` | :math:`\vert 11\rangle`       |
#     +-------------------------+-------------------------------+
#
# .. raw:: html
#
#     </div>
#
# Up to a global phase, this corresponds to the `CZ` gate. Together with the `RX` and `RY` gates, we have a universal set of gates,
# since the `CNOT` gate can be expressed in terms of `CZ` via the equation
# 
# .. figure:: ../demonstrations/neutral_atoms/cnot_and_cz.png
#    :align: center
#    :width: 60%
#
#    .. 
#
# Challenges and future improvements
# ----------------------------------
#
# Great, this all seems to work like a charm... at least in theory. In practice, however, there are still challenges to overcome. 
# We've managed to efficiently prepare qubits, apply gates, and measure, satisfying Di Vincenzo's second, fourth, and fifth criteria.
# However, as is most quantum architectures, there are some challenges to overcome with regard to scalability and decoherence times.
#
# While we are able to trap many atoms with our current laser technology, scaling optical tweezer arrays to thousands of qubits
# poses an obstacle. We rely on spatial modulators to divide our laser beams, but this also reduces the strength of the tweezers. At 
# we split a laser beam too much, tweezers become too weak to contain an atom. Of course, we could simply use more laser sources, 
# but the spatial requirements # for the hardware would also grow. Alternatively, we can use laser sources with higher intensity, 
# but such technology is still being developed. Another solution is to use photons through optical fibres to communicate between
# different processors, allowing for further connectivity and scalability.
#
# But the number one nemesis of quantum hardware engineers is decoherence. Quantum states are short-lived in the presence of external
# influences. We can never achieve a perfect vacuum in the chamber, and the particles and charges around the atoms will destroy
# our carefully crafted quantum states in a matter of microseconds. But we need time to move the atoms around and apply the pulses, 
# so we must be extremely quick to win against decoherence. Overall, improving our register preparation, gates, and measurement speeds
# and precisions is of the essence to improve on neutral-atom technology. 
#
# Conclusion
# ----------
#
# Neutral-atom quantum hardware is a promising and quickly developing technology which we should keep an eye on. The ability to
# easily create custom qubit topologies is its main strength, and its weaknesses are actually no different from other quantum 
# architectures. We can easily program neutral-atom devices using pulses, for which PennyLane is of great help. If you want to 
# learn more, check out our tutorials on the Aquila device, neutral atom configurations, and pulse programming. And do take a look 
# at the references below to dive into much more detail about the topics introduced here. 
#
# References
# ~~~~~~~~~~
#
# .. [#DiVincenzo2000]
#
#     D. DiVincenzo. (2000) "The Physical Implementation of Quantum Computation",
#     `Fortschritte der Physik 48 (9–11): 771–783
#     <https://onlinelibrary.wiley.com/doi/10.1002/1521-3978(200009)48:9/11%3C771::AID-PROP771%3E3.0.CO;2-E>`__.
#     (`arXiv <https://arxiv.org/abs/quant-ph/0002077>`__)