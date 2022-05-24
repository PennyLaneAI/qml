r""".. _photonics:

Photonic quantum computers
=============================

.. meta::
    :property="og:description": Photon-based quantum computers
    :property="og:image": https://pennylane.ai/qml/_images/photonics_tn.png

.. related::
   tutorial_pasqal Quantum computation with neutral atoms

*Author: Alvaro Ballon. Posted: 31 May 2022. Last updated: 31 May 2022.*

To create a functional quantum computer, we need to gather and control a 
large number of qubits. This feat has proven difficult, although significant 
progress has been made using trapped ions, superconducting circuits, 
and many other technologies. Scalability—the ability to put many 
qubits together—is limited because individual qubits in a multi-qubit 
system lose their quantum properties quickly. This phenomenon, 
known as decoherence, happens due to the interactions of the qubits with 
each other and their surroundings. One way to get scalable qubits is to 
use photons (particles of light). Photons rarely affect each other and 
their quantum state is not easily destroyed, so we may be onto something!

Indeed, many approaches to use photons for quantum
computing have been proposed. We will focus on *linear optical quantum computing*,
an approach that has already achieved quantum advantage. It 
is being developed further by Xanadu, PsiQuantum, and other institutions
around the globe. Linear optical quantum computing does not use qubits directly. 
Instead, it's based on qumodes: states of 
many photons that are superpositions of infinitely many basis states. But no need to panic. 
By the end of this demo, you will be able to explain how photonic 
qumode devices can accomplish 
the same tasks as qubit-based devices. You will learn how to prepare,
measure and manipulate the quantum states of light to achieve 
universal quantum computations. Moreover, you will identify 
the strengths and weaknesses of photonic devices in terms of 
Di Vincenzo's criteria, introduced in the blue box below.

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

Our journey will start by defining the simplest states of light, known as Gaussian states. We will also
describe how we can perform simple gates and measurements on such states. The next step is to  
figure out simple ways to generate the more general non-Gaussian states, needed for universal
quantum computing. We'll see that we end up needing only 
a special type of non-Gaussian states, known as GKP states. 
Finally, we will bring all the concepts together to understand how qubit-based
computations can be performed by using qumodes. Let's get started!

"""

##############################################################################
#
# Gaussian states of light
# ~~~~~~~~~~~~~~~~~~~~~~~~
#
# Why are the quantum states of light so durable? Photons seldom interact with each other, 
# which means we can easily avoid uncontrolled interactions that destroy their quantum 
# state. However, to build a universal quantum computer, we need multi-qubit gates, 
# which means that photons must be made to communicate with each other somehow! We *can* make photons 
# affect each other, but for now, let's focus on the quantum states of light that 
# we can theoretically obtain when photons don't need to interact. Such states are called 
# **Gaussian states**, and they can be fabricated with a 100% success rate
# using common optical devices. 
#
#
# To precisely define a Gaussian state, we need a mathematical representation for states 
# of light. As is the case with qubits, states of light are represented by a linear 
# combination of basis vectors. But unlike qubits, two basis vectors aren't enough. Why not?
# The reason is that light is characterized by its so-called *position and momentum quadratures* :math:`x` and :math:`p`, 
# captured by the operators :math:`\hat{X}` and :math:`\hat{P}` respectively. 
# 
# .. note::
#
#    The position and momentum quadratures :math:`x` and :math:`p` do not represent the position and momentum 
#    of a photon. They describe a state of many photons (e.g., a laser beam), and they are related
#    to the amplitude and phase of light. The names come from the fact that the quadrature observables :math:`\hat{X}`
#    and :math:`\hat{P}` satisfy 
#    
#    .. math:: \left[ \hat{X},\hat{P}\right]=i\hbar,
#
# which is the same relation satisfied by conventional position and momentum in quantum mechanics.
# As a consequence, the standard deviations of the measurements of :math:`x` and :math:`p` satisfy the
#    uncertainty relation
#
#    .. math:: \Delta x \Delta p \geq 1,
# 
#    where we work in units where :math:`\hbar = 2.` Sometimes the word "quadratures" is omitted for simplicity.
# 
# Upon measurement, the quadratures can take any real value, which means that :math:`\hat{X}` and 
# :math:`\hat{P}` have infinitely many eigenvectors. Therefore, to describe a quantum state of light 
# :math:`\left\lvert \psi \right\rangle` we need infinitely many basis vectors! 
# 
# Such a representation of a quantum state is called a **qumode**. For example, we write
#
# .. math:: \left\lvert \psi \right\rangle = \int_\mathbb{R}\psi(x)\vert x \rangle dx,
#
# where :math:`\vert x \rangle` is the eigenstate of :math:`\hat{X}` with eigenvalue 
# :math:`x`, and :math:`\psi` is a complex-valued function known as the *wave function*. 
# A similar expansion can be done in terms of the eigenstates :math:`\vert p \rangle` of :math:`\hat{P}`.
#
# So how do we define a Gaussian state using this representation? It is a state that is completely 
# determined by the average values :math:`\bar{x}` and :math:`\bar{p}` of the position and momentum quadratures, 
# as well as their standard deviations :math:`\Delta x` and :math:`\Delta p`. The most trivial (and boring) 
# Gaussian state is the *vacuum state*—the state with no photons. Let us see what happens if 
# we sample measurements of the quadratures :math:`\hat{X}` and :math:`\hat{P}` when light is in the vacuum state. 
# We can use PennyLane's ``default.gaussian`` device, which lets us create, manipulate, and measure Gaussian 
# states of light. Let's first call the usual imports,
#

import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

##############################################################################
#
# and define the device. Note that the number of wires corresponds to the number of qumodes.

dev = qml.device('default.gaussian', wires = 1, shots=1000)

##############################################################################
#
# We would like to know how the measured values of position and momentum 
# are distributed in the :math:`x-p` space, usually called *[phase space](https://en.wikipedia.org/wiki/Optical_phase_space)*. 
# The initial state in ``default.gaussian`` is the vacuum state, so the circuits 
# to measure the quadratures need not contain any operations, except for measurements! 
# We plot 1000 measurement results for both :math:`x` and :math:`p`.

@qml.qnode(dev)
def vacuum_measure_x():
    return qml.sample(qml.X(0)) #Samples X quadratures
@qml.qnode(dev)
def vacuum_measure_p():
    return qml.sample(qml.P(0)) #Samples P quadrature

#Sample measurements in phase space
x_sample = vacuum_measure_x().numpy()
p_sample = vacuum_measure_p().numpy()

#Import some libraries for a nicer plot
from scipy.stats import gaussian_kde
from numpy import vstack as vstack

#Point density calculation
xp = vstack([x_sample,p_sample])
z = gaussian_kde(xp)(xp)

#Sort the points by density
sorted = z.argsort()
x, y, z = x_sample[sorted], p_sample[sorted], z[sorted]

#Plot
fig, ax = plt.subplots()
ax.scatter(x,y, c=z, s=50, cmap='turbo')
plt.title("Vacuum state",fontsize=12)
ax.set_ylabel("Momentum", fontsize=11)
ax.set_xlabel("Position", fontsize=11)
ax.set_aspect('equal', adjustable='box')
plt.show()

##############################################################################
#
# We observe that the values of the quadratures are distributed around the 
# origin with a spread of approximately 1. We can check these eyeballed values explicitly, 
# using a device without shots this time. 

dev_exact=qml.device('default.gaussian', wires=1) #No explicit shots gives analytic calculations
@qml.qnode(dev_exact)
def vacuum_mean_x():
    return qml.expval(qml.X(0)) #Returns exact expecation value of x
@qml.qnode(dev_exact)
def vacuum_mean_p():
    return qml.expval(qml.P(0)) #Returns exact expectation value of p
@qml.qnode(dev_exact)
def vacuum_var_x():
    return qml.var(qml.X(0)) #Returns exact variance of x
@qml.qnode(dev_exact)
def vacuum_var_p():
    return qml.var(qml.P(0)) #Returns exact variane of p

#Print calculated statistical quantities
print("Expectation value of x-quadrature: {}".format(vacuum_mean_x()))
print("Expectation value of p-quadrature: {}".format(vacuum_mean_p()))
print("Variance of x-quadrature: {}".format(vacuum_var_x()))
print("Variance of p-quadrature: {}".format(vacuum_var_p()))

##############################################################################
#
# What other Gaussian states are there? The states produced by lasers are called *coherent states*, 
# which are also Gaussian with :math:`\Delta x = \Delta p = 1`. The vacuum is but an example of a 
# coherent state. Coherent states, in general, can have non-zero expectation values for the 
# quadratures (i.e., they are not centred around the origin). 
# 
# .. note::
#
#    More generally, a coherent state is one that saturates the uncertainty relation. That is,
#
#    .. math:: \Delta x \Delta p = 1.
# 
# The ``default.gaussian`` device allows for the easy preparation of coherent states 
# through ``qml.CoherentState``. This function takes two parameters :math:`\alpha` and :math:`\phi`, 
# where :math:`\alpha=\sqrt{\vert\bar{x}\vert^2+\vert\bar{p}\vert^2}`is the magnitude 
# and :math:`\phi` is the polar angle of the point :math:`(\bar{x}, \bar{p}).`
# Let us plot sample quadrature measurements for a coherent state.
 
@qml.qnode(dev)
def measure_coherent_x(alpha,phi):
    qml.CoherentState(alpha,phi,wires=0) #Prepares coherent state
    return qml.sample(qml.X(0)) #Measures X quadrature

@qml.qnode(dev)
def measure_coherent_p(alpha,phi):
    qml.CoherentState(alpha,phi,wires=0) #Prepares coherent state
    return qml.sample(qml.P(0)) #Measures P quadrature

#Choose alpha and phi and sample 1000 measurements
x_sample_coherent = measure_coherent_x(2,np.pi/3).numpy()
p_sample_coherent = measure_coherent_p(2,np.pi/3).numpy()

#Plot as before
xp = vstack([x_sample_coherent,p_sample_coherent])
z1 = gaussian_kde(xp)(xp)

sorted = z1.argsort()
x, y, z = x_sample_coherent[sorted], p_sample_coherent[sorted], z1[sorted]

fig, ax1 = plt.subplots()
ax1.scatter(x,y, c=z, s=50, cmap='turbo')
ax1.set_title("Coherent State",fontsize=12)
ax1.set_ylabel("Momentum", fontsize=11)
ax1.set_xlabel("Position", fontsize=11)
ax1.set_aspect('equal', adjustable='box')
plt.show()

##############################################################################
#
# But where does the name Gaussian come from? If we plot the density 
# of quadrature measurements for a coherent state in three dimensions, we obtain the following plot.
# 
# .. figure:: ../demonstrations/photonics/vacuum_wigner.png
#    :align: center
#    :width: 70% 
#    
#    ..
#
#    Density of measurement results in phase space for a coherent state
# 
# The density has the shape of a 2-dimensional Gaussian surface, hence the name. **For Gaussian 
# states only**, the density is exactly equal to the so-called [*Wigner function*](https://en.wikipedia.org/wiki/Wigner_quasiprobability_distribution) :math:`W(x,p)`, 
# defined using the wave function :math:`\psi(x)`:
#
# .. math:: W(x,p) = \frac{1}{\pi\hbar}\int_{-\infty}^{\infty}\psi^{*}(x+y)\psi(x-y)e^{2ipy/\hbar}dy.
#
# Since the Wigner function satisfies  
# 
# .. math:: \int_{\mathbb{R}^2}W(x,p)dxdp = 1
#
# and is positive for Gaussian states,
# it is *almost as if* the values of the momentum and position had an underlying classical 
# probability distribution, save for the fact that the quadratures can't be measured simultaneously. 
# For this reason, Gaussian states are considered to be "classical" states of light. Now we're ready 
# for the technical definition of a Gaussian state.
# 
# .. admonition:: Definition
#     :class: defn
#
#     A qumode is said to be in a **Gaussian state** if its Wigner function is a two-dimensional
#     Gaussian function
#
# Gaussian operations
# ~~~~~~~~~~~~~~~~~~~~~~~~
#
# We have only learned about two types of Gaussian states so far. The vacuum state can be obtained 
# by simply doing nothing and a coherent state can be produced 
# by a laser, so we already have these at hand. But how can we obtain any Gaussian state of our liking? 
# This is achieved through *Gaussian operations*, which transform a Gaussian state to 
# another Gaussian state. These operations are relatively easy to implement in a lab 
# using some of the optical elements introduced in the table below.
# 
# .. rst-class:: docstable
#
#     +---------------------+--------------------------------------------------------------+---------------------------------------------------------------------+
#     | .. centered::       | .. centered::                                                | .. centered::                                                       |
#     |  Element            |  Diagram                                                     |   Description                                                       |
#     +=====================+==============================================================+=====================================================================+
#     | Waveguide           | .. figure:: ../demonstrations/photonics/Waveguide.png        | A long strip of material that contains and guides                   |
#     |                     |    :align: center                                            | electromagentic waves. In photonics, optical fibres are used.       |
#     |                     |    :width: 70%                                               |                                                                     | 
#     +---------------------+--------------------------------------------------------------+---------------------------------------------------------------------+
#     | Thermo-optic        | .. figure:: ../demonstrations/photonics/Thermo-optic.png     | A waveguide with a resistive material inside. When it heats up,     |
#     | phase shifter       |    :align: center                                            | the properties of the waveguide change, which allows us to          |
#     |                     |    :width: 70%                                               | shift the phase of light in a controlled way.                       |
#     +---------------------+--------------------------------------------------------------+---------------------------------------------------------------------+
#     | Beamsplitter        | .. figure:: ../demonstrations/photonics/Beam_splitter.png    | An element with two entry and two exit ports. It transmits a        |
#     |                     |    :align: center                                            | fraction :math:`t` of the photons coming in through either port,    |
#     |                     |    :width: 100%                                              | and reflects a fraction :math:`r=1-t`. It entangles two qumodes     |
#     |                     |                                                              | coming in through each port.                                        |
#     +---------------------+--------------------------------------------------------------+---------------------------------------------------------------------+
#
# What if, for example, we would like to change the expectation value of the :math:`x` 
# quadrature without changing anything else about the state? This can be done 
# via the *displacement operator*, implemented in PennyLane via ``qml.Displacement``. 
# Let's see the effect of this operation on an intial coherent state.

@qml.qnode(dev)
def displace_coherent_x(alpha,phi,x):
    qml.CoherentState(alpha,phi,wires=0) #Create coherent state
    qml.Displacement(x,0, wires=0) #Second argument is the displacement direction in phase space
    return qml.sample(qml.X(0))

@qml.qnode(dev)
def displace_coherent_p(alpha,phi,x):
    qml.CoherentState(alpha,phi,wires=0)
    qml.Displacement(x,0, wires=0)
    return qml.sample(qml.P(0))

#We plot both the initial and displaced state
initial_x=displace_coherent_x(3,np.pi/3,0) #initial state amounts to 0 displacement
initial_p=displace_coherent_p(3,np.pi/3,0)
displaced_x=displace_coherent_x(3,np.pi/3,1) #set a parameter x=1 to displace in x-direction
displaced_p=displace_coherent_p(3,np.pi/3,1)
#Plot as before
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12,5),gridspec_kw={'width_ratios': [1, 1]})
fig.tight_layout()
xp1 = vstack([initial_x,initial_p])
z1 = gaussian_kde(xp1)(xp1)
sorted1=z1.argsort()
x1,y1,z1= initial_x[sorted1], initial_p[sorted1], z1[sorted1]
xp2 = vstack([displaced_x,displaced_p])
z2 = gaussian_kde(xp2)(xp2)
sorted2=z2.argsort()
x2,y2,z2= displaced_x[sorted2], displaced_p[sorted2], z2[sorted2]
ax1.scatter(x1,y1,c=z1, s=50,cmap='turbo')
ax2.scatter(x2,y2,c=z2, s=50,cmap='turbo')
ax1.set_title("Initial state",fontsize=16)
ax1.set_ylabel("Momentum", fontsize=14)
ax1.set_xlabel("Position", fontsize=14)
ax2.set_title("After displacement",fontsize=16)
ax2.set_ylabel("Momentum", fontsize=14)
ax2.set_xlabel("Position", fontsize=14)
plt.show()
 
##############################################################################    
#
# Note that setting :math:`x=1` gives a displacement of 2 units in the x-direction. 
# But how do we make a displacement in the lab? One method 
# is shown below, which uses a beam splitter and a source of high-intensity coherent light.
#
# .. figure:: ../demonstrations/photonics/Displacement.png
#    :align: center
#    :width: 70% 
#
#    ..
#
#    Experimental implementation of the displacement operator on an arbitrary state
#
# We can check that this setup implements a displacement operator using PennyLane. This time,
# we need two qumodes, since we rely on combining the qumode we want to displace with a 
# coherent state in a beamsplitter. The target state is another coherent state

dev2 = qml.device('default.gaussian', wires = 2, shots=1000)

@qml.qnode(dev2)
def disp_optics(z,x):
    qml.CoherentState(z,0,wires=0) #Coherent state to mix with initial state
    qml.CoherentState(3,np.pi/3,wires=1) #Target state (low amplitude coherent state)
    qml.Beamsplitter(np.arccos(1-x**2/z**2),0,wires=[0,1]) #Beamsplitter 
    return qml.sample(qml.X(1)) #Measure x quadrature
@qml.qnode(dev2)
def mom_optics(z,x):
    qml.CoherentState(z,0,wires=0)
    qml.CoherentState(3,np.pi/3,wires=1)
    qml.Beamsplitter(np.arccos(1-x**2/z**2),0,wires=[0,1])
    return qml.sample(qml.P(1)) #Measure p quadrature

#Plot quadrature measurement before and after implementation of displacement
initial_x=disp_optics(100,0) #Initial state corresponds to beam splitter with t=0 (x=0)
initial_p=mom_optics(100,0)  #Amplitude of coherent state = 100 must be large
displaced_x=disp_optics(100,1)
displaced_p=mom_optics(100,1) #Set some non-trivial t
#Plot as before
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12,5),gridspec_kw={'width_ratios': [1, 1]})
fig.tight_layout()
xp1 = vstack([initial_x,initial_p])
z1 = gaussian_kde(xp1)(xp1)
sorted1=z1.argsort()
x1,y1,z1= initial_x[sorted1], initial_p[sorted1], z1[sorted1]
xp2 = vstack([displaced_x,displaced_p])
z2 = gaussian_kde(xp2)(xp2)
sorted2=z2.argsort()
x2,y2,z2= displaced_x[sorted2], displaced_p[sorted2], z2[sorted2]
ax1.scatter(x1,y1,c=z1, s=50,cmap='turbo')
ax2.scatter(x2,y2,c=z2, s=50,cmap='turbo')
ax1.set_title("Initial state",fontsize=16)
ax1.set_ylabel("Momentum", fontsize=14)
ax1.set_xlabel("Position", fontsize=14)
ax2.set_title("After displacement",fontsize=16)
ax2.set_ylabel("Momentum", fontsize=14)
ax2.set_xlabel("Position", fontsize=14)
plt.show()

############################################################################## 
#
# We see that we get a displaced state. The amount of displacement can be adjusted by
# changing the parameters of the beamsplitter. 
# Similarly, we can implement rotations in phase space using ``qml.Rotation``, which 
# simply amounts to changing the phase of light using a thermo-optic phase shifter. 
# 
# We have focused so far on changing the mean values of :math:`x` and :math:`p`, 
# but what if we also want to change the spread of the quadratures? That corresponds to 
# creating *squeezed states*, which is more difficult than simply using 
# beamsplitters and phase shifters. It requires shining light through non-linear 
# materials, where the state of light will undergo unitary evolution in a 
# way that changes :math:`\Delta x` and :math:`\Delta p.`
# 
# .. figure:: ../demonstrations/photonics/Squeezer.png
#    :align: center
#    :width: 70%
#
#    ..
#
#    A non-linear material can work as a squeezer
#
# We won't go into detail here, 
# but we note that the technology to produce these is quite mature. 
# In PennyLane, we can generate squeezed states through the squeezing 
# operator ``qml.Squeezing``. This function depends on the squeezing parameter 
# :math:`r` which tells us how much the variance in :math:`x` and :math:`p` changes, and :math:`\phi`, 
# which rotates the state in phase space. Let's take a look at how squeezing changes the distribution
# of quadrature measurements. 

@qml.qnode(dev)
def measure_squeezed_x(r):
    qml.Squeezing(r,0,wires=0)
    return qml.sample(qml.X(0))
@qml.qnode(dev)
def measure_squeezed_p(r):
    qml.Squeezing(r,0,wires=0)
    return qml.sample(qml.P(0))

#Choose alpha and phi and sample 1000 measurements
x_sample_squeezed = measure_squeezed_x(0.4).numpy()
p_sample_squeezed = measure_squeezed_p(0.4).numpy()

#Plot as before
xp = vstack([x_sample_squeezed,p_sample_squeezed])
z = gaussian_kde(xp)(xp)

sorted_meas = z.argsort()
x, y, z = x_sample_squeezed[sorted_meas], p_sample_squeezed[sorted_meas], z[sorted_meas]

fig, ax1 = plt.subplots(figsize=(7,7))
ax1.scatter(x,y, c=z, s=50, cmap='turbo')
ax1.set_title("Squeezed State",fontsize=12)
ax1.set_ylabel("Momentum", fontsize=11)
ax1.set_xlabel("Position", fontsize=11)
ax1.set_xlim([-4,4])
plt.show()

##############################################################################
#
# This confirms that squeezing changes the variances of the quadratures.
# 
# .. note::
#
#    The squeezed states produced above satisfy :math:`\Delta x \Delta p = 1`,
#    which means they are Gaussian states as well. We won't need to use any 
#    more general Gaussian states.
#
#
# Measuring quadratures
# ~~~~~~~~~~~~~~~~~~~~~
#
# Now that we know how to manipulate Gaussian states, we would like to perform measurements 
# on them. So far, we have taken for granted that we can measure the quadratures 
# :math:`\hat{X}` and :math:`\hat{P}`. But how do we actually measure them using optical elements? 
# We will need a measuring device known as a photon counter. These contain a piece of a 
# photoelectric material, where each outer electron can be stimulated by a photon. 
# The more photons that are incident on the photon counter, the more electrons that are 
# freed in the material, which in turn form an electric current. Mathematically,
#
# .. math:: I = qN,
#
# where :math:`I` is the electric current, :math:`N` is the number of photons, and :math:`q` is a detector-dependent 
# proportionality constant. Hence, measuring the current amounts to measuring the number of photons!
#
# The number of photons in a quantum state of light is not fixed. It is measured by the quantum 
# photon-number observable :math:`\hat{N}`, which has eigenstates denoted :math:`\vert 0 \rangle, \vert 1\rangle, \vert 2 \rangle,...`. 
# These states, known as *Fock states*, do have a well-defined number of photons: 
# repeated measurements of :math:`\hat{N}` on the same state will yield the same output. 
# The natural number :math:`n` in the Fock state :math:`\vert n \rangle` denotes the only possible 
# result we would get upon measuring the photon number. But nothing prevents light from being in a superposition of 
# Fock states. For example, when we measure :math:`\hat{N}` for the state
#
# .. math:: \vert \psi \rangle = \frac{1}{\sqrt{3}}\left(\vert 0 \rangle + \vert 1 \rangle + \vert 2 \rangle\right),
#
# we get 0, 1, or 2 photons, each with probability :math:`1/3`. 
#
# Except for the vacuum :math:`\vert 0 \rangle,` **Fock states are not Gaussian**. Gaussian states are, in general, 
# superpositions of Fock States. For example, let's measure the expected photon-number for some squeezed state:

dev3 = qml.device('default.gaussian', wires = 1)
@qml.qnode(dev3)
def measure_n_coherent(alpha,phi):
    qml.Squeezing(alpha,phi,wires=0)
    return qml.expval(qml.NumberOperator(0))

coherent_expval = measure_n_coherent(1,np.pi/3)
print("Expected number of photons: {}".format(coherent_expval))

############################################################################## 
#
# Since the expectation value is not an integer number, the measurement results cannot have been all the same integer.
# This squeezed state cannot be a Fock state!
#
# Now we know how to measure the number of photons. But what about the promised 
# quadratures? We can do this through a combination of photon counting and a beamsplitter, 
# as shown in the diagram below. 
#
# .. figure:: ../demonstrations/photonics/Homodyne.png
#    :align: center
#    :width: 70%
#
#    ..
#
#    Measuring quadratures using photon counters
#
# Let's code this setup using PennyLane and check that it amounts to the measurement of quadratures.

dev_exact2=qml.device('default.gaussian', wires = 2)
@qml.qnode(dev_exact2)
def measurement(a,phi):
    qml.Displacement(a,phi,wires=0) #Implement displacement using PennyLane
    return qml.expval(qml.X(0))

@qml.qnode(dev_exact2)
def measurement2(a, theta, alpha, phi):
    qml.Displacement(a,theta, wires=0) #We choose the initial state to be a displaced vacuum
    qml.CoherentState(alpha, phi, wires = 1) #Prepare coherent state as second qumode
    qml.Beamsplitter(np.pi/4, 0, wires = [0,1]) #Interfere both states
    return qml.expval(qml.NumberOperator(0)), qml.expval(qml.NumberOperator(1)) #Read out photon counters

print("Expectation value of x-quadrature after displacement: {}".format(round(measurement(3,0).numpy(),2)))
print("Expected number of photons in each detector:\n")
print("Detector 1: {}".format(round(measurement2(3,0,1,0)[0].numpy(),2)))
print("Detector 2: {}".format(round(measurement2(3,0,1,0)[1].numpy(),2)))
print("Difference between photon numbers detected: {}".format(round(measurement2(3,0,1,0)[1].numpy(),2)-round(measurement2(3,0,1,0)[0].numpy(),2)))

##############################################################################
#
# Although PennyLane does not allow us to sample ``qml.NumberOperator``, trying the above 
# with many input states should convince you that this setup, known as *homodyne measurement*, 
# allows us to measure the quadratures :math:`\hat{X}` and :math:`\hat{P}`. Feel free to play around
# changing the values of :math:`\phi` and :math:`a`!
#
# Beyond Gaussian states
# ~~~~~~~~~~~~~~~~~~~~~~
#
# We've learned a lot about Gaussian states now, but they don't seem to have many quantum properties. 
# They are described by an underlying probability distribution of the quadratures, given by the Wigner 
# function, so how is this different from classical states? These are legitimate concerns and, 
# indeed, to build  photonic quantum computer we need both entangled states *and* non-Gaussian states. 
#
# The former are not a problem, however, since beamsplitters already entangle the input states! 
# Let us set on the more challenging mission to find a way to prepare non-Gaussian states. 
# All of the operations that we have learned so far—displacements, rotations, squeezing—are Gaussian. 
# Do we need some kind of strange material that will implement a non-Gaussian operation? That's certainly 
# a possibility, and some examples can be found in the Kerr and Cubic phase interactions. But relying 
# on these non-linear materials is far from optimal, since we don't have much freedom to manipulate 
# the setup into getting an arbitrary non-Gaussian state. 
#
# But there's one non-Gausian operation that's been right in front of our eyes all this time. 
# Photon counters take a Gaussian state and collapse it into a Fock state (although this destroys the photons); 
# therefore, photon number detection is not a Gaussian operation. Combined with squeezed states and beamsplitters, 
# we have all the ingredients to produce any non-Gaussian state that we like. 
#
# Let's explore how this works. The main idea is to tweak a particular photonic circuit known 
# as a *Gaussian Boson Sampler*, which is shown below:
#
# .. figure:: ../demonstrations/photonics/GBS.png
#    :align: center
#    :width: 70%
#
#    ..
#
#    A Gaussian Boson Sampling circuit
#
# Gaussian boson sampling is interesting on its own. The output probabilities that we just obtained 
# allow us to calculate the Hafnian of a matrix, which is classically a hard problem 
# (see `this tutorial <https://pennylane.ai/qml/demos/tutorial_qgbs.html>`__ for an in-depth discussion). 
# This was the great accomplishment of Jiuzhang, 
# the first photonic quantum computer to achieve quantum advantage. But the most interesting 
# application comes from removing the detector in the last wire, as shown below.
#
# .. figure:: ../demonstrations/photonics/GKP_Circuit.png
#    :align: center
#    :width: 70%
#
#    ..
#
#    Circuit to produce non-Gaussian states probabilistically
#
# Generalizations of circuits like the above can, after photon detection of the other qumodes, 
# produce a good approximation of any non-Gaussian state that we want. The reason is that the 
# final state for the circuit is an entangled qumode, and we apply a non-Gaussian operation to some of the 
# qumodes. This measurement affects the remaining qumode and turns it into a non-Gaussian state. This is 
# the magic of quantum mechanics! For example, the choice of parameters  
# 
# .. math:: t_1 = 0.8624, \quad t_2=0.7688, \quad t_3 = 0.7848, \quad S_1 = -1.38, \quad S_2 = -1.22, \quad S_3 = 0.780 \quad S_4 = 0.196,
#
# for this generalized GBS circuit produces the state (expressed as a combination of Fock states)
#
# .. math:: \vert \psi \rangle = S(0.196)\left(0.661 \vert 0\rangle -0.343 \vert 2\rangle + 0.253\vert 4\rangle -0.368\vert 6\rangle
#             +0.377 \vert 8\rangle + 0.323 \vert 10\rangle + 0.325\vert 12\rangle\right)
#
# but only when the detectors measure 5 and 7 photons.  This state's Wigner function is shown below:
#
# .. figure:: ../demonstrations/photonics/gkp_wigner.png
#    :align: center
#    :width: 70%
#
#    ..
#
#    Wigner function of non-Gaussian state
#
# It does not have the shape of a Gaussian and moreover, it can be negative, a tell-tale feature of  
# non-Gaussian states (we can only interpret this function as a true probability distribution for the case of Gaussian states!). 
# The only issue is that the non-Gaussian state is produced only with some probability, that is,
# when the detectors measure some particular number of photons. But, at the very least, we can 
# be sure that we have obtained the non-Gaussian state we wanted in that case, and otherwise we 
# just discard the qumode. For more precise calculations, you can check out `this 
# tutorial <https://the-walrus.readthedocs.io/en/latest/gallery/gkp.html>`__ from PennyLane's sister library The Walrus, 
# which is optimized for simulating this type of circuit.  
#
# Encoding qubits into qumodes
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# It's great that we can manipulate quantum states of light so freely, but we haven't discussed how 
# to use them for quantum computing. What we would like is a way to encode qubits into qumodes, so 
# that we can run any algorithm designed for qubit-based quantum computers using qumodes. 
# Surely there's more than one way to encode a two-dimensional subspace into an infinite-dimensional 
# one. The only problem is that most of these encodings are extremely sensitive to the noise 
# affecting the large Hilbert space. A way that has proven to be quite robust to errors is to 
# encode qubits in states of light is using a special type of non-Gaussian states called *GKP states*.
#
# GKP states are states that are linear combinations of the following two basis states:
#
# .. math:: \vert 0 \rangle_{GKP} = \sum_{n} \vert 2n\pi\rangle_x,
# .. math:: \vert 1 \rangle_{GKP} = \sum_{n} \vert (2n+1)\pi\rangle_x,
#
# where the subscript :math:`x` means that the kets in the sum are eigenstates of the quadrature observable 
# :math:`\hat{X}`. Therefore, an arbitrary qubit :math:`\vert \psi \rangle = \alpha\vert 0 \rangle + \beta\vert 1 \rangle` 
# can be expressed through the qumode as
#
# .. math:: \vert \psi \rangle_{GKP} = \alpha\vert 0 \rangle_{GKP} + \beta\vert 1 \rangle_{GKP}.
#
# Producing these GKP states is physically impossible. But we can produce approximate versions of them and still
# calculate with great precision. In fact, the non-Gaussian state that we wrote in the previous section is 
# one of these approximate GKP states. We already have all the tools to produce them!
#
# We can remain within the subspace spanned by the GKP basis states by restricting the operations we apply
# on our qumodes. For example, we see that applying a displacement by :math:`\sqrt{\pi}` to :math:`\vert 0 \rangle_{GKP}` gives the state 
# :math:`\vert 1 \rangle_{GKP}`, and vice versa. Therefore, the displacement operator corresponds to the qubit 
# bit-flip gate :math:`X`. Similarly, a rotation operator by :math:`\pi/2` implements the Hadamard gate. 
# The figure below gives more detail on how to implement all the gates we need for 
# universal quantum computation using optical gates on  exact GKP states.
#
# .. rst-class:: docstable
#
#     +---------------------+--------------------------------------------------------------+---------------------------------------------------------------------+
#     | .. centered::       | .. centered::                                                | .. centered::                                                       |
#     |  Qumode Gate        |  Optical Diagram                                             |  Qubit gate on GKP states                                           |
#     +=====================+==============================================================+=====================================================================+
#     | Displacement        | .. figure:: ../demonstrations/photonics/Displacement.png     | Pauli X gate if displacement is in :math:`x` direction.             |
#     |                     |    :align: center                                            | Pauli Z gate if displacement is in :math:`p` direction              |
#     |                     |    :width: 70%                                               |                                                                     | 
#     +---------------------+--------------------------------------------------------------+---------------------------------------------------------------------+
#     | Rotation            | .. figure:: ../demonstrations/photonics/Rotation.png         | Hadamard gate for :math:`\phi=\pi/2`.                               |
#     |                     |    :align: center                                            |                                                                     |
#     |                     |    :width: 70%                                               |                                                                     |
#     +---------------------+--------------------------------------------------------------+---------------------------------------------------------------------+
#     | Continuous variable | .. figure:: ../demonstrations/photonics/CV_ctrlz.png         | CNOT                                                                |
#     | CNOT                |    :align: center                                            |                                                                     |
#     |                     |    :width: 100%                                              |                                                                     |
#     |                     |                                                              |                                                                     |
#     +---------------------+--------------------------------------------------------------+---------------------------------------------------------------------+
#
# The above already allow for a universal set of gates on the GKP qubits. Even if their effect is approximate, they are quick
# and quite straightforward to implement with our current technology.  Therefore, we have all the ingredients to build a universal
# quantum computer using photons, summarized in the formula:
#
# .. figure:: ../demonstrations/photonics/formula_qc.png
#    :align: center
#    :width: 70%
#
# The state of the art
# ~~~~~~~~~~~~~~~~~~~~
#
# We have now learned the basics of how to build a quantum computer using photonics. So what challenges 
# are there to overcome for scaling further? 
# Let us analyze what we have learned in terms of Di Vincenzo's criteria. 
# 
# From what we've learned above, the third, fourth, and fifth criteria are satisfied by linear optical
# quantum computers. There is room for improvement for gate precision and qubit measurements.
# But the main challenges are posed by the first and second criteria. 
# We do have well-defined and scalable qubits thanks to our ability to produce GKP states. 
# But scalability does present a challenge that some other technologies bypass. The quantum gates need to be 
# physically built, as opposed to, for example, using lasers to change the qubit's state (see tutorials on 
# `trapped ions <https://pennylane.ai/qml/demos/tutorial_trapped_ions.html>`__ and
# `superconducting qubits <https://pennylane.ai/qml/demos/tutorial_sc_qubits.html>`__). 
# As a consequence, more powerful photonic quantum computers will occupy more physical space. 
#
# The main challenge lies in the second criterion: the ability to *prepare a qubit*. 
# We need GKP states, but these cannot be prepared deterministically; we have to get a bit lucky. 
# We can bypass this by *multiplexing*, that is, using many 
# Gaussian Boson Sampling circuits in parallel. But we would need many of these circuits,  
# which occupy physical space and require photon detectors held at low temperatures. Moreover, the GKP states
# that we produce are not exact. Producing them with better fidelity needs larger circuits, 
# which in turn exponentially decreases the probability of obtaining them. 
# 
# What can we do instead? Xanadu is currently following a hybrid approach.
# When we fail to produce a GKP state, the output of a Gaussian Boson Sampling circuit is
# a squeezed state entangled with other qumodes. Should we just get rid of it? Not at all! Strongly-entangled squeezed
# states are still a precious resource, so we shouldn't just throw them away. Indeed, using other encodings beyond GKP allows us
# to use these highly-entangled squeezed states as a resource for quantum computing, although they're more prone to error. This approach
# reduces the amount of multiplexing needed to perform quantum computations.
#
# .. figure:: ../demonstrations/photonics/chip.png
#    :align: center
#    :width: 40%
#
#    ..
#
#    Xanadu's X8 Gaussian Boson Sampling chip. Variants of this chip can be used to generate approximate GKP states.
# 
# 
# Conclusion
# ~~~~~~~~~~ 
#
# The approach of photonic devices to quantum computing is quite different from their qubit-based counterparts. Recent theoretical
# and technological developments have given a boost to their status as a scalable approach, although the generation of qubits
# remains a challenge to overcome. The variety of ways that we can encode 
# qubits into photonic states leave plenty of room for creativity, and opens the
# door for further research and engineering breakthroughs. If you would like to learn more about photonics, make sure to 
# check out the `Strawberry Fields demos <https://strawberryfields.ai/photonics/demonstrations.html>`__, 
# as well as the references listed below. 
#
# References
# ----------
#
# .. [#DiVincenzo2000]
#
#     D. DiVincenzo. (2000) "The Physical Implementation of Quantum Computation",
#     `Fortschritte der Physik 48 (9–11): 771–783
#     <https://onlinelibrary.wiley.com/doi/10.1002/1521-3978(200009)48:9/11%3C771::AID-PROP771%3E3.0.CO;2-E>`__.
#     (`arXiv <https://arxiv.org/abs/quant-ph/0002077>`__)
#
#
# About the author
# ----------------
#
##############################################################################
# .. bio:: Alvaro Ballon
#    :photo: ../_static/Alvaro.png
#
#    Alvaro Ballon is a quantum computing educator at Xanadu. His work involves making the latest developments
#    in quantum computing and PennyLane accessible to the community. 
#


