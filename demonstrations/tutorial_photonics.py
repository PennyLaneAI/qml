r""".. _photonics:

Photonic quantum computers
=============================

.. meta::
    :property="og:description": Photon-based quantum computers
    :property="og:image": https://pennylane.ai/qml/_images/photonics_tn.png

.. related::
   tutorial_pasqal Quantum computation with neutral atoms

*Author: PennyLane dev team. Posted: 17 May 2022. Last updated: 17 May 2022.*

To create a functional quantum computer, we need to gather and control a 
large number of qubits. This feat has proven difficult, although significant 
progress has been made using trapped ions, superconducting circuits, 
and many other technologies. Scalability — the ability to put many 
qubits together — is limited because individual qubits in a multi-qubit 
system lose their quantum properties quickly. This phenomenon, 
known as decoherence, happens due to the interactions of the qubits with 
each other and their surroundings. One way to get scalable qubits is to 
use photons (particles of light). Photons rarely affect each other and 
their quantum state is not easily destroyed, so we may be on to something!

The theory for how photonic quantum computers work is different from what 
you may be used to. While there are some proposals to use polarization 
states of photons, some of the most succesful implementations don't 
use photonic qubit states directly. Instead, we use qumodes: states of 
light that are superpositions of infinitely many basis states. The 
formalism that deal with qumodes is known as continuous variable 
quantum mechanics. But no need to panic. By the end of this demo, 
you will be able to explain how photonic qumode devices can accomplish 
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

"""

##############################################################################
#
# Gaussian states of light
# ~~~~~~~~~~~~~~~~~~~~~~~~
#
# Why are the quantum states of light so durable? Photons seldom interact with each other, 
# which means we can easily avoid uncontrolled interactions that destroy their quantum 
# state. However, to build a universal quantum computer, we need multi-qubit gates, 
# which means that photons must communicate with each other! We *can* make photons 
# affect each other, but for now, let's focus on the quantum states of light that 
# we can theoretically obtain when photons don't need to interact. Such states are called 
# **Gaussian states**, and they can be fabricated with a 100% success rate
# using common optical devices. 
#
#
# To precisely define a Gaussian state is, we need a mathematical representation for states 
# of light. As is the case with qubits, states of light are represented by a linear 
# combination of basis vectors. But unlike qubits, two basis vectors aren't enough. 
# Light is characterized by its position and momentum *quadratures* :math:`x` and :math:`p`, 
# measured by the observables :math:`\hat{X}` and :math:`\hat{P}` respectively. These names can 
# be confusing, since the quadratures *do not* represent the position and momentum 
# of a photon. They are actually related to the intensity and phase of light. The 
# quadratures can take any real value, so :math:`\hat{X}` and :math:`\hat{P}` must have 
# infinitely many eigenvectors. Therefore, to describe a quantum state of light 
# :math:`\left\lvert \psi \right\rangle` we need infinitely many basis vectors! Such 
# representation of a quantum state is called a **qumode**. For example, we write
#
# .. math:: \left\lvert \psi \right\rangle = \int_\mathbb{R}\psi(x)\vert x \rangle dx,
#
# where :math:`\vert x \rangle` is the eigenstate of :math:`\hat{X}` with position quadrature 
# equal to :math:`x`, and :math:`\psi` is a complex-valued function known as the *wave function*. 
# A similar expansion can be done in terms of the eigenstates :math:`\vert p \rangle` of :math:`\hat{P}`.
#
# So how do we define a Gaussian state using this representation? It is a state that is completely 
# determined by the average values :math:`\bar{x}` and :math:`\bar{p}` of the position and momentum quadratures, 
# as well as their standard deviations :math:`\Delta x` and :math:`\Delta p`. The most trivial (and boring) 
# Gaussian state is the *vacuum state* -the state with no photons. Let us see what happens if 
# we sample measurements of the quadratures :math:`\hat{X}` and :math:`\hat{P}` when light is in the vacuum state. 
# We use the ``default.gaussian`` device, which lets us create, manipulate, and measure Gaussian 
# states of light. Let's first call the usual imports,
#

import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

##############################################################################
#
# and define the device. Note that the number of wires corresponds to the number of qumodes.

dev = qml.device('default.gaussian', wires = 1, shots=500)

##############################################################################
#
# We would like to know how the measured values of position and momentum 
# are distributed in the :math:`x-p space`, usually called *phase space*. 
# The initial state in default.gaussian is the vacuum state, so the circuits 
# to measure the quadratures are empty! Then we proceed to plot 500 
# measurement results for both :math:`x` and :math:`p`.

@qml.qnode(dev)
def vacuum_measure_x():
    return qml.sample(qml.X(0)) #Samples X quadratures
@qml.qnode(dev)
def vacuum_measure_p():
    return qml.sample(qml.P(0)) #Samples P quadrature

# We plot the sample measurements in phase space
x_sample = vacuum_measure_x().numpy()
p_sample = vacuum_measure_p().numpy()
fig, ax1 = plt.subplots(figsize=(9,6))
ax1.scatter(x_sample, p_sample)
ax1.set_title("Vacuum state",fontsize=16)
ax1.set_ylabel("Momentum", fontsize=14)
ax1.set_xlabel("Position", fontsize=14)
ax1.set_aspect('equal', adjustable='box')
plt.show()

##############################################################################
#
# We observe that the values of the quadratures are distributed around the 
# origin with a spread of approximately 1. We can check these eyeballed values explicitly, 
# using a device without shots this time. 

dev_exact=qml.device('default.gaussian', wires=1)
@qml.qnode(dev_exact)
def vacuum_mean_x():
    return qml.expval(qml.X(0))
@qml.qnode(dev_exact)
def vacuum_mean_p():
    return qml.expval(qml.P(0))
@qml.qnode(dev_exact)
def vacuum_var_x():
    return qml.var(qml.X(0))
@qml.qnode(dev_exact)
def vacuum_var_p():
    return qml.var(qml.P(0))

print(vacuum_mean_x())
print(vacuum_mean_p())
print(vacuum_var_x())
print(vacuum_var_p())

##############################################################################
#
# What other gaussian states are there? The states produced by lasers are called *coherent states*, 
# which are also gaussian with :math:`\Delta x = \Delta p = 1`. The vacuum is but an example of a 
# coherente state, but these, in general, may have non-zero expectation values for the 
# quadratures. The default.gaussian device allows for the easy preparation of coherent states 
# through `qml.CoherentState`. This function takes two parameters :math:`\alpha` and :math:`\phi`, 
# which are the polar coordinates of the point :math:`(\bar{x}, \bar{p})` in phase space. Let us plot
# sample measurement of the quadratures for a coherent state.
 
@qml.qnode(dev)
def measure_coherent_x(alpha,phi):
    qml.CoherentState(alpha,phi,wires=0)
    return qml.sample(qml.X(0))

@qml.qnode(dev)
def measure_coherent_p(alpha,phi):
    qml.CoherentState(alpha,phi,wires=0)
    return qml.sample(qml.P(0))

x_sample_coherent = measure_coherent_x(2,np.pi/3).numpy()
p_sample_coherent = measure_coherent_p(2,np.pi/3).numpy()

fig, ax1 = plt.subplots(figsize=(9,6))
ax1.scatter(x_sample_coherent, p_sample_coherent)
ax1.set_title("x and p measurements",fontsize=16)
ax1.set_ylabel("Momentum", fontsize=14)
ax1.set_xlabel("Position", fontsize=14)
ax1.set_aspect('equal', adjustable='box')
plt.show()

##############################################################################
#
# But where does the name Gaussian come from? If we plot the density 
# of quadrature measurements for a coherent state, we obtain the following plot.
# 
# .. figure:: ../demonstrations/photonics/vacuum_wigner.png
#    :align: center
#    :width: 70% 
# 
# The density has the shape of a 2-dimensional Gaussian surface, hence the name. For Gaussian 
# states only, the density is exactly equal to the Wigner function :math:`W(x,p)`, 
# defined using the wave function :math:`\psi(x)`:
#
# .. math:: W(x,p) = \frac{1}{\pi\hbar}\int_{-\infty}^{\infty}\psi^{*}(x+y)\psi(x-y)e^{2ipy/\hbar}dy.
#
# Since the Wigner function is positive and integrates to 1 for Gaussian states,
# it is *almost as if* the values of the momentum and position had an underlying classical 
# probability distribution, save for the fact that the quadratures can't be measured simultaneously. 
# For this reason, Gaussian states are considered to be classical. Now we're ready 
# for the technical definition of a Gaussian state: **a qumode is said to be Gaussian if its 
# Wigner function is a 2-dimensional Gaussian function.** 
#
# Gaussian operations
# ~~~~~~~~~~~~~~~~~~~~~~~~
#
# We have only learned about two Gaussian states of light so far. The vacuum state can be obtained 
# by simply doing nothing (provided all lights are turned off) and a coherent state is simply 
# a pulse of laser light. How can we obtain any Gaussian state of our liking? 
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
#     |                     |    :align: center                                            | fraction :math:`t` of the photons coming in, and reflects the rest. |
#     |                     |    :width: 100%                                              | It can entangle states of light if it takes in two qumodes          |
#     |                     |                                                              |                                                                     |
#     +---------------------+--------------------------------------------------------------+---------------------------------------------------------------------+
#
# What if, for example, I would like to change the expectation value of the :math:`x` 
# quadrature without changing anything else about the state? This can be done 
# via the *displacement operator*, implemented in PennyLane via qml.Displacement. 
# Let's see the effect of this operation on an intial coherent state.

@qml.qnode(dev)
def displace_coherent_x(alpha,phi,x):
    qml.CoherentState(alpha,phi,wires=0)
    qml.Displacement(x,0, wires=0)
    return qml.sample(qml.X(0))

@qml.qnode(dev)
def displace_coherent_p(alpha,phi,x):
    qml.CoherentState(alpha,phi,x)
    qml.Displacement(x,0, wires=0)
    return qml.sample(qml.P(0))

displaced_x=displace_coherent_x(3,np.pi/3,1)
displaced_p=displace_coherent_x(3,np.pi/3,1)
fig, ax1 = plt.subplots(figsize=(9,6))
ax1.scatter(displaced_x, displaced_p)
ax1.set_title("after displacement",fontsize=16)
ax1.set_ylabel("Momentum", fontsize=14)
ax1.set_xlabel("Position", fontsize=14)
ax1.set_aspect('equal', adjustable='box')
plt.show()
 
##############################################################################    
#
# Exactly as we expected. But how do we make a displacement in the lab? One method 
# is shown below, which uses a beam splitter and a source of high-intensity coherent light
#
# .. figure:: ../demonstrations/photonics/Displacement.png
#    :align: center
#    :width: 70% 
#
# We can check that this setup implements a displacement operator using PennyLane. This time,
# we need two qumodes, since we rely on combining the qumode we want to displace with a 
# coherent state in a beam-splitter

dev2 = qml.device('default.gaussian', wires = 2, shots=500)

@qml.qnode(dev2)
def disp_optics(z,x):
    qml.CoherentState(z,0,wires=0)
    qml.CoherentState(3,np.pi/3,wires=1)
    qml.Beamsplitter(np.arccos(1-x**2/z**2),0,wires=[0,1])
    return qml.sample(qml.X(1))
@qml.qnode(dev2)
def mom_optics(z,x):
    qml.CoherentState(z,0,wires=0)
    qml.CoherentState(3,np.pi/3,wires=1)
    qml.Beamsplitter(np.arccos(1-x**2/z**2),0,wires=[0,1])
    return qml.sample(qml.P(1))

displaced_x=disp_optics(100,1)
displaced_p=mom_optics(100,1)
fig, ax1 = plt.subplots(figsize=(9,6))
ax1.scatter(displaced_x, displaced_p)
ax1.set_title("after displacement",fontsize=16)
ax1.set_ylabel("Momentum", fontsize=14)
ax1.set_xlabel("Position", fontsize=14)
ax1.set_aspect('equal', adjustable='box')
plt.show()

############################################################################## 
#
# Similarly, we can implement rotations in phase space using ``qml.Rotation``, which 
# simply amounts to changing the phase of light using a thermo-optic phase shifter. 
# 
# We have # focused on changing the expectation values of :math:`x` and :math:`p`, 
# but what if we also want to change the spread of the quadratures? That corresponds to 
# creating *squeezed states*, which is more difficult than simply using 
# beamsplitters and phase shifters. It requires shining light through non-linear 
# materials, where the state of light will undergo unitary evolution in a 
# way that changes :math:`\Delta x` and :math:`\Delta p` 
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
# In PennyLane, we can generate squeezed states thorugh the squeezing 
# operator qml.Squeezing. This function depends on the squeezing parameter 
# :math:`r` which tells us how much the variance in :math:`x` is reduced, and phi 
# which rotates the state in phase space. (Need to show a code block and a Wigner function)
#
# Measuring quadratures
# ~~~~~~~~~~~~~~~~~~~~~
#
# Now that we know how to manipulate Gaussian states, we would like to perform measurements 
# on them. So far, we have taken for granted that we can measure the quadratures 
# :math:`\hat{X}` and :math:`\hat{P}`. But how do we actually measure them using optical elements? 
# We will need a measuring device known as a photon counter. These are made of 
# photoelectric materials, where each outer electron can be stimulated by a photon. 
# The more photons that are incident on the photon counter, the more electrons are 
# pulled out of an atom, which in turn form an electric current in the material. Mathematically,
#
# .. math:: I = qN
#
# where :math:`I` is the electric current, :math:`N` is the number of photons, and :math:`q` is a detector-dependent 
# proportionality constant. Hence, measuring the current amounts to measuring the number of photons!
#
# The number of photons in a quantum state of light is not fixed. It is measured by the quantum 
# photon-number observable :math:`\hat{N}`, which has eigenstates denoted :math:`\vert 0 \rangle, \vert 1\rangle, \vert 2 \rangle,...`. 
# These states, known as *Fock states*, do have a well-defined number of photons: 
# repeated measurements of :math:`\hat{N}` on the same state will yield the same output. 
# The natural number :math:`n` in the Fock state :math:`\vert n \rangle` denotes the only possible 
# result we would get upon measuring :math:`N`. But nothings prevent light from being in a superposition of 
# Fock states. For example, when we measure :math:`\hat{N}` for the state
#
# .. math:: \vert \psi \rangle = \frac{1}{\sqrt{3}}\left(\vert 0 \rangle + \vert 1 \rangle + \vert 2 \rangle\right),
#
# we get 0, 1, or 2 photons, each with probability :math:`1/3`. 
#
# Except for the vacuum :math:`\vert 0 \rangle,` **Fock states are not Gaussian**. Gaussian states are, in general, 
# superpositions of Fock States. For example, let's measure the photon number for some squeezed state:

dev3 = qml.device('default.gaussian', wires = 1)
@qml.qnode(dev3)
def measure_n_coherent(alpha,phi):
    qml.Squeezing(alpha,phi,wires=0)
    return qml.expval(qml.NumberOperator(0))

measure_n_coherent(1,np.pi/3)

############################################################################## 
#
# Since the expectation value is not an integer number, the measurement results cannot have been all the same integer!
#
# .. figure:: ../demonstrations/photonics/Homodyne.png
#    :align: center
#    :width: 70%
#
#    ..
#
#    Measuring quadratures using photon counters
#
# Now we know how to measure the number of photons, which is good and all. But what about the promised 
# quadratures? We can do this through a combination of quadrature measurement and a beamsplitter, 
# as shown in the diagram below. 


dev_exact2=qml.device('default.gaussian', wires = 2)
@qml.qnode(dev_exact2)
def measurement(a,phi):
    qml.Displacement(a,phi,wires=0)
    return qml.expval(qml.X(0))

@qml.qnode(dev_exact2)
def measurement2(a, theta, alpha, phi):
    qml.Displacement(a,theta, wires=0)
    qml.CoherentState(alpha, phi, wires = 1)
    qml.Beamsplitter(np.pi/4, 0, wires = [0,1])
    return qml.expval(qml.NumberOperator(0)), qml.expval(qml.NumberOperator(1))

print(measurement(3,0))
print(measurement2(3,0,1,0))

############################################################################## 
#
# Although PennyLane does not allow us to sample NumberOperator, hopefully trying the above 
# with many input states will convince you that this setup, known as **Homodyne Measurement**, 
# allows us to measure the quadratures :math:`\hat{X}` and :math:`\hat{P}`. 
#
# Beyond Gaussian states
# ~~~~~~~~~~~~~~~~~~~~~~
#
# We've learned a lot about Gaussian states now, but they don't seem to have many quantum properties. 
# They are described by an underlying probability distribution of the quadratures, given by the Wigner 
# function, so how is this different from classical states? These are legitimate concerns and, 
# indeed, to build  photonic quantum computer we need both entangled *and* non-Gaussian states. 
# The former are not a problem, however, since beamsplitters already entangle the input states!
#
# Let us then on the more challenging mission to find a way to prepare non-Gaussian states. 
# All of the operations that we have learned so far: displacements, rotations, squeezing, are Gaussian. 
# Do we need some kind of strange material that will implement a non-Gaussian operation? That's certainly 
# a possibility, and some examples can be found in the Kerr and Cubic phase interactions. But relying 
# on these non-linear materials is far from optimal, since we don't have much freedom to manipulate 
# the setup into getting an arbitrary non-Gaussian state. 
#
# But there's one non-Gausian operation that's been right in front of our eyes all this time. 
# Photon counters take a Gaussian state and collapse it into a Fock state; therefore, photon 
# number detection is not a Gaussian operation. Combined with squeezed states and beamsplitters, 
# we have all the ingredients to produce any non-Gaussian state that we like. 
#
# Let's explore how this works. The main idea is to tweak a particular photonic circuit known 
# as *Gaussian Boson Sampling*, which is shown below:
#
# .. figure:: ../demonstrations/photonics/GBS.png
#    :align: center
#    :width: 70%
#
#    ..
#
#    The Gaussian Boson Sampling circuit
#
# Gaussian boson sampling is interesting on its own. The output probabilities that we just obtained 
# allow us to calculate the Hafnian of a matrix, which is classically a hard problem 
# (see this tutorial for an in-depth discussion). This was the great accomplishment of Jiuzhang, 
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
# produce a good approximation of any non-Gaussian state that we want. For example, 
# the choice of parameter (write parameters) produces the state
#
# *Write GKP state*
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
# non-Gaussian states (we can only interpret this function as a probability for Gaussian states!). 
# The only issue is that the non-Gaussian state is produced only with some probability, that is,
# when the detectors measure some particular number of photons. But, at the very least, we can 
# be sure that we have obtained the non-Gaussian state we wanted in that case, and otherwise we 
# just discard the qumode. PennyLane is not the optimal tool to work with non-Gaussian states, 
# but you can check out this The Walrus tutorial, which is optimized for simulating this type of 
# circuit.  
#
# Encoding qubits into qumodes
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# It's great that we can manipulated quantum states of light so freely, but we haven't discussed how 
# to use them for quantum computing. What we would like is a way to encode qubits into qumodes, so 
# that we can run any algorithm designed for qubit-based quantum computers using qumodes. 
# Surely there's more than one way to encode a two dimensional space into an infinite-dimensional 
# one. The only problem is that most of these encodings are extremely sensitive to the noise 
# affecting the large Hilbert space. A way that has proven to be quite robust to errors is to 
# encode qubits in states of light is using a special type of non-Gaussian states called *GKP states*.
# In fact the non-Gaussian state that we wrote in the previous section is (approximately) a GKP state!
#
# GKP states are states that are linear combinations of the following two basis states:
#
# .. math:: \vert 0 \rangle_{GKP} = \sum_{n} \vert 2n\pi\rangle_x,
# .. math:: \vert 1 \rangle_{GKP} = \sum_{n} \vert (2n+1)\pi\rangle_x,
#
# where the subscrpit :math:`x` means that the kets in the sum are eigenstates of the quadrature observable 
# :math:`\hat{X}`. We see that applying a displacement by :math:`\sqrt{\pi}` to :math:`\vert 0 \rangle_{GKP}` gives the state 
# :math:`\vert 1 \rangle_{GKP}`, and viceversa. Therefore, the displacement operator corresponds to the qubit 
# bit-flip gate :math:`X`. Similarly, a rotation operator by :math:`\pi/2` implements the Hadamard gate. 
# The figure below gives more detail on how to implement all the gates we need for 
# universal quantum computation using optical gates on GKP states.
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
#     +---------------------+--------------------------------------------------------------+---------------------------------------------------------------------+
#
#
# The state of the art
# ~~~~~~~~~~~~~~~~~~~~
#
# We have now learned the basics of how to build a quantum computer using photonics. So what is preventing this approach
# from scaling further? In terms of Di Vincenzo's criteria, it is the first one, the ability to *prepare a qubit*, that poses
# a challenge. We need GKP states, but these cannot be prepared deterministically; we have to get a bit lucky. We can bypass this
# by *multiplexing*, that is, using many Gaussian Boson Sampling circuits in parallel. But we would need a lot of these circuits,  
# which do require the photon detectors to be held at low temperatures. Moreover, generating more precise GKP states 
# requires more qumodes, which in turn exponentially decreases the probability of obtaining them.
# 
# What can we do instead? When we fail to produce a GKP state, the output of a Gaussian Boson Sampling circuit is
# a squeezed state entangled with other qumodes. Should we just get rid of it? Not at all! Strongly entangled squeezed
# states are still a precious resource, so we shouldn't just throw them away. Indeed, using other encodings beyond GKP allows us
# to use these highly entangled squeezed states as a resource for quantum computing, although they're more prone to error. This approach,
# currently used by Xanadu, reduces the amount of multiplexing needed to perform quantum computations.
# 
# Conclusion
# ~~~~~~~~~~ 
#
# The approach of photonic devices to quantum computing is quite different from their qubit-based counterparts. Recent theoretical
# and technological developments have given a boost to their status as a scalable approach, although the generation of qubits
# remians troublesome. The variety of ways that we can encode qubits into photons state leave plenty of room for creativity, and open the
# door for further research and engineering breakthroughs. If you would like to learn more about photonics, make sure to 
# check out the Strawberry Fields demos, as well that the references listed below. 
#
#
#
#
#

