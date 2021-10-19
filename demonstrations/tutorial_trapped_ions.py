r"""

A guide to build a quantum computer. Part I: Trapped ions
=========================================================


"""


##############################################################################
# The race for building a quantum computer that everyone can use started
# decades ago, and we can say that this goal has been reached. Users
# worldwide can now access a variety of quantum computers through the
# Cloud. However, using a current quantum computer does not present
# intrinsic advantages for everyday applications. But the race for
# supremacy is on, and the competitors are moving fast. Hundreds of
# research institutions, including universities and privately funded
# companies, are figuring out the best way to build a scalable quantum
# computer. Many have been successful to a degree. A select few have
# claimed quantum advantage: the ability to solve problems quickly that a
# classical computer would take too long to solve. Nonetheless, we are
# still far from implementing the most ambitious and useful quantum
# algorithms.
# 
# The different players are using various technologies to build a quantum
# computer. The three main approaches are to use **ion traps,
# superconducting circuits, or photons**. Each of these has advantages and
# disadvantages, and discussing whether one of them is superior leads to a
# neverending debate. All of them pose complex technological challenges,
# which we can only solve through innovation, inventiveness, hard work,
# and a bit of luck. It is difficult to predict whether these problems are
# solvable in a given timeframe. More often than not, our predictions have
# been wrong, but new knowledge is always produced in the process.
# Therefore, the purpose of the series of tutorials is not to speculate.
# Instead, we intend to summarize what we have learned about the viable
# realizations of quantum computers and identify the obstacles that need
# to be overcome.
# 
# In this first part of the series, we introduce **trapped ion quantum
# computers**. It is the preferred technology used by research groups at
# prestigious universities like Oxford and ETH and companies like
# Honeywell and IonQ. In particular, Honeywell has achieved a quantum
# volume of 64, the largest in the market. As the name suggests, the
# qubits are ions trapped in a potential well and are manipulated using
# lasers. Trapped ions have long coherence times, which means that the
# qubits are long-lived. Additionally, they can easily interact with their
# neighbours. However, they do have some severe scalability problems,
# which we will analyze later on.
# 

##############################################################################
# What makes a good quantum computer?
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 

##############################################################################
# To build a quantum computer, we will find it helpful to formulate some
# concrete objectives. In the year 2000, the physicist David DiVincenzo
# proposed a wishlist for the experimental characteristics of a quantum
# computer. **DiVincenzo’s criteria** have since become the main guideline
# for physicists and engineers in the quantum computing industry and
# academia. They are based on the features that made classical computers
# so successful. However, they are not always easy to implement on a
# quantum computer. Even though they were formulated two decades ago, they
# are still relevant today. Let us write them down and outline how trapped
# ions satisfy these criteria.
# 

##############################################################################
# 1. **Well-characterized and scalable qubits**. Every quantum computing
#    algorithm uses qubits: quantum mechanical systems that can be in two
#    primary states. However, many of the quantum systems that we find in
#    nature are not two-level systems. In the case of trapped ions, we
#    deal with atomic energy levels, which are infinitely many. Using
#    ingenious techniques, we can restrict ourselves to the transitions
#    between two energy levels, effectively making a qubit. The selection
#    of atomic levels is made such that the excited state is long-lived.
#    Such choices are not unique, and we will see that they come with
#    different advantages and drawbacks.
# 
# 2. **Qubit initialization**. It is important that we are able to prepare
#    every physical system that we are working on in a stable quantum
#    state, denoted in general by :math:`\left\lvert 0 \right\rangle`. We
#    can use lasers to induce the electrons in the ion to occupy a
#    specific stable energy level. This process is called optical pumping.
# 
# 3. **Long coherence times**. One of the main obstacles in building a
#    quantum computer is isolating the qubits. If we do not, the
#    environment will change their state by interacting with them too
#    much. This phenomenon is called decoherence and is the primary source
#    of error in quantum mechanical experiments. However, the qubits
#    cannot be completely isolated. After all, to carry out quantum
#    computations, we need to be able to manipulate them. Therefore, we
#    would like to have qubits that are resistant to decoherence. Ion
#    traps give us a choice of energy levels, and some electrons can stay
#    in these for a long time despite not being completely isolated.
# 
# 4. **Universal set of gates**. These include two types of gates, each of
#    which present unique challenges in their implementation. We would
#    like them to be implemented in short periods of time. Indeed,
#    decoherence will destroy our state if we do not act fast.
# 
#    -  **Single-qubit gates**: We need to make arbitrary operations on
#       single qubits since this is required for a quantum computer that
#       can run all possible logical operations. In ion traps, this is
#       done by manipulating the ions with laser pulses.
# 
#    -  **Two-qubit gates**: In particular, we must be able to entangle
#       qubits. Entanglement is required to make our quantum computer a
#       universal Turing Machine and to achieve quantum advantage. Often,
#       this is one of the most difficult challenges when building a
#       quantum computer. Moreover, we would like to perform operations on
#       any two qubits, which is even more complicated. We can implement
#       two-qubit gates in trapped ions by making an ion chain oscillate
#       using specific frequencies of light. However, this procedure
#       becomes less precise as the chain becomes longer.
# 
# 5. **Measurement of individual qubits**. At the end of a quantum
#    algorithm, we always need to read a result through a measurement. In
#    quantum mechanics, measurements are probabilistic, so we may need to
#    measure several times and derive results from the obtained
#    statistics. A very convenient feature of the trapped ion paradigm is
#    that we can measure without losing the ion, and this means that we
#    can keep reusing the qubits we already have without the need to
#    generate more. All we will need to do is prepare them in the ground
#    state again.
# 

##############################################################################
# How to trap an ion
# ~~~~~~~~~~~~~~~~~~
# 

##############################################################################
# Why do we use ions, that is, charged atoms, as qubits? The main reason
# is to contain them in one precise location using electric fields; we
# could never do this with a neutral atom since electric forces do not
# affect them. The history of ion traps goes back to 1953 when Wolfgang
# Paul proposed his now-called Paul trap. This technique awarded Paul and
# Dehmelt the 1989 Physics Nobel Prize since it is used to make highly
# precise atomic clocks. Current trapped ion quantum computers extensively
# use the Paul trap, but Paul won the prize six years before such an
# application was proposed!
# 


##############################################################################
# It is not easy to create electric fields that contain the ion in one
# small region of space. From Laplace’s equation in electrostatics, we can
# show that it is not possible to create a confining potential with only
# time independent electric fields. The sought for potential would look
# like this:
# 

# %matplotlib inline 
import matplotlib.pyplot as plt
import numpy as np 
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D

u,v = np.mgrid[-6:6:100j, -6:6:100j]
x = u
y = v
z = 2*u**2+2*v**2

fig = plt.figure(figsize=(12,12));

ax = fig.add_subplot(111, projection='3d');
ax.plot_surface(
x, y, z,  rstride=1, cstride=1, cmap='viridis', alpha=0.4,linewidth=0);
ax._axis3don = False
ax.view_init(elev=10, azim=40);
ax.set_aspect('auto');

##############################################################################
# However, such potentials are not allowed in electrostatics. Instead, we
# can obtain saddle-point potentials:
# 

# %matplotlib inline 
import matplotlib.pyplot as plt
import numpy as np 
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D

u,v = np.mgrid[-6:6:100j, -6:6:100j]
x = u
y = v
z = 2*v**2-2*u**2

fig = plt.figure(figsize=(12,12));

ax = fig.add_subplot(111, projection='3d');
ax.plot_surface(
x, y, z,  rstride=1, cstride=1, cmap='viridis', alpha=0.4,linewidth=0);
ax._axis3don = False
ax.view_init(elev=0, azim=30);
ax.set_aspect('auto');


##############################################################################
# This potential is problematic, since the ion is contained in one
# direction but could escape in the perpendicular direction. Therefore,
# the solution is to use time dependent electric fields. What would
# happen, for example, if we rotated the potential plotted above? We can
# imagine that if the saddle potential rotates at a specific frequency,
# then it will catch the ion as it tries to escape in the concave
# direction of the potential. Explicitly, the electric potential that we
# generate is given by
# 
# .. math:: \Phi = \frac{1}{2}\left(u_x x^2 + u_y y^2 + u_z z^2\right) + \frac{1}{2}\left(v_x x^2 + v_y y^2 + v_z z^2\right)\cos(\Omega t+\phi)
# 
# The parameters :math:`u_i`, :math:`v_i` and :math:`\phi` need to be
# adjusted to the charge and mass of the ion we want to contain and to the
# frequency :math:`\Omega` with which our potential rotates. We have to do
# this tuning of parameters very carefully, since the ion could escape if
# we do not apply the right forces at the right time.
# 

##############################################################################
# We want to make a quantum computer, so having one qubit cannot be
# enough. We would like as many as we can possibly afford! The good news
# is that we have the technology to trap many ions and put them close
# together in a one dimensional array, called an **ion chain**. Why do we
# need this particular configuration? In order to manipulate the qubits,
# we need the system of ions to absorb photons. However, shooting a photon
# at an ion can cause some unwanted relative motion between ions in the
# chain. Proximity between qubits will cause unwanted interactions, which
# could modify their state. Happily, there is a solution to this issue:
# place the ions in a sufficiently spaced one-dimensional array, and
# **cool them all down to the point where their motion in space is
# quantized**. In this scenario, photons that would bring the ion to their
# excited states will not cause any relative motion. Instead, all ions
# will recoil together. This phenomenon is called the **Mossbauer
# effect**. We will see later that by carefully tuning the frequency of
# the photons, we can cause some relative motion between the ions. This
# user-controlled motion is precisely what we need to perform quantum
# operations with two qubits.
# 

##############################################################################
# Trapped ions as robust qubits
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 


##############################################################################
# Now that we know how to trap ions, we would like to use them as qubits.
# Would any ion that we choose work well as a qubit? In fact, only a
# select few isotopes of particular elements will do the trick. The reason
# is that our qubit basis states are the ground and excited states of an
# electron in the atom, and we need to be able to transition between them
# using laser light. Therefore, we would like the atom to have an excited
# state that is long-lived, and also one that we may manipulate using
# frequencies that lasers can produce. Thanks to semiconductor laser
# technology, we have a wide range of frequencies that we can use in the
# visible and infrarred range, so getting a desired frequency is not too
# much of a problem. The best ions for our purposes are single-charged
# ions in Group II of the periodic table, such as Calcium-40, Berilium-9,
# and Barium-138, commonly used in university laboratories. The rare earth
# Yterbium-171 is used by IonQ and Honeywell.
# 


##############################################################################
# Having chosen the ions that will act as our qubits, we need to be able
# to prepare them in a stable fiducial state, known as the **ground
# state** and denoted by :math:`\left\lvert 0 \right\rangle`. The
# preparation is done by a procedure called optical pumping. To see how
# this works, let us take Calcium-40 as an example. In this case, the
# electron has two stable states, denoted by
# :math:`\left\lvert S_{1/2}, m_j=-1/2\right\rangle`,
# :math:`\left\lvert S_{1/2}, m_j=1/2\right\rangle`. We do not know which
# state the electron is in, and we would like make sure that the electron
# is in the
# :math:`\left\lvert 0\right\rangle = \left\lvert S_{1/2}, m_j=-1/2\right\rangle`
# state, since it is our fiducial state of choice. However, quantum
# mechanics forbids a direct transition between these two stable states.
# Instead, we take a detour: by using circularly polarized laser light of
# a particular wavelegnth (397nm for Calcium-40), we excite
# :math:`\left\lvert S_{1/2}, m_j=1/2\right\rangle` into a short lived
# excited state :math:`\left\lvert P_{1/2}, m_j=-1/2\right\rangle`. This
# light does not estimulate any other transitions in the ion, so that an
# electron that was already in the ground state will remain there. The
# theory of quantum mechanical angular momentum tells us that, in a matter
# of nanoseconds, the excited electron decays to our desired ground state
# with probability 1/3, but returns to
# :math:`\left\lvert S_{1/2}, m_j=1/2\right\rangle` otherwise. For this
# reason, we need to repeat the procedure many times, gradually “pumping”
# the electrons in all (or the vast majority of) our ions to the ground
# state.
# 

##############################################################################
# What about the other basis qubit state? This will be a long-lived state,
# denoted by :math:`\left\lvert 1 \right\rangle`. For the Calcium-40 ion,
# there is a metastable state, denoted by :math:`D_{5/2}`, with a
# half-life of about 1 second. While apparently short, most quantum
# operations can be performed in a time scale of micro to milliseconds.
# Nevertheless, the short lifespan is a source of error since a percentage
# of the excited states could decay much quicker. The energy difference
# between the ground and excited state corresponds to a laser frequency of
# 729nm, achievable with an infrarred laser. Therefore, we call this an
# **optical qubit**. An alternative is to use an ion, such as Calcium-43,
# that has a hyperfine structure. This means that there are two states
# separated by a very small energy gap, which makes the higher energy
# state have a virtually infinite lifespan. We can use a procedure similar
# to optical pumping to transition between these two states, so while
# coherence times are longer for these **hyperfine qubits**, gate
# implementation is harder and needs a lot of precision.
# 


##############################################################################
# We have now learned how trapped ions make for very stable qubits that
# allow us to implement many quantum operations without decohering too
# soon. We have also learned how to prepare these qubits in a stable
# ground state. Does this mean that we have already satisfied DiVincezo’s
# first, second, and third criteria? We have definitely satisfied the
# second one, since optical pumping is a very robust method. However, we
# have mostly been focusing on a single qubit. Since we have not discussed
# scalability yet, we have not fully satisfied the first criterion.
# Introducing more ions will also pose additional challenges to satisfying
# the third criterion. For now, let us focus on how to satisfy criteria 4
# and 5, and we will come back to these issues once we discuss what
# happens when we deal with multiple ions.
# 


##############################################################################
# Non-demolition measurements
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 

##############################################################################
# Let us now discuss the last step in a quantum computation: measuring the
# qubits. Since it takes quite a bit of work to trap an ion, it would be
# ideal if we could measure the state of our qubits without getting rid of
# the ion. We definitely do not want to trap ions again after performing
# one measurement. Such a measurement is called a **non-demolition
# measurement**, and for trapped ions, they are easy enough to carry out.
# 


##############################################################################
# The measurement method uses a similar principle to that of optical
# pumping. Once again, and continuing with the Calcium-40 example, we make
# use of the auxiliary state :math:`\left\lvert 1 \right\rangle`. This
# time, we shine a laser light wavelength of 397 nm that drives the
# transition from :math:`\left\lvert 0 \right\rangle` to the auxiliary
# state. The transition is short lived, so we will measure
# :math:`\left\lvert 0 \right\rangle` if we see the ion glowing: it
# continuously emits light at a frequency of 397 nm. Consersely, if the
# ion is dark, we assign will have measured the result corresponding to
# state :math:`\left\lvert 1\right\rangle`. To see the photons emitted by
# the ions, we need to collect the photons using a lens and a
# photomultiplier, which is a device that transforms weak light signals
# into electric currents.
# 


##############################################################################
# In light of all these considerations, we have satisfied the fifth
# criterion. Via a careful experimental arrangement, we can detect the
# emission of photons of each atom individually. In reality, there is also
# some uncertainty in the measurement, since there can be spontaneous
# emission of photons by the excited state
# :math:`\left\lvert 1 \right\rangle`. But since such events are uncommon,
# ion traps do achieve high-fidelity measurements of output states. Next,
# we will focus on the foruth criterion: controlling the qubits to perform
# quantum operations.
# 


##############################################################################
# Rabi oscillations to manipulate single qubits
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 

##############################################################################
# How do we make single-qubit quantum gates? So far, we have been using
# lasers to excite states, but is there a way to put the electron in the
# ion in a superposition of the ground and excited states? Since our aim
# is to change the energy state of an electron, we have no choice but to
# continue using lasers to shoot photons at it, tuning the frequency to
# that of the energy gap. To understand how we would achieve a state
# superposition by interacting with the ion using light, let us take a
# look at the interaction Hamiltoian between the ion and surrounding
# radiation. After many simplifications involving the rotating wave
# approximation, the interaction part of the Hamiltonian is given by:
# 
# .. math:: \hat{H}=\frac{\hbar\Omega}{2}\left(S_+ e^{i\varphi}+S_{-}e^{-i\varphi}\right).
# 

##############################################################################
# Here, :math:`\Omega` is the Rabi frequency. It is defined by
# :math:`\Omega=\mu_m B/2\hbar`, where :math:`B` is the applied magnetic
# field due to the laser and :math:`\mu_m` is the magnetic moment of the
# ion. The phase :math:`\varphi` measures the phase of the light at the
# position of the atom at time zero. Using Schrodinger’s equation, we know
# that an initial state :math:`\left\lvert 0 \right\rangle` will evolve
# into the following time dependent state:
# 
# .. math:: \left\lvert \psi(t)\right\rangle = \exp(-i \hat{H} t/\hbar)\left\lvert 0 \right\rangle
# 

import pennylane as qml
from pennylane import numpy as np
from scipy.linalg import expm

Omega=100
S_plus = np.array([[0,0],[1,0]])
S_minus = np.array([[0,1],[0,0]])

def evolution(phi,t):
    Ham = Omega/2*(S_plus*np.exp(1j*phi)+S_minus*np.exp(-1j*phi))
    return expm(-1j*Ham*t)

dev = qml.device("default.qubit",wires=1)

@qml.qnode(dev)
def circuit(phi,t,state=0):
    
    if state ==1:
        qml.PauliX(wires=0)
    
    qml.QubitUnitary(evolution(phi,t),wires=0)
    
    return qml.state()



##############################################################################
# *Ok I haven’t finished this, but the idea is to show that for
# appropriate values of t and phi we can obtain T gates and Hadamard
# gates, for example*.
# 


##############################################################################
# In fact, we can solve the Schrodinger equation explicitly (feel free to
# do this if you want to pratice solving differential equations!). If we
# do this, we can deduce that the ground state
# :math:`\left\lvert 0 \right\rangle` evolves to
# 
# .. math:: \left\lvert \psi_0(t) \right\rangle = \cos\left(\frac{\Omega t}{2}\right)\left\lvert 0 \right\rangle -i\sin\left(\frac{\Omega t}{2}\right) e^{-i\varphi}\left\lvert 1 \right\rangle .
# 
# This solution is called a **Rabi oscillation**. We observe that we can
# obtain an arbitrary superposition of qubits by adjusting the duration of
# the interaction and the phase. This means that we can produce any
# single-qubit gate! To be more precise, let us see what would happen if
# the initial state was :math:`\left\lvert 1 \right\rangle`. As before, we
# can show that the evolution is given by
# 
# .. math:: \left\lvert \psi_1(t) \right\rangle = -i\sin\left(\frac{\Omega t}{2}\right)e^{i\varphi}\left\lvert 0 \right\rangle +\cos\left(\frac{\Omega t}{2}\right)\left\lvert 1 \right\rangle .
# 

##############################################################################
# Therefore, the unitary induced by a laser pulse of amplitude :math:`B`,
# duration :math:`t`, and phase :math:`\varphi` on an ion with magnetic
# moment :math:`\mu_m` is
# 
# .. math::  U(\Omega,\varphi,t)=\left( \begin{array}{cc} \cos\left(\frac{\Omega t}{2}\right) & -i\sin\left(\frac{\Omega t}{2}\right)e^{-i\varphi} \\ -i\sin\left(\frac{\Omega t}{2}\right)e^{i\varphi} & \cos\left(\frac{\Omega t}{2}\right)\end{array}\right)
# 
# We can then calculate the characteristics of the pulse needed to
# implement a general rotation of the form
# 
# .. math::  R(\theta,\phi)=\left( \begin{array}{cc} \cos\left(\frac{\theta}{2}\right) & i\sin\left(\frac{\theta}{2}\right)e^{i\phi} \\ i\sin\left(\frac{\theta}{2}\right)e^{-i\phi} & \cos\left(\frac{\theta}{2}\right)\end{array}\right)
# 
# Do you see now how we calculated the values of :math:`t` and
# :math:`\phi` to generate the gates before? Indeed, it suffices to choose
# :math:`t=\theta/\Omega` and :math:`\phi=-\varphi` to implement a general
# rotation.
# 


##############################################################################
# As we see, for typical Rabi frequencies, the gates can be implemented in
# a few milliseconds. This means that, even for the seemingly short
# lifespans of optical qubits, we can implement quantum algorithms
# involving many gates. As a consequence, we have now satisfied half of
# criterion 4. The second half is not theoretically difficult to
# implement, but it can be experimentally hard, and for some years it was
# challenging to implement two qubit gates in a reasonable period of time.
# 


##############################################################################
# Entangling ions
# ~~~~~~~~~~~~~~~
# 

##############################################################################
# Scalability challenges
# ~~~~~~~~~~~~~~~~~~~~~~
# 


##############################################################################
# The state of the art
# ~~~~~~~~~~~~~~~~~~~~
# 

