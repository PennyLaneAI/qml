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

*Author: Alvaro Ballon — Posted: XX March 2023.*

In the last few years, a new quantum technology has gained the attention of the quantum computing
community. Thanks to recent developments in optical-tweezer technology,
neutral atoms can be used as robust and versatile qubits. In 2020, a collaboration between
various academic institutions produce a neutral-atom device with a whooping 256 qubits 😲! It is 
no surprise that this family of devices have gained traction in the private sector, with startups such 
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
Afterwards, we will learn how to perform measurements on the atom's states. Finally, we will explore the state-of-the-art 
in neutral-atom devices and the work that still needs to be done to scale this up technology even further.

"""

##############################################################################
#
# Trapping individual atoms
# -------------------------
#
# In our cousin demo about trapped-ion technologies, we learned that we can trap individual charged
# atoms by carefully controlled electric fields. But neutral atoms, by definition, have no charge, 
# so they can't be affected by electric fields. How can we even hope to manipulate them indivudually?
# It turns out that the technology to do this has been around for decades! 
# **Optical tweezers**—highly focused laser beams—can grab small objects and hold them in place. 
# Let's see how they are able to do this.
#
# Laser beams are nothing but electromagnetic waves, that is, oscillating electric and magnetic
# fields. It would seem that a neutral atom could not be affected by them—but it can! To understand how, we need
# to keep in mind two facts. First, in a laser beam, light is more intense at the center of the beam and 
# and it dims progressively as we go towards the edges. This means that the average
# strength of the electric fields is higher closer to the center of the beam. 
# Secondly, as small as neutral atoms are, they're not just
# points. They do carry charges that can move around relative to each other when we expose them to electric fields.
#
# The consequence of these two observations is that, if an atom inside a laser beam tries to escape towards the edge
# of the beam, the negative charges will be pulled towards the center of the beam, while the positive charges are pushed
# away. But, since the electric fields are stronger towards the center, the
# negative charges are pulled more strongly, so more negative charge will accumulate in the center. 
# Therefore, these negative charges will pull the positive charge that's trying to escape back to the middle. You can 
# look at the figure below to gain a bit more intuition.
# 
# In the last decade, optical tweezer technology has evolved to the point where we can move atoms around
# into customizable arrays (check out :doc:`this tutorial </demos/tutorial_pascal>` and have some fun doing this!).
# This means that we have a lot of freedom in how  and when our atom-encoded qubits interact with each other. Sounds 
# like a dream come true! However, there is a big challenge to address: unlike, for example, trapped ions,
# neutral atoms do not interact strongly with each other. Therefore, implementing two-qubit gates—which are 
# needed for universality—poses a bit of a challenge, 
# Later, we will see how to get around this so that we can, indeed, perform universal computations using an
# array of neutral atoms. But first, let us understand how neutral atoms can be used as qubits.
#
# Encoding a qubit in an atom
# ---------------------------
#
# To encode a qubit in a neutral atom, we need to have access to two distinct atomic quantum states. The most 
# easily accessible quantum states in an atom are the electronic energy states.  We would like to **switch
# one electron between two different energy states**, which means that we must make sure not to affect other
# electrons when we manipulate the atom. For this reason, the most ideal atoms to work with are those with 
# one valence electron, i.e. one "loose" electron that is not too tightly bound to the nucleus. 
# 
# .. note::
#
#    In some cases, such as the devices built by Atom Computing,
#    qubits are not encoded in atomic energy levels, but in so called nuclear-spin
#    energy levels instead. Such qubits are known as **nuclear spin qubits**. In this demo, we will not focus
#    on the physics of these qubits. However, similar principles to those we'll outline in this demo
#    for qubit preparation, control, and measurement will apply for this type of qubit.
# 
# 
# A common choice is the Rubidium
# atom, given that it's commonly used in atomic physics and we have the appropriate technology to change their
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
#    excited state . If a photon hits an electron, it will go to that
#    higher energy state. When the light stimulus is removed, the excited
#    electrons will return to stable states. The time it takes them to do
#    so depends on the particular excited state they are in since,
#    sometimes, the laws of physics will make it harder for electrons to
#    jump back on their own.
#
#    .. figure:: ../demonstrations/trapped_ions/atomic.png
#       :align: center
#       :width: 60%
#
#       ..
#
# But even if we've chosen one electron in the atom, we need to make sure that we are effectively
# working with only two energy levels in that atom. This ensures that we have a qubit!
# One of the energy levels will be a ground state
# for the valence electron, which we call *fiducial state* and denote by :math:`\lvert 0 \rangle.` 
# The other energy level will be # an excited state that is long-lived, known as a hyperfine state, denoted by :math:`\lvert 1 \rangle.`  
# We'll induce transitions between these two states using light whose energy matches the energy difference between
# these atomic levels.
#
#
# Initializing the qubits
# -----------------------
#
# We have chosen our atom and its energy levels, so the easy part is over! But there are still some difficult tasks
# ahead of us. In particular, we need to isolate individual atoms inside our optical 
# tweezers *and* make sure that they are all in the same initial state—known as the **fiducial
# ground state**—as required, by di Vincenzo's second criterion. This fiducial state
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
# let's figure out how all the electrons end up in the same energy state. It turns out that Rubidium is 
# the ideal atom not only because it has one valence electron, but also because it has a **closed optical loop.** 
# Let's look at the picture below to understand it the meaning of this term.
#
# Rubidium-85 has two ground states :math:`\vert 0\rangle` and :math:`\vert \bar{0}\rangle`, which are excited 
# using the laser to two excited states :math:`\vert 1\rangle` and :math:`\vert \bar{1}\rangle` respectively. However, 
# both of these excited states will decay to :math:`\vert 0\rangle` with high probability. This means that no
# matter what ground state the electrons occupied initially, they will be driven to the same ground state
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
# measuring the state of our system. One might wonder... isn't measurement the last step in a quantum circuit?
# Aren't we skipping ahead a little bit? Not really! Once we have our initial state, we should measure it to 
# verify that we have indeed prepared the fiducial state. After all, some of the steps we carried out to
# prepare the atoms aren't really foolproof; there are two issues that we need to address.
# 
# The first problem is that traps are designed to trap *at most* one atom. This means that zome traps might contain
# **no** atoms! Indeed, in the lab, it's usually the case that half of the traps aren't filled. The second issue is
# that laser cooling is not deterministic, which means that some atoms may not be in the ground state. We would like
# to exclude those from our initial state. Happily, there is a simple solution that addresses these two problems.
#
# To verify that a neutral atom is in the fiducial state :math:`\left 0 \rangle`, we shine a photon on it that stimulates 
# the transition between this state to some short-lived excited state :math:`\left h \rangle`. Electrons excited in this way will 
# promptly decay to the state $\left 0 \rangle$ again, emitting light. The electrons that were in some state different than
# :math:`\left 0 \rangle`, never get excited, since the photon does not have the right energy. And, of course, nothing will happen
# in traps where there is no atom. The net result is that atoms in the ground state will shine, while others won't. This
# phenomenon, known as fluoresence, is also used in trapped ion technologies. The same method can be used at the end of 
# a quantum computation to measure the final state of the atoms in the computational basis.
#
# Single qubit gates 
# ------------------
#
# Driving excitations with pulses
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# To make our valence electron change quantum states, we need to act on it with a *light pulse*. A light pulse refers
# to a short burst of light whose amplitude and phase are carefully controlled over time. When a pulse of light whose frequency
# matches with that of the electron is shone upon the atom, then the *Hamiltonian* describing this interaction is 
# 
#
#
# Where :math:`Omega(t)` is the time-dependent amplitude of the pulse, and :math:`\phi` is the phase. 
#
#
#
#
#
#
# Two qubit-gates
# ---------------
#
# The Rydberg blockade
# ~~~~~~~~~~~~~~~~~~~~
#
# The Ctrl-Z gate
# ~~~~~~~~~~~~~~~
#
#
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
#
# .. [#Weedbrook2012]
#
#     C. Weedbrook, et al. (2012) "Gaussian Quantum Information",
#     `Rev. Mod. Phys. 84, 621
#     <https://journals.aps.org/rmp/abstract/10.1103/RevModPhys.84.621>`__.
#     (`arXiv <https://arxiv.org/abs/1110.3234>`__)
#
# .. [#Sabouri2021]
#
#     S. Sabouri, et al. (2021) "Thermo Optical Phase Shifter With Low Thermal Crosstalk for SOI Strip Waveguide"
#     `IEEE Photonics Journal vol. 13, no. 2, 6600112
#     <https://ieeexplore.ieee.org/document/9345963>`__.
#
# .. [#Paris1996]
#
#     M. Paris. (1996) "Displacement operator by beam splitter",
#     `Physics Letters A, 217 (2-3): 78-80
#     <https://www.sciencedirect.com/science/article/abs/pii/0375960196003398?via%3Dihub>`__.
#
# .. [#Braunstein2005]
#
#     S. Braunstein, P. van Loock. (2005) "Quantum information with continuous variables",
#     `Rev. Mod. Phys. 77, 513
#     <https://journals.aps.org/rmp/abstract/10.1103/RevModPhys.77.513>`__.
#     (`arXiv <https://arxiv.org/abs/quant-ph/0410100>`__)
#
# .. [#Hamilton2017]
#
#     C. Hamilton , et al. (2017) "Gaussian Boson Sampling",
#     `Phys. Rev. Lett. 119, 170501
#     <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.119.170501>`__.
#     (`arXiv <https://arxiv.org/abs/1612.01199>`__)
#
# .. [#Zhong2020]
#
#     H.S. Zhong, et al. (2020) "Quantum computational advantage using photons",
#     `Science 370, 6523: 1460-1463
#     <https://www.science.org/doi/10.1126/science.abe8770>`__.
#     (`arXiv <https://arxiv.org/abs/2012.01625>`__)
#
# .. [#Madsen2020]
#
#     L. Madsen, et al. (2022) "Quantum computational advantage with a programmable photonic processor"
#     `Nature 606, 75-81
#     <https://www.nature.com/articles/s41586-022-04725-x>`__.
#
# .. [#Tzitrin2020]
#
#     I. Tzitrin, et al. (2020) "Progress towards practical qubit computation using approximate Gottesman-Kitaev-Preskill codes"
#     `Phys. Rev. A 101, 032315
#     <https://journals.aps.org/pra/abstract/10.1103/PhysRevA.101.032315>`__.
#     (`arXiv <https://arxiv.org/abs/1910.03673>`__)
#
# .. [#Gottesman2001]
#
#     D. Gotesman, A. Kitaev, J. Preskill. (2001) "Encoding a qubit in an oscillator",
#     `Phys. Rev. A 64, 012310
#     <https://journals.aps.org/pra/abstract/10.1103/PhysRevA.64.012310>`__.
#     (`arXiv <https://arxiv.org/abs/quant-ph/0008040>`__)
#
# .. [#Bourassa2021]
#
#     E. Bourassa, et al. (2021) "Blueprint for a Scalable Photonic Fault-Tolerant Quantum Computer",
#     `Quantum 5, 392
#     <https://quantum-journal.org/papers/q-2021-02-04-392/>`__.
#     (`arXiv <https://arxiv.org/abs/2010.02905>`__)
#
# About the author
# ~~~~~~~~~~~~~~~~
# .. include:: ../_static/authors/alvaro_ballon.txt
