r"""
Finding Ground State Energies via Quantum Phase Estimation
==========================================================

 .. meta::
    :property="og:description": This demo introduces to the
    Quantum Phase Estimation algorithm and shows users how to implement a
    QPE subroutine in a quantum program with PennyLane for estimating
    molecular energies.
    :property="og:image": https://pennylane.ai/qml/demonstrations/function_fitting_qsp/cover.png 
                    
*Authors: Davide Castaldo and Aleksei Malyshev (Xanadu residents). Posted: 25 July 2022*
"""
######################################################################
#
# Hi! It seems that you were wandering around the PennyLane website and
# opened the demo called “Finding Ground State Energies via Quantum Phase
# Estimation”. We guess this happened because you’ve heard that:
#
# 1. Quantum computers will make finding ground state energies easier.
# 2. The  quantum phase estimation (QPE) algorithm is likely to be involved — and you are keen to learn how ground state energies and QPE go together.
#
# Well, in this case you
# You've come to the right place, because in this demo we will:
#
# 1. Explain the connection between ground state energies and quantum phases.
# 2. Recap the quantum phase estimation algorithm which is good at, well, estimating quantum phases.
# 3. Show how to make the QPE estimate phases related to ground state energies.
# 4. Mention some caveats.
# 5. Of course, provide the PennyLane implementation!
#
#

######################################################################
# Relating energies to quantum phases
# ===================================
# 
# Suppose we have a system described by a Hamiltonian :math:`\hat{H}` and
# :math:`|n\rangle` are its eigenstates,
# i.e. :math:`\hat{H}|n\rangle = E_n|n\rangle, \ E_n \in \mathbb{R}`. Let us now allow the system
# evolve for a time :math:`t`. The time-dependent Schrödinger equation
# tells us that the evolution operator is
# :math:`\hat{U}(t) = e^{-\frac{i\hat{H}t}{\hbar}}` which can be decomposed as:
#
# .. math:: \hat{U}(t) = \sum_{n} e^{-\frac{i E_n t}{\hbar}} |n\rangle\langle n|
#
# Now, let’s act with :math:`\hat{U}(t)` on any of
# Hamiltonian eigenstates: 
#
# .. math::  \hat{U}(t)|m\rangle = \sum_{n} e^{-\frac{i E_n t}{\hbar}} |n\rangle \underbrace{\langle n|m\rangle}_{\delta_{nm}} = \sum_{n} e^{-\frac{i E_n t}{\hbar}} |n\rangle \delta_{nm} = e^{-\frac{i E_m t}{\hbar}} |m\rangle.
#
# We have used the orthogonality condition for Hamiltonian eigenstates
# :math:`{\langle n|m\rangle} = \delta_{nm}`, where :math:`\delta_{nm} = 1` if :math:`n = m` and
# :math:`\delta_{nm} = 0` otherwise.
#
# What we have just obtained? Well, our derivation shows
# that as the system evolves, its eigenstates basically stay the same — up
# to a global quantum phase. Importantly, the acquired phase is not
# random, for each eigenstate it is related to its energy in a known way.
# 
# Okay, now it is clear what the recipe for finding the ground state
# energies might look like. Suppose we have the ground state of some
# system at our disposal — then, we can just let it evolve for some known
# time :math:`t`, measure the accumulated phase
# :math:`\varphi_0 = -\frac{E_{0}t}{\hbar}` and deduce the energy from it
# as :math:`E_{0} = -\frac{\hbar \varphi_0}{t}`. Woop-woop, isn't it great!
#
# However, at least three possible complications might immediately come to your mind:
# 
# 1. **How on Earth we could measure the global phase?**
# Indeed, any textbook on quantum physics will tell you that the global
# phase for an isolated system is *unobservable*, i.e. it can’t be detected by any measurements
# performed on the system. Well, our system need not be
# isolated, right? We can make it interact with another, *ancilla* system
# controlled by us. Then, the global phase will (in a sense) become local
# for a composite quantum system and we will be able to detect it — and
# that is exactly what the QPE algorithm does in a clever way. So, bear
# with us, we’ll come back to the QPE shortly.
# 
# 2. **Phases are defined modulo** :math:`2\pi` **— how on Earth we can be
# sure we have measured the correct one?** You are right, we can’t.
# However, we can easily get overcome this with a couple of tricks which we will explain later.
# 
# 3. **How on Earth we can have the ground state at our disposal when we
# even don’t know its energy?**  Alright, this is a serious one.
# Finding the ground state of a system is indeed barely simpler than finding the ground state energy, and so it
# seems ridiculous to presume that we can use the former to get the
# latter. However, as it turns out having *a good approximation* of the
# ground state also suffices in many cases, although it complicates
# matters to an extent. We will demonstrate later how the this inaccuracy
# manifests itself and how to deal with its consequences.
# 
# All in all, for now we have the high-level picture: measuring the
# quantum phase of an evolved state brings us information about its energy.
# So let’s proceed to a more detailed consideration of the low-level
# components.
# 

######################################################################
# Quantum Phase Estimation
# ========================
# 
# As we have already mentioned, the Quantum Phase Estimation algorithm
# does a seemingly impossible thing: it allows us to figure out the
# global phase acquired by an evolved quantum state. Let us formalize this
# statement a bit and say that in the simplest form QPE solves the
# following problem:
# 
# **Problem 1.** Given a unitary :math:`\hat{U}` and its eigenstate
# :math:`|u\rangle` such that
# :math:`\hat{U} |u\rangle = e^{i 2\pi \varphi_{u}} |u\rangle`, find the
# corresponding value of :math:`\varphi_{u} \in [0, 1)`.
# 
# .. raw:: html
# 
#    <!-- Just in case, don't be afraid of an ominous requirement for $\varphi_{u}$ to be a $k$-bit binary fraction — it just makes the QPE succeed after one run. We will mention later that if $\varphi_{u}$ is not a $k$-bit binary fraction, then it will take several executions of QPE to get $\varphi_{u}$ to a desired accuracy. -->
# 
# Now, let’s go through the main stages of QPE. First, let’s have a look
# at the circuit implementing the algorithm.
# 


######################################################################
# .. figure:: ../demonstrations/qpe/qpe_circuit.png
#    :align: center
#    :alt: QPE Circuit
#    :width: 70%
#

######################################################################
# You can see that the circuit has two registers of qubits. The first one
# consists of :math:`K` qubits and is a *readout* register (and it’s not
# accidental that it has as many qubits as there are bits
# in the binary representation of :math:`\varphi_{u}`). The second
# register is called *system* register and it has :math:`N` qubits, where
# :math:`n` is the number of qubits in the considered Hamiltonian
# :math:`\hat{H}`. Perhaps, you can already guess that we will encode the
# state :math:`|u\rangle` in the system register, entangle it in some
# complicated way with the readout register, measure the latter and *read
# out* the value of :math:`\varphi_{u}`. More formally we can say that QPE
# includes four main steps:
# 
# 0. **Initialization.** We prepare the system register in the state
# :math:`|u\rangle` (let us call the preparation gate :math:`\hat{P}`).
# Also, we apply Hadamard gate to each qubit in the readout register,
# which creates a uniform superposition of all computational basis
# states in that register. By the end of this stage the state of the
# system is:
#
# .. math:: |\Psi_0\rangle = \left(\frac{|0\rangle + |1\rangle}{\sqrt{2}}\right)^{\otimes K} \otimes |u\rangle
#
#
# 1. **Phase encoding.** Here we apply a sequence of :math:`K` controlled
# unitaries. The :math:`k`-th unitary is controlled by :math:`k`-th
# qubit in the readout register and applies the operator
# :math:`\hat{U}^{2^{k - 1}}` to the system register. As a result, the
# information about quantum phase :math:`\varphi_{u}` appears in the
# readout register. To see that, let’s consider the action of
# :math:`k`-th unitary on :math:`k`-th qubit of the readout register
# and the system register (i.e. on qubits actually affected by the
# unitary):
#
# .. math:: \widehat{CU^{2^{k - 1}}} \left[\left(\frac{|0\rangle_k + |1\rangle_k}{\sqrt{2}}\right)\otimes |u\rangle\right] &= \frac{1}{\sqrt{2}}\left[|0\rangle_k \otimes |u\rangle + |1\rangle_k \otimes \hat{U}^{2^{k - 1}} |u\rangle \right] = \\
#                                                    &= \frac{1}{\sqrt{2}}\left[|0\rangle_k \otimes |u\rangle + |1\rangle_k \otimes e^{i 2 \pi \cdot 2^{k - 1} \varphi_u} |u\rangle \right] = \\
#                                                    &= \left[\left(\frac{|0\rangle_k + e^{i 2 \pi \cdot 2^{k - 1} \varphi_u}|1\rangle_k}{\sqrt{2}}\right)\otimes |u\rangle\right]
#
#
# As a result, by the end of this stage the information
# about the phase :math:`\varphi_{u}` is encoded in the first register
# and the full system state is:
# 
# .. math:: |\Psi_1\rangle = \left[ \otimes_{k=1}^{K} \left(\frac{|0\rangle_k + e^{i 2 \pi \cdot 2^{k - 1} \varphi_u} |1\rangle_k}{\sqrt{2}}\right)\right] \otimes |u\rangle.
#
#
# The approach we used is called *phase kickback* and
# as you have just seen it indeed *kicks back* the global phase from
# one register to another.
#
#
# 2. **Phase decoding.** Now the QPE algorithm does seemingly little but
# in fact achieves a lot. Namely, it applies the *inverse* quantum
# Fourier transform (quantum FT, QFT) to the readout register. Why it does so? Well, if
# we expand the brackets in the expression for :math:`|\Psi_1\rangle`,
# we can see that it is equivalent to:
#
# .. math:: |\Psi_1\rangle = \left[ \otimes_{k=1}^{K} \left(\frac{|0\rangle_k + e^{i 2 \pi \cdot 2^{k - 1} \varphi_u} |1\rangle_k}{\sqrt{2}}\right)\right] \otimes |u\rangle = \sum_{j=0}^{2^K - 1} e^{i \cdot 2 \pi \varphi_u j} |j\rangle |u\rangle.
#
#
# Here by :math:`|j\rangle` we mean a state of the
# readout register given by the binary expansion of :math:`j`, i.e. if
# :math:`j = j_K \cdot 2^{K - 1} + \ldots + j_2 \cdot 2^1 + j_1 \cdot 2^0`,
# then :math:`|j\rangle = |j_1 j_2 \ldots j_K\rangle` (it differs from the
# ordinary PennyLane convention, but in practice this wouldn't matter — for now we
# are interested in understanding the QPE, not implementing it).
# 
# Now, if we define :math:`\omega = 2 \pi \varphi_u`, then we can see that
# the amplitudes of the readout register encode the function
# :math:`f(j) = e^{i \omega j}`. Does it remind you of anything? Well, if
# we merely rename :math:`j` to :math:`t`, we obtain
# :math:`f(t) = e^{i \omega t}`, which looks exactly like a complex
# sinusoidal wave. Hence, it is not a big surprise that we want to apply
# the inverse Fourier transform (IFT) to this function and thus extract
# the value of :math:`\omega` (we apply the IFT here because the
# conventionally defined direct FT outputs :math:`\omega` for
# :math:`g(t) = e^{-i \omega t}`). The only difference is that in our case
# “the time” is discrete, and therefore we will have to use the inverse
# *discrete* Fourier Transform — and in the language of quantum computing
# it means that we have to apply a unitary corresponding to the inverse QFT, just
# because *by definition* QFT performs Fourier transform on the state amplitudes.
# 
# Long story short, the state of the system at the end of this stage is:
#
# .. math:: |\Psi_2\rangle = \sum_{l = 0}^{2^K - 1} \alpha_l |l\rangle |u\rangle,
#
# where :math:`\alpha_l := \mathfrak{F}^{-1}\left[e^{i 2 \pi \varphi_u j}\right](l)`,
# i.e. to calculate :math:`\alpha_l` one applies the inverse DFT (denoted as :math:`\mathfrak{F}^{-1}`)
# to the function :math:`e^{i 2 \pi \varphi_u j}` and then takes its value at the
# point (at the discrete frequency) :math:`l`.
#
# Phew, let’s proceed to the final stage! 
#
#
# 3. **Measurement.** On the one hand, this stage is very
# straightforward: we just measure the readout register. On the other
# hand, simple as it sounds, we need to figure out what the measurement
# result would be.
# 
# First, let’s suppose :math:`\varphi_u` can be exactly represented as a
# :math:`K`-bit binary fraction,
# i.e. :math:`\varphi_u = \varphi_{u, K} 2^{-1} + \varphi_{u, K - 1} 2^{-2} + \ldots \varphi_{u, 1} 2^{-K}`,
# where :math:`\varphi_{u, k} \in \{0, 1\} \ \forall k \in [1..K]`.
# Equivalently, there exists an integer
# :math:`l_{\varphi_{u}} \in [0..2^K - 1]` such that
# :math:`\varphi_{u} \cdot 2^{K} = l_{\varphi_{u}}`. In this case the
# expression for the amplitudes of the readout register is simple:
#
# .. math:: \alpha_l = \begin{cases}1, \text{if } l=l_{\varphi_{u}} \\ 0, \text{otherwise.} \end{cases}
#
# Indeed, in this case we apply the inverse FFT to a function
# :math:`f(j) = e^{i 2 \pi \varphi_{u} j}` which is represented by a *single*
# complex harmonic. Again, *by definition* of the inverse FFT, it will be non-zero only at a single
# point — namely at the angular frequency of this harmonic. Hence, the measurement of the readout
# register always brings it to the state :math:`|l_{\varphi_{u}}\rangle`,
# from which we can deduce the value of :math:`\varphi_{u}`.
# 
# Now, let’s suppose :math:`\varphi_u` *cannot* be exactly represented as
# a :math:`K`-bit binary fraction. Let :math:`\tilde{\varphi}_{u}` be the
# best :math:`K`-bit approximation of :math:`\varphi_u` such that
# :math:`\tilde{\varphi}_{u} < \varphi_u`
# (i.e. :math:`\delta := \varphi_u - \tilde{\varphi}_{u} < 2^{-K}`). In
# this case after some algebra juggling we can obtain the following
# expression for :math:`\alpha_l`:
#
# .. math:: \alpha_l = \frac{1}{2^K}\left(\frac{1 - e^{i 2\pi \left(\delta \cdot 2^K - l\right)}}{1 - e^{i 2\pi\left(\delta - \frac{l}{2^K}\right)}}\right).
#
# In general, there is more than one non-zero
# :math:`\alpha_l`, and therefore after the measurement we will be
# observing the readout register in different states :math:`|l\rangle`
# with probabilities :math:`\left|\alpha_l\right|^2`. Thus, the QPE
# algorithm becomes *stochastic*, i.e. from run to run we will be seeing
# different results.
# 
# How bad is that? To answer this question, let’s plot
# :math:`\left|\alpha_l\right|^2` for :math:`K = 10` and :math:`\varphi_u`
# equal to, say, :math:`0.27` (in this case
# :math:`\tilde{\varphi}_u = 0.0100010100_2` and we would like to obtain
# :math:`|l_{\tilde{\varphi}_u}\rangle = |276\rangle` as the state of the
# readout register after the measurement).
#

######################################################################
# .. figure:: ../demonstrations/qpe/qpe_leaking.png
#    :align: center
#    :alt: Leaking probabilities
#    :width: 70%
#

######################################################################
# We see that the probability to measure the “best approximation” value of
# :math:`l_{\tilde{\varphi}_u}=276` is not equal to 1, in some sense it
# “leaked” to other possible values of :math:`l`. In fact,
# :math:`\left|\alpha_{276} \right|^2 \approx 0.4` and the measurement
# result will be equal to :math:`l_{\tilde{\varphi}_u}` in less than half
# of QPE runs. Moreover, if you look at the inset, you can notice that the
# measurement result :math:`|277\rangle` is almost as probable as
# :math:`|276\rangle`, since
# :math:`\left|\alpha_{277} \right|^2 \approx 0.4` too.
# 
# Now you may wonder: once the probabilities are involved, how on Earth we
# can be sure that we measured something *relevant*? We agree,
# non-deterministic QPE may sound like a headache, but let us argue it’s
# not that hopeless. Indeed, have a look at the plot one more time: the
# probability leaks, but it still is mostly concentrated around the
# “optimal” value :math:`l_{\tilde{\varphi}_u}`. Hence, even though we
# might be getting wrong values for the least significant bits, we are
# likely to get the values of the most significant ones correctly. More
# formally, the probability to get an error in
# :math:`K^\prime < K` the most significant bits of :math:`\varphi_u` is
# equal to
# :math:`p = \sum_{l: \left|l - l_{\tilde{\varphi}_u} \right| > 2^{K - K^\prime}} \left|\alpha_l\right|^2`
# and it can bounded from above as
# :math:`p < \frac{1}{2\left(2^{K - K^\prime} - 2\right)}`.
# 
# As a result, it becomes clear how to mitigate the stochastic nature of
# QPE: we accept that QPE can give us an erroneous result, but instead we
# aim to make the probability of an error to be small. In particular, if
# we want to estimate :math:`K` most significant bits of :math:`\varphi_u`
# with probability of success at least :math:`1-\varepsilon`, we just need
# to add
# :math:`\Delta = \left \lceil \log_2\left(2 + \frac{1}{2\varepsilon} \right) \right\rceil`
# additional qubits to the readout register — one can check by a simple
# substitution to the formula above that in this case
# :math:`p < \varepsilon`.


######################################################################
# Aaand… We are done with understanding the QPE algorithm! Now let’s just
# briefly go over the two complications we mentioned before and then we
# are set up and ready to actually use QPE and get the molecular ground
# state energies!
# 
# Non-uniqueness of the quantum phase
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# In our description of QPE we explicitly required that
# :math:`\varphi_u \in [0, 1)`, but in principle
# :math:`\varphi_0 = -\frac{E_0 t}{2\pi \hbar}` doesn’t necessarily
# satisfy this condition. Fortunately, with classical quantum chemistry
# methods we can usually find rather good bounds :math:`E_0^{\min}` and
# :math:`E_0^{\max}` for :math:`E_0` such that
# :math:`E_0^{\min} \leq E_0 \leq E_0^{\max}`. These give us two possible
# workarounds:
#
# 1.  Use :math:`E_0^{\min}` and :math:`E_0^{\max}` to bound
# the absolute value of :math:`E_0` from above as :math:`|E_0| \leq C_0`
# and then choose such evolution time :math:`t` that
# :math:`t \leq \frac{2 \pi \hbar}{C_0}` — in this case :math:`\varphi_0`
# is guaranteed to be in :math:`[0, 1)`.
#
#
# 2. Due to the fact that phases
# are defined modulo :math:`2 \pi` the phase :math:`\tilde{\varphi}_0`
# estimated by QPE relates to :math:`\varphi_0` as follows:
#
# .. math:: \varphi_0 = \tilde{\varphi}_0 + m, \ m \in \mathbb{Z}.
#
# As a matter of fact, it is usually possible to find such
# :math:`E_0^{\min}` and :math:`E_0^{\max}` that only one integer
# :math:`m` would satisfy the relation between :math:`\varphi_0` and
# :math:`\tilde{\varphi}_0`, and thus the former would be uniqely obtained
# from the latter.
# 


######################################################################
# Initial state preparation
# ~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# Finally, let’s resolve one of the main concerns: how to prepare the
# ground state :math:`|0\rangle` corresponding to the quantum phase
# :math:`\varphi_0` ? As we have announced, this is not strictly required.
# Indeed, suppose that at the Stage 0 of QPE we prepare not an eigenstate
# of :math:`\hat{U}` but rather some superposition of its eigenstates:
#
# .. math:: |\Psi_0\rangle = \sum_{n} c_n |u_n\rangle, \ \text{where } \hat{U}|u_n\rangle = e^{i 2\pi \varphi_n} |u_n\rangle.
# 
# 
# Then, if we juggle some algebra again, we will be able to see that at
# the end of QPE phase decoding stage we have the following state in the
# readout register:
#
# .. math:: |\Psi_2\rangle = \sum_{n} c_n \sum_{l = 0}^{2^K - 1} \alpha_l^{(n)} |l\rangle |u_n\rangle.
#
# Basically it looks like a superposition of QPE results
# for different eigenstates of :math:`\hat{U}`. In particular, if we have
# previously chosen the parameters of our circuit such that the error
# probability is :math:`\varepsilon`
# 

######################################################################
# Quantum phase estimation with PennyLane
# =======================================
# 
# Now we look into detail how to compute molecular energies in PennyLane
# with a Quantum Phase Estimation algorithm. We will focus on the
# H\ :math:`_3^{+}` molecule at the equilibrium geometry.
# 
# First of all let us build our
# `molecule <https://pennylane.readthedocs.io/en/stable/code/qml_qchem.html>`__
# object:
# 

import numpy as np
import scipy

from matplotlib import pyplot as plt

import pennylane as qml
from pennylane import numpy as np
from pennylane import qchem

#### build your molecular hamiltonian ###

symbols = ['H', 'H', 'H']

BOHR_TO_ANGSTROM = 1.88973
geometry = BOHR_TO_ANGSTROM * np.array([[ 0.0056957528, 0.0235477326, 0.0],
                                        [0.5224540730, 0.8628715457, 0.0],
                                        [0.9909500019, -0.0043172515, 0.0]])

mol = qchem.Molecule(symbols, geometry, charge=1)
hamiltonian = qchem.diff_hamiltonian(mol)(geometry)


######################################################################
# When initializing the molecule object it is important to remember that
# the default unit for distances in PennyLane is the Bohr.
# 
# At this point we have to build our circuit, this is very easy using the
# qml.templates circuit provided by PennyLane!
# 

def build_qpe_unitary(hamiltonian, 
                      evol_time=1.0):    
    # Molecular hamiltonian in the computational basis
    matrix = qml.utils.sparse_hamiltonian(hamiltonian).toarray() 
    
    # Return unitary evolution operator    
    return scipy.linalg.expm(- 1j * matrix * evol_time)

qpe_unitary = build_qpe_unitary(hamiltonian)

def build_qpe_circuit(qpe_unitary,
                      state_prep_circuit,
                      target_wires,
                      estimation_wires,
                      device):
    @qml.qnode(device)
    def circuit():
        state_prep_circuit()
        qml.templates.qpe.QuantumPhaseEstimation(qpe_unitary,
                                                 target_wires=target_wires,
                                                 estimation_wires=estimation_wires)
        
        return qml.probs(wires=estimation_wires)
    
    return circuit

n_system = 2 * mol.n_orbitals # Number of qubits in the system register aka N
n_readout = 7 # Number of qubits in the readout register aka K
n_qubits = n_system + n_readout

# Indices of the qubits belonging to the system register 
target_wires = list(range(n_system)) 
# Indices of the qubits belonging to the readout register
estimation_wires = list(range(n_system, n_qubits)) 

def build_hf_prep_circuit(n_system,
                          n_electrons,
                          target_wires):
    def hf_prep_circuit():
        hf_state = qchem.hf_state(electrons=2, orbitals=len(target_wires))
        qml.BasisState(hf_state, wires=target_wires)
        
    return hf_prep_circuit

device = qml.device('lightning.qubit', wires=n_qubits)
hf_prep_circuit = build_hf_prep_circuit(n_system, 
                                        mol.n_electrons.item(),
                                        target_wires)
hf_qpe_circuit = build_qpe_circuit(qpe_unitary,
                                   hf_prep_circuit,
                                   target_wires,
                                   estimation_wires,
                                   device)


######################################################################
# The quantum function follows very closely the steps reported in Fig. 1.
# Particularly, we have a first step that concerns the initialization of
# the system register. We chose as initial state for our system the
# Hartree Fock (HF) wavefunction which is a good approximation of the exact ground state.
# 
# Then we use the
# `template <https://pennylane.readthedocs.io/en/stable/code/api/pennylane.QuantumPhaseEstimation.html>`__
# for the QPE circuit implemented in PennyLane which automatically
# provides the gates corresponding to the phase encoding and decoding
# steps of the algorithm.
# 
# In order to correctly use the template we have to provide a unitary
# matrix as a numpy array that corresponds to the evolution operator. This
# can be done constructing the molecular Hamiltonian expressed as a
# combination of Pauli matrices and then obtaining its representation in
# the computational basis which we can finally exponentiate.
# 
# The ``qpe_circuit`` function returns the probabilities of the bitstring
# states (lexicographically ordered) belonging to the readout register.
# 
# Now we are almost done, what is missing is a function that converts the
# outcome of the circuit into an actual estimate for the molecular ground
# state energy.
# 
# This is given by the following function:
# 

def compute_energy(probs, n_readout):
    return -(2 * np.pi * np.argmax(probs) / 2**n_readout)


######################################################################
# As we can notice, beside using the most sampled bitstring as per our
# estimate of the phase, we also have to multiply it by :math:`-2\pi` and
# divide it by the overall number of bistrings spanning the readout qubit
# register space. This is a result of the prefactor in the QFT operator
# and of the convention used by the PennyLane implementation for which the
# eigenvalue :math:`\lambda_u` we want to estimate is written as
# :math:`\lambda_u = e^{i 2 \pi \varphi_u}`.
# 
# Another detail that is important to keep in mind 
# is that the estimate of our eigenvalue provided by the QPE
# is made modulo :math:`2\pi`.
# 
# In general, therefore, to convert the estimated phase into the correct
# eigenvalue we use a function like the following:
# 

def extended_compute_energy(probs, n_readout, m, t):
    return -(2 * np.pi * (np.argmax(probs) / (2**n_readout * t) + m))


######################################################################
# Now we have taken into account that the estimation is made modulo
# :math:`2\pi` by adding the term :math:`2m\pi` and that the eigenvalue of
# the evolution operator depends also on the evolution time length.
# 
# For the purpose of the demo the two functions are equivalent as we have
# set :math:`t=1` and we will look at an eigenvalue whose modulus falls
# within the :math:`(0, 2\pi)` range meaning that :math:`m=0`.
# Nonetheless, if we were interested in a different scenario, an easy way
# to go would to look at the HF value for the energy by calling
# ``qml.hf.hf_energy(mol)(geometry)`` and from that choosing the right
# value of :math:`m` to use.
# 
# To see everything in action we now just need two lines of code:
# 

probs = {}
energies = {}
qpe_circuits = {}

qpe_circuits['HF'] = hf_qpe_circuit
probs['HF'] = qpe_circuits['HF']()
energies['HF'] = compute_energy(probs['HF'], n_readout)
print(f"Energy QPE = {energies['HF']}")


######################################################################
# The output estimated energy is :math:`E_{\rm QPE} = -1.27627` Ha which
# has to be compared with exact value at the Full Configuration Interaction level
# (i.e., exact diagonalization of the Hamiltonian) of :math:`E_{\rm FCI} = -1.27443` Ha.
# 
# In the next section we will look at the effects of changing the initial
# state and increasing the number of readout qubits.
# 


######################################################################
# Changing the initial state
# ==========================
# 
# In the previous section we have seen how to implement a QPE routine to
# estimate the ground state energy of the H\ :math:`_3^{+}` molecule.
# It is now time to consider
# possible scenarios one might encounter when using QPE for quantum
# chemistry and how to do it cleverly!
# 
# The first thing we will look at is the effect of the initial quantum
# state on the energy estimation and on the measurement result.
# 
# In particular, we saw in the introduction that the
# probability of measuring a bitstring is influenced by two factors: 
# 1. How well the binary representation of the string approximates a given
# eigenvalue.
# 2. The overlap of a given eigenvector with the initial
# state prepared into the system register.
# 
# Now we will consider two different scenarios to compare with the results
# obtained previously by initializing the system register into the
# :math:`|HF\rangle` state. First we will initialize the state randomly
# applying a random gate to a doubly excited (w.r.t. the HF configuration)
# Slater determinant, then we will look at the opposite scenario in which
# we prepare the system register in order to encode the actual FCI state.
# 
# To do so we define the two following functions:
# 

from scipy.stats import unitary_group

def build_random_prep_circuit(n_system,
                              target_wires):
    def random_prep_circuit():
        # And apply a random unitary matrix
        qml.QubitUnitary(unitary_group.rvs(2**n_system),
                         wires=target_wires) 
        
    return random_prep_circuit

def build_fci_prep_circuit(n_system, 
                           n_electrons,
                           target_wires):
    # FCI state is a HF state after a sequence of two
    # double excitations 
    hf_prep_circuit = build_hf_prep_circuit(n_system,
                                            n_electrons,
                                            target_wires)
    theta_fci = [0.14497247, 0.17033575]
    def fci_prep_circuit():
        hf_prep_circuit()
        
        # Particle conserving double excitation (0 -> 2, 1 -> 3)
        qml.DoubleExcitation(theta_fci[0], wires=[0, 1, 2, 3]) 
        
        # Particle conserving double excitation (0 -> 4, 1 -> 5)
        qml.DoubleExcitation(theta_fci[1], wires=[0, 1, 4, 5]) 
        
    return fci_prep_circuit


######################################################################
# As we can see the new functions include additional gates in the state
# preparation part. In the first function we have added the random gate
# feeding ``qml.QubitUnitary`` with a randomly generate unitary matrix.
# In the second function, to prepare the FCI state we are using the result of a VQE calculation;
# for a closer look at how to implement the VQE with PennyLane see the
# tutorial `A brief overview of
# VQE <https://pennylane.ai/qml/demos/tutorial_vqe.html>`__. The structure
# of the parameterized ansatz is obtained by using an
# `adaptive <https://pennylane.ai/qml/demos/tutorial_adaptive_circuits.html>`__
# procedure discarding excitations not relevant for the ground state
# description.
# 
# To check how a different initial state affects the result of the QPE we
# both compute the estimate for the GS energy as previously done and also
# plot the probability distributions for measuring the bitstring state of
# the readout register:
# 

prep_circuits = {}
prep_circuits['random'] = build_random_prep_circuit(n_system,
                                                    target_wires)
prep_circuits['FCI'] = build_fci_prep_circuit(n_system,
                                              mol.n_electrons.item(),
                                              target_wires)

for method, prep_circuit in prep_circuits.items():
    qpe_circuits[method] = build_qpe_circuit(qpe_unitary,
                                             prep_circuit,
                                             target_wires,
                                             estimation_wires,
                                             device)
    probs[method] = qpe_circuits[method]()
    energies[method] = compute_energy(probs[method], n_readout)
    print(f"Energy QPE-{method}: {energies[method]}")


######################################################################
# The energy obtained with the initial state corresponding to the FCI
# state is again :math:`E_{\rm QPE-FCI} = -1.27627` while the estimate of
# the randomly initialized state is :math:`E_{\rm QPE-Random} = -0.73631`.
# The random initial state preparation has profoundly changed our
# estimate! O.o
# 
# To understand this behavior let’s look at the distribution of the
# measurements:
# 

from matplotlib import pyplot as plt
method_to_linestyle = {
    'HF': '-',
    'FCI': '--',
    'random': '-.'
}
with plt.xkcd():
    fig, ax = plt.subplots()
    
    for method in probs:
        ax.plot(probs[method], 
                label=f"Init. state: {method}",
                linestyle=method_to_linestyle[method])

    ax.set_xlabel("Readout bitstring index ($l$)")
    ax.set_ylabel(r"Measurement probability ($\left|\alpha_l\right|^2$)")
    ax.legend()
    plt.show()


######################################################################
# Initializing the system register with a random state implies that the
# measurement distribution is dramatically changed as now we have non
# negligible overlaps with many more eigenstates than previously. This occurs because we are
# generating a state with contributions from
# determinants with a different number of electrons.
# 
# Altogether all this completely spoils the estimate of the QPE circuit
# even for a simple system as H\ :math:`_3^{+}` :/. On the other hand the
# result of feeding the QPE circuit with the exact eigenstate is that we
# are able to get a very good estimate of the true ground state energy
# with a probability close to 100%! This is what happens when a powerful
# tool such as the VQE is used as a state preparation routine to be
# combined with the clever estimation strategy provided by the QPE.
# 
# Take home message: always keep an eye on your initial state!
# 
# In the next section we will go greedy and will look at the accuracy
# improvements that we can get by adding more qubits in the readout
# register :3!
# 


######################################################################
# Improving the estimate accuracy
# ===============================
# 
# Depending on the application different levels of accuracy may
# be needed in our estimate of the energy. In quantum chemistry we usually want
# to estimate our energies with a precision that is below 1 kcal/mol to
# ensure quantitative predictions. This value is approximatively equal to
# 0.0016 Ha.
# 
# This means that the previous energy estimate, although apparently very
# precise, is not enough for our purposes!
# 
# How do we increase the accuracy of our estimate? It is very
# simple: we only need to add more qubits to the readout register. Indeed,
# by doing so, we will increase the resolution with which we are able to
# estimate our phase as the number of binary fractions that we are able to
# represent is given by :math:`2^{k}` thus exponentially increasingly with
# the number of qubits.
# 
# Let’s compute the error of our estimate for an increasing number of
# readout qubits:
# 

FCI_ENERGY = -1.27443

max_n_additional = 6
errors = []
for n_additional in range(max_n_additional):
    n_qubits = n_system + n_readout + n_additional
    device = qml.device('lightning.qubit', wires=n_qubits)
    estimation_wires = list(range(n_system, n_qubits)) 
    qpe_circuit = build_qpe_circuit(qpe_unitary,
                                    hf_prep_circuit,
                                    target_wires,
                                    estimation_wires,
                                    device)
    probs = qpe_circuit()
    errors.append(compute_energy(qpe_circuit(),
                                 n_readout + n_additional))
                 
errors = np.asarray(errors) - FCI_ENERGY

with plt.xkcd():
    fig, ax = plt.subplots()

    ax.plot(list(range(n_readout, n_readout + max_n_additional)), 
               np.abs(errors), 
               label=f"Init. state: HF", 
               linestyle='-.')

    ax.set_xlabel(r"$K + \Delta$")
    ax.set_ylabel(r"$\left|E_0 - E_{\rm FCI} \right|$ [Ha]")
    ax.legend()
    ax.set_yscale('log')
    plt.show()


######################################################################
# Amazing! Adding more qubits we are able to achieve chemical accuracy in
# the estimation of our molecular energies :D
# 


######################################################################
# Conclusion
# ==========
# 
# In this tutorial we have reviewed the Quantum Phase Estimation algorithm
# and we have seen how to use it to compute molecular energies.
# Furthermore we have seen it in action both through varying the initial state
# preparation step and the number of qubits to estimate the desired
# eigenvalue. Now it’s time for you to step in and try to use the QPE in
# different settings with PennyLane!
# 

######################################################################
#
# [2] And, as it turns out, it is a pretty valuable feature, 
# so QPE is an important subroutine of many advanced quantum algorithms.
#
# [3] For a more detailed analysis of all the derivations presented in this
# section please look at Quantum Information and Quantum computation by
# Michael A. Nielsen and Isaac L. Chuang.


######################################################################
# .. bio:: Davide Castaldo
#    :photo: ../_static/authors/jay_soni.png
#
#    Davide is a PhD student at the University of Padova where he is studying
# how to use quantum computers to simulate molecular systems. Currently he
# is also working at Xanadu as part of the residency program. He is a firm
# believer that one day racoons will take over humans in Toronto.
#
#
#
# .. bio:: Aleksei Malyshev
#    :photo: ../_static/authors/jay_soni.png
#
#    Aleksei's Bio.
#
#
#
