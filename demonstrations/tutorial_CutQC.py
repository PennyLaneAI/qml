"""
CutQC: Simulating Large Quantum Circuits on a Small Quantum Computer
====================================================================

"""


######################################################################
# Jimmy works for a quantum computing company based in Canada, that has
# just released its 3-qubit photonic-based quantum computer. Knowing the
# prospects of quantum computers, Jack who is the manager for an AI
# company also based in Canada, approached Jimmy to help with a quantum
# machine learning (qml) problem they would like to solve using their
# quantum computer. Solving this problem is key to the future advancement
# of the AI company as well as the quantum computing company. Jimmy
# accepted to help! Upon analyzing the problem however, Jimmy realized
# that the algorithm to solve it would require more qubits (about
# 5-qubits) than their current quantum computer can offer. He is confused
# and doesn’t know how to go about solving the problem.
#
# This tutorial is meant to help people like Jimmy, who have some
# knowledge in linear algebra (with good understanding of concepts such as
# Hermitian, eigenvalue, Hilbert space, self-adjoint, bra-ket notation,
# and eigenprojector), figure out how to use their small quantum computers
# (qc) to run large quantum circuits requiring more qubits than their qc
# can provide. Knowledge of Pennylane and Qiskit is also required. While
# Pennylane is used to do most of the work, we use Qiskit to very our
# results. It is based on the results from [1] and [2]. At the end of this
# tutorial, you should be able to:
#
# -  Explain the basic idea behind qubit cutting.
# -  Implement circuit cutting for one qubit
# -  Extend the idea of cutting to multi-qubit quantum circuits
#


######################################################################
# **Background**
# --------------
#


######################################################################
# **Qubits – Aliens that live on the planet called Bloch Sphere**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


######################################################################
# A qubit is to a quantum computer what a bit is to a classical computers.
# It is the basic unit of information in a quantum computer. As in a
# classical computer where information is encoded as :math:`0`\ ’s and
# :math:`1`\ ’s, a quantum computer encodes its information as
# :math:`\left | {0} \right \rangle` and
# :math:`\left | {1} \right \rangle` states, except that in addition,it
# can also encode information as a linear combination of the
# :math:`\left | {0} \right \rangle` and
# :math:`\left | {1} \right \rangle` states (e.g. the state
# :math:`\frac{1}{\sqrt{2}}(\left | {0} \right \rangle + \left | {1} \right \rangle)`.
# Qubits of this type are said to be in a superposition of states. A more
# general representation of a qubit is given in equation
# (:raw-latex:`\ref{eq1}`). Interested readers who are curious to know how
# this came about can look up [3] and [4]
#
# :raw-latex:`\begin{equation}\label{eq1}\tag{1}
# \begin{split}
# \left |{\psi} \right \rangle &= \cos{\left (\frac{\theta}{2} \right )}\left |{0} \right\rangle + e^{i\phi}\sin{\left (\frac{\theta}{2} \right )}\left |{1} \right\rangle
# \end{split}
# \end{equation}`
#
# **Observables**
# ^^^^^^^^^^^^^^^
#
# Measurement gives us useful information about the state of a qubit. One
# may be interested in measuring the property (e.g. position, spin,
# energy, momentum, etc.) of a quantum system (in this case a qubit), and
# we refer to this measurable property of the qubit as an **observable**.
# For example, and as we will see later, we may be interested in knowing
# the :math:`x`-, :math:`y`- or :math:`z`-component (position) of the
# qubit. Observables are **Hermitian matrices** with eigenvalues that
# represent the set of possible measurement outcomes. The most commonly
# used observables are the Pauli :math:`X`, :math:`Y`, and :math:`Z`
# obsevables – and they will be useful for the purpose of this tutorial.
#
# The corresponding matrix for the respective Pauli observables are as
# shown below.
#
# :raw-latex:`\begin{equation}\label{eq2}\tag{2}
# \begin{split}
#     X &= \left (
#     \begin{matrix}
#      0 & 1 \\
#     1 & 0
#     \end{matrix}
#     \right ) \\
#     \\
#     Y &= \left (
#     \begin{matrix}
#      0 & -i \\
#     i & 0
#     \end{matrix}
#     \right ) \\
#     \\
#     Z &= \left (
#     \begin{matrix}
#      1 & 0 \\
#     0 & -1
#     \end{matrix}
#     \right )
# \end{split}
# \end{equation}`
#
# The eigenvalues of any of the Pauli observable are :math:`+1` and
# :math:`-1`. Given that these matrices are Hermitian, if we measure any
# of the observables for a given qubit state, then the outcome will either
# be a :math:`+1` or a :math:`-1`. After measurement, the state of the
# qubit collapses to the corresponding eigenvector. For example, if we
# measured the Pauli-:math:`Z` observable, the state will collapse into a
# :math:`\left | 0 \right \rangle` state for eigenvalue :math:`+1` or a
# :math:`\left | 1 \right \rangle` state for measurement outcome with
# eigenvalue :math:`-1`. The same applies to Pauli-:math:`X` with
# eigenvectors :math:`\left | + \right \rangle` and
# :math:`\left | - \right \rangle` and Pauli-:math:`Y` with eigenvectors
# :math:`\left | +i \right \rangle` and :math:`\left | -i \right \rangle`.
# Where:
#
# :raw-latex:`\begin{equation}\label{eq3}\tag{3}
# \begin{split}
# \left | \pm \right \rangle &= \frac{1}{\sqrt{2}} \left( \left | 0 \right \rangle \pm \left | 1 \right \rangle \right) \\
# \left | \pm i \right \rangle &= \frac{1}{\sqrt{2}} \left( \left | 0 \right \rangle \pm i\left | 1 \right \rangle \right)
# \end{split}
# \end{equation}`
#
# **Expectation Value**
# ^^^^^^^^^^^^^^^^^^^^^
#
# Measurement is probabilistic and therefore, to ensure consistency in
# measurement outcomes, we must take several measurements. The weighted
# average of the outcomes over many measurements of any observable is know
# as the **expectation** of the that observable for a given qubit state.
# Given the state :math:`\left | \psi \right\rangle`, we can compute the
# expectation values for the different Pauli observables as follows:
#
# :raw-latex:`\begin{equation}\label{eq4}\tag{4}
# \begin{split}
# \left\langle X \right\rangle &= \left\langle \psi | X | \psi \right\rangle = \sin {\theta}\sin{\phi} \\
# \left\langle Y \right\rangle &= \left\langle \psi | Y | \psi \right\rangle = \sin{\theta}\cos{\phi} \\
# \left\langle Z \right\rangle &= \left\langle \psi | Z | \psi \right\rangle = \cos{\theta} \\
# \end{split}
# \end{equation}`
#
# **The Bloch Sphere**
# ^^^^^^^^^^^^^^^^^^^^
#
# Equation (:raw-latex:`\ref{eq4}`) look very much like the spherical
# coordinate system where the position of a point is specified by three
# numbers: **radius** :math:`r`, **inclination** :math:`\theta`,
# **azimuth** :math:`\phi`. In our case, the radius is :math:`1` and the
# resulting sphere is known as the **Bloch Sphere**. Bloch spheres
# represent the states of a single qubit on a spherical surface. Each
# qubit state vector corresponds to a 3D real vector on the surface of the
# sphere. The parameters :math:`\theta` and :math:`\phi`, are respectively
# the colatitude with respect to the z-axis and the longitude with respect
# to the x-axis [5], [6], and they specify the location of the state
# :math:`\left | \psi \right\rangle` at a point :math:`p` on the Bloch
# Sphere. The point :math:`p` is defined by equation
# (:raw-latex:`\ref{eq6}`).
#
# :raw-latex:`\begin{equation}\label{eq6}\tag{6}
# \begin{split}
# p &= \left( \sin {\theta}\sin{\phi}, \sin{\theta}\cos{\phi}, \cos{\theta} \right)\\
# &= (p_x, p_y, p_z)
# \end{split}
# \end{equation}`
#
# The numbers :math:`p_x` , :math:`p_y`, :math:`p_z` of the point
# :math:`p` are the expectations of the three Pauli matrices :math:`X`,
# :math:`Y`, :math:`Z`, allowing one to identify the three coordinates
# with :math:`x`, :math:`y`, and :math:`z` axes. The antipodal points
# along any of the axis corresponds to the eigenstate of the corresponding
# Pauli matrix. For example, the antipodal points along the zenith
# (:math:`z`-axis) are the eigenstates, :math:`\left | 0 \right \rangle`
# and :math:`\left | 1 \right \rangle` of Pauli-:math:`Z` matrix. The same
# applies to the other two coordinates as shown in the diagram below.
#


######################################################################
# |Bloch-Sphere|
#
# .. raw:: html
#
#    <p style="text-align: center">
#
# Figure 1. The Bloch Sphere
#
# .. raw:: html
#
#    </p>
#
# .. |Bloch-Sphere| image:: ./figs/bloch-sphere.png
#


######################################################################
# **Density Matrix**
# ^^^^^^^^^^^^^^^^^^
#
# For any :math:`2`-level quantum system, every pure quantum state (a
# state which can be described by a single ket vector in a Hilbert space)
# is located on the surface of the Bloch Sphere while mixed quantum states
# (a statistical ensemble of pure states) live inside the Bloch Sphere
# [4], [7]. An arbitrary state, :math:`\left | \psi \right\rangle` for a
# single qubit can be expressed as a linear combination of the Pauli
# matrices as shown in equation (:raw-latex:`\ref{eq7}`). This results in
# a :math:`2\times 2` self-adjoint matrices known as the **density
# matrix** which describes the quantum state of a physical system.
#
# :raw-latex:`\begin{equation}\label{eq7}\tag{7}
# \begin{split}
# \rho &= \frac{1}{2} \left ( I + p\cdot\sigma \right )
# \end{split}
# \end{equation}`
#
# where from equation (`6 <#mjx-eqn-eq6>`__):
#
# .. math::
#
#
#    p = (p_x, p_y, p_z)
#
# and
#
# .. math::
#
#
#    \sigma = (X, Y, Z)
#
# Substituting for :math:`p` and :math:`\sigma` in equation
# (`7 <#mjx-eqn-eq7>`__) gives:
#
# :raw-latex:`\begin{equation}\label{eq8}\tag{8}
# \begin{split}
# \rho &= \frac{1}{2} \left(\begin{matrix}1+p_z & p_x - i p_y\\ p_x+i p_y & 1-p_z\end{matrix}\right)
# \end{split}
# \end{equation}`
#


######################################################################
# **Circuit Cutting**
# ~~~~~~~~~~~~~~~~~~~
#
# Another way to express the density matrix is:
#
# :raw-latex:`\begin{equation}\label{eq9}\tag{9}
# \begin{split}
# \rho &= \frac{Tr(\rho I)I + Tr(\rho X)X + Tr(\rho Y)Y + Tr(\rho Z)Z}{2}
# \end{split}
# \end{equation}`
#
# Where:
#
# :raw-latex:`\begin{equation}\label{eq10}\tag{10}
# \begin{split}
# Tr(\rho I) &= 1 \\
# Tr(\rho X) &= p_x \\
# Tr(\rho Y) &= p_y \\
# Tr(\rho Z) &= p_z
# \end{split}
# \end{equation}`
#
# Note that the trace of the different Pauli matrices correspond to a
# measurement for the expectation value of that Pauli matrix.
#
# Pauli matrices can be expressed in terms of their eigenprojectors and
# eigenvalues as:
#
# :raw-latex:`\begin{equation}\label{eq11}\tag{11}
# \begin{split}
# I &= \left | {0} \right\rangle \left\langle {0} \right | + \left | {1} \right\rangle \left\langle {1} \right | \\
# X &= \left | {+} \right\rangle \left\langle {+} \right | - \left | {-} \right\rangle \left\langle {-} \right | \\
# Y &= \left | {+i} \right\rangle \left\langle {+i} \right | - \left | {-i} \right\rangle \left\langle {-i} \right | \\
# Z &= \left | {0} \right\rangle \left\langle {0} \right | - \left | {1} \right\rangle \left\langle {1} \right |
# \end{split}
# \end{equation}`
#
# If we substitute equation (:raw-latex:`\ref{eq11}`) into
# (:raw-latex:`\ref{eq9}`) we have:
#
# :raw-latex:`\begin{equation}\label{eq12}\tag{12}
# \begin{split}
# \rho &= \frac{1}{2} \left [ Tr(\rho I)\left | {0} \right\rangle \left\langle {0} \right | + Tr(\rho I)\left | {1} \right\rangle \left\langle {1} \right | + Tr(\rho X)\left | {+} \right\rangle \left\langle {+} \right | - Tr(\rho X)\left | {-} \right\rangle \left\langle {-} \right | \\
# + Tr(\rho Y)\left | {+i} \right\rangle \left\langle {+i} \right | - Tr(\rho Y)\left | {-i} \right\rangle \left\langle {-i} \right | + Tr(\rho Z)\left | {0} \right\rangle \left\langle {0} \right | - Tr(\rho Z)\left | {1} \right\rangle \left\langle {1} \right | \right ]
# \end{split}
# \end{equation}`
#
# Equation (:raw-latex:`\ref{eq12}`), can thus be expressed more
# succinctly as:
#
# :raw-latex:`\begin{equation}\label{eq13}\tag{13}
# \begin{split}
# \rho &= \frac{1}{2}\sum_{i=1}^{8} c_i Tr(\rho O_i)\rho_i
# \end{split}
# \end{equation}`
#
# Here, we have denoted Pauli matrices by :math:`O_i`, their
# eigenprojectors by :math:`\rho_i` and their corresponding eigenvalues by
# :math:`c_i`. To explain equation (:raw-latex:`\ref{eq13}`), the term
# :math:`Tr(\rho O_i)\rho_i` is considered as a quantum map that can be
# implemented by first measuring the Pauli observable :math:`O_i` and then
# preparing the corresponding eigenstate :math:`\rho_i`.
#
# In general, if we picture the time evolution (flow) of a qubit state
# from time :math:`u` to time :math:`v` as a line between point :math:`u`
# and point :math:`v`, then if we cut this line at some point, we can
# recover the qubit state at point :math:`v` using equation
# (:raw-latex:`\ref{eq13}`) as follows:
#


######################################################################
# |CutQC|
#
# .. raw:: html
#
#    <p style="text-align: center">
#
# Figure 2. CutQC
#
# .. raw:: html
#
#    </p>
#
# .. |CutQC| image:: ./figs/cutqc.png
#


######################################################################
# This means that after cutting the qubit into two parts as shown above,
# we measure the Pauli observable :math:`O_i` for the part (or
# subcircuit-:math:`u`) connected to the point :math:`u` and then
# initialize the qubit of :math:`v` (subcircuit-:math:`v`) with eigenstate
# :math:`\rho_i`. Note that we have delibrately ommited the constant
# :math:`\frac{1}{2}` for simplicity.
#
# Further simplification of equation (`12 <#mjx-eqn-eq12>`__) gives:
#
# .. math::
#
#
#    \rho = \frac{\rho_1 + \rho_2 + \rho_3 + \rho_4}{2}
#
# where:
#
# :raw-latex:`\begin{equation}\label{eq14}\tag{14}
# \begin{split}
# \rho_1 &= \left[Tr(\rho I) + Tr(\rho Z)\right]\left | {0} \right\rangle \left\langle {0} \right | \\
# \rho_2 &= \left[Tr(\rho I) - Tr(\rho Z)\right]\left | {1} \right\rangle \left\langle {1} \right | \\
# \rho_3 &= Tr(\rho X)\left[2\left | {+} \right\rangle \left\langle {+} \right | - \left | {0} \right\rangle \left\langle {0} \right | - \left | {1} \right\rangle \left\langle {1} \right | \right] \\
# \rho_4 &= Tr(\rho Y)\left[2\left | {+i} \right\rangle \left\langle {+i} \right | - \left | {0} \right\rangle \left\langle {0} \right | - \left | {1} \right\rangle \left\langle {1} \right | \right]
# \end{split}
# \end{equation}`
#
# Where we have made the following substitutions:
#
# :raw-latex:`\begin{equation}\label{eq15}\tag{15}
# \begin{split}
# \left | {+} \right\rangle \left\langle {+} \right | - \left | {-} \right\rangle \left\langle {-} \right | &= 2\left | {+} \right\rangle \left\langle {+} \right | - \left | {0} \right\rangle \left\langle {0} \right | - \left | {1} \right\rangle \left\langle {1} \right | \\
# \left | {+i} \right\rangle \left\langle {+i} \right | - \left | {-i} \right\rangle \left\langle {-i} \right | &= 2\left | {+i} \right\rangle \left\langle {+i} \right | - \left | {0} \right\rangle \left\langle {0} \right | - \left | {1} \right\rangle \left\langle {1} \right |
# \end{split}
# \end{equation}`
#
# This means that we only have to do four measurements
# :math:`\left(Tr(\rho I), Tr(\rho X), Tr(\rho Y), Tr(\rho Z) \right)` for
# subcircuit-:math:`u` and initialize subcircuit-:math:`v` with only four
# states: :math:`\left | {0} \right\rangle`,
# :math:`\left | {1} \right\rangle`, :math:`\left | {+} \right\rangle` and
# :math:`\left | {+i} \right\rangle`
#


######################################################################
# **Examples**
# ~~~~~~~~~~~~
#
# Consider a single qubit circuit as shown below measured in the
# computational basis. Our goal is to cut the circuit at the point between
# the :math:`X`-gate and the :math:`H`-gate, reconstruct the probability
# distribution of the cut circuit to be approximately equal to that of the
# original circuit using CutQC method.
#


######################################################################
# |1-Qubit CutQC|
#
# .. raw:: html
#
#    <p style="text-align: center">
#
# Figure 3. 1-Qubit CutQC
#
# .. raw:: html
#
#    </p>
#
# .. |1-Qubit CutQC| image:: ./figs/cutqc_single_qubit_02.png
#


######################################################################
# First, lets reproduce the uncut-circuit using Pennylane and compute what
# the probability outcome of the measurement would be
#


######################################################################
# **Helper Functions**
# ^^^^^^^^^^^^^^^^^^^^
#


def prob_dict(probs):
    """
    returns the dictionary of a given probability distribution
    Inputs: probs = np.array(circ_cuttingI()) --- probabilities from running subcircuit-1 (upstream circuit).
            should be an array not a tensor
    Output: Dictionay --- keys = binary of states, value = value of probability the state
    """
    n = int(np.log2(len(probs)))  # length of subcircuit 1 (or upstream circuit)
    probs_dic = {}  # dictionary of all the probability values keys are the binary value of states
    exp_probs = {}  # dictionary for the probability values that contribute to the expectation value

    # Build a probability dictionary for the probability outcome
    for i in range(len(probs)):
        key, value = np.binary_repr(i, n), float(probs[i])
        probs_dic[key] = value

    return probs_dic


######################################################################
# **Probability outcome of measuring the original 1-qubit uncut-Circuit using Pennylane**
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#

# Import packages that would be used
import pennylane as qml
import numpy as np

# Create the Qnode to run our circuit
dev = qml.device("default.qubit", wires=1)


@qml.qnode(dev)

# Create the function that implements the circuit
def circ():
    qml.PauliX(wires=0)
    qml.Hadamard(wires=0)
    return qml.probs(wires=0)


fig, ax = qml.draw_mpl(circ)()

probs_uncut = circ()
prob_dict(probs_uncut)


######################################################################
# **Probability outcome of measuring the original 1-qubit uncut-Circuit using Qiskit**
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#

from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram

backend = Aer.get_backend("qasm_simulator")
qc = QuantumCircuit(1)
qc.x(0)
qc.h(0)

qc = qc.reverse_bits()  # Here we reversed the bitstrings to match that of Pennylane

qc.measure_all()

job = execute(qc, backend, shots=1024)

result = job.result()
counts = result.get_counts()

qc.draw("mpl")

plot_histogram(counts)


######################################################################
# From the result above, the probability distribution for the original
# 1-qubit circuit is
# :math:`\{"0": 0.4999999999999999, "1": 0.4999999999999999\}` in
# pennylane and approximately the same result is obtained using Qiskit.
# This means that there is equal probrobability (:math:`0.5`) of measuring
# the circuit and finding it to be in states
# :math:`\left | {0} \right \rangle` and
# :math:`\left | {1} \right \rangle`
#


######################################################################
# **Applying CutQC to the 1-qubit Circuit**
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#


######################################################################
# The result of cutting the circuit above is as shown below:
#


######################################################################
# |CutQC circuit cutting|
#
# .. raw:: html
#
#    <p style="text-align: center">
#
# Figure 4. CutQC circuit cutting
#
# .. raw:: html
#
#    </p>
#
# .. |CutQC circuit cutting| image:: ./figs/1Qubit-circuit-cutting-02.png
#


######################################################################
# Going by equation (`14 <#mjx-eqn-eq14>`__), we can reduce the number of
# terms needed to compute CutQC probabilities for the circuit from
# :math:`8` terms to :math:`4` terms each for subcircuit-:math:`u` and
# subcircuit-:math:`v` as shown below:
#


######################################################################
# |image0|
#
# .. raw:: html
#
#    <p style="text-align: center">
#
# Figure 5. CutQC subcircuit terms needed
#
# .. raw:: html
#
#    </p>
#
# .. |image0| image:: ./figs/1Qubit-Circuit-Cutting-4terms-03.png
#


######################################################################
# We would have to compute the expectation values for the Pauli
# observables of subcircuit-:math:`u` and the probabilities for
# subcircuit-:math:`v` given the different initialization of the qubit
# state with the Pauli eigenvectors, and afterward, combine the results
# according to (`14 <#mjx-eqn-eq14>`__) to obtain the actual probability
# of the original circuit.
#


######################################################################
# **Compute Subcircuit-:math:`u` Terms**
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#

# Create the Qnode to run our circuit
dev = qml.device("default.qubit", wires=1)


@qml.qnode(dev)

# Create the function that implements subcircuit-u
def subcirc_u(ob):
    """
    Input: Pauli observable to measure
    Output: Expectation value
    """
    qml.PauliX(wires=0)
    return qml.expval(ob)


# compute expectation for the observable, identity
ob = qml.Identity(wires=0)
fig, ax = qml.draw_mpl(subcirc_u)(ob)

exp_I = float(subcirc_u(ob))
print("The expectation value for the Identity observable is: ", exp_I)

# compute expectation for the observable, Pauli-X
ob = qml.PauliX(wires=0)
fig, ax = qml.draw_mpl(subcirc_u)(ob)

exp_X = float(subcirc_u(ob))
print("The expectation value for the Pauli-X observable is: ", exp_X)


# compute expectation for the observable, Pauli-Y
ob = qml.PauliY(wires=0)
fig, ax = qml.draw_mpl(subcirc_u)(ob)

exp_Y = float(subcirc_u(ob))
print("The expectation value for the Pauli-Y observable is: ", exp_Y)

# compute expectation for the observable, Pauli-Z
ob = qml.PauliZ(wires=0)
fig, ax = qml.draw_mpl(subcirc_u)(ob)

exp_Z = float(subcirc_u(ob))
print("The expectation value for the Pauli-Z observable is: ", exp_Z)


######################################################################
# **Compute Subcircuit-:math:`v` Terms**
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#

# Create the Qnode to run our circuit
dev = qml.device("default.qubit", wires=1)


@qml.qnode(dev)

# Create the function that implements subcircuit-v
def subcirc_v(init):
    """
    Input: Pauli eigenstates - "zero", "one", "plus", and "plus_i"
    Output: probability distrubution measured in the computational basis
    """
    if init == "one":  # If initized to the |1> state
        qml.PauliX(wires=0)
    elif init == "plus":  # If initized to the |+> state
        qml.Hadamard(wires=0)
    elif init == "plus_i":  # If initized to the |+i> state
        qml.Hadamard(wires=0)
        qml.S(wires=0)
    qml.Hadamard(wires=0)
    return qml.probs(wires=0)


# compute probabilities for the |0> state
init = "zero"
fig, ax = qml.draw_mpl(subcirc_v)(init)

prob_zero = np.array(subcirc_v(init))
print("The probability for the state |0>: ", prob_zero)

# compute probabilities for the |1> state
init = "one"
fig, ax = qml.draw_mpl(subcirc_v)(init)

prob_one = np.array(subcirc_v(init))
print("The probability for the state |1>: ", prob_one)

# compute probabilities for the |+> state
init = "plus"
fig, ax = qml.draw_mpl(subcirc_v)(init)

prob_plus = np.array(subcirc_v(init))
print("The probability for the state |+>: ", prob_plus)

# compute probabilities for the |+i> state
init = "plus_i"
fig, ax = qml.draw_mpl(subcirc_v)(init)

prob_plus_i = np.array(subcirc_v(init))
print("The probability for the state |+i>: ", prob_plus_i)


######################################################################
# **Combining results from Subcircuit-:math:`u` and Subcircuit-:math:`v` Terms**
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#


######################################################################
# We will combine the subcircuits using the equation:
#
# .. math::
#
#
#    \rho = \frac{\rho_1 + \rho_2 + \rho_3 + \rho_4}{2}
#
# where:
#
# .. math::
#
#
#    \rho_1 = \left[Tr(\rho I) + Tr(\rho Z)\right]\left | {0} \right\rangle \left\langle {0} \right | \\
#    \rho_2 = \left[Tr(\rho I) - Tr(\rho Z)\right]\left | {1} \right\rangle \left\langle {1} \right | \\
#    \rho_3 = Tr(\rho X)\left[2\left | {+} \right\rangle \left\langle {+} \right | - \left | {0} \right\rangle \left\langle {0} \right | - \left | {1} \right\rangle \left\langle {1} \right | \right] \\
#    \rho_4 = Tr(\rho Y)\left[2\left | {+i} \right\rangle \left\langle {+i} \right | - \left | {0} \right\rangle \left\langle {0} \right | - \left | {1} \right\rangle \left\langle {1} \right | \right]
#
# Here the traces :math:`Tr(\rho I)`, :math:`Tr(\rho X)`,
# :math:`Tr(\rho Y)`, and :math:`Tr(\rho IZ)` are the results we got for
# the respective expection values we computed for subcircuit-:math:`u`
# while the eigenprojectors
# :math:`\left | {0} \right\rangle \left\langle {0} \right |`,
# :math:`\left | {1} \right\rangle \left\langle {1} \right |`,
# :math:`\left | {+} \right\rangle \left\langle {+} \right |` and
# :math:`\left | {+I} \right\rangle \left\langle {+i} \right |` are the
# results we got from initializing subcircuit-:math:`u` with the
# eigenvectors states: :math:`\left | {0} \right\rangle`,
# :math:`\left | {1} \right\rangle`, :math:`\left | {+} \right\rangle` and
# :math:`\left | {+i} \right\rangle` respectively.
#
# So, let’s put everything together:
#


def compute_prob(exp_I, exp_X, exp_Y, exp_Z, prob_zero, prob_one, prob_plus, prob_plus_i):
    prob_rho_1 = (exp_I + exp_Z) * prob_zero
    prob_rho_2 = (exp_I - exp_Z) * prob_one
    prob_rho_3 = (exp_X) * (2 * prob_plus - prob_zero - prob_one)
    prob_rho_4 = (exp_Y) * (2 * prob_plus_i - prob_zero - prob_one)

    prob_rho = (prob_rho_1 + prob_rho_2 + prob_rho_3 + prob_rho_4) / 2
    return prob_rho


combined_probs = compute_prob(
    exp_I, exp_X, exp_Y, exp_Z, prob_zero, prob_one, prob_plus, prob_plus_i
)
print("The combined probabilities from CutQC is: ", prob_dict(combined_probs))

print("The probabilities from the uncut circuit is: ", prob_dict(probs_uncut))


######################################################################
# This confirms that we can get the same result with CutQC as we would get
# without cutting the circuit.
#


######################################################################
# **CutQC for multi-qubit quantum circuit**
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#


######################################################################
# It may seem less intuitive to understand the power of CutQC with only
# 1-qubit circuit. To get a better insight into the power of CutQC, let’s
# revisit Jimmy’s problem that informed this tutorial. Recall that Jimmy
# has a 3-qubit photonic quantum computer, but would like to run a quantum
# circuit requiring 5 qubits.
#
# Jimmy’s 5-qubits circuit problem is as shown below:
#


######################################################################
# |5-Qubits Quantum Circuit|
#
# .. raw:: html
#
#    <p style="text-align: center">
#
# Figure 6. 5-Qubits Quantum Circuit
#
# .. raw:: html
#
#    </p>
#
# .. |5-Qubits Quantum Circuit| image:: ./figs/cutqc-5qubits.png
#


######################################################################
# **Goal:**
# '''''''''
#
# -  Our goal is to run this quantum circuit using Jimmy’s 3-qubits
#    photonic quantum computer.
#


######################################################################
# **Verification: Probability outcome of measuring the original 5-qubit (multi-qubit) uncut-Circuit using Qiskit**
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#


backend = Aer.get_backend("qasm_simulator")
multi_qc = QuantumCircuit(5)

for i in range(5):
    multi_qc.h(i)

multi_qc.cz(0, 1)

for j in range(2, 5):
    multi_qc.t(j)

multi_qc.cz(0, 2)
multi_qc.rx(np.pi / 2, 0)
multi_qc.rx(np.pi / 2, 1)
multi_qc.rx(np.pi / 2, 4)
multi_qc.cz(2, 4)
multi_qc.cz(2, 3)
multi_qc.t(0)
multi_qc.t(1)
multi_qc.ry(np.pi / 2, 4)

for i in range(5):
    multi_qc.h(i)

multi_qc = multi_qc.reverse_bits()  # Here we reversed the bitstrings to match that of Pennylane

multi_qc.measure_all()

multi_job = execute(multi_qc, backend, shots=2048)

multi_result = multi_job.result()
multi_counts = multi_result.get_counts()

multi_qc.draw("mpl")

plot_histogram(multi_counts, figsize=(16, 6))


######################################################################
# **Implementing CutQC for multi-qubit circuit using Pennylane**
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#


######################################################################
# **Step 1:**
# ^^^^^^^^^^^
#
# -  Chose a qubit and find a point on the qubit to cut. Here, we have
#    chosen qubit :math:`q_2` at the location between the first two qubit
#    gates connected to it as shown below:
#


######################################################################
# |image0|
#
# .. raw:: html
#
#    <p style="text-align: center">
#
# Figure 7. Circuit cutting for 5-qubits quantum Circuit
#
# .. raw:: html
#
#    </p>
#
# .. |image0| image:: ./figs/5qubits-cuts-02.png
#


######################################################################
# The cut on the original circuit resulted in two subcircuits:
# subcircuit-:math:`u` and subcircuit-:math:`v`. Each subcircuit has a
# maximum of 3-qubits, so we can run each subcircuit separately on Jimmy’s
# 3-qubits photonic quantum computer using CutQC. This is good new! But
# how can we do this and still get the same result as the original
# circuit? This will be our focus in the following sections.
#


######################################################################
# **Step 2:**
# ^^^^^^^^^^^
#
# -  Identify the terms to compute for the two subcircuits. The terms we
#    need are summarized in the figure below:
#


######################################################################
# |Subcircuit terms required for CutQC|
#
# .. raw:: html
#
#    <p style="text-align: center">
#
# Figure 8. Subcircuit terms required for CutQC
#
# .. raw:: html
#
#    </p>
#
# .. |Subcircuit terms required for CutQC| image:: ./figs/5qubits-terms-02.png
#


######################################################################
# **Rules for computing expectation vaules for subcircuit-:math:`u`**
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# -  As can be seen from the above figure, the terms needed are not
#    different from those we computed for 1-qubit circuit. However, one
#    thing worth noting is that because we are working with a multi-qubit
#    circuit, the expectation values for subcircuit-:math:`u` and the
#    probabilities for subcircuit-:math:`v` will depend on the state for
#    which we would like to determine its probability.
#
# -  Since there are 5-qubits in the original circuit, the total number of
#    states for which we would have to compute probabilities for would be
#    :math:`2^5=32` states; starting from the first state,
#    :math:`\left | 00000 \right\rangle` to the last state,
#    :math:`\left | 11111 \right\rangle`.
#
# -  Lets say we are interest in computing the probabilities for the
#    state: :math:`\left | 00100 \right\rangle`. Since we cut the circuit
#    at qubit :math:`q_2`, subcircuit-:math:`u` qubits would be
#    :math:`\left | 001 \right\rangle` and subcircuit-:math:`v` qubits
#    would be :math:`\left | 100 \right\rangle`. Notice that both
#    subcircuits share qubit :math:`q_2` with state
#    :math:`\left | 1 \right\rangle`.
#
# -  Next would be to compute the expectation value for
#    subcircuit-:math:`u` for only the states:
#    :math:`\left | 000 \right\rangle` and
#    :math:`\left | 001 \right\rangle`. Also, notice that qubit
#    :math:`q_2` has taken on two states: :math:`\left | 0 \right\rangle`
#    and :math:`\left | 1 \right\rangle` while the states of other qubits
#    remain the same.
#
# -  The implication is that, we would run subcircuit-:math:`u`, measure
#    the uncut qubits in the computational basis, and measure the cut
#    qubit in any of the Pauli observable’s (matrix) basis. From the
#    resulting probability distribution obtained for the total possible
#    states (in this case :math:`2^3=8` states), we are interested in the
#    probabilities of the states: :math:`\left | 000 \right\rangle` and
#    :math:`\left | 001 \right\rangle`. It is the probabilities for these
#    states that we would use to compute the expectation value for the
#    Pauli observable of interest.
#
# -  Except for the :math:`I` observable whose eigenvalues are both
#    :math:`+1`, for all other Pauli observables, to compute their
#    expectation values, we multiply the probability of the state where
#    the cut qubit state is :math:`\left | 1 \right\rangle` (in this case,
#    the probability for the state :math:`\left | 001 \right\rangle`) with
#    the eigenvalue :math:`-1` and all other states where the cut qubit
#    state is :math:`\left | 0 \right\rangle` with :math:`+1` (in this
#    case, the probability for the state
#    :math:`\left | 100 \right\rangle`) or simply do nothing to it. Add
#    the results together and that is the expectation value of the
#    observable of interest.
#


######################################################################
# Let’s define a function that computes the expectation of any of the Pauli observables for subcircuit-:math:`u` given the probability distribution for that observable in subcircuit-:math:`u`,and the state we wish to find its probability
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#


def expVal(state, ob):
    """
    Function to compute the expectation of a Pauli observables from the probability outcome (probs)
    of subcircuit-u values with respect to a given state (the state you want to find its probability value)
    Inputs: state = "00000"  --- state we want to find its probability in the uncut circuit
            ob = observable ("I", "X", "Y", "Z") to compute its expectation value.
    Output: Float --- Expectation value
    """

    # Define the quantum node and compute the probabilities for subcircuit-u with respest to the observable of interest
    dev = qml.device("default.qubit", wires=3)

    @qml.qnode(dev)
    def subcirc_u():
        for i in range(3):
            qml.Hadamard(wires=i)
        qml.CZ(wires=[0, 1])
        qml.T(wires=2)
        qml.CZ(wires=[0, 2])
        qml.RX(np.pi / 2, wires=0)
        qml.RX(np.pi / 2, wires=1)
        qml.T(wires=0)
        qml.T(wires=1)
        for i in range(2):
            qml.Hadamard(wires=i)

        # Apply the right measurement basis. Note that the I and Z measurement bases are same, so we do nothing
        if ob == "X":  # measure in the X-basis
            qml.Hadamard(wires=2)
        elif ob == "Y":
            qml.adjoint(qml.S)(wires=2)  # measure in the Y-basis
            qml.Hadamard(wires=2)
        return qml.probs(range(3))

    probs = np.array(subcirc_u())  # the probability destribution output from running subcircuit-u
    n = int(np.log2(len(probs)))  # length of subcircuit-u (or upstream circuit)
    probs_dic = {}  # dictionary of all the probability values keys are the binary value of states
    exp_probs = {}  # dictionary for the probability values that contribute to the expectation value

    # Build a probability dictionary for the probability outcome
    for i in range(len(probs)):
        key, value = np.binary_repr(i, n), probs[i]
        probs_dic[key] = value

        # Pick only probabilities that contribute to the expectation value
        if list(key)[0 : n - 1] == list(state)[0 : n - 1]:
            exp_probs[key] = value

    exp_val = 0  # place holder for the expectation value
    for key in exp_probs.keys():
        if key[-1] == "1" and ob != "I":
            exp_val -= exp_probs[key]
        else:
            exp_val += exp_probs[key]
    return exp_val


######################################################################
# Let’s compute the expectation values of subcircuit-:math:`u` for the example state: :math:`\left | 00100 \right\rangle`
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#

obs = ["I", "X", "Y", "Z"]
exp_dict = {}
state = "00100"
for ob in obs:
    expectation = expVal(state, ob)
    exp_dict[ob] = expectation
print("The expectation values of the Pauli observables for the state |00100> is: ", exp_dict)


######################################################################
# **Rules for computing probabilities for subcircuit-:math:`v`**
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# -  Going by the above example, say we would like to compute the
#    probability for the state :math:`\left | 00100 \right\rangle`.
#    Subcircuit-:math:`v` would be :math:`\left | 100 \right\rangle`.
#    After initializing the cut qubit to the state of interest, run the
#    circuit to compute the probability distribution for all the states
#    (in this case :math:`2^3=8` states).
#
# -  Out of the 8 states, the probability of interest is the probability
#    for the state: :math:`\left | 100 \right\rangle`. This is the
#    probability that would be used to compute the final probability for
#    the state of interest: :math:`\left | 00100 \right\rangle`.
#


######################################################################
# Let’s define another a function to compute the probabilities of subcircuit-:math:`v` with respect to the state of interest.
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#


def probs_cut(init, state):
    """
    This function computes the probabilities of subcircuit-v for different initialization of Pauli
    eigenvector for the cut qubit
    Input: init -- str ("zero" for state |0>, "one" for state |1>, "plus" for state |+>, "plus_i" for state |+i>)
           state -- the state we would like to compute its probability
    Output: array of probability destributions for subcircuit-v
    """
    # Define a Qnode and compute the probabilities of subcircuit-v for different initiations
    dev = qml.device("default.qubit", wires=3)

    @qml.qnode(dev)
    def subcirc_v():
        # initialize the cut qubit to the Pauli eigenstate of interest
        if init == "one":
            qml.PauliX(wires=0)
        elif init == "plus":
            qml.Hadamard(wires=0)
        elif init == "plus_i":
            qml.Hadamard(wires=0)
            qml.S(wires=0)

        # Build the subcircuit-v with the other gates from the original circuit
        for i in range(1, 3):
            qml.Hadamard(wires=i)

        for j in range(1, 3):
            qml.T(wires=j)
        qml.RX(np.pi / 2, wires=2)
        qml.CZ(wires=[0, 2])
        qml.CZ(wires=[0, 1])
        qml.RY(np.pi / 2, wires=2)
        for i in range(3):
            qml.Hadamard(wires=i)
        return qml.probs(range(3))

    probs = np.array(subcirc_v())  # the probability destribution output from running subcircuit-v
    n = int(np.log2(len(probs)))  # length of subcircuit-v (or upstream circuit)
    probs_dic = {}  # dictionary of all the probability values keys are the binary value of states
    prob = {}  # dictionary for the probability values that contribute to the expectation value

    # Build a probability dictionary for the probability outcome
    for i in range(len(probs)):
        key, value = np.binary_repr(i, n), probs[i]
        probs_dic[key] = value

        # Pick only probability of the state for subcircuit-v that contribute to the overall probability
        s = state[n - 1 :]

    return probs_dic[s]


inits = ["zero", "one", "plus", "plus_i"]
probs_dict = {}
state = "00100"
for init in inits:
    prob = probs_cut(init, state)
    probs_dict[init] = prob
print("The expectation values of the Pauli observables for the state |00100> is: ", probs_dict)


######################################################################
# Let’s combine the results from subcircuit-:math:`u` and subcircuit-:math:`v`
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

# Results from subcircuit-u
exp_I = exp_dict["I"]
exp_X = exp_dict["X"]
exp_Y = exp_dict["Y"]
exp_Z = exp_dict["Z"]

# Results from subcircuit-v
prob_zero = probs_dict["zero"]
prob_one = probs_dict["one"]
prob_plus = probs_dict["plus"]
prob_plus_i = probs_dict["plus_i"]
print(
    "The probability for the state, 00100:",
    compute_prob(exp_I, exp_X, exp_Y, exp_Z, prob_zero, prob_one, prob_plus, prob_plus_i),
)


######################################################################
# **Let’s put it all together - Compute the probabilities for all the states**
# ----------------------------------------------------------------------------
#


######################################################################
# -  Creat a function to put all the pieces together.
#


def cutqc_prob(state):
    # Compute expectation values for subcircuit-u
    obs = ["I", "X", "Y", "Z"]
    exp_dict = {}
    for ob in obs:
        expectation = expVal(state, ob)
        exp_dict[ob] = expectation

    # Results from subcircuit-u
    exp_I = exp_dict["I"]
    exp_X = exp_dict["X"]
    exp_Y = exp_dict["Y"]
    exp_Z = exp_dict["Z"]

    # Compute probabilities for subcircuit-v
    inits = ["zero", "one", "plus", "plus_i"]
    probs_dict = {}
    for init in inits:
        prob = probs_cut(init, state)
        probs_dict[init] = prob

    # Results from subcircuit-v
    prob_zero = probs_dict["zero"]
    prob_one = probs_dict["one"]
    prob_plus = probs_dict["plus"]
    prob_plus_i = probs_dict["plus_i"]

    return compute_prob(exp_I, exp_X, exp_Y, exp_Z, prob_zero, prob_one, prob_plus, prob_plus_i)


######################################################################
# -  compute the probabilities for all the states
#

n = 5
cutqc_probs = {}
for i in range(2**5):
    state = np.binary_repr(i, n)
    cutqc_probs[state] = cutqc_prob(state)
print("The probability result from CutQC is: ", cutqc_probs)


######################################################################
# It is easier to read from below:
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#

cutqc_probs


######################################################################
# **Compare CutQC Result with the result from the original uncut circuit**
# ------------------------------------------------------------------------
#

dev = qml.device("default.qubit", wires=5)


@qml.qnode(dev)
def circ_full():
    for i in range(5):
        qml.Hadamard(wires=i)
    qml.CZ(wires=[0, 1])
    for j in range(2, 5):
        qml.T(wires=j)
    qml.CZ(wires=[0, 2])
    qml.RX(np.pi / 2, wires=0)
    qml.RX(np.pi / 2, wires=1)
    qml.RX(np.pi / 2, wires=4)
    qml.CZ(wires=[2, 4])
    qml.CZ(wires=[2, 3])
    qml.T(wires=0)
    qml.T(wires=1)
    qml.RY(np.pi / 2, wires=4)
    for i in range(5):
        qml.Hadamard(wires=i)
    return qml.probs(range(5))


fig, ax = qml.draw_mpl(circ_full)()

probs_full = np.array(circ_full())
probs_full

prob_dict(probs_full)

cutqc_probs

import matplotlib.pyplot as plt

original_probs = prob_dict(probs_full)

X = np.arange(len(original_probs))
fig, ax = plt.subplots(figsize=(16, 6))

ax.bar(X, cutqc_probs.values(), width=0.5, color="g", align="center")

ax.bar(X - 0.5, original_probs.values(), width=0.5, color="b", align="center")

ax.legend(("CutQC Probabilities", "Original Probabilities"))
plt.xticks(X, original_probs.keys(), rotation=90)
plt.title("Comparing Original Vs CutQC Probabilities", fontsize=17)
plt.xlabel("States")
plt.ylabel("Probabilities")
plt.show()


######################################################################
# Recall that from Qiskit we got the following plot:
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

plot_histogram(multi_counts, figsize=(16, 6))


######################################################################
# **This confirms that CutQC works and Jimmy can relax knowing that he can
# solve the problem of running large quantum circuits on his 3-qubit
# photonic quantum computer!**
#


######################################################################
# References
# ==========
#
# .. raw:: html
#
#    <!-- BIBLIOGRAPHY START -->
#
# .. container:: csl-bib-body
#
# .. raw:: html
#
#    <!-- BIBLIOGRAPHY END -->
#


######################################################################
# [1] T. Peng, A. W. Harrow, M. Ozols, and X. Wu, “Simulating large
# quantum circuits on a small quantum computer,” Physical Review Letters,
# vol. 125, no. 15, p. 150504, 2020.
#
# [2] W. Tang, T. Tomesh, M. Suchara, J. Larson, and M. Martonosi, “CutQC:
# Using Small Quantum Computers for Large Quantum Circuit Evaluations,” in
# Proceedings of the 26th ACM International Conference on Architectural
# Support for Programming Languages and Operating Systems, New York, NY,
# USA: Association for Computing Machinery, 2021, pp. 473–486. [Online].
# Available: https://doi.org/10.1145/3445814.3446758
#
# [3] “Bloch sphere - Wikipedia.” Accessed: Mar. 31, 2022. [Online].
# Available: https://en.wikipedia.org/wiki/Bloch_sphere
#
# [4] P. Viswanath, “Quantum States And The Bloch Sphere,” Quantum
# Untangled, Feb. 10, 2021.
# https://medium.com/quantum-untangled/quantum-states-and-the-bloch-sphere-9f3c0c445ea3
# (accessed Mar. 31, 2022).
#
# [5] “Spherical coordinate system,” Wikipedia. Mar. 09, 2022. Accessed:
# Mar. 31, 2022. [Online]. Available:
# https://en.wikipedia.org/w/index.php?title=Spherical_coordinate_system&oldid=1076199402
#
# [6] “Xanadu Quantum Codebook.” https://codebook.xanadu.ai/ (accessed
# Apr. 01, 2022)
#
