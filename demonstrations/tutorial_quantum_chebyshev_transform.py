r"""Intro to the Quantum Chebyshev Transform
=============================================================

This demo is inspired by the paper `"Quantum Chebyshev transform: mapping, embedding, learning and sampling distributions" <https://arxiv.org/abs/2306.17026>`__ [#williams2023]_ wherein the authors describe a workflow for quantum Chebyshev-based model building. 
They demonstrate the use of the Chebyshev basis space in generative modeling for probability distributions. A proposed protocol for learning and sampling multivariate probability distributions that arise in high-energy physics also makes use of the Chebyshev basis [#delejarza2025]_.
Crucial to the implementation of learning models in Chebyshev space is the quantum Chebyshev transform (QChT), which is used to swap between the computational basis and the Chebyshev basis. 

We will start by discussing Chebyshev polynomials and why you may want to work in Chebyshev space. Then we will show how the QChT can be implemented in PennyLane. 


What are Chebyshev polynomials?
---------------------------------------

`Chebyshev polynomials <https://en.wikipedia.org/wiki/Chebyshev_polynomials#As_a_basis_set>`__ of the first kind :math:`T_n(x)` are a set of orthogonal polynomials that are complete on the interval :math:`[-1,1]`. They can be defined as 

.. math::
  T_n(x) \equiv \cos(n \arccos(x))

where :math:`n` is the order of the polynomial. We can write out the first few orders explicitly.

.. math::
  T_0(x) &= 1 \\
  T_1(x) &= x \\
  T_2(x) &= 2x^2-1 \\
  T_3(x) &= 4x^3 - 3x \\
  T_4(x) &= 8x^4 - 8x^2 + 1 \\
  &\ \,\vdots \\
  T_{n+1}(x) &= 2xT_n(x) - T_{n-1}(x)

The recursion relation in the last line can be used to compute the next orders. 
Observe that odd and even order :math:`T_n` are odd and even functions, respectively. 
The roots of the :math:`T_n(x)` occur at the values 

.. math::
  x_n^\mathrm{Ch} = \cos\left(\frac{2k+1}{2n}\pi\right)\,, \quad k=0, ..., n-1\,. 

These are known as the `Chebyshev nodes <https://en.wikipedia.org/wiki/Chebyshev_nodes>`__.

.. figure:: ../_static/demonstration_assets/quantum_chebyshev_transform/chebyshev_polynomials.png
    :align: center
    :width: 60%
    :target: javascript:void(0)

The nodes are plotted above along with the corresponding polynomials. Note that the polynomials are normalized such that $T_n(1)=1$, and they satisfy a discrete orthogonality condition on the nodes of $T_N(x)$ in the following way for $k$, $\ell<N$

.. math::
  \sum^{N-1}_{j=0}T_k(x_j^\mathrm{Ch})T_\ell(x_j^\mathrm{Ch}) =  
    \begin{cases}
      0 & k \neq \ell\,,\\
      N & k = \ell = 0\,,\\
      N/2 & k = \ell \neq 0
    \end{cases}

The Chebyshev polynomials have a lot of *nice* properties. Because they are complete, any function :math:`f(x)` on the interval :math:`x\in [-1,1]` can be expanded in :math:`T_n(x)` up to order :math:`N` as :math:`f(x) = \sum_{j=0}^N a_j T_j(x)`. \
To do this process numerically for a discrete set of sampling points would take :math:`\mathcal{O}(N^2)` operations for a general set of complete functions 🐌. 
However, because of the way the Chebyshev polynomials are defined in terms of cosine, the `discrete Chebyshev transformation (DChT) <https://en.wikipedia.org/wiki/Discrete_Chebyshev_transform>`__ can be related to the `discrete cosine transform (DCT) <https://en.wikipedia.org/wiki/Discrete_cosine_transform>`__ to leverage the efficiency of the `fast-Fourier-transform <https://en.wikipedia.org/wiki/Fast_Fourier_transform>`__-style algorithms, which take :math:`\mathcal{O}(N \log N)` operations 🚀.

The DChT is sampled on the nodes of :math:`T_N(x)`, which are non-equidistant on the :math:`[-1,1]` interval. 
This non-uniform sampling has more resolution at the boundary, but less in the middle. 
This can be a benefit if you are, for example, solving a differential equation and expect more interesting features at the boundary, so the extra resolution there is useful. 
In general, working in the Chebyshev basis can have advantages over the Fourier basis for polynomial decomposition.


Quantum Chebyshev basis
---------------------------------------
The quantum Chebyshev transform (QChT) circuit described in Ref. [#williams2023]_ maps :math:`2^N` computational states :math:`\{|x_j\rangle\}_{j=0}^{2^N-1}` to Chebyshev states :math:`\{|\tau\rangle(x_j^\mathrm{Ch})\}_{j=0}^{2^N-1}` which have amplitudes given by the Chebyshev polynomials of the first kind. 

.. math::
  |\tau(x)\rangle = \frac1{2^{N/2}}T_0(x)|0\rangle + \frac1{2^{(N-1)/2}}\sum_{k=1}^{2^N-1}T_k(x)|k\rangle

where :math:`|k\rangle` are the computational basis states and :math:`N` is the number of qubits. These states are orthonormal at the Chebyshev nodes due to the orthogonality condition, that is

.. math::
  \langle\tau(x_j^\mathrm{Ch})|\tau(x_{j'}^\mathrm{Ch})\rangle = \delta_{j, j'}

The squared overlap if one of the variables is not at a node can be derived analytically as

.. math:: 
  |\langle\tau(x_j^\mathrm{Ch})|\tau(x)\rangle|^2 = \frac{\left(T_{2^N+1}(x_j^\mathrm{Ch})T_{2^N}(x)-T_{2^N}(x_j^\mathrm{Ch})T_{2^N+1}(x)\right)^2}{2^{2N}(x_j^\mathrm{Ch}-x)^2}

The goal is to design a circuit that applies the operation :math:`\mathcal{U}_\mathrm{QChT} = \sum_{j=0}^{2^N-1} |\tau(x_j^\mathrm{Ch})\rangle\langle x_j|`.


Designing the transform circuit
---------------------------------------
Let's start from the end and look at the circuit diagram generated from the code we want to write. 
An ancilla qubit is required, which will be the :math:`0` indexed qubit, and the rest compose the state :math:`|x\rangle` which starts in the computational basis, shown below as :math:`|\psi\rangle`. We demonstrate for :math:`N=4` non-ancilla qubits.

.. figure:: ../_static/demonstration_assets/quantum_chebyshev_transform/qcht_diagram_4qubits.png
    :align: center
    :width: 100%
    :target: javascript:void(0)

The intuition for the structure of the above circuit comes from the link between the DChT and the DCT. 
Notice the use of the `quantum Fourier transform (QFT) <https://pennylane.ai/qml/demos/tutorial_qft/>`__ applied on all qubits. 
The QChT is an extended QFT circuit with some added interference and mixing of the elements.

Let's break down the circuit above into pieces that we will use inside our circuit function. 
First, a Hadamard gate is applied to the ancilla, and then a CNOT ladder is applied, controlled on the ancilla. 
To start, we will define a function for the CNOT ladder.

"""

import pennylane as qml

# number of qubits (non-ancilla)
N = 4


def CNOT_ladder():
    for wire in range(1, N + 1):
        qml.CNOT([0, wire])


#############################################
# After the initial CNOT ladder comes an :math:`N+1` QFT circuit, which can be implemented using ``qml.QFT``.
#
# Next are phase rotations and shifts.
# In particular, there is a phase shift on the ancilla by :math:`-\pi/2^{(N+1)}` followed by a :math:`Z` rotation of :math:`-\pi(2^N - 1)/2^{N+1}`.
# The other other qubits are rotated in :math:`Z` by :math:`\pi/2^{(j+1)}`, where :math:`j` is the index of the qubit as labelled in the circuit diagram.

import numpy as np

pi = np.pi


def rotate_phases():
    """shift the ancilla's phase and rotate the jth qubit by
    pi/2^(j+1) in Z"""
    qml.PhaseShift(-pi / 2 ** (N + 1), wires=0)
    qml.RZ(-pi * (2**N - 1) / 2 ** (N + 1), wires=0)
    for wire in range(1, N + 1):
        qml.RZ(pi / 2 ** (wire + 1), wires=wire)


#############################################
# Now a permutation of the qubits is used to reorder them.
# This is built using a multicontrolled NOT gate applied to each qubit from the initial state, which is controlled on the ancilla and all qubits with larger index than the target.
# The multicontrolled NOT gate can be implemented using a multicontrolled Pauli X gate.
# Let's see what that looks like.


def permute_elements():
    """reorders amplitudes of the conditioned states"""
    for wire in reversed(range(1, N + 1)):
        control_wires = [0] + list(range(wire + 1, N + 1))
        qml.MultiControlledX(wires=(*control_wires, wire))


#############################################
# In the above code, we use ``reversed`` to loop over the qubits in reverse order, to apply the controlled gate to the last qubit first.
# After the permutation is another CNOT ladder, which we already have a function for.
#
# The last part is a phase adjustment of the ancilla qubit: a phase shift of :math:`-\pi/2`, followed by a rotation in :math:`Y` by :math:`\pi/2` and a multicontrolled :math:`X` rotation by :math:`\pi/2`.
# All of the other qubits control the :math:`X` rotation, but the control is sandwiched by Pauli :math:`X` operators.
# We can implement the multicontrolled :math:`X` rotation by using the function ``qml.ctrl`` on ``qml.RX``, specifying the target wire in ``qml.RX`` and the control wires as the second argument of ``qml.ctrl``.


def adjust_phases():
    """adjusts the phase of the ancilla qubit"""
    qml.PhaseShift(-pi / 2, wires=0)
    qml.RY(-pi / 2, wires=0)
    # first Pauli Xs
    for wire in range(1, N + 1):
        qml.PauliX(wires=wire)
    # controlled RX gate
    qml.ctrl(qml.RX(pi / 2, wires=0), range(1, N + 1))
    # second Pauli Xs
    for wire in range(1, N + 1):
        qml.PauliX(wires=wire)


#############################################
# All together, we can construct the circuit.
# We have added :class:`qml.BasisState` to initialize the input in any computational basis state with the optional argument ``state``.


def QChT():
    qml.Hadamard(wires=0)
    CNOT_ladder()
    qml.QFT(wires=range(N + 1))
    rotate_phases()
    permute_elements()
    CNOT_ladder()
    adjust_phases()


dev = qml.device("default.qubit")


@qml.qnode(dev)
def circuit(state=0):
    qml.BasisState(state=state, wires=range(1, N + 1))
    QChT()
    return qml.state()


#############################################
# Finally, we can reproduce the circuit diagram shown at the beginning of this section using ``qml.draw_mpl``.


def circuit_to_draw():
    qml.BasisState(state=0, wires=range(1, N + 1))
    QChT()


fig, ax = qml.draw_mpl(circuit_to_draw, decimals=2, style="pennylane")()
fig.show()

#############################################
# Note the new function is defined only to remove the returned ``qml.state``, simplifying the circuit drawn.

#############################################
# Testing the QChT
# ----------------
# With our QChT circuit, let's see if the orthonormality described earlier holds.
# To do this, we'll use the computational state :math:`|7\rangle`, which will transform into :math:`|\tau(x_7^\mathrm{Ch})\rangle`.
# Then, we will compute the overlap at the nodes with all other :math:`|\tau(x_j^\mathrm{Ch})\rangle`.

j = 7  # initial state in computational basis

# compute state after transform
total_state = circuit(state=j)

# reduce state size, effectively removing the ancilla
state = total_state[: 2**N] * np.sqrt(2)


# compute nodes
def ch_node(j, N):
    return np.cos(pi * (2 * j + 1) / 2 ** (N + 1))


js = list(range(int(len(state))))
nodes = [ch_node(i, N) for i in js]

# compute overlap with other basis states using np.inner()
overlaps = []
for i in js:
    state_i = circuit(state=i)[: 2**N] * np.sqrt(2)
    overlaps.append(np.inner(state, state_i))

#############################################
# Now we plot the squared overlaps at the nodes computed from the circuit transformation and compare to the definition, plotting the squared overlaps at all values of :math:`x`.

import matplotlib.pyplot as plt


def T_n(x, n):
    """Chebyshev polynomial of order n"""
    return np.cos(n * np.arccos(x))


def overlap_sq(x, xp, N):
    """computes the squared overlap"""
    numerator = T_n(xp, 2**N + 1) * T_n(x, 2**N) - T_n(xp, 2**N) * T_n(x, 2**N + 1)
    return numerator**2 / (2 ** (2 * N)) / (xp - x) ** 2


plt.style.use("pennylane.drawer.plot")
fig, ax = plt.subplots()
ax.set(xlabel="x", ylabel="Squared Overlap")

# plot squared overlaps computed in circuit
ax.plot(nodes, np.abs(overlaps) ** 2, marker="o", label="circuit")

# plot expected squared overlaps
xs = np.linspace(-1, 1, 1000)
ax.plot(xs, [overlap_sq(x, nodes[j], N) for x in xs], label="expectation")

ax.legend()
plt.show()


#############################################
# Almost orthonormal. We note that the amplitude at the corresponding node approaches :math:`1` if the number of qubits is increased.
#
# Let's also see if the amplitudes of the state in the computational basis agree with expectation.
# To do this, we just modify our ``circuit`` function to return the probabilities of each of the computational basis states (ignoring the ancilla).


@qml.qnode(dev)
def circuit(state=None):
    qml.BasisState(state=state, wires=range(1, N + 1))
    QChT()
    return qml.probs(wires=range(1, N + 1))


probs = circuit(state=j)

# computational basis indices
x = range(2**N)


def tau_amplitudes(x, k, N):
    """computes the expected amplitud es of tau"""
    if k == 0:
        prefactor = 1 / 2 ** (N / 2)
    else:
        prefactor = 1 / 2 ** ((N - 1) / 2)
    return prefactor * np.cos(k * np.arccos(x))


fig, ax = plt.subplots()
ax.plot(x, probs, "o", label="circuit")
ax.plot(x, [tau_amplitudes(nodes[j], xs, N) ** 2 for xs in x], label="expectation")
ax.set(xlabel=r"|k$\rangle$", ylabel="Probability")
ax.legend()
plt.show()

#############################################
# The circuit output probabilities are exactly what we want.
#
#
# Conclusion
# ----------
# In this tutorial, we've gone through how to implement the QChT from a recent paper by Williams *et al.*, and tested the circuit output by looking at the state amplitudes and the orthonormality.
# Further work could test the phase of the output to make sure it matches what we expect the QChT to output.
# One could also implement the quantum Chebyshev feature map from the same paper, which prepares a state in the Chevyshev space via a parameter :math:`x``.
#
#
# References
# ----------
#
# .. [#williams2023]
#
#   Chelsea A. Williams, Annie E. Paine, Hsin-Yu Wu, Vincent E. Elfving and Oleksandr Kyriienk. "Quantum Chebyshev transform: mapping, embedding, learning and sampling distributions." `arxiv:2306.17026 <https://arxiv.org/abs/2306.17026>`__ (2023).
#
# .. [#delejarza2025]
#
#   Jorge J. Martínez de Lejarza, Hsin-Yu Wu, Oleksandr Kyriienko, Germán Rodrigo, Michele Grossi. "Quantum Chebyshev probabilistic models for fragmentation functions." `arxiv:2503.16073 <https://arxiv.org/abs/2503.16073>`__ (2025).
#
# About the author
# ----------------
#
