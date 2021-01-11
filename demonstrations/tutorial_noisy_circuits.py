r"""

Noisy circuits
==============

.. meta::
    :property="og:description": Learn how to simulate nosiy quantum circuits
    :property="og:image":

.. related::

    tutorial_noisy_circuit_optimization Optimizing noisy circuits with Cirq
    pytorch_noise PyTorch and noisy devices

In this demonstration, you'll learn how to simulate noisy circuits using built-in functionality in
PennyLane. We'll cover the basics of noisy channels and density matrices, then use example code to
simulate noisy circuits. PennyLane, the library for differentiable quantum computations, has
unique features that enable us to compute *gradients* of noisy channels. So you'll also learn how to employ
channel gradients to optimize noise parameters in a circuit.

We're putting the N in NISQ. 🇳
"""

##############################################################################
#
# Noisy operations
# ----------------
# Noise is any unwanted transformation that corrupts the intended
# output of a quantum computation. It can be separated into two categories. **Coherent
# noise** is described by unitary operations that maintain the purity of the output quantum state.
# A common source are systematic errors originating from imperfectly-calibrated devices that do
# not exactly apply the desired gates. **Incoherent noise** is more problematic: it originates from a
# quantum computer becoming entangled with the environment, resulting in
# mixed states as outputs. Incoherent noise leads to
# outputs that are always random, regardless of what basis we measure in.
#
# Mixed states are described by `density matrices <https://en.wikipedia.org/wiki/Density_matrices>`__.
# This means that, compared to pure states that
# are described as vectors, a different method is needed for simulating mixed-state quantum
# computations. This is the purpose of PennyLane's ``default.mixed`` device, which provides native
# support for mixed states and for simulating all types of noise.
#
# Let's use ``default.mixed`` to simulate a simple circuit for preparing the
# Bell state :math:`|\psi\rangle=\frac{1}{\sqrt{2}}(|00\rangle+|11\rangle)`. We ask the QNode to
# return the expectation value of :math:`Z_0\otimes Z_1`:
#
import pennylane as qml
from pennylane import numpy as np

qml.enable_tape()
dev = qml.device('default.mixed', wires=2)
ZZ = qml.PauliZ(0) @ qml.PauliZ(1)


@qml.qnode(dev)
def circuit():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.expval(ZZ)


print(f"QNode output = {circuit():.4f}")

######################################################################
# The device stores the output state as a density matrix. In this case, since we haven't
# introduced any noise, the density matrix is equal to the rank-one
# projector :math:`|\psi\rangle\langle\psi|, where :math:`|\psi\rangle=\frac{1}{\sqrt{
# 2}}(|00\rangle + |11\rangle)`.

print(np.real(dev.state))

######################################################################
# Just as mixed states are represented by density matrices, incoherent noise is modelled by
# *quantum channels*, which are linear, completely positive, and trace-preserving (`CPTP
# <https://www.quantiki.org/wiki/channel-cp-map>`__) maps. A convenient strategy for representing
# quantum channels is to employ Kraus operators :math:`K = \{K_i\}` satisfying the condition
# :math:`\sum_i K_{i}^{\dagger} K_i = \mathcal{I}`. From an initial state :math:`\rho`, the output
# state after the action of a channel :math:`E` is:
#
# .. math::
#
#     \rho' = E(\rho) = \sum_i K_i \rho K_{i}^{\dagger}.
#
# The action of a quantum channel can thus be interpreted as applying the transformation
# :math:`K_i` with probability :math:`p_i = \text{Tr}[K_i \rho K_{i}^{\dagger}]`. For
# example, consider the bit flip channel. It describes a transformation that flips the state of
# a qubit (applies an X gate) with probability :math:`p`. Its Kraus operators are
#
# .. math::
#
#     K_0 = (1-p)\begin{pmatrix}1 & 0\\ 0 & 1\end{pmatrix}, \quad
#     K_1 = p\begin{pmatrix}0 & 1\\ 1 & 0\end{pmatrix}.
#
# This channel can be implemented in PennyLane using the :func:`qml.BitFlip <pennylane.BitFlip>` operation.
# Let's see what happens when we simulate this type of noise acting on
# both qubits in the circuit. We'll evaluate the QNode for different bit flip probabilitites.
#


@qml.qnode(dev)
def bitflip_circuit(p):
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    qml.BitFlip(p, wires=0)
    qml.BitFlip(p, wires=1)
    return qml.expval(ZZ)


ps = [0.001, 0.01, 0.1, 0.2]
for p in ps:
    print(f"QNode output = {bitflip_circuit(p):.4f}")

print(f"Output state = \n{np.real(dev.state)}")
######################################################################
# The circuit behaves quite differently in the presence of noise! This will be familiar to anyone
# that has run an algorithm on quantum hardware. It also motivates why error mitigation and error
# correction are so important.
#
# Besides the bit flip channel, PennyLane supports several other noisy channels that are commonly
# used to describe experimental imperfections: :func:`~.pennylane.PhaseFlip`,
# :func:`~.pennylane.AmplitudeDamping`, :func:`~.pennylane.GeneralizedAmplitudeDamping`,
# :func:`~.pennylane.PhaseDamping`, and the :func:`~.pennylane.DepolarizingChannel`. You can also
# build your own custom channel using the operation :func:`~.pennylane.QubitChannel` by
# specifying its Kraus matrices.
#
# For example, the depolarizing channel is a generalization of
# the bit flip and phase flip channels, where each of the three possible Pauli errors can be
# applied to a single qubit. A circuit modelling the effect of depolarizing noise in
# preparing a Bell state is implemented below.


@qml.qnode(dev)
def depolarizing_circuit(p):
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    qml.DepolarizingChannel(p, wires=0)
    qml.DepolarizingChannel(p, wires=1)
    return qml.expval(ZZ)


ps = [0.001, 0.01, 0.1, 0.2]
for p in ps:
    print(f"QNode output = {depolarizing_circuit(p):.4f}")

######################################################################
# Modelling the noise that occurs in real experiments requires careful consideration. PennyLane
# offers the flexibility to experiment with different combinations of noisy channels to mimic the
# performance of quantum algorithms when deployed on real devices.
#
# Channel gradients
# -----------------
#
# The ability to compute gradients of any operation is an essential ingredient of differentiable
# quantum programming. In PennyLane, it is possible to
# compute gradients of noisy channels and optimize them inside variational circuits. Analytical
# gradients exist for channels whose Kraus operators are proportional to unitary
# matrices [#johannes]_. In other cases, gradients are evaluated using finite difference.
#
# To illustrate this property, we'll consider an elementary example. We'll aim to train the noise
# parameters of a circuit in order to reproduce an observed expectation value. So suppose that we
# run the circuit to prepare a Bell state
# on a hardware device and observe that the expectation value of :math:`Z_0\otimes Z_1` is
# not equal to 1, as would occur with an ideal device. In the experiment, it is known that the
# major source of noise is amplitude damping, as would happen for example due to photon loss.
# This process projects a state to :math:`|0\rangle` with probability :math:`p`. What damping
# parameter explains this outcome?
#
# We can answer this question by optimizing the channel parameters to reproduce the experimental
# observation! Since the parameter :math:`p` is a probability, we use a sigmoid function to
# ensure that the trainable parameters give rise to a valid channel parameter (i.e., a number between 0 and 1).
#
ev = 0.7781  # observed expectation value


def sigmoid(x):
    return 1/(1+np.exp(-x))


@qml.qnode(dev)
def damping_circuit(x):
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    qml.AmplitudeDamping(sigmoid(x), wires=0)  # p = sigmoid(x)
    qml.AmplitudeDamping(sigmoid(x), wires=1)
    return qml.expval(ZZ)

######################################################################
# We optimize the circuit with respect to a simple cost function that attains its minimum when
# the output of the QNode is equal to the experimental value:


def cost(x):
    return (damping_circuit(x) - ev)**2

######################################################################
# All that remains is to optimize the parameter. We use a straightforward gradient descent
# method.


opt = qml.GradientDescentOptimizer(stepsize=10)
steps = 50
x = 0.0

for i in range(steps):
    x = opt.step(cost, x)

p = sigmoid(x)
print(f"QNode output = {damping_circuit(x):.4f}")
print(f"Observed expectation value = {ev}")
print(f"Optimized noise parameter = {p:.4f}")

######################################################################
# Voilà! We've trained the noisy channel to reproduce the experimental observation. 😎
#
# Developing quantum algorithms that leverage the power of NISQ devices requires serious
# consideration of the effects of noise. With PennyLane, you have access to tools that can
# help you design, simulate, and optimize noisy quantum circuits. We look forward to seeing what
# the quantum community can achieve with them! 🚀
#
# References
# ----------
#
# .. [#johannes]
#
#     Johannes Jakob Meyer, Johannes Borregaard, and Jens Eisert, "A variational toolbox for quantum
#     multi-parameter estimation." `arXiv:2006.06303 (2020) <https://arxiv.org/abs/2006.06303>`__.
#
#
#
