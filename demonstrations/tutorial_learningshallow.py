r"""Learning shallow quantum circuits
=====================================

In a recent paper `Learning shallow quantum circuits <https://arxiv.org/abs/2401.10095>`_ [#Huang]_, Huang et al
introduce and prove performance bounds on efficient algorithms to learn shallow, constant depth unitaries.
At the heart of the paper lie local inversions that locally undo a quantum circuit, as well as a circuit "sewing"
technique that let's one construct a global inversion from local ones.
We are going to review these concepts and showcase how to implement them in PennyLane.

Introduction
------------

Shallow, constant depth quantum circuit are provably powerful [#Bravyi]_.
At the same time, quantum neural networks are known to be difficult to train [#Anschuetz].
The authors of [#Huang]_ tackle the question of whether or not shallow circuits are efficiently learnable.
They go through an exhaustive list of relevant learning scenarios, like learning :math:`U` directly,
learning :math:`\hat{U}` s.t. :math:`\hat{U}|0^{\otimes n} \rangle = U|0^{\otimes n} \rangle`, for general and restricted cases in terms of
available gates and locality criteria.
At the heart of the solutions to all these scenarios lies local inversions and sewing them together to form a global inversion.

If you, like me, struggle to grasp these concepts from 88 pages of Lemmas and Theorems without examples, this demo is for you.
We hope to absolve you from some of these difficulties by providing some concrete examples with PennyLane code.


Local Inversions
----------------

A local inversion is a unitary circuit that locally disentangles one qubit after a previous, different unitary entangled them. Let us 
make an explicit example in PennyLane after some boilerplate imports. Let us look at a very shallow unitary circuit
:math:`U^\text{test} = \text{CNOT}_{(0, 1)}\text{CNOT}_{(2, 3)}\text{CNOT}_{(1, 2)} H^{\otimes n}`
"""
import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt

def U_test():
    for i in range(4):
        qml.Hadamard(i)
    qml.CNOT((1, 2))
    qml.CNOT((2, 3))
    qml.CNOT((0, 1))

def draw(func, *args, **kwargs):
    with qml.tape.QuantumTape() as tape:
        func(*args, **kwargs)
    print(qml.drawer.tape_mpl(tape))

draw(U_test)


##############################################################################
# And let us locally invert it. For that, we just follow the light-cone of the qubit that we want to invert
# and perform the inverse operations in reverse order. After performing :math:`U^\text{test}` and the inversion of qubit 
# :math:`0`, :math:`V_0` we find the reduced state :math:`|0 \times 0|` on the correct qubit.

def V_0():
    qml.CNOT((0, 1))
    qml.CNOT((1, 2))
    qml.Hadamard(0)
    
dev = qml.device("default.qubit")

@qml.qnode(dev)
def local_inversion():
    U_test()   # some shallow unitary circuit
    V_0()      # supposed to disentangle qubit 0
    return qml.density_matrix(wires=[0])

print(np.allclose(local_inversion(), np.array([[1., 0.], [0., 0.]])))

##############################################################################
# Here we can just construct the local inversions from knowing the circuit structure of :math:`U^\text{test}`.
# We are interested in scenarios where we are trying to learn an unknown :math:`U`.
# While learning a global inversion :math:`V` s.t. :math:`U V = \mathbb{1}` is difficult, learning a local 
# inversion :math:`V_0` s.t. :math:`\text{tr}_{\neq 0}\left[U V_0\right] = \mathbb{1}_0` is provably feasible
# and can be done for all qubits. More on that later. For the moment, let us assume that we have learned these local inversions and continue with sewing.
#
# Let us also construct variants of the other local inversions as we will need them in the next section.

def V_1():
    qml.CNOT((0, 1))
    qml.CNOT((1, 2))
    qml.Hadamard(1)

def V_2():
    qml.CNOT((2, 3))
    qml.CNOT((1, 2))
    qml.Hadamard(2)

def V_3():
    qml.CNOT((2, 3))
    qml.CNOT((1, 2))
    qml.Hadamard(3)


##############################################################################
# Circuit Sewing
# --------------
#
# It is highly non-trivial in general to recombine these local inversions into a global inversion (which constitutes a variant of the `quantum marginal problem <https://arxiv.org/abs/1404.1085>`_).
# So how does knowing local inversions :math:`\{V_0, V_1, V_2, V_3\}` help us with solving the original goal of finding a global inversion :math:`U V = \mathbb{1}`?
# The authors introduce a clever trick that they coin circuit sewing. It works by moving the decoupled qubit to an ancilla register and restoring ("repairing") the unitary on the remaining wires. 
# Let us walk through the steps.
#
# We already saw how to decouple qubit :math:`0` in ``local_inversion()``. We continue by swapping out
# the decoupled wire with an ancilla qubit and "repairing" the circuit by applying :math:`V^\dagger_0`.
# This is called repairing because :math:`V_1` can now decouple qubit 1, which would not be possible if we just applied :math:`V_0` and :math:`V_1` consecutively.
# We label all ancilla wires by ``i+n`` to have an easy 1-to-1 correpsondence and we see that qubit ``1`` is successfully decoupled.
# For completeness, we also check that the swapped out qubit (now moved to wire ``0+n``) is decoupled still.
#

n = 4 # number of qubits

@qml.qnode(dev)
def sewing_1():
    U_test()   # some shallow unitary circuit
    V_0()      # supposed to disentangle qubit 0
    qml.SWAP((0, n))
    qml.adjoint(V_0)()
    V_1()
    return qml.density_matrix(wires=[1]), qml.density_matrix(wires=[n])

r1, rn = sewing_1()
print(f"Sewing step 3")
print(f"rho_3 = |0x0| {np.allclose(r1, np.array([[1, 0], [0, 0]]))}")
print(f"rho_0+n = |0x0| {np.allclose(rn, np.array([[1, 0], [0, 0]]))}")

##############################################################################
# We can continue this process for all qubits. Let us be tedious and do all steps one by one.

@qml.qnode(dev)
def sewing_2():
    U_test()   # some shallow unitary circuit
    V_0()      # supposed to disentangle qubit 0
    qml.SWAP((0, n))
    qml.adjoint(V_0)()
    V_1()
    qml.SWAP((1, n + 1))
    qml.adjoint(V_1)()
    V_2()
    return qml.density_matrix(wires=[2]), qml.density_matrix(wires=[n]), qml.density_matrix(wires=[n + 1])

r2, rn, rn1 = sewing_2()
print(f"Sewing step 3")
print(f"rho_3 = |0x0| {np.allclose(r2, np.array([[1, 0], [0, 0]]))}")
print(f"rho_0+n = |0x0| {np.allclose(rn, np.array([[1, 0], [0, 0]]))}")
print(f"rho_1+n = |0x0| {np.allclose(rn1, np.array([[1, 0], [0, 0]]))}")

@qml.qnode(dev)
def sewing_3():
    U_test()   # some shallow unitary circuit
    V_0()      # supposed to disentangle qubit 0
    qml.SWAP((0, n))
    qml.adjoint(V_0)()
    V_1()
    qml.SWAP((1, n + 1))
    qml.adjoint(V_1)()
    V_2()
    qml.SWAP((2, n + 2))
    qml.adjoint(V_2)()
    V_3()
    return qml.density_matrix(wires=[3]), qml.density_matrix(wires=[n]), qml.density_matrix(wires=[n + 1]), qml.density_matrix(wires=[n + 2])

r3, rn, rn1, rn2 = sewing_3()
print(f"Sewing step 3")
print(f"rho_3 = |0x0| {np.allclose(r3, np.array([[1, 0], [0, 0]]))}")
print(f"rho_0+n = |0x0| {np.allclose(rn, np.array([[1, 0], [0, 0]]))}")
print(f"rho_1+n = |0x0| {np.allclose(rn1, np.array([[1, 0], [0, 0]]))}")
print(f"rho_2+n = |0x0| {np.allclose(rn2, np.array([[1, 0], [0, 0]]))}")

##############################################################################
# After one final swap and repair, we arrive at a state that has all original wires decoupled. We just need to move them back to their original position
# with a global SWAP. 
# But not just that, we also now know that, globally, the original :math:`U^\text{test}` is inverted.

@qml.qnode(dev)
def sewing_final():
    U_test()   # some shallow unitary circuit
    V_0()      # supposed to disentangle qubit 0
    qml.SWAP((0, n))
    qml.adjoint(V_0)()
    V_1()
    qml.SWAP((1, n + 1))
    qml.adjoint(V_1)()
    V_2()
    qml.SWAP((2, n + 2))
    qml.adjoint(V_2)()
    V_3()
    qml.SWAP((3, n + 3))
    qml.adjoint(V_3)()
    for i in range(n):
        qml.SWAP((i + n, i))
    return qml.density_matrix([0, 1, 2, 3])

psi0 = np.eye(2**4)[0] # |0>^n
np.allclose(sewing_final(), np.outer(psi0, psi0))

##############################################################################
# Overall, we have constructed
#
# ..math:: U_\text{sewed} = U^\dagger \otimes U.
#
# Depending on whether we want to apply :math:`U` or :math:`U^\dagger` on the first register, we perform the global swap.


##############################################################################
#
# Numerical experiment
# --------------------
#
# The actual learning in this procedure happens in obtaining the local inversions :math:`\{V_0, V_1, V_2, V_3\}`.
# The paper relies on existence proofs from gate synthesis [#Shende]_, whereas no explicit construction is given.
# Let us here look at an explicit example by constructing a target unitary of some structure and Ansätze for 
# the local inversions that has a different structure. There are many ways to obtain local inversions, this is just one we find convenient.
# 
# First, let us construct a target unitary

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

n = 4
wires = range(n)
U_params = jax.random.normal(jax.random.PRNGKey(0), shape=(2, n//2), dtype=float)

def U_target(wires):
    for i in range(n):
        qml.Hadamard(wires=wires[i])
    # brick-wall ansatz
    for i in range(0, n, 2):
        qml.IsingXX(U_params[0, i], wires=(wires[i], wires[(i+1)%len(wires)]))
    for i in range(1, n, 2):
        qml.IsingXX(U_params[1, i], wires=(wires[i], wires[(i+1)%len(wires)]))

draw(U_target, wires)

##############################################################################
# Ansatz

n_layers = 2

def V_i(params, wires):

    for i in range(n):
        qml.RX(params[0, 0, i], i)
    for i in range(n):
        qml.RY(params[0, 1, i], i)
    
    for ll in range(n_layers):
        for i in range(0, n, 2):
            qml.CNOT((wires[i], wires[i+1]))
        for i in range(1, n, 2):
            qml.CNOT((wires[i], wires[(i+1)%n]))
        for i in range(n):
            qml.RX(params[ll+1, 0, i], i)
        for i in range(n):
            qml.RY(params[ll+1, 1, i], i)

params = jax.random.normal(jax.random.PRNGKey(10), shape=(n_layers+1, 2, n), dtype=float)

draw(V_i, params, wires)

##############################################################################
# some more boiler plate code
import optax
from datetime import datetime
from functools import partial
X, Y, Z = qml.PauliX, qml.PauliY, qml.PauliZ

def run_opt(value_and_grad, theta, n_epochs=100, lr=0.1, b1=0.9, b2=0.999, verbose=True):

    optimizer = optax.adam(learning_rate=lr, b1=b1, b2=b2)
    opt_state = optimizer.init(theta)

    energy = np.zeros(n_epochs)
    gradients = []
    thetas = []

    @jax.jit
    def step(opt_state, theta):
        val, grad_circuit = value_and_grad(theta)
        updates, opt_state = optimizer.update(grad_circuit, opt_state)
        theta = optax.apply_updates(theta, updates)

        return opt_state, theta, val

    t0 = datetime.now()
    ## Optimization loop
    for n in range(n_epochs):
        opt_state, theta, val = step(opt_state, theta)

        energy[n] = val
        thetas.append(
            theta
        )
    t1 = datetime.now()
    if verbose:
        print(f"final loss: {val}; min loss: {np.min(energy)}; after {t1 - t0}")
    
    return thetas, energy, gradients

dev = qml.device("default.qubit")

@qml.qnode(dev, interface="jax")
def qnode_i(params, i):
    U_target(wires)
    V_i(params, wires)
    return [qml.expval(P(i)) for P in [X, Y, Z]]
    
@partial(jax.jit, static_argnums=1)
@jax.value_and_grad
def cost_i(params, i):
    X, Y, Z = qnode_i(params, i)
    return X**2 + Y**2 + (1-Z)**2

##############################################################################
# obtain local inversions

params_i = []
for i in range(n):
    cost = partial(cost_i, i=i)
    thetas, _, _ = run_opt(cost, params)
    params_i.append(thetas[-1])

for i in range(n):
    X_res, Y_res, Z_res = qnode_i(params_i[i], i)
    print(f"Bloch sphere coordinates of qubit {i} after inversion: {X_res:.5f}, {Y_res:.5f} {Z_res:.5f}")

##############################################################################
# Sew them together to invert U

def U_sew():
    for i in range(n):
        # local sewing: inversion, exchange, heal
        V_i(params_i[i], range(n))
        qml.SWAP((i, i+n))
        qml.adjoint(V_i)(params_i[i], range(n))

    # global SWAP
    for i in range(n):
        qml.SWAP((i, i+n))

@qml.qnode(dev, interface="jax")
def sewing_test():
    U_target(range(n))
    U_sew()
    return qml.density_matrix(range(4))

print(np.allclose(sewing_test(), np.outer(psi0, psi0), atol=1e-1))

##############################################################################
# text
#
# Conclusion
# ----------
#
# Conclusions



##############################################################################
# 
# References
# ----------
#
# .. [#Huang]
#
#     Hsin-Yuan Huang, Yunchao Liu, Michael Broughton, Isaac Kim, Anurag Anshu, Zeph Landau, Jarrod R. McClean
#     "Learning shallow quantum circuits"
#     `arXiv:2401.10095 <https://arxiv.org/abs/2401.10095>`__, 2024.
#
# .. [#Bravyi]
#
#     Sergey Bravyi, David Gosset, Robert Koenig
#     "Quantum advantage with shallow circuits"
#     `arXiv:1704.00690 <https://arxiv.org/abs/1704.00690>`__, 2017.
#
# .. [#Anschuetz]
#
#     Eric R. Anschuetz, Bobak T. Kiani
#     "Beyond Barren Plateaus: Quantum Variational Algorithms Are Swamped With Traps"
#     `arXiv:2205.05786 <https://arxiv.org/abs/2205.05786>`__, 2022.
#
# .. [#Shende]
#
#     Vivek V. Shende, Stephen S. Bullock, Igor L. Markov
#     "Synthesis of Quantum Logic Circuits"
#     `arXiv:quant-ph/0406176 <https://arxiv.org/abs/quant-ph/0406176>`__, 2004.
#

##############################################################################
# About the author
# ----------------
# .. include:: ../_static/authors/korbinian_kottmann.txt
