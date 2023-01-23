r"""Differentiable pulse programming with qubits in PennyLane
=============================================================

.. meta::
    :property="og:description": Simulating differentialble pulse programs in PennyLane with qubits
    :property="og:image": https://pennylane.ai/qml/_images/pauli_shadows.jpg

*Author: Korbinian Kottmann — Posted: 20 February 2023.

In this demo we are going to introduce pulse gates and differentiable pulse programming, showcase
the current functionality in PennyLane and run the ctrl-VQE algorithm for an example molecule.

Topic1
------
Reference docs :doc:`tutorial_classical_shadows`
Reference literature, ctrl vqe paper MET [#Asthana2022]_
.. math:: \sum_i c_i k_i

Pulses in quantum computers
---------------------------
# Introduce pulses, channels, pulse gates, time dependent Schrodinger equation
In non-measurement-based quantum computers such as superconducting and ion trap systems, qubits are realized through physical systems with a discrete set of energy levels.
For example, transmon qubits realize an anharmonic oscillator whose ground and first excited states can serve as the two energy
levels of a qubit. Such a qubit can be controlled via an electromagnetic field tuned to its energy gap. In general, this
electromangnetic field can be altered in time, leading to a time-dependent Hamiltonian interaction :math:`H(t)`.
We call driving the system with such an electromagnetic field for a fixed time window a pulse sequence. During a pulse sequence, the state evolves according
to the time-dependent Schrodinger equation

.. math:: \frac{d}{dt}|\psi\rangle = -i H(t) |\psi\rangle

realizing a unitary evolution :math:`U(t_0, t_1)` from times :math:`t_0` to :math:`t_1` of the input state, i.e. 
:math:`|\psi(t_1)\rangle = U(t_0, t_1) |\psi(t_0)\rangle`.

In non-measurement-based digital quantum computers, the amplitude and frequencies of predefined pulse sequences are
fine tuned to realize the native gates of the quantum computer. More specifically, the Hamiltonian interaction :math:`H(t)`
is tuned such that the respective evolution :math:`U(t_0, t_1)` realizes for example a Pauli or CNOT gate.

Pulse programming in PennyLane
------------------------------

A user of a quantum computer typically operates on the higher and more abstract gate level.
Future fault tolerance quantum computers require this abstraction to allow for error correction.
For noisy and intermediate sized quantum computers, the abstraction of decomposing quantum algorithms
into a fixed native gate set can be a hindrance and unnecessarily increase execution time, therefore leading
to more decoherence. The idea of differentiable pulse programming is to optimize quantum circuits on the pulse
level and therefore allowing for the shortest interaction sequence a hardware system allows.

In PennyLane, we can now simulate arbitrary qubit system interactions to explore the possibilities of such pulse programs.
We can create a time-dependent Hamiltonian :math:`H(p, t)= \sum_i f_i(p, t) H_i` with envelope :math:`f_i(p, t)` that may
depend on parameters :math:`p` and constant operators :math:`H_i` in PennyLane in the following way
"""

import pennylane as qml
import jax.numpy as jnp

def f1(p, t):
    return jnp.polyval(p, t)
def f2(p, t):
    return p[0] * jnp.sin(p[1] * t)

Ht = f1 * qml.PauliX(0) + f2 * qml.PauliY(1)

##############################################################################
# Note that when constructing such a Hamiltonian, the ``callable`` functions are 
# expected to have the fixed signature ``(p, t)``, such that the Hamiltonian itself
# can be called via ``H((p1, p2), t)``. 

p = (jnp.ones(5), jnp.array([1., jnp.pi]))
print(Ht(p, 0.5))

##############################################################################
# We can construct more complicated Hamiltonians like :math:`\sum_i X_i X_{i+1} + \sum_i f_i(p, t) Z_i` using ``qml.ops.dot``
# in the following way.

coeffs = [1.] * 2
coeffs += [lambda p, t: jnp.polyval(p, t) for _ in range(3)]
ops = [qml.PauliX(i) @ qml.PauliX(i+1) for i in range(2)]
ops += [qml.PauliZ(i) for i in range(3)]

Ht = qml.ops.dot(coeffs, ops)
p = tuple(jnp.ones(3) for _ in range(3))
print(Ht(p, 0.5))

##############################################################################
# PennyLane also provides a variety of convenience functions to enable for example piece-wise-constant parametrizations,
# i.e. defining the function values at fixed time bins as parameters.

# PWC example

##############################################################################
# Researchers interested in more specific hardware systems can simulate them using the specific Hamiltonian interactions.
# For an example of a transmon qubit system, scroll down to the ctrl-VQE example.


##############################################################################
# More text 
# Gradients of pulse gates
# ------------------------
# Can compute them on hardware with parameter shift rule [#Leng2022]_#



import pennylane as qml
import pennylane.numpy as np
import jax.numpy as jnp
import jax

import optax # for jax optimization

import matplotlib.pyplot as plt
from datetime import datetime

symbols = ["H", "H"]
coordinates = np.array([[0., 0., 0.],[0., 0., 1.5]])

basis_set = "sto-3g"
H, n_wires = qml.qchem.molecular_hamiltonian(
    symbols,
    coordinates,
    charge=0,
    mult=1,
    basis=basis_set,
    mapping="bravyi_kitaev",
    method="pyscf",
)

print(f"no qubits: {n_wires}")
coeffs, obs = jnp.array(H.coeffs), H.ops
H_obj = qml.Hamiltonian(coeffs, obs, grouping_type="qwc")
H_obj_m = qml.matrix(H_obj)
E_exact = np.min(np.linalg.eigvalsh(H_obj_m))

##############################################################################
# We are considering a coupled transmon qubit system with the constant drift term Hamiltonian 
#
# .. math:: H_D = \sum_q \omega_q a_q^\dagger a_q + \sum_q \frac{\delta_q}{2} a^\dagger_q a^\dagger_q a_q a_q + \sum_{\braket{pq}} g_{pq} a^\dagger_p a_q
# 
# with bosonic creation and annihilation operators. We are only going to consider qubit subspace such that the quadratic term is zero.
# The order of magnitude of the resonance frequencies $\omega_q$ and coupling strength $g_{pq}$ are taken from [#Asthana2022]_.

def a(wires):
    return 0.5*qml.PauliX(wires) + 0.5j* qml.PauliY(wires)
def ad(wires):
    return 0.5*qml.PauliX(wires) - 0.5j* qml.PauliY(wires)

omega = 4.8 * jnp.ones(n_wires)
g = 0.01 * jnp.ones(n_wires-1)

H_D = qml.op_sum(*[qml.s_prod(omega[i], ad(i) @ a(i)) for i in range(n_wires)])
H_D += qml.op_sum(*[qml.s_prod(g[i], ad(i) @ a(i+1)) for i in range(n_wires-1)])

##############################################################################
# The system is driven under the control term
#
# .. math:: H_C(t) = \sum_q \Omega_q(t) \left(e^{i\nu_q t} a_q + e^{-i\nu_q t} a^\dagger_q \right)
# 
# with the (real) time-dependent amplitude :math:`\Omega(t)` and frequency :math:`\nu_q` of the drive.

# TODO use official convenience functions once merged
def pwc(t1, t2):
    def wrapped(params, t):
        N = len(params)
        idx = jnp.array(N/(t2 - t1) * (t - t1), dtype=int) # corresponding sample
        return params[idx]

    return wrapped

def f(t1, t2, sgn=1.):
    # assuming p = (len(t_bins) + 1) for the frequency nu
    def wrapped(p, t):
        return pwc(t1, t2)(p[:-1], t) * jnp.exp(sgn*1j*p[-1]*t)
    return wrapped

t1 = 0.
t2 = 15.

fs = [f(t1, t2, 1.) for i in range(n_wires)]
fs += [f(t1, t2, -1.) for i in range(n_wires)]
ops = [a(i) for i in range(n_wires)]
ops += [ad(i) for i in range(n_wires)]
# ops = [qml.PauliX(i) for i in range(n_wires)]
# ops += [qml.PauliY(i) for i in range(n_wires)]

H_C = qml.ops.dot(fs, ops)

H_pulse = H_D + H_C
t_bins = 200 # number of time bins
p = jnp.array([jnp.ones(t_bins + 1) for _ in range(n_wires)])
dev = qml.device("default.qubit", wires=range(n_wires))

@jax.jit
@qml.qnode(dev, interface="jax")
def qnode(p, t=15.):
    qml.evolve(H_pulse)(params=(*p, *p), t=t)
    return qml.expval(H_obj)


def abs_diff(p):
    """compute |p_i - p_i-1|^2"""
    #p = jnp.array(p)
    return jnp.mean(jnp.abs(jnp.diff(p, axis=1))**2)
def cost_fn(p):
    C_exp = qnode(p)
    C_par = jnp.mean(jnp.abs(p)**2)
    C_der = abs_diff(p)
    return C_exp + C_par + C_der
theta = jnp.array([jnp.ones(t_bins) for _ in range(n_wires)])

n_epochs = 50
cosine_decay_scheduler = optax.cosine_decay_schedule(0.5, decay_steps=n_epochs, alpha=0.95)
optimizer = optax.adam(learning_rate=0.1) # cosine_decay_scheduler
opt_state = optimizer.init(theta)

value_and_grad = jax.jit(jax.value_and_grad(cost_fn, argnums=0))

energy = np.zeros(n_epochs)
cost = np.zeros(n_epochs)
theta_i = [theta]

## Optimization loop

t0 = datetime.now()
_ = value_and_grad(theta)
t1 = datetime.now()
print(f"grad and val compilation time: {t1 - t0}")

for n in range(n_epochs):
    val, grad_circuit = value_and_grad(theta)
    updates, opt_state = optimizer.update(grad_circuit, opt_state)
    theta = optax.apply_updates(theta, updates)

    energy[n] = qnode(theta)
    cost[n] = val
    theta_i.append(theta)

    if not n%5:
        print(f"{n+1} / {n_epochs}; energy: {energy[n]}; cost: {val}")

# TODO: how can the energy be smaller than the exact diagonalization one
fig, axs = plt.subplots(ncols=2, figsize=(10,4))
ax = axs[0]
ax.plot(cost,".:", label="cost")
ax.plot(energy, "x:", label="energy")
#ax.plot(cost - energy, "x:", label="control contribution")
ax.plot([0, n_epochs], [E_exact]*2, "--", label="exact", color="grey")
ax.legend()

ax = axs[1]
for i in range(n_wires):
    ax.plot(theta[i],"-", label=f"qubit {i}", alpha=0.5)
ax.legend()
plt.show()

##############################################################################
#
# Conclusion
# ----------
# Conclusion
#
#
# References
# ----------
#
# .. [#Asthana2022]
#
#     Ayush Asthana, Chenxu Liu, Oinam Romesh Meitei, Sophia E. Economou, Edwin Barnes, Nicholas J. Mayhall
#     "Minimizing state preparation times in pulse-level variational molecular simulations."
#     `arXiv:2203.06818 <https://arxiv.org/abs/2203.06818>`__, 2022.
#
# .. [#Leng2022] 
# 
#     Jiaqi Leng, Yuxiang Peng, Yi-Ling Qiao, Ming Lin, Xiaodi Wu
#     Differentiable Analog Quantum Computing for Optimization and Control
#     `arXiv:2210.15812 <https://arxiv.org/abs/2210.15812>`__, 2022#

##############################################################################
# About the author
# ----------------
# .. include:: ../_static/authors/korbinian_kottmann.txt
