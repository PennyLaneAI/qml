r"""Differentiable pulse programming with qubits in PennyLane
=============================================================

.. meta::
    :property="og:description": Simulating differentialble pulse programs in PennyLane with qubits
    :property="og:image": https://pennylane.ai/qml/_images/pauli_shadows.jpg

*Author: Korbinian Kottmann — Posted: 20 February 2023.

In this demo we are going to introduce pulse gates and differentiable pulse programming, showcase
the current functionality in PennyLane and run the ctrl-VQE algorithm for an example molecule.

Pulses in quantum computers
---------------------------

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
level with the aim of achieving the shortest interaction sequence a hardware system allows.

In PennyLane, we can now simulate arbitrary qubit system interactions to explore the possibilities of such pulse programs.
First, we need to define the time-dependent Hamiltonian :math:`H(p, t)= \sum_i f_i(p, t) H_i` with envelope :math:`f_i(p, t)` that may
depend on parameters :math:`p` and constant operators :math:`H_i`. In PennyLane, we can do this intuitively in the following way.
"""

import pennylane as qml
import pennylane.numpy as np
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt

def f1(p, t):
    return jnp.polyval(p, t)
def f2(p, t):
    return p[0] * jnp.sin(p[1] * t)

Ht = f1 * qml.PauliX(0) + f2 * qml.PauliY(1)

##############################################################################
# Note that when constructing such a Hamiltonian, the ``callable`` functions are 
# expected to have the fixed signature ``(p, t)``, such that the Hamiltonian itself
# can be called via ``H((p1, p2), t)``. 

params = (jnp.ones(5), jnp.array([1., jnp.pi]))
print(Ht(params, 0.5))

##############################################################################
# We can construct more complicated Hamiltonians like :math:`\sum_i X_i X_{i+1} + \sum_i f_i(p, t) Z_i` using :func:`~pennylane.ops.dot`.
# We use two sinusodials with random frequencies as the time-dependent parametrization.

coeffs = [jnp.array(1.)] * 2
coeffs += [lambda p, t: jnp.sin(p[0]*t) + jnp.sin(p[1]*t) for _ in range(3)]
ops = [qml.PauliX(i) @ qml.PauliX(i+1) for i in range(2)]
ops += [qml.PauliZ(i) for i in range(3)]

Ht = qml.ops.dot(coeffs, ops)

# random coefficients
key = jax.random.PRNGKey(777)
subkeys = jax.random.split(key, 3)
params = [jax.random.uniform(subkeys[i], shape=[2], maxval=5) for i in range(3)]
print(Ht(params, 0.5))

##############################################################################
# We can visualize the Hamiltonian interaction by plotting the time-dependent envelopes.

def draw(H, params, t, resolution=200):
    ts = jnp.linspace(0, t, resolution)
    fs = H.coeffs_parametrized
    ops = H.ops_parametrized
    n_channels = len(fs)
    fig, axs = plt.subplots(nrows=n_channels, figsize=(5,2*n_channels))
    for n in range(n_channels):
        ax = axs[n]
        ax.plot(ts, fs[n](params[n], ts))
        ax.set_ylabel(f"{ops[n].__repr__()}")
    axs[0].set_title(f"Drift term: {H.H_fixed()}")
    plt.show()

draw(Ht, params, 4.)

##############################################################################
# PennyLane also provides a variety of convenience functions to enable for example piece-wise-constant parametrizations,
# i.e. defining the function values at fixed time bins as parameters.

# PWC example

##############################################################################
# Researchers interested in more specific hardware systems can simulate them using the specific Hamiltonian interactions.
# For example, we will later simulate a transmon qubit system in the ctrl-VQE example in the last section of this demo.
#
# A pulse program is then executed by using the :func:`~pennylane.ops.evolve` transform to create the evolution
# gate :math:`U(t_0, t_1)`, which implicitly depends on the parameters ``p``.

dev = qml.device("default.qubit", range(4))

ts = jnp.array([0., 3.])
H_obj = sum([qml.PauliZ(i) for i in range(4)])

@jax.jit
@qml.qnode(dev, interface="jax")
def qnode(params):
    qml.evolve(Ht)(params, ts)
    return qml.expval(H_obj)

print(qnode(params))

##############################################################################
# We used the decorator ``jax.jit`` to just-in-time compile this execution. This means the first execution will typically take a little longer with the
# benefit that all following executions will be significantly faster. JIT-compiling is optional, and one can remove the decorator when only single executions
# are of interest.
#
# Gradients of pulse programs
# ---------------------------
# Internally, this program solves the the time-dependent Schrodinger equation using the `Dopri5 <https://en.wikipedia.org/wiki/Dormand%E2%80%93Prince_method>`_ solver for
# ordinary differential equations (ODEs). In particular, the step sizes between :math:`t_0` and :math:`t_1` are chosen adaptively to stay within a given error tolerance.
# We can backpropagate through this ODE solver and obtain the gradient via ``jax.grad``.

print(jax.grad(qnode)(params))

##############################################################################
# Alternatively, one can compute the gradient with the parameter shift rule [#Leng2022]_, which is particularly interesting for
# real hardware execution. In classical simulations, however, backpropagation is recommended.
# 
# Variational quantum eigensolver with pulse programming
# ------------------------------------------------------
# We can now use those gradients to perform the variational quantum eigensolver on the pulse level (ctrl-VQE) as is done in [#Asthana2022]_. 
# First, we define the molecular Hamiltonian whose energy estimate we want to minimize. We are choosing :math:`H_2` as a simple example.


symbols = ["H", "O", "H"]
coordinates = np.array([-0.0399, -0.0038, 0.0, 1.5780, 0.8540, 0.0, 2.7909, -0.5159, 0.0])

basis_set = "sto-3g"
H, n_wires = qml.qchem.molecular_hamiltonian(
    symbols,
    coordinates,
    charge=0,
    mult=1,
    basis=basis_set,
    active_electrons=4,
    active_orbitals=3,
)

print(f"number of qubits: {n_wires}")
coeffs, obs = H.coeffs, H.ops
H_obj = qml.Hamiltonian(jnp.array(coeffs), obs)

##############################################################################
# For such small systems, we can of course compute the exact ground state energy.
# We will later use this to determine if our ctrl-VQE algorithm was successful.

H_obj_m = qml.matrix(H_obj)
E_exact = np.min(np.linalg.eigvalsh(H_obj_m))

##############################################################################
# As a realistic physical system to simulate, we are considering a coupled transmon qubit system with the constant drift term Hamiltonian 
#
# .. math:: H_D = \sum_q \omega_q a_q^\dagger a_q + \sum_q \frac{\delta_q}{2} a^\dagger_q a^\dagger_q a_q a_q + \sum_{\braket{pq}} g_{pq} a^\dagger_p a_q
# 
# with bosonic creation and annihilation operators. We are only going to consider the qubit subspace such that the quadratic term proportional to :math:`\delta_q` is zero.
# The order of magnitude of the resonance frequencies :math:`\omega_q` and coupling strength :math:`g_{pq}` are taken from [#Asthana2022]_.

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
# We let :math:`\Omega(t)` be a piece-wise-constant real function that is optimized alongside the frequencies :math:`\nu_q`.

# TODO use official convenience functions once merged
def pwc(t1, t2):
    def wrapped(params, t):
        N = len(params)
        idx = jnp.array(N/(t2 - t1) * (t - t1), dtype=int) # corresponding sample
        return params[idx]

    return wrapped

def envelope(t1, t2, sign=1.):
    # assuming p = (len(t_bins) + 1) for the frequency nu
    def wrapped(p, t):
        return pwc(t1, t2)(p[:-1], t) * jnp.exp(sign*1j*p[-1]*t)
    return wrapped

t1 = 0.
t2 = 15.

fs = [envelope(t1, t2, 1.) for i in range(n_wires)]
fs += [envelope(t1, t2, -1.) for i in range(n_wires)]
ops = [a(i) for i in range(n_wires)]
ops += [ad(i) for i in range(n_wires)]

H_C = qml.ops.dot(fs, ops)

##############################################################################
# Overall, we end up with the time-dependent parametrized Hamiltonian :math:`H(p, t) = H_D + H_C(p, t)`
# under which the system is evolved for the given time window.

H_pulse = H_D + H_C

dev = qml.device("default.qubit", wires=range(n_wires))

@jax.jit
@qml.qnode(dev, interface="jax")
def qnode(p, t=15.):
    qml.evolve(H_pulse)(params=(*p, *p), t=t)
    return qml.expval(H_obj)

##############################################################################
# Our aim is to minimize the energy expectation value :math:`\langle H_\text{obj} \rangle` but we also need to take 
# certain physical constraints into account. For example, the amplitude of the physical driving field :math:`\Omega(t)` cannot 
# reach arbitrarily large values and cannot change arbitrarily quickly in time. We therefore add two penalty terms quantifying the 
# mean absolute square of the amplitude and changes in the amplitude. The overall cost function is thus
# 
# .. math:: \mathcal{C}(p) = \langle H_\text{obj} \rangle + \sum_{ki} |p^k_i|^2 + \sum_{ki} |p^k_i - p^k_{i-1}|^2.
#

def abs_diff(p):
    """compute |p_i - p_i-1|^2"""
    return jnp.mean(jnp.abs(jnp.diff(p[:, :-1], axis=1))**2)

def cost_fn(p):
    C_exp = qnode(p)                # expectation value
    C_par = jnp.mean(jnp.abs(p)**2) # parameter values
    C_der = abs_diff(p)             # derivative values
    return C_exp + 10*C_par + 10*C_der


##############################################################################
# We now have all the ingredients to run our ctrl-VQE program. We use the adam implementation in ``optax`` for optimizations in ``jax`` for our optimization loop.
import optax 
from datetime import datetime

t_bins = 200 # number of time bins
theta = jnp.array([jnp.ones(t_bins, dtype=float) for _ in range(n_wires)])

n_epochs = 50
optimizer = optax.adam(learning_rate=0.1)
opt_state = optimizer.init(theta)

value_and_grad = jax.jit(jax.value_and_grad(cost_fn, argnums=0))

energy = np.zeros(n_epochs)
cost = np.zeros(n_epochs)
theta_i = [theta]

## Optimization loop

time0 = datetime.now()
_ = value_and_grad(theta)
time1 = datetime.now()
print(f"grad and val compilation time: {time1 - time0}")

for n in range(n_epochs):
    val, grad_circuit = value_and_grad(theta)
    updates, opt_state = optimizer.update(grad_circuit, opt_state)
    theta = optax.apply_updates(theta, updates)

    energy[n] = qnode(theta)
    cost[n] = val
    theta_i.append(theta)

    if not n%5:
        print(f"{n+1} / {n_epochs}; energy: {energy[n]}; cost: {val}")

##############################################################################
# We see that the energy converges relatively quickly to the desired exact result.

fig, axs = plt.subplots(nrows=2, figsize=(5,5), sharex=True)
ax = axs[0]
ax.plot(energy, ".:", label="energy")
ax.plot([0, n_epochs], [E_exact]*2, "--", label="exact", color="grey")
ax.set_ylabel("Energy")

ax = axs[1]
ax.plot(cost,".:", label="cost")
ax.set_xlabel("epoch")
ax.set_ylabel("Cost")

plt.show()

##############################################################################
# We can also visualize the paths for the envelopes for each qubit take in time.

# this plot doesnt make sense atm due to complex valued envelopes
draw(H_pulse, theta, 15.)

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
