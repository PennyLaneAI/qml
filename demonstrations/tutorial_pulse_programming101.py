r"""Differentiable pulse programming with qubits in PennyLane
=============================================================

.. meta::
    :property="og:description": Simulating differentialble pulse programs in PennyLane with qubits
    :property="og:image": https://pennylane.ai/qml/_images/pauli_shadows.jpg

*Author: Korbinian Kottmann — Posted: 20 February 2023.

In this demo we are going to introduce pulse operations in PennyLane and run the
ctrl-VQE algorithm for an example molecule.

Pulses in quantum computers
---------------------------

In many quantum computing architectures such as superconducting and ion trap systems, qubits are realized through physical systems with a discrete set of energy levels.
For example, transmon qubits realize an anharmonic oscillator whose ground and first excited states can serve as the two energy
levels of a qubit. Such a qubit can be controlled via an electromagnetic field tuned to its energy gap. In general, this
electromangnetic field can be altered in time, leading to a time-dependent Hamiltonian interaction :math:`H(t)`.
We call driving the system with such an electromagnetic field for a fixed time window a *pulse sequence*. During a pulse sequence, the state evolves according
to the time-dependent Schrödinger equation

.. math:: \frac{d}{dt}|\psi\rangle = -i H(t) |\psi\rangle

following a unitary evolution :math:`U(t_0, t_1)` of the input state from time :math:`t_0` to :math:`t_1`, i.e. 
:math:`|\psi(t_1)\rangle = U(t_0, t_1) |\psi(t_0)\rangle`.

In non-measurement-based digital quantum computers, the amplitude and frequencies of predefined pulse sequences are
fine tuned to realize the native gates of the quantum computer. More specifically, the Hamiltonian interaction :math:`H(t)`
is tuned such that the respective evolution :math:`U(t_0, t_1)` realizes for example a Pauli or CNOT gate.

Pulse programming in PennyLane
------------------------------

A user of a quantum computer typically operates on the higher and more abstract gate level.
Future fault-tolerant quantum computers require this abstraction to allow for error correction.
For noisy and intermediate sized quantum computers, the abstraction of decomposing quantum algorithms
into a fixed native gate set can be a hindrance and unnecessarily increase execution time, therefore leading
to more noise in the computation. The idea of differentiable pulse programming is to optimize quantum circuits on the pulse
level with the aim of achieving the shortest interaction sequence a hardware system allows.

In PennyLane, we can simulate arbitrary qubit system interactions to explore the possibilities of such pulse programs.
First, we need to define the time-dependent Hamiltonian :math:`H(p, t)= \sum_i f_i(p, t) H_i` with constant operators :math:`H_i` and driving fields :math:`f_i(p, t)` that may
depend on parameters :math:`p`. In PennyLane, we can do this in an intuitive way:
"""

import pennylane as qml
import pennylane.numpy as np
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt

# remove jax CPU/GPU warning
jax.config.update('jax_platform_name', 'cpu')

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
# We can construct more complicated Hamiltonians like :math:`\sum_i X_i X_{i+1} + \sum_i f_i(p, t) Z_i` using :func:`qml.ops.dot <pennylane.ops.dot>`.
# We use two sinusodials with random frequencies as the time-dependent parametrization for each :math:`Z_i`.

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

ts = jnp.linspace(0., 5., 100)
fs = Ht.coeffs_parametrized
ops = Ht.ops_parametrized
n_channels = len(fs)
fig, axs = plt.subplots(nrows=n_channels, figsize=(5,2*n_channels))
for n in range(n_channels):
    ax = axs[n]
    ax.plot(ts, fs[n](params[n], ts))
    ax.set_ylabel(f"Z_{n}")
axs[0].set_title(f"Drift term: X_0 X_1 + X_1 X_2")
plt.show()

##############################################################################
# PennyLane also provides a variety of convenience functions to enable for example piece-wise-constant parametrizations,
# i.e. defining the function values at fixed time bins as parameters.

# PWC example

##############################################################################
# Researchers interested in more specific hardware systems can simulate them using the specific Hamiltonian interactions.
# For example, we will simulate a transmon qubit system in the ctrl-VQE example in the last section of this demo.
#
# A pulse program is then executed by using the :func:`~.pennylane.ops.evolve` transform to create the evolution
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
# We used the decorator ``jax.jit`` to compile this execution just-in-time. This means the first execution will typically take a little longer with the
# benefit that all following executions will be significantly faster. JIT-compiling is optional, and one can remove the decorator when only single executions
# are of interest.
#
# Gradients of pulse programs
# ---------------------------
# Internally, this program solves the the time-dependent Schrödinger equation using the `Dopri5 <https://en.wikipedia.org/wiki/Dormand%E2%80%93Prince_method>`_ solver for
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
# First, we define the molecular Hamiltonian whose energy estimate we want to minimize. 
# We are going to look at :math:`H_3^+` as a simple example and load it from the `PennyLane quantum datasets <https://pennylane.ai/qml/datasets.html>`_ website.
# 

# symbols = ["H", "H", "H"]
# coordinates = np.array([-0.0399, -0.0038, 0.0, 1.5780, 0.8540, 0.0, 2.7909, -0.5159, 0.0])

# basis_set = "sto-3g"
# H, n_wires = qml.qchem.molecular_hamiltonian(
#     symbols,
#     coordinates,
#     basis=basis_set,
#     method='pyscf',
#     mult=2
# )

# coeffs, obs = H.coeffs, H.ops
# H_obj = qml.Hamiltonian(jnp.array(coeffs), obs)

data = qml.data.load("qchem", molname="HeH+", basis="STO-3G", bondlength=0.82)[0]
H_obj = data.tapered_hamiltonian
n_wires = len(H_obj.wires)

##############################################################################
# For such small systems, we can of course compute the exact ground state energy.
# We will later use this to determine if our ctrl-VQE algorithm was successful.

H_obj_m = qml.matrix(H_obj)
E_exact = data.fci_energy #np.min(np.linalg.eigvalsh(H_obj_m))

##############################################################################
# As a realistic physical system to simulate, we are considering a coupled transmon qubit system with the constant drift term Hamiltonian 
#
# .. math:: H_D = \sum_q \omega_q a_q^\dagger a_q + \sum_q \frac{\delta_q}{2} a^\dagger_q a^\dagger_q a_q a_q + \sum_{\braket{pq}} g_{pq} a^\dagger_p a_q
# 
# with bosonic creation and annihilation operators. The quadratic part propotional to :math:`\delta_q` is describing the anharmonic contribution to higher energy levels.
# We are only going to consider the qubit subspace such that this term is zero.
# The order of magnitude of the resonance frequencies :math:`\omega_q` and coupling strength :math:`g_{pq}` are taken from [#Asthana2022]_ (in GHz).

def a(wires):
    return 0.5*qml.PauliX(wires) + 0.5j* qml.PauliY(wires)
def ad(wires):
    return 0.5*qml.PauliX(wires) - 0.5j* qml.PauliY(wires)

omega = jnp.array([4.8080, 4.8333])
g = jnp.array([0.01831, 0.02131])

H_D = qml.ops.dot(omega, [ad(i) @ a(i) for i in range(n_wires)])
H_D += qml.ops.dot(g, [ad(i) @ a((i+1)%n_wires) + ad((i+1)%n_wires) @ a(i) for i in range(n_wires)])

##############################################################################
# The system is driven under the control term
#
# .. math:: H_C(t) = \sum_q \Omega_q(t) \left(e^{i\nu_q t} a_q + e^{-i\nu_q t} a^\dagger_q \right)
# 
# with the (real) time-dependent amplitudes :math:`\Omega_q(t)` and frequencies :math:`\nu_q` of the drive.
# We let :math:`\Omega(t)` be a piece-wise-constant real function that is optimized alongside the frequencies :math:`\nu_q`.
# Further, the amplitude of :math:`\Omega(t)` is restricted to :math:`20` MHz.

# TODO use official convenience functions once merged
def pwc(T):
    def wrapped(params, t):
        N = len(params)
        idx = jnp.array(N/(T) * t, dtype=int) # corresponding sample
        return params[idx]

    return wrapped

def normalize(z):
    """eq. (8) in https://arxiv.org/pdf/2210.15812.pdf"""
    def S(x):
        return (1-jnp.exp(-x))/(1+jnp.exp(-x))
    absz = jnp.abs(z)
    argz = jnp.angle(z)
    return S(absz) * jnp.exp(1j*argz)

def drive_field(T, omega, sign=1.):
    # assuming len(p) = len(t_bins) + 1 for the frequency nu
    def wrapped(p, t):
        #amp = jnp.clip(pwc(T)(p[:-1], t), -0.02, 0.02)
        amp = pwc(T)(p[:-1], t)
        phase = jnp.exp(sign*1j*p[-1]*t)
        return amp * phase

    return wrapped

duration = 15.

fs = [drive_field(duration, omega[i], 1.) for i in range(n_wires)]
fs += [drive_field(duration, omega[i], -1.) for i in range(n_wires)]
ops = [a(i) for i in range(n_wires)]
ops += [ad(i) for i in range(n_wires)]

H_C = qml.ops.dot(fs, ops)

##############################################################################
# Overall, we end up with the time-dependent parametrized Hamiltonian :math:`H(p, t) = H_D + H_C(p, t)`
# under which the system is evolved for the given time window.

H_pulse = H_D + H_C

dev = qml.device("default.qubit", wires=range(n_wires))

t_bins = 100 # number of time bins
key = jax.random.PRNGKey(666)
theta = 0.02*jax.random.uniform(key, shape=jnp.array([n_wires, t_bins + 1]))

# KK meta comment:
# The step sizes are chosen adaptively, so there is in principle no need to provide 
# explicit time steps. However, because the pwc function can be discontinuous it makes
# sense to force the solver to evaluate the points of the evolution.
# The error is still guaranteed to stay within the tolerance by using adaptive steps in between.
ts = jnp.linspace(0., duration, t_bins)

@qml.qnode(dev, interface="jax")
def qnode(theta, t=ts):
    qml.BasisState(data.tapered_hf_state, wires=H_obj.wires)
    qml.evolve(H_pulse)(params=(*theta, *theta), t=t)
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

def cost_fn(params):
    # params.shape = (n_wires, {n_bins}+1)
    p = params[:, :-1]
    C_exp = qnode(params)           # expectation value
    C_par = jnp.mean(jnp.abs(p)**2) # parameter values
    C_der = abs_diff(p)             # derivative values
    return C_exp + 3*C_par + 3*C_der

##############################################################################
# We now have all the ingredients to run our ctrl-VQE program. We use the adam implementation in ``optax``, a package for optimizations in ``jax``.
import optax 
from datetime import datetime

n_epochs = 50
optimizer = optax.adabelief(learning_rate=0.2) #adabelief
opt_state = optimizer.init(theta)

value_and_grad = jax.jit(jax.value_and_grad(cost_fn))

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
        print(f"mean grad: {jnp.mean(grad_circuit)}")
        print(f"{n+1} / {n_epochs}; energy: {energy[n]}; cost: {val}")

##############################################################################
# We see that the energy converges relatively quickly to the desired exact result.

fig, axs = plt.subplots(nrows=2, figsize=(5,5), sharex=True)
ax = axs[0]
ax.plot(energy, ".:", label="energy")
ax.plot([0, n_epochs], [E_exact]*2, ":", label="exact diag", color="grey")
ax.set_ylabel("Energy")
ax.legend()

ax = axs[1]
ax.plot(cost,".:", label="cost")
ax.set_xlabel("epoch")
ax.set_ylabel("Cost")

plt.tight_layout()
plt.show()

##############################################################################
# We can also visualize the envelopes for each qubit in time. 
# Because the field is complex valued with :math:`\Omega(t) e^{-i\nu_q t}` we plot only
# :math:`\Omega(t)` and indicate the numerical value of :math:`\nu_q`.


fs = H_pulse.coeffs_parametrized[:n_wires]
n_channels = len(fs)
fig, axs = plt.subplots(nrows=n_channels, figsize=(5,2*n_channels))
for n in range(n_channels):
    ax = axs[n]
    amp = fs[n](theta[n], ts)
    ax.plot(ts, jnp.sign(amp) * jnp.abs(amp), ".:", label=f"$\\nu$_{n}: {jnp.angle(fs[n](theta[n], 1.))/jnp.pi:.3}/$\\pi$")
    ax.set_ylabel(f"amplitude_{n}")
    ax.legend()

plt.tight_layout()
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
