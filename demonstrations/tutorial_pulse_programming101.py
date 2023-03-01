r"""Differentiable pulse programming with qubits in PennyLane
=============================================================

.. meta::
    :property="og:description": Simulating differentialble pulse programs in PennyLane with qubits
    :property="og:image": https://pennylane.ai/qml/_images/pulse_illustration.png

Author: Korbinian Kottmann — Posted: 20 February 2023.

In this demo we are going to introduce pulse programming with qubits in PennyLane and run the
ctrl-VQE algorithm on a two-qubit Hamiltonian for the HeH+ molecule. The overall idea is to continuously
manipulate qubits with electromagnetic pulses in time. These pulses can be optimized to achieve a task like
minimizing the expectation value of an observable.

|

.. figure:: ../demonstrations/pulse_programming101/pulse_illustration.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

|

Pulses in quantum computers
---------------------------

In many quantum computing architectures such as `superconducting <https://pennylane.ai/qml/demos/tutorial_sc_qubits.html>`_, `ion trap <https://pennylane.ai/qml/demos/tutorial_trapped_ions.html>`_ 
and `neutral atom Rydberg <https://pennylane.ai/qml/demos/tutorial_pasqal.html>`_ systems,
qubits are realized through physical systems with a discrete set of energy levels.
For example, transmon qubits realize an anharmonic oscillator whose ground and first excited states can serve as the two energy
levels of a qubit. Such a qubit can be controlled via an electromagnetic field tuned to its energy gap. In general, this
electromagnetic field can be altered in time, leading to a time-dependent Hamiltonian interaction :math:`H(t)`.
We call driving the system with such an electromagnetic field for a fixed time window a *pulse sequence*. During a pulse sequence, the state evolves according
to the time-dependent Schrödinger equation

.. math:: \frac{d}{dt}|\psi\rangle = -i H(t) |\psi\rangle

from an initial state :math:`|\psi(t_0)\rangle` to a final state :math:`|\psi(t_1)\rangle`. This process corresponds to a unitary evolution :math:`U(t_0, t_1)`
of the input state from time :math:`t_0` to :math:`t_1`, i.e. :math:`|\psi(t_1)\rangle = U(t_0, t_1) |\psi(t_0)\rangle`.

In most digital quantum computers (with the exception of `measurement-based <https://pennylane.ai/qml/demos/tutorial_mbqc.html>`_ architectures), the amplitude and frequencies of predefined pulse sequences are
fine tuned to realize the native gates of the quantum computer. More specifically, the Hamiltonian interaction :math:`H(t)`
is tuned such that the respective evolution :math:`U(t_0, t_1)` realizes for example a Pauli or CNOT gate.

Pulse programming in PennyLane
------------------------------

A user of a quantum computer typically operates on the higher and more abstract gate level.
Future fault-tolerant quantum computers require this abstraction to allow for error correction.
For noisy and intermediate sized quantum computers, the abstraction of decomposing quantum algorithms
into a fixed native gate set can be a hindrance and unnecessarily increase execution time, therefore leading
to more noise in the computation. The idea of differentiable pulse programming is to optimize quantum circuits on the pulse
level instead, with the aim of achieving the shortest interaction sequence a hardware system allows.

In PennyLane, we can simulate arbitrary qubit system interactions to explore the possibilities of such pulse programs.
First, we need to define the time-dependent Hamiltonian :math:`H(p, t)= \sum_i f_i(p_i, t) H_i` with constant operators :math:`H_i` and control fields :math:`f_i(p_i, t)` that may
depend on parameters :math:`p`. In PennyLane, we can do this in the following way:
"""

import pennylane as qml
import pennylane.numpy as np
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt

# Set to float64 precision and remove jax CPU/GPU warning
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

def f1(p, t):
    # polyval(p, t) evaluates a polynomial of degree N=len(p)
    # i.e. p[0]*t**(N-1) + p[1]*t**(N-2) + ... + p[N-2]*t + p[N-1]
    return jnp.polyval(p, t)


def f2(p, t):
    return p[0] * jnp.sin(p[1] * t)


Ht = f1 * qml.PauliX(0) + f2 * qml.PauliY(1)

##############################################################################
# Note that when constructing such a Hamiltonian, the ``callable`` functions are
# expected to have the fixed signature ``(p, t)``. The Hamiltonian on the other hand
# is passed a ``tuple`` or ``list`` of the parameters for each of the functions in the same
# order the Hamiltonian was constructed.

p1 = jnp.ones(5)              # parameters for f1
p2 = jnp.array([1.0, jnp.pi]) # parameters for f2
t = 0.5                       # some fixed point in time
print(Ht((p1, p2), t))        # order of parameters p1, p2 matters

##############################################################################
# We can construct general Hamiltonians of the form :math:`\sum_i H_i^d + \sum_i f_i(p, t) H_i`
# using :func:`qml.dot <pennylane.dot>`. Such a time-dependent Hamiltonian consists time-indepdenent drift terms :math:`H_i^d`
# and time dependent control terms :math:`f_i(p, t) H_i` with scalar complex-valued function :math:`f_i(p, t)` in time that is 
# parametrized by some parameters :math:`p`.
# In the following we are going to construct :math:`\sum_i X_i X_{i+1} + \sum_i f_i(p, t) Z_i`

coeffs = [1.0] * 2
coeffs += [lambda p, t: jnp.sin(p[0] * t) + jnp.sin(p[1] * t) for _ in range(3)]
ops = [qml.PauliX(i) @ qml.PauliX(i + 1) for i in range(2)]
ops += [qml.PauliZ(i) for i in range(3)]

Ht = qml.dot(coeffs, ops)

# random coefficients
key = jax.random.PRNGKey(777)
subkeys = jax.random.split(key, 3)
params = [jax.random.uniform(subkeys[i], shape=[2], maxval=5) for i in range(3)]
print(Ht(params, 0.5))

##############################################################################
# We can visualize the Hamiltonian interaction by plotting the time-dependent envelopes. We refer to the drift term as all constant terms in time, i.e. :math:`\sum_i X_i X_{i+1}`,
# and plot the envelopes :math:`f_i(p, t)` of the time dependent terms :math:`\sum_i f_i(p, t) Z_i`.

ts = jnp.linspace(0.0, 5.0, 100)
fs = Ht.coeffs_parametrized
ops = Ht.ops_parametrized
n_channels = len(fs)
fig, axs = plt.subplots(nrows=n_channels, figsize=(5, 2 * n_channels), gridspec_kw={"hspace": 0})
for n in range(n_channels):
    ax = axs[n]
    ax.plot(ts, fs[n](params[n], ts))
    ax.set_ylabel(f"Z_{n}")
axs[0].set_title(f"Drift term: $X_0 X_1 + X_1 X_2$")
axs[-1].set_xlabel("time t")
plt.tight_layout()
plt.show()

##############################################################################
#
# A pulse program is then executed by using the :func:`~.pennylane.evolve` transform to create the evolution
# gate :math:`U(t_0, t_1)`, which implicitly depends on the parameters ``p``.

dev = qml.device("default.qubit.jax", range(4))

ts = jnp.array([0.0, 3.0])
H_obj = sum([qml.PauliZ(i) for i in range(4)])


@jax.jit
@qml.qnode(dev, interface="jax")
def qnode(params):
    qml.evolve(Ht)(params, ts)
    return qml.expval(H_obj)


print(qnode(params))

##############################################################################
# We used the decorator ``jax.jit`` to compile this execution just-in-time. This means the first execution will typically take a little longer with the
# benefit that all following executions will be significantly faster, see the `jax docs on jitting <https://jax.readthedocs.io/en/latest/jax-101/02-jitting.html>`_. JIT-compiling is optional, and one can remove the decorator when only single executions
# are of interest.
#
# Researchers interested in more specific hardware systems can simulate them using the specific Hamiltonian interactions.
# For example, we will simulate a transmon qubit system in the ctrl-VQE example in the last section of this demo.
#
# Gradients of pulse programs
# ---------------------------
# Internally, pulse programs in PennyLane solve the time-dependent Schrödinger equation using the `Dopri5 <https://en.wikipedia.org/wiki/Dormand%E2%80%93Prince_method>`_ solver for
# ordinary differential equations (ODEs). In particular, the step sizes between :math:`t_0` and :math:`t_1` are chosen adaptively to stay within a given error tolerance.
# We can backpropagate through this ODE solver and obtain the gradient via ``jax.grad``.

print(jax.grad(qnode)(params))

##############################################################################
# Alternatively, one could consider computing the gradient with the parameter shift rule [#Leng2022]_, which is particularly interesting for
# real hardware execution. In classical simulations, however, backpropagation is recommended.


##############################################################################
# Piece-wise-constant parametrizations
# ------------------------------------
# PennyLane also provides a variety of convenience functions to create, for example, piece-wise-constant parametrizations
# defining the function values at fixed time bins as parameters. We can construct such a callable with :func:`~pennylane.pulse.pwc`
# by providing a ``timespan`` argument which is expected to be either a total time (``float``) or a start and end time (``tuple``).

timespan = 10.0
coeffs = [qml.pulse.pwc(timespan) for _ in range(2)]

##############################################################################
# This creates a callable with signature ``(p, t)`` that returns ``p[int(len(p)*t/duration)]``, such that the passed parameters are the function values
# for different time bins.

key = jax.random.PRNGKey(777)
subkeys = jax.random.split(key, 2)
theta = [jax.random.uniform(subkeys[i], shape=[shape], maxval=5) for i, shape in enumerate([4, 10])]

ts = jnp.linspace(0.0, 10.0, 100)[:-1]
fig, axs = plt.subplots(nrows=2, sharex=True)
for i in range(2):
    ax = axs[i]
    ax.plot(ts, coeffs[i](theta[i], ts), ".-")
    ax.set_ylabel(f"coeffs[{i}]")
ax.set_xlabel("time t")
plt.show()

##############################################################################
# Note how the number of time bins is implicitly defined through the length of the parameters,
# which we chose to be ``4`` and ``10`` in the example above, respectively.
# We can use these callables as before to construct a :func:`~.pennylane.pulse.ParametrizedHamiltonian`.

ops = [qml.PauliX(i) for i in range(2)]
H = qml.pulse.ParametrizedHamiltonian(coeffs, ops)
print(H(theta, 0.5))

##############################################################################
# Variational quantum eigensolver with pulse programming
# ------------------------------------------------------
# We can now use the ability to access gradients to perform the variational quantum eigensolver on the pulse level (ctrl-VQE) as is done in [#Mitei]_.
# For a more general introduction to VQE see :doc:`tutorial_vqe`.
# First, we define the molecular Hamiltonian whose energy estimate we want to minimize.
# We are going to look at :math:`\text{HeH}^+` as a simple example and load it from the `PennyLane quantum datasets <https://pennylane.ai/qml/datasets.html>`_ website.
# We are going to use the tapered Hamiltonian which makes use of symmetries to reduce the number of qubits, see :doc:`tutorial_qubit_tapering` for details.

data = qml.data.load("qchem", molname="HeH+", basis="STO-3G", bondlength=1.5)[0]
H_obj = data.tapered_hamiltonian
H_obj = qml.Hamiltonian(jnp.array(H_obj.coeffs), H_obj.ops)
E_exact = data.fci_energy
n_wires = len(H_obj.wires)

##############################################################################
# As a realistic physical system with pulse level control, we are considering a coupled transmon qubit system with the constant drift term Hamiltonian
#
# .. math:: H_D = \sum_q \omega_q a_q^\dagger a_q - \sum_q \frac{\delta_q}{2} a^\dagger_q a^\dagger_q a_q a_q + \sum_{\braket{pq}} g_{pq} a^\dagger_p a_q
#
# with bosonic creation and annihilation operators. The anharmonicity :math:`\delta_q` is describing the contribution to higher energy levels.
# We are only going to consider the qubit subspace and hence set this term to zero.
# The order of magnitude of the resonance frequencies :math:`\omega_q` and coupling strength :math:`g_{pq}` are taken from [#Mitei]_ (in GHz).


def a(wires):
    return 0.5 * qml.PauliX(wires) + 0.5j * qml.PauliY(wires)


def ad(wires):
    return 0.5 * qml.PauliX(wires) - 0.5j * qml.PauliY(wires)


omega = 2 * jnp.pi * jnp.array([4.8080, 4.8333])
g = 2 * jnp.pi * jnp.array([0.01831, 0.02131])

H_D = qml.dot(omega, [ad(i) @ a(i) for i in range(n_wires)])
H_D += qml.dot(
    g, [ad(i) @ a((i + 1) % n_wires) + ad((i + 1) % n_wires) @ a(i) for i in range(n_wires)]
)

##############################################################################
# The system is driven under the control term
#
# .. math:: H_C(t) = \sum_q \Omega_q(t) \left(e^{i\nu_q t} a_q + e^{-i\nu_q t} a^\dagger_q \right)
#
# with the (real) time-dependent amplitudes :math:`\Omega_q(t)` and frequencies :math:`\nu_q` of the drive.
# We let :math:`\Omega(t)` be a piece-wise-constant real function whose values are optimized.
# In principle one can also optimize the drive frequency :math:`\nu_q`, but we already find good results by setting it to the qubit frequencies.
# We restrict the amplitude to :math:`\pm 20 \text{MHz}` to abide by realistic hardware constraints.


def drive_field(T, omega, sign=1.0):
    def wrapped(p, t):
        amp = jnp.clip(qml.pulse.pwc(T)(p, t), -0.02, 0.02)
        phase = jnp.exp(sign * 1j * omega * t)
        return amp * phase

    return wrapped


duration = 15.0

fs = [drive_field(duration, omega[i], 1.0) for i in range(n_wires)]
fs += [drive_field(duration, omega[i], -1.0) for i in range(n_wires)]
ops = [a(i) for i in range(n_wires)]
ops += [ad(i) for i in range(n_wires)]

H_C = qml.dot(fs, ops)

##############################################################################
# Overall, we end up with the time-dependent parametrized Hamiltonian :math:`H(p, t) = H_D + H_C(p, t)`
# under which the system is evolved for the given time window of ``15ns``. Note that we are expressing time
# in nanoseconds (:math:`10^{-9}` s) and frequencies in gigahertz (:math:`10^{9}` Hz), such that both
# exponents cancel.

H_pulse = H_D + H_C

##############################################################################
# Now we define the ``qnode`` that computes the expectation value of the molecular Hamiltonian.

dev = qml.device("default.qubit.jax", wires=range(n_wires))

@qml.qnode(dev, interface="jax")
def qnode(theta, t=duration):
    qml.BasisState(list(data.tapered_hf_state), wires=H_obj.wires)
    qml.evolve(H_pulse)(params=(*theta, *theta), t=t)
    return qml.expval(H_obj)


##############################################################################
# We now have all the ingredients to run our ctrl-VQE program. We use the ``adabelief`` implementation in `optax <https://optax.readthedocs.io/en/latest/>`_, a package for optimizations in ``jax``.
# The success of the optimization is sensitive to the initial values of the parameters.
# We showcase a good random seed here. In reality, this optimization easily gets stuck in local minima
# such that we would have to repeat the experiment multiple times with different random initializations.
# On the other hand, the Hartree-Fock state is usually a good starting point, i.e. choosing all parameters to zero.
# However, this results in a near-zero gradient in our case. This is why we choose a trade-off by reducing
# the initial amplitude of the random values.
#
# Further, we note that with the increase in the number of parameters due to the continuous evolution, the optimization
# becomes harder. In particular, besides the random initialization, the optimization is also very sensitive to the choice of
# optimizer and learning rate. We systematically tried a variety of combinations and provide one possible choice leading to good results.
#
# We choose ``t_bins = 100`` segments for the piece-wise-constant parametrization of the pulses.

t_bins = 100  # number of time bins

key = jax.random.PRNGKey(999)
theta = 0.01 * jax.random.uniform(key, shape=jnp.array([n_wires, t_bins]))

import optax
from datetime import datetime

n_epochs = 100
optimizer = optax.adabelief(learning_rate=1e-2)
opt_state = optimizer.init(theta)

value_and_grad = jax.jit(jax.value_and_grad(qnode))

energy = np.zeros(n_epochs + 1)
energy[0] = qnode(theta)
gradients = np.zeros(n_epochs)

## Compile the evaluation and gradient function and report compilation time
time0 = datetime.now()
_ = value_and_grad(theta)
time1 = datetime.now()
print(f"grad and val compilation time: {time1 - time0}")

## Optimization loop
for n in range(n_epochs):
    val, grad_circuit = value_and_grad(theta)
    updates, opt_state = optimizer.update(grad_circuit, opt_state)
    theta = optax.apply_updates(theta, updates)

    energy[n + 1] = val
    gradients[n] = np.mean(np.abs(grad_circuit))

    if not n % 10:
        print(f"{n+1} / {n_epochs}; energy discrepancy: {val-E_exact}")
        print(f"mean grad: {gradients[n]}")

##############################################################################
# We see that we have converged well within chemical accuracy after half the number of epochs.

fig, ax = plt.subplots(nrows=1, figsize=(5, 3), sharex=True)

y = np.array(energy) - E_exact
ax.plot(y, ".:", label="$\\langle H_{{obj}}\\rangle - E_{{FCI}}$")
ax.fill_between([0, len(y)], [1e-3] * 2, 5e-4, alpha=0.2, label="chem acc.")
ax.set_yscale("log")
ax.set_ylabel("Energy ($E_H$)")
ax.set_xlabel("epoch")
ax.legend()

plt.tight_layout()
plt.show()

##############################################################################
# We can also visualize the envelopes for each qubit in time.
# We only plot the real amplitude :math:`\Omega(t)` without the qubit frequency modulation.
# Note that we obtain bang-bang like solutions as indicated in [#Asthana2022]_, making it
# likely we are close to the minimal evolution time with ``15ns``.


fs = H_pulse.coeffs_parametrized[:n_wires]
n_channels = len(fs)
ts = jnp.linspace(0, duration, t_bins)
fig, axs = plt.subplots(nrows=n_channels, figsize=(5, 2 * n_channels), sharex=True)
for n in range(n_channels):
    ax = axs[n]
    label = f"$\\nu_{n}$: {omega[n]/2/jnp.pi:.3}/$2\\pi$"
    ax.plot(ts, np.clip(theta[n], -0.02, 0.02), ".:", label=label)
    ax.set_ylabel(f"$amp_{n}$ (GHz)")
    ax.legend()
ax.set_xlabel("t (ns)")

plt.tight_layout()
plt.show()

##############################################################################
#
# Conclusion
# ----------
# Pulse programming is an exciting new field within noisy quantum computing. By skipping the digital abstraction, one can
# write variational programs on the hardware level, potentially minimizing the computation time. This then ideally allows for effectively deeper
# circuits on noisy hardware.
# On the other hand, the possibility to continuously vary the Hamiltonian interaction in time significantly increases
# the parameter space. A good parametrization trading off flexibility and the number of parameters is therefore necessary as systems scale up.
# Further, the increased flexibility also affects the search space in Hilbert space that pulse gates can reach.
# Barren plateaus in variational quantum algorithms are typically due to a lack of a good inductive bias in the ansatz, i.e. having a search space that is too large.
# It is therefore crucial to find physically motivated ansätze for pulse programs.
#
# References
# ----------
#
# .. [#Leng2022]
#
#     Jiaqi Leng, Yuxiang Peng, Yi-Ling Qiao, Ming Lin, Xiaodi Wu
#     "Differentiable Analog Quantum Computing for Optimization and Control"
#     `arXiv:2210.15812 <https://arxiv.org/abs/2210.15812>`__, 2022
#
# .. [#Mitei]
#
#     Oinam Romesh Meitei, Bryan T. Gard, George S. Barron, David P. Pappas, Sophia E. Economou, Edwin Barnes, Nicholas J. Mayhall
#     "Gate-free state preparation for fast variational quantum eigensolver simulations: ctrl-VQE"
#     `arXiv:2008.04302 <https://arxiv.org/abs/2008.04302>`__, 2020
#
# .. [#Asthana2022]
#
#     Ayush Asthana, Chenxu Liu, Oinam Romesh Meitei, Sophia E. Economou, Edwin Barnes, Nicholas J. Mayhall
#     "Minimizing state preparation times in pulse-level variational molecular simulations."
#     `arXiv:2203.06818 <https://arxiv.org/abs/2203.06818>`__, 2022.
#
#
#


##############################################################################
# About the author
# ----------------
# .. include:: ../_static/authors/korbinian_kottmann.txt
