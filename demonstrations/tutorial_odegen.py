r"""Evaluating analytic gradients of pulseprograms on quantum computers
=======================================================================

.. meta::
    :property="og:description": Differentiate pulse gates on hardware
    :property="og:image": https://pennylane.ai/qml/_images/thumbnail_tutorial_pulse_on_hardware.png

.. related::
   ahs_aquila Pulse programming on Rydberg atom hardware
   tutorial_pulse_programming101 Differentiable pulse programming with qubits in PennyLane
   oqc_pulse Differentiable pulse programming on OQC Lucy

*Author: Korbinian Kottmann — Posted: 31 November 2023.*

Abstract

This is an abstract.

Introduction
============

Many contemporary quantum computers are operated by steering the qubit state through an
electromagnetic pulse. This can be modeled by means of a time-dependent Hamiltonian 

.. math:: H(\theta, t) = \sum_q f_q(\theta, t) H_q

with time-dependent, parametrized pulse envelopes :math:`f_q(\theta, t)`. A prominent example
is superconducting qubit platforms as described in :the demo on differentiable pulse programming <tutorial_pulse_programming101>`_
or :the demo about OQC's Lucy <oqc_pulse>`_. Such a drive then induces a unitary evolution :math:`U(\theta)` according
to the time-dependent Schrödinger equation.

The parameters :math:`\theta` of :math:`H(\theta, t)` determine the shape and strength of the pulse,
and can be subject to optimization in applications like the variational quantum eigensolver (VQE) [#Meitei].
Gradient based optimization on hardware is possible by utilizing the stochastic 
parameter shift (SPS) rule introduced in [#Banchi] and [#Leng]. However, this method is intrinsically stochastic
and may require a large number of shots.

In this demo, we are going to explain and showcase ODEgen [#Kottmann], a new analytic method that utilizes classical 
ordinary differential equation (ODE) solvers for computing gradients of quantum pulse programs
on hardware.


ODEgen vs. SPS
==============

We are interested in cost functions of the form

.. math:: \mathcal{L}(\theta) = \langle 0 | U(\theta)^\dagger H_\text{obj} U(\theta) | 0 \rangle

where we compute the expectation value of some objective Hamiltonian :math:`H_\text{obj}` (think molecular Hamiltonian to estimate the ground state energy of a molecule).
For simplicity, we assume a sole pulse gate :math:`U(\theta)` (we will discuss the case of multiple gates later). Further, let us assume the so-called pulse generators
:math:`H_q` in :math:`H(\theta, t)` to be Pauli words, which will make SPS rule below a bit more digestible (the general case is described in [#Kottmann]).

SPS
---

We can compute the gradient of :math:`\mathcal{L}` by means of the stochastic parameter shift rule via

.. math:: \frac{\partial}{\partial \theta_j} \mathcal{L}(\theta) = \int_0^T d\tau \sum_q \frac{\partial f_q(\theta, \tau)}{\partial \theta_j} \left(\tilde{\mathcal{L}}^+_q(\tau) - \tilde{\mathcal{L}}^-_q(\tau) \right).

The :math:`\tilde{\mathcal{L}}^\pm_q(\tau) = \langle \psi_q^\pm(\tau) | H_\text{obj} | \psi_q^\pm(\tau) \rangle` are the original expectation values with
shifted evolutions :math:`| \psi_q^\pm(\tau) \rangle = U(T, \tau) e^{-i\left(\pm\frac{\pi}{4}\right)H_q} U(\tau, 0)` (:math:`U(t_1, t_0)` indicates the evolution from time :math:`t_0` to :math:`t_1`).
In practice, the integral is approximated via Monte Carlo integration

.. math:: \frac{\partial}{\partial \theta_j} \mathcal{L}(\theta) \approx \frac{1}{N_s} \sum_{\tau \in \mathcal{U}([0, T])} \sum_q \frac{\partial f_q(\theta, \tau)}{\partial \theta_j} \left(\tilde{\mathcal{L}}^+_q(\tau) - \tilde{\mathcal{L}}^-_q(\tau) \right)

where :math:`N_s` is the number of Monte Carlo samples for the integration. The larger the number of samples, the better the approximation.

ODEgen
------

In contrast, the recently introduced ODEgen method [#Kottmann] has the advantage that it circumvents the need for Monte Carlo sampling by off-loading the complexity
introduced by the time-dynamics to a differentiable ODE solver. Let me walk you throug hthe basic steps.



Variational Quantum Algorithms
==============================

We want to put ODEgen and SPS head to head in a variational quantum algorithm with the same available quantum resources.
For this, we are going to perform the variational quantum eigensolver (VQE) on a ``HeH+`` molecular Hamiltonian taken from
the PennyLane `quantum dataset <datasets>`.

"""
from copy import copy
import pennylane as qml
import numpy as np
import jax.numpy as jnp
import jax

import optax
from datetime import datetime

jax.config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt

bond_distance = 1.78
data = qml.data.load("qchem", molname="HeH+", basis="STO-3G", bondlength=bond_distance)[0]
H_obj = data.tapered_hamiltonian
H_obj = qml.dot(jnp.array(H_obj.coeffs), H_obj.ops)
E_exact = data.fci_energy

##############################################################################
# We are going to consider a system of transmon qubits
# 
# .. math:: H(\theta, t) = - \sum_q \frac{\omega_q}{2} Z_q + \sum_q \Omega(t) \sin(\nu_q t + \phi_t) Y_q + \sum_{q, p \in \mathcal{C}} \frac{g_{qp}}{2} (X_q X_p + Y_q Y_p).
# 
# The first term describes the single qubits with frequencies :math:`\omega_q`. 
# The second term desribes the driving with drive frequencies :math:`\nu_q` and phases :math:`\phi_q`, where the latter
# may be varying in time. You can check out our :doc:`recent demo on driving qubits on OQC's luc </demos/oqc_pulse`_ if 
# you want to learn more about the details of controlling transmon qubits.
# The third term describes the coupling between neighboring qubits. We only have two qubits and a simple topology of 
# :math:`\mathcal{C} = ((0, 1))`.
# The coupling is necessary to generate entanglement, which is achieved with cross-resonant driving in fixed-coupling 
# transmon systems, as is the case here.
#


# values and parametrization from https://arxiv.org/pdf/2210.15812.pdf
# all in units of 10^9
qubit_freq = jnp.pi * 2 * np.array([5.23, 5.01])
eps = np.array([32.9, 31.5]) #10^9
max_amp = jnp.array([0.955, 0.987]) # much larger than in ctrl-vqe paper
connections = [(0, 1)]
coupling = 0.0123
wires = [0, 1]
n_wires = len(wires)
dt = 0.22
timespan = dt * 720 # 360


##############################################################################
# We are going to take the parametrization from [#Leng]_ 
# 
# .. math:: f_q(\theta, t) = \sum_{j=0}^d \theta_j \mathcal{P}\left( \frac{2t}{T} - 1 \right)
# 
# in terms of legendre polynomials :math:`\mathcal{P}` up to degree :math:`d=4`.
# For this we take their 
# `coefficients <https://en.wikipedia.org/wiki/Legendre_polynomials#Rodrigues'_formula_and_other_explicit_formulas>`_
# and create callables with trainable :math:`d+1` complex coefficients for each qubit, separated in :math:`n2(d+1)` real numbers.
#

def normalize(x):
    """Differentiable normalization to +/- 1 outputs (shifted sigmoid)"""
    return (1 - jnp.exp(-x))/(1 + jnp.exp(-x))

legendres = jnp.array([
    [0, 0, 0, 0, 1],
    [0, 0, 0, 1, 0],
    0.5*jnp.array([0, 0, 3, 0, -1]),
    0.5*jnp.array([0, 5, 0, -3, 0]),
    1/8*jnp.array([35, 0, -30, 0, 3])
])
leg_func = jax.jit(jax.vmap(jnp.polyval, [0, None]))
dLeg = len(legendres)

def amp(timespan, omega, max_amp):
    def wrapped(p, t):
        pr, pi = p[:dLeg], p[dLeg:]
        par = pr + 1j * pi
        leg_vals = leg_func(legendres, 2*t/timespan - 1)
        z = par @ leg_vals
        res = normalize(z) * jnp.angle(z)
        res = max_amp * jnp.real(jnp.exp(1j*omega*t) * res) # eq. (27)
        return res
    return wrapped

H_D = qml.dot(0.5*eps, [qml.Identity(i) - qml.PauliZ(i) for i in wires])
H_D += coupling/2 * (qml.PauliX(0) @ qml.PauliX(1) + qml.PauliY(0) @ qml.PauliY(1)) # implicit factor 2 due to redundancy in formula

fs = [amp(timespan, qubit_freq[i], max_amp[i]) for i in range(n_wires)]
ops = [qml.PauliX(i) for i in wires]

H_C = qml.dot(fs, ops)
H = H_D + H_C

##############################################################################
# We can now define the cost function that computes the expectation value of 
# the molecular Hamiltonian after evolving the state with the parametrized pulse Hamiltonian.
# As is standard in quantum chemistry, we initialize the Hartree Fock state by flipping both qubits.
# We then define the two separate qnodes with ODEgen and SPS as their differentiation methods, respectively.


atol=1e-5

dev = qml.device("default.qubit", wires=n_wires)

def circuit(params):
    # Hartree Fock state (see data.tapered_hf_state)
    qml.PauliX(0)
    qml.PauliX(1)
    # Evolve state according to parametrized pulse Hamiltonian
    qml.evolve(H, atol=atol)(params, t=timespan)
    return qml.expval(H_obj)

def f(params, tau):
    return [fs[i](params[i], tau) for i in range(len(fs))]

num_split_times = 20

cost_ps8 = qml.QNode(
    circuit,
    dev, 
    interface="jax", 
    diff_method=qml.gradients.stoch_pulse_grad, 
    num_split_times=8, 
    use_broadcasting=True
)
cost_ps20 = qml.QNode(
    circuit,
    dev, 
    interface="jax", 
    diff_method=qml.gradients.stoch_pulse_grad, 
    num_split_times=20, 
    use_broadcasting=True
)
cost_odegen = qml.QNode(
    circuit,
    dev, 
    interface="jax", 
    diff_method=qml.gradients.pulse_odegen
)

value_and_grad_sps8 = jax.value_and_grad(cost_ps8)
value_and_grad_sps20 = jax.value_and_grad(cost_ps20)
value_and_grad_odegen = jax.value_and_grad(cost_odegen)

##############################################################################
# We note that for as long as we are in simulation, there is no difference between the gradients obtained
# from direct backpropagation and using ODEgen.

theta0 = jnp.ones((n_wires, 2*dLeg))
cost_jax = qml.QNode(circuit, dev, interface="jax")
value_and_grad_jax = jax.jit(jax.value_and_grad(cost_jax))

res0, grad0 = value_and_grad_jax(theta0)
res1, grad1 = value_and_grad_odegen(theta0)
np.allclose(res0, res1), np.allclose(grad0, grad1, atol=1e-2)

##############################################################################
# This allows us to use direct backpropagation here, which is always faster in simulation.
# We now have all ingredients to run VQE with ODEgen and SPS. We define the following standard
# optimization loop using `optax`. We start the optimization from the same random initial values.

def run_opt(value_and_grad, theta, n_epochs=100, lr=0.1, b1=0.9, b2=0.999, E_exact=0., verbose=True):

    optimizer = optax.adam(learning_rate=lr, b1=b1, b2=b2)
    opt_state = optimizer.init(theta)

    energy = np.zeros(n_epochs)
    gradients = []
    thetas = []

    @jax.jit
    def partial_step(grad_circuit, opt_state, theta):
        updates, opt_state = optimizer.update(grad_circuit, opt_state)
        theta = optax.apply_updates(theta, updates)

        return opt_state, theta


    t0 = datetime.now()
    ## Optimization loop
    for n in range(n_epochs):
        # val, theta, grad_circuit, opt_state = step(theta, opt_state)
        val, grad_circuit = value_and_grad(theta)
        opt_state, theta = partial_step(grad_circuit, opt_state, theta)

        energy[n] = val
        gradients.append(
            grad_circuit
        )
        thetas.append(
            theta
        )
    t1 = datetime.now()
    if verbose:
        print(f"final loss: {val - E_exact}; min loss: {np.min(energy) - E_exact}; after {t1 - t0}")
    
    return thetas, energy

lr = 0.1
n_epochs = 150
key = jax.random.PRNGKey(0)
theta0 = jax.random.normal(key, shape=(n_wires, 2*dLeg))

thetaf_odegen, energy_odegen = run_opt(value_and_grad_jax, theta0, lr=lr, verbose=1, n_epochs=n_epochs, E_exact=E_exact)
thetaf_sps, energy_sps = run_opt(value_and_grad_sps8, theta0, lr=lr, verbose=1, n_epochs=n_epochs, E_exact=E_exact)

plt.plot(np.array(energy_sps) - E_exact)
plt.plot(np.array(energy_odegen) - E_exact)
plt.yscale("log")
plt.show()


##############################################################################
# Conclusion
# ==========
# Text. #



##############################################################################
# 
#
#
# References
# ----------
#
# .. [#Kottmann]
#
#     Korbinian Kottmann, Nathan Killoran
#     "Evaluating analytic gradients of pulse programs on quantum computers"
#     `arXiv:2309.16756 <https://arxiv.org/abs/2309.16756>`__, 2023.
#
# .. [#Krantz]
#
#     Philip Krantz, Morten Kjaergaard, Fei Yan, Terry P. Orlando, Simon Gustavsson, William D. Oliver
#     "A Quantum Engineer's Guide to Superconducting Qubits"
#     `arXiv:1904.06560 <https://arxiv.org/abs/1904.06560>`__, 2019.
#
# .. [#Meitei]
#
#     Oinam Romesh Meitei, Bryan T. Gard, George S. Barron, David P. Pappas, Sophia E. Economou, Edwin Barnes, Nicholas J. Mayhall
#     "Gate-free state preparation for fast variational quantum eigensolver simulations: ctrl-VQE"
#     `arXiv:2008.04302 <https://arxiv.org/abs/2008.04302>`__, 2019.
#
# ..  [#Banchi]
#
#     Leonardo Banchi, Gavin E. Crooks
#     "Measuring Analytic Gradients of General Quantum Evolution with the Stochastic Parameter Shift Rule"
#     `arXiv:2005.10299 <https://arxiv.org/abs/2005.10299>`__, 2020
#
# ..  [#Leng]
#
#     Jiaqi Leng, Yuxiang Peng, Yi-Ling Qiao, Ming Lin, Xiaodi Wu
#     "Differentiable Analog Quantum Computing for Optimization and Control"
#     `arXiv:2210.15812 <https://arxiv.org/abs/2210.15812>`__, 2022
#
#
##############################################################################
# About the author
# ----------------
# .. include:: ../_static/authors/korbinian_kottmann.txt
