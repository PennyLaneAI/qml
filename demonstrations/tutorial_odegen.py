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





"""
import pennylane as qml
import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt

X, Y, Z = qml.PauliX(0), qml.PauliY(0), qml.PauliZ(0)

omega = 2 * jnp.pi * 5.

def amp(nu):
    def wrapped(p, t):
        return jnp.pi * jnp.sin(nu*t + p)
    return wrapped

H = -omega/2 * qml.PauliZ(0)
H += amp(omega) * qml.PauliY(0)

@jax.jit
@qml.qnode(qml.device("default.qubit", wires=1), interface="jax")
def trajectory(params, t):
    qml.evolve(H)((params,), t, return_intermediate=True)
    return [qml.expval(op) for op in [X, Y, Z]]

ts = jnp.linspace(0., 1., 10000)
res0 = trajectory(0., ts)
res1 = trajectory(jnp.pi/2, ts)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot(*res0, "-", label="$\\phi=0$")
ax.plot(*res1, "-", label="$\\phi=\\pi/2$")
ax.legend()

##############################################################################
# .. figure:: ../demonstrations/oqc_pulse/qubit_rotation.png
#     :align: center
#     :width: 70%
#     :alt: Single qubit rotations with different phases leading to different effective rotation axes
#     :target: javascript:void(0);


##############################################################################
# text #




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
