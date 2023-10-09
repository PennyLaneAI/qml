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

|

.. figure:: ../demonstrations/oqc_pulse/qubit_rotation.png
    :align: center
    :width: 70%
    :alt: Illustration of how single qubit rotations are realized by Z-precession and Rabi oscillation
    :target: javascript:void(0);

|

Introduction
============

tesxt


Transmon Physics
================

text

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
# .. [#Rahamim]
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
#
##############################################################################
# About the author
# ----------------
# .. include:: ../_static/authors/korbinian_kottmann.txt
