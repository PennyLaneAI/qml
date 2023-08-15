r"""Differentiable pulse programming on OQC Lucy
================================================

.. meta::
    :property="og:description": Perform differentiable pulse gates on superconducting qubit hardware through PennyLane
    :property="og:image": https://pennylane.ai/qml/_images/thumbnail_tutorial_pulse_on_hardware.png

.. related::
   ahs_aquila Pulse programming on Rydberg atom hardware
   tutorial_pulse_programming101 Differentiable pulse programming with qubits in PennyLane

*Author: Korbinian Kottmann — Posted: 05 September 2023.*

Abstract

|

.. figure:: ../demonstrations/ahs_aquila/aquila_demo_image.png
    :align: center
    :width: 70%
    :alt: Illustration of robotic hand controlling Rubidium atoms with electromagnetic pulses
    :target: javascript:void(0);

|


Heading0
-------------------------------------

text

.. figure:: ../demonstrations/ahs_aquila/rydberg_blockade_diagram.png
    :align: center
    :figwidth: 55%
    :width: 95%
    :alt: A diagram of the energy levels for the ground, single excitation, and double excitation states
    :target: javascript:void(0);

    Figure caption

"""

import pennylane as qml
import numpy as np
import jax.numpy as jnp
import jax

from scipy.optimize import curve_fit

import matplotlib.pyplot as plt

wire = 1
dev_sim = qml.device("default.qubit.jax", wires=[wire])
dev_lucy = qml.device("braket.aws.qubit",
    device_arn="arn:aws:braket:eu-west-2::device/qpu/oqc/Lucy",
    wires=range(8), 
    shots=1000,
    parallel=True
)

qubit_freq = dev_lucy.pulse_settings["qubit_freq"][wire]

T = 40      # pulse duration
amp = 0.6   # pulse amplitude
phi0 = 0.   # pulse phase

H0 = qml.pulse.transmon_interaction(
    qubit_freq = [qubit_freq],
    connections = [],
    coupling = [],
    wires = [wire]
)
Hd0 = qml.pulse.transmon_drive(qml.pulse.constant, qml.pulse.constant, qubit_freq, wires=[wire])
H = H0 + Hd0

def qnode0(params, duration):
    qml.evolve(H)(params, t=duration, atol=1e-12)
    return qml.expval(qml.PauliZ(wire))

qnode_sim = jax.jit(qml.QNode(qnode0, dev_sim, interface="jax"))
qnode_lucy = qml.QNode(qnode0, dev_lucy, interface="jax")

def fit_sinus(x, y, initial_guess=[1., 0.1, 1]):
    """[A, omega, phi]"""
    x_fit = np.linspace(np.min(x), np.max(x), 500)

    # Define the function to fit (sinusoidal)
    def sinusoidal_func(x, A, omega, phi):
        return A * np.sin(omega * x + phi)

    # Perform the curve fit
    params, _ = curve_fit(sinusoidal_func, np.array(x), np.array(y), maxfev = 10000, p0=initial_guess)

    # Generate the fitted curve
    y_fit = sinusoidal_func(x_fit, *params)
    return x_fit, y_fit, params

# t0, t1, num_ts = 10., 25., 20
# x_lucy = jnp.linspace(t0, t1, num_ts)
# name = f"data/calibration_duration_-{t0}-{t1}-{num_ts}_phi-{phi0}-amp-{amp}-allwires"
# params = jnp.array([amp, phi0])

# y_lucy = [qnode_lucy(params, t) for t in ts]

# np.savez(name, x=x_lucy, y=y_lucy)

dat = np.load("1-qubit-vqe/data/calibration_duration_-10.0-25.0-20_phi-0.0-amp-0.6-allwires.npz", allow_pickle=True) #'data/calibration_duration_-10.0-25.0-20_phi-0.0-amp-0.6-allwires'
x_lucy, y_lucy = dat["ts"], dat["calibration"]

x_lucy_fit, y_lucy_fit, coeffs_fit_lucy = fit_sinus(x_lucy, y_lucy, [1., 0.1, 1])


plt.plot(x_lucy, y_lucy, "x:", label="data")
plt.plot(x_lucy_fit, y_lucy_fit, "-", color="tab:blue", label=f"{coeffs_fit_lucy[0]:.3f} sin({coeffs_fit_lucy[1]:.3f} t + {coeffs_fit_lucy[2]:.3f})", alpha=0.4)

params_sim = jnp.array([amp, phi0])
x_sim = jnp.linspace(10., 15., 50)
y_sim = jax.vmap(qnode_sim, (None, 0))(params_sim, x_sim)
x_fit, y_fit, coeffs_fit_sim = fit_sinus(x_sim, y_sim, [1., 3.7, 2.])

plt.plot(x_sim, y_sim, "x-", label="sim")
plt.plot(x_fit, y_fit, "-", color="tab:orange", label=f"{coeffs_fit_sim[0]:.3f} sin({coeffs_fit_sim[1]:.3f} t + {coeffs_fit_sim[2]:.3f})", alpha=0.4)
plt.legend()
plt.ylabel("<Z>")
plt.xlabel("t1")

##############################################################################
# .. figure:: ../demonstrations/oqc_pulse/calibration0.png
#     :align: center
#     :width: 40%
#     :alt: The layout of the 3 atoms defined by `coordinates`
#     :target: javascript:void(0);

attenuation = coeffs_fit_lucy[1] / coeffs_fit_sim[1]


##############################################################################
# .. rst-class:: sphx-glr-script-out
#
#  Out:
#
#  .. code-block:: none
#
#      coordinates: [(0, 0), (5, 0), (2.5, 4.330127018922194)]


##############################################################################
# .. rst-class:: sphx-glr-script-out
#
#  Out:
#
#  .. code-block:: none
#
#      coordinates: [(0, 0), (5, 0), (2.5, 4.330127018922194)]

##############################################################################
# .. figure:: ../demonstrations/ahs_aquila/rydberg_blockade_coordinates.png
#     :align: center
#     :width: 40%
#     :alt: The layout of the 3 atoms defined by `coordinates`
#     :target: javascript:void(0);

##############################################################################
# .. figure:: ../demonstrations/ahs_aquila/rydberg_blockade_coordinates.png
#     :align: center
#     :width: 40%
#     :alt: The layout of the 3 atoms defined by `coordinates`
#     :target: javascript:void(0);

######################################################################
# Conclusion
# ----------
#
# conclusion
#
#
# References
# ----------
#
# .. [#Semeghini]
#
#     G. Semeghini, H. Levine, A. Keesling, S. Ebadi, T.T. Wang, D. Bluvstein, R. Verresen, H. Pichler,
#     M. Kalinowski, R. Samajdar, A. Omran, S. Sachdev, A. Vishwanath, M. Greiner, V. Vuletic, M.D. Lukin
#     "Probing topological spin liquids on a programmable quantum simulator"
#     `arxiv.2104.04119 <https://arxiv.org/abs/2104.04119>`__, 2021.
#
# .. [#Lienhard]
#
#     V. Lienhard, S. de Léséleuc, D. Barredo, T. Lahaye, A. Browaeys, M. Schuler, L.-P. Henry, A.M. Läuchli
#     "Observing the Space- and Time-Dependent Growth of Correlations in Dynamically Tuned Synthetic Ising
#     Models with Antiferromagnetic Interactions"
#     `arxiv.2104.04119 <https://arxiv.org/abs/1711.01185>`__, 2018.
#
# .. [#BraketDevGuide]
#
#     Amazon Web Services: Amazon Braket
#     "Hello AHS: Run your first Analog Hamiltonian Simulation"
#     `AWS Developer Guide <https://docs.aws.amazon.com/braket/latest/developerguide/braket-get-started-hello-ahs.html>`__
#
# .. [#Asthana2022]
#
#     Alexander Keesling, Eric Kessler, and Peter Komar
#     "AWS Quantum Technologies Blog: Realizing quantum spin liquid phase on an analog Hamiltonian Rydberg simulator"
#     `Amazon Quantum Computing Blog <https://aws.amazon.com/blogs/quantum-computing/realizing-quantum-spin-liquid-phase-on-an-analog-hamiltonian-rydberg-simulator/>`__, 2021.

##############################################################################
# About the author
# ----------------
# .. include:: ../_static/authors/lillian_frederiksen.txt
