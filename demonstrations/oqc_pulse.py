r"""Differentiable pulse programming on OQC Lucy
================================================

.. meta::
    :property="og:description": Perform differentiable pulse gates on superconducting qubit hardware through PennyLane
    :property="og:image": https://pennylane.ai/qml/_images/thumbnail_tutorial_pulse_on_hardware.png

.. related::
   ahs_aquila Pulse programming on Rydberg atom hardware
   tutorial_pulse_programming101 Differentiable pulse programming with qubits in PennyLane

*Author: Korbinian Kottmann — Posted: 05 October 2023.*

Abstract

|

.. figure:: ../demonstrations/ahs_aquila/aquila_demo_image.png
    :align: center
    :width: 70%
    :alt: Illustration of robotic hand controlling Rubidium atoms with electromagnetic pulses
    :target: javascript:void(0);

|


Transmon Physics
----------------

Oxford Quantum Circuit's Lucy is a quantum computer with 8 superconducting transmon qubits based on the coaxmon design #Rahamim.
In order to control a transmon qubit, it is driven by an electromagnetic microwave pulse. This can be modeled by the Hamiltonian

$$ H(t) = - \frac{\omega_q}{2} Z_q + \Omega(t) \sin(\nu_q t + \phi) Y_q $$

of the driven qubit with qubit frequency $\omega_q$, drive amplitude $\Omega(t)$, drive frequency $\nu_q$ and phase $\phi$.
The first term leads to a constant precession around the Z axis on the Bloch sphere, whereas the second term introduces
the so-called Rabi oscillations between $|0\rangle$ and $|1\langle$. This can be seen by the following simple simulation,
where we evolve the state in the Bloch sphere from $|0\rangle$ with a constant pulse of $\Omega(t) = 2 \pi \text{GHz}$
for $1\text{ns}$.
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
#     :width: 40%
#     :alt: Single qubit rotations with different phases leading to different effective rotation axes
#     :target: javascript:void(0);


##############################################################################
# We can see that for a fixed time, we land on a different longitude of the Bloch sphere. 
# We can therefore control the rotation axis of the logical gate by setting the phase $\phi$
# of the drive. Another way of seeing this is by fixing the pulse duration and looking at the
# final state for different amplitudes.

def amp(nu):
    def wrapped(p, t):
        return p[0] * jnp.sin(nu*t + p[1])
    return wrapped

H1 = -omega/2 * qml.PauliZ(0)
H1 += amp(omega) * Y

@jax.jit
@qml.qnode(qml.device("default.qubit", wires=1), interface="jax")
def trajectory(Omega0, phi):
    qml.evolve(H1)([[Omega0, phi]], 20.)
    return [qml.expval(op) for op in [X, Y, Z]]

phi = 0
Omegas = jnp.linspace(0., 1., 10000)
res0 = jax.vmap(trajectory, [0, None])(Omegas, phi)
res1 = jax.vmap(trajectory, [0, None])(Omegas, jnp.pi/2)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot(*res0, "-", label="$\\Phi=0$")
ax.plot(*res1, "-", label="$\\Phi=\\pi/2$")
ax.legend()
plt.savefig("qubit_rotation2.png", dpi=500)

##############################################################################
# .. figure:: ../demonstrations/oqc_pulse/qubit_rotation2.png
#     :align: center
#     :width: 40%
#     :alt: Single qubit rotations with different phases leading to different effective rotation axes
#     :target: javascript:void(0);

##############################################################################
# #



##############################################################################
# 



##############################################################################
# 



##############################################################################
# 

















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
# .. [#Rahamim]
#
#     J. Rahamim, T. Behrle, M. J. Peterer, A. Patterson, P. Spring, T. Tsunoda, R. Manenti, G. Tancredi, P. J. Leek
#     "Double-sided coaxial circuit QED with out-of-plane wiring"
#     `arXiv:1703.05828 <https://arxiv.org/abs/1703.05828>`__, 2017.
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
