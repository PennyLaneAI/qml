r"""Differentiable pulse programming on OQC Lucy
================================================

.. meta::
    :property="og:description": Perform differentiable pulse gates on superconducting qubit hardware through PennyLane
    :property="og:image": https://pennylane.ai/qml/_images/thumbnail_tutorial_pulse_on_hardware.png

.. related::
   ahs_aquila Pulse programming on Rydberg atom hardware
   tutorial_pulse_programming101 Differentiable pulse programming with qubits in PennyLane

*Author: Korbinian Kottmann — Posted: 05 October 2023.*

Pulse-level access of quantum computers offers many interesting new avenues in
quantum optimal control, variational quantum algorithms and device-aware algorithm design.
We now have the possibility to run hardware-level circuits combined with standard gates on a
physical device in ``PennyLane`` via ``AWS Braket`` on OQC's Lucy quantum computer. We explain
the physical principles and how to access them in PennyLane in this demo.

|

.. figure:: ../demonstrations/oqc_pulse/qubit_rotation.png
    :align: center
    :width: 70%
    :alt: Illustration of how single qubit rotations are realized by Z-precession and Rabi oscillation
    :target: javascript:void(0);

|

Introduction
============

Pulse level access to quantum computers provides new opportunities to parametrize gates in variational quantum algorithms.
For a general introduction to differentiable pulse programming, see our `recent demo <tutorial_pulse_programming101>`_.
Additionally to accessing `neutral atom quantum computers by Quera through PennyLane and aws <ahs_aquila>`_, we now 
also have the possibility to access OQC's Lucy, a 8-qubit superconducting quantum computer with a ring-like connectivity.
Through the `PennyLane-Braket plugin <https://amazon-braket-pennylane-plugin-python.readthedocs.io/en/latest/>`_,
we can now have the possibility to design custom pulse gates that control the physical qubits on the lowest hardware level.
A neat feature of controlling this device is the possibility to combine _digital_ gates like :math:`\text{CNOT}, H, R_x, R_y, R_z` with _pulse_ gates.
Further, this allows differentiating parametrized pulse gates natively on hardware via our recently introduced ``ODEgen`` method [#Kottmann], which we
will discuss in detail in a future demo.

In this demo, we are going to explore the physical principles for hardware level control of transmon qubits and run custom pulse gates on quantum hardware, i.e.
OQC Lucy via the `pennylane-braket plugin <https://amazon-braket-pennylane-plugin-python.readthedocs.io/en/latest/>`__.

.. note::

    To access remote services on Amazon Braket, you must first
    `create an account on AWS <https://aws.amazon.com/braket/getting-started/>`__ and also follow the
    `setup instructions <https://github.com/aws/amazon-braket-sdk-python>`__ for accessing Braket from Python.


Transmon Physics
================

Oxford Quantum Circuit's Lucy is a quantum computer with 8 superconducting transmon qubits based on the coaxmon design [#Rahamim]_.
In order to control a transmon qubit, it is driven by an electromagnetic microwave pulse. This can be modeled by the Hamiltonian

.. math:: H(t) = - \frac{\omega_q}{2} Z_q + \Omega(t) \sin(\nu_q t + \phi) Y_q

of the driven qubit with qubit frequency :math:`\omega_q`, drive amplitude :math:`\Omega(t)`, drive frequency :math:`\nu_q` and phase :math:`\phi`.
See, for example, reference [#Krantz]_ for a good derivation and review.
The first term leads to a constant precession around the Z axis on the Bloch sphere, whereas the second term introduces
the so-called Rabi oscillation between :math:`|0\rangle` and :math:`|1\rangle`. 

This can be seen by the following simple simulation,
where we evolve the state in the Bloch sphere from :math:`|0\rangle` with a constant pulse of :math:`\Omega(t) = 2 \pi \text{GHz}`
for :math:`1\text{ns}`.
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
# We can see that for a fixed time, we land on a different longitude of the Bloch sphere. 
# We can therefore control the rotation axis of the logical gate by setting the phase :math:`\phi`
# of the drive. Another way of seeing this is by fixing the pulse duration and looking at the
# final state for different amplitudes and two phases shifted by :math:`\pi/2`.

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

##############################################################################
# .. figure:: ../demonstrations/oqc_pulse/qubit_rotation2.png
#     :align: center
#     :width: 70%
#     :alt: Single qubit rotations with different phases leading to different effective rotation axes
#     :target: javascript:void(0);

##############################################################################
# Rabi oscillation calibration
# ============================
# 
# Because every execution on the device costs money, we want to make sure that we can leverage classical 
# simulation as best as possible. For this, we calibrate the attenuation :math:`\nu` between the voltage output
# that we set on the device, :math:`V_0`, and the actual voltage the superconducting qubit receives, :math:`V_\text{device} = \nu V_0`.
# The attenuation :math:`\nu` accounts for all losses between the arbitrary waveform generator (AWG) that outputs the signal in
# the lab at room temperature and all wires that lead to the cooled down chip in a cryostat.
# 
# We start by setting up the real device and a simulation device and perform all measurements on qubit 5.

wire = 5
dev_sim = qml.device("default.qubit.jax", wires=[wire])
dev_lucy = qml.device("braket.aws.qubit",
    device_arn="arn:aws:braket:eu-west-2::device/qpu/oqc/Lucy",
    wires=range(8), 
    shots=1000,
)

qubit_freq = dev_lucy.pulse_settings["qubit_freq"][wire]

##############################################################################
# #

H0 = qml.pulse.transmon_interaction(
    qubit_freq = [qubit_freq],
    connections = [],
    coupling = [],
    wires = [wire]
)
Hd0 = qml.pulse.transmon_drive(qml.pulse.constant, qml.pulse.constant, qubit_freq, wires=[wire])

def circuit(params, duration):
    qml.evolve(H0 + Hd0)(params, t=duration)
    return qml.expval(qml.PauliZ(wire))

qnode_sim = jax.jit(qml.QNode(circuit, dev_sim, interface="jax"))
qnode_lucy = qml.QNode(circuit, dev_lucy, interface="jax")

##############################################################################
# We are going to fit the resulting Rabi oscillations to a sinusoid. For this we use 
# a little helper function.

from scipy.optimize import curve_fit
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

##############################################################################
# We can now execute the same constant pulse for different evolution times and see Rabi oscillation
# in the evolution of :math:`\langle Z_5 \rangle`.

t0, t1, num_ts = 10., 25., 20
phi0 = 0.
amp0 = 0.3
x_lucy = np.linspace(t0, t1, num_ts)
params = jnp.array([amp0, phi0])

y_lucy = [qnode_lucy(params, t) for t in x_lucy]

##############################################################################
# And we compare that to the same pulses in simulation.


x_lucy_fit, y_lucy_fit, coeffs_fit_lucy = fit_sinus(x_lucy, y_lucy, [1., 0.6, 1])

plt.plot(x_lucy, y_lucy, "x:", label="data")
plt.plot(x_lucy_fit, y_lucy_fit, "-", color="tab:blue", label=f"{coeffs_fit_lucy[0]:.3f} sin({coeffs_fit_lucy[1]:.3f} t + {coeffs_fit_lucy[2]:.3f})", alpha=0.4)

params_sim = jnp.array([amp0, phi0])
x_sim = jnp.linspace(10., 15., 50)
y_sim = jax.vmap(qnode_sim, (None, 0))(params_sim, x_sim)
x_fit, y_fit, coeffs_fit_sim = fit_sinus(x_sim, y_sim, [2., 1., -np.pi/2])

plt.plot(x_sim, y_sim, "x-", label="sim")
plt.plot(x_fit, y_fit, "-", color="tab:orange", label=f"{coeffs_fit_sim[0]:.3f} sin({coeffs_fit_sim[1]:.3f} t + {coeffs_fit_sim[2]:.3f})", alpha=0.4)
plt.legend()
plt.ylabel("<Z>")
plt.xlabel("t1")

plt.show()

##############################################################################
# .. figure:: ../demonstrations/oqc_pulse/calibration0.png
#     :align: center
#     :width: 70%
#     :alt: Rabi oscillation for different pulse lengths.
#     :target: javascript:void(0);
# 
# We see that the oscillation on the real device is significantly slower due to the attenuation.
# We can estimate it by ratio of the measured Rabi frequency for simulation and device execution.#

attenuation = np.abs(coeffs_fit_lucy[1] / coeffs_fit_sim[1])
print(attenuation)

##############################################################################
# .. rst-class:: sphx-glr-script-out
#
#  Out:
#
#  .. code-block:: none
#
#      0.14381682156995643

##############################################################################
# We can now plot the same comparison above but with the attenuation factored in and see a
# better match between simulation and device execution.

plt.plot(x_lucy, y_lucy, "x:", label="data")
plt.plot(x_lucy_fit, y_lucy_fit, "-", color="tab:blue", label=f"{coeffs_fit_lucy[0]:.3f} sin({coeffs_fit_lucy[1]:.3f} t + {coeffs_fit_lucy[2]:.3f})", alpha=0.4)

params_sim = jnp.array([attenuation * amp0, phi0])
x_sim = jnp.linspace(10., 25., 50)
y_sim = jax.vmap(qnode_sim, (None, 0))(params_sim, x_sim)
x_fit, y_fit, coeffs_fit_sim = fit_sinus(x_sim, y_sim, [2., 0.5, -np.pi/2])

plt.plot(x_sim, y_sim, "x-", label="sim")
plt.plot(x_fit, y_fit, "-", color="tab:orange", label=f"{coeffs_fit_sim[0]:.3f} sin({coeffs_fit_sim[1]:.3f} t + {coeffs_fit_sim[2]:.3f})", alpha=0.4)
plt.legend()
plt.ylabel("<Z>")
plt.xlabel("t1")

plt.show()

##############################################################################
# .. figure:: ../demonstrations/oqc_pulse/calibration1.png
#     :align: center
#     :width: 70%
#     :alt: Rabi oscillation for different pulse lengths.
#     :target: javascript:void(0);

##############################################################################
# In particular, we see a match in both Rabi frequencies. The error in terms of the magnitude of the Rabi oscillation
# may be due to different sources. For one, the qubit has a read-out fidelity of :math:`93\%`, according to the vendor. 
# Another possible source is classical and quantum cross-talk not considered in our model. We suspect the main source
# for error beyond read-out fidelity to come from excitations to higher levels, caused by strong amplitudes and rapid
# changes in the signal.



##############################################################################
# X-Y Rotations
# =============
#
# We now want to experiment with performing X-Y-rotations by setting the phase.
# For that, we compute expectation values of :math:`\langle X \rangle`, :math:`\langle Y \rangle`, and :math:`\langle Z \rangle`
# while changing the phase :math:`\phi` at a fixed duration of :math:`15`ns and output amplitude of :math:`0.3` (arbitrary unit :math:`\in [0, 1]`).

def amplitude(p, t):
    return attenuation * p
Hd_attenuated = qml.pulse.transmon_drive(amplitude, qml.pulse.constant, qubit_freq, wires=[wire])

@jax.jit
@qml.qnode(dev_sim, interface="jax")
def qnode_sim(params, duration=15.):
    qml.evolve(H0 + Hd_attenuated)(params, t=duration, atol=1e-12)
    return [qml.expval(qml.PauliX(wire)), qml.expval(qml.PauliY(wire)), qml.expval(qml.PauliZ(wire))]

@qml.qnode(dev_lucy, interface="jax")
def qnode_lucy(params, duration=15.):
    qml.evolve(Hd0)(params, t=duration)
    return [qml.expval(qml.PauliX(wire)), qml.expval(qml.PauliY(wire)), qml.expval(qml.PauliZ(wire))]


phi0, phi1, n_phis = -np.pi, np.pi, 20
amp0 = 0.3
x_lucy = np.linspace(phi0, phi1, n_phis)
y_lucy = [qnode_lucy([amp0, phi]) for phi in x_lucy]

##############################################################################
# With the attenuation explicitly taken into account, we can now achieve a good comparison
# between simulation and device execution.

fig, axs = plt.subplots(ncols=2, figsize=(8, 4))

ax = axs[0]
ax.plot(x_lucy, y_lucy[:, 0], "x-", label="$\\langle X \\rangle$")
ax.plot(x_lucy, y_lucy[:, 1], "x-", label="$\\langle Y \\rangle$")
ax.plot(x_lucy, y_lucy[:, 2], "x-", label="$\\langle Z \\rangle$")
ax.plot(x_lucy, np.sum(y_lucy**2, axis=1), ":", label="$\\langle X \\rangle^2 + \\langle Y \\rangle^2 + \\langle Z \\rangle^2$")
ax.set_xlabel("$\\phi$")
ax.set_title(f"OQC Lucy qubit {wire}")
ax.set_ylim((-1, 1))

x_sim = x_lucy
params_sim = jnp.array([[amp0, phi] for phi in x_sim])
y_sim = np.array(jax.vmap(qnode_sim)(params_sim))

ax = axs[1]
ax.plot(x_sim, y_sim[0], "x-", label="$\\langle X \\rangle$")
ax.plot(x_sim, y_sim[1], "x-", label="$\\langle Y \\rangle$")
ax.plot(x_sim, y_sim[2], "x-", label="$\\langle Z \\rangle$")
ax.plot(x_sim, np.sum(y_sim**2, axis=0), ":", label="$\\langle X \\rangle^2 + \\langle Y \\rangle^2 + \\langle Z \\rangle^2$")

ax.set_xlabel("$\\phi$")
ax.set_title("Simulation")
ax.legend()

##############################################################################
# .. figure:: ../demonstrations/oqc_pulse/calibration2.png
#     :align: center
#     :width: 70%
#     :alt: Rabi oscillation for different pulse lengths.
#     :target: javascript:void(0);

##############################################################################
# As expected, we see a constant :math:`\langle Z \rangle` contribution, as changing :math:`\phi` delays the precession around the Z-axis
# and we land on a fixed latitude. What is changed is the longitude, leading to different rotation axes in the X-Y-plane.



##############################################################################
# Conclusion
# ==========
# Can conclude as soon as device works as expected.#



##############################################################################
# 
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
