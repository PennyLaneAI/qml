r"""Evaluating analytic gradients of pulseprograms on quantum computers
=======================================================================


Are you tired of spending precious quantum resources on computing stochastic gradients of quantum pulse programs?
ODEgen allows you to compute analytic gradients with high accuracy at lower cost! Learn about how ODEgen achieves this
and convince yourself with a numerical demonstration in this demo.

|

.. figure:: ../demonstrations/odegen/odegen_fig1.png
    :align: center
    :width: 100%
    :alt: Illustration of the ODEgen gradient method for pulse gates, compared to the stochastic parameter-shift rule
    :target: javascript:void(0);

    Illustration of ODEgen and stochastic parameter-shift for computing gradients of quantum pulse programs.
    ODEgen offloads the complexity induced by the time-dynamics to a classical ODE solver, whereas SPS performs
    Monte-Carlo sampling in time.

|

Introduction
------------

Many contemporary quantum computers are operated by steering the qubit state through an
electromagnetic pulse. This can be modeled by means of a time-dependent Hamiltonian 

.. math:: H(\theta, t) = H_\text{drift} + \sum_{j=1}^{N_g} f_j(\theta, t) H_j

with time-dependent, parametrized pulse envelopes :math:`f_j(\theta, t)` and a constant drift term :math:`H_\text{drift}`. A prominent example
is superconducting qubit platforms as described in the :doc:`demo on differentiable pulse programming </demos/tutorial_pulse_programming101>`
or :doc:`the demo about OQC's Lucy </demos/oqc_pulse>`. Such a drive for some time window then induces a unitary evolution :math:`U(\theta)` according
to the time-dependent Schrödinger equation.

The parameters :math:`\theta` of :math:`H(\theta, t)` determine the shape and strength of the pulse,
and can be subject to optimization in applications like the variational quantum eigensolver (VQE) [#Meitei]_.
Gradient based optimization on hardware is possible by utilizing the stochastic 
parameter-shift (SPS) rule introduced in [#Banchi]_ and [#Leng]_. However, this method is intrinsically stochastic
and may require a large number of shots.

In this demo, we are going to take a look at the recently introduced ODEgen method for computing analytic gradients 
of pulse gates [#Kottmann]_. It utilizes classical 
ordinary differential equation (ODE) solvers for computing gradient recipes of quantum pulse programs
that can be executed on hardware.


ODEgen vs. SPS
--------------

We are interested in cost functions of the form

.. math:: \mathcal{L}(\theta) = \langle 0 | U(\theta)^\dagger H_\text{obj} U(\theta) | 0 \rangle

where we compute the expectation value of some objective Hamiltonian :math:`H_\text{obj}` (think quantum many-body Hamiltonian whose ground state energy we want to estimate).
For simplicity, we assume a sole pulse gate :math:`U(\theta)`. Further, let us assume the so-called pulse generators
:math:`H_q` in :math:`H(\theta, t)` to be Pauli words, which will make SPS rule below a bit more digestible. For more details on the general cases we refer to the original paper [#Kottmann]_.

SPS
~~~

We can compute the gradient of :math:`\mathcal{L}` by means of the stochastic parameter-shift rule via

.. math:: \frac{\partial}{\partial \theta_j} \mathcal{L}(\theta) = \int_0^T d\tau \sum_q \frac{\partial f_q(\theta, \tau)}{\partial \theta_j} \left(\tilde{\mathcal{L}}^+_q(\tau) - \tilde{\mathcal{L}}^-_q(\tau) \right).

The :math:`\tilde{\mathcal{L}}^\pm_q(\tau) = \langle \psi_q^\pm(\tau) | H_\text{obj} | \psi_q^\pm(\tau) \rangle` are the original expectation values with
shifted evolutions :math:`| \psi_q^\pm(\tau) \rangle = U(T, \tau) e^{-i\left(\pm\frac{\pi}{4}\right)H_q} U(\tau, 0)`. Here, we use the notation :math:`U(t_1, t_0)`
to indicate the evolution is going from time :math:`t_0` to :math:`t_1`.
In practice, the integral is approximated via Monte Carlo integration

.. math:: \frac{\partial}{\partial \theta_j} \mathcal{L}(\theta) \approx \frac{1}{N_s} \sum_{\tau \in \mathcal{U}([0, T])} \sum_q \frac{\partial f_q(\theta, \tau)}{\partial \theta_j} \left(\tilde{\mathcal{L}}^+_q(\tau) - \tilde{\mathcal{L}}^-_q(\tau) \right)

where the :math:`\tau \in \mathcal{U}([0, T])` are sampled uniformly between :math:`0` and :math:`T`, and :math:`N_s` is the number 
of Monte Carlo samples for the integration. The larger the number of samples, the better the approximation. 
This comes at the cost of more quantum resources :math:`\mathcal{R}`
in form of the number of distinct expectation values executed on the quantum device,

.. math:: 

    \mathcal{R}_\text{SPS} = 2 N_s N_g.

ODEgen
~~~~~~

In contrast, the recently introduced ODEgen method [#Kottmann]_ has the advantage that it circumvents the need for Monte Carlo sampling by off-loading the complexity
introduced by the time-dynamics to a differentiable ODE solver.

The first step of ODEgen is writing the derivative of a pulse unitary :math:`U(\theta)` as

.. math:: \frac{\partial}{\partial \theta_j} U(\theta) = -i U(\theta) \mathcal{H}_j

with a so-called effective generator :math:`\mathcal{H}_j` for each of the parameters :math:`\theta_j`.
We can obtain :math:`\mathcal{H}_j` classically by computing both :math:`\frac{\partial}{\partial \theta_j} U(\theta)` 
and :math:`U(\theta)` in a forward and backward pass through the ODE solver. We already use such a solver in PennyLane for simulating pulses in :class:`~.pennylane.pulse.ParametrizedEvolution`.
Here, we use it to generate parameter-shift rules that can be executed on hardware.

The next step is to decompose each effective generator into a basis the quantum computer can understand, and, in particular, can execute.
We choose the typical Pauli basis and write

.. math:: \mathcal{H}_j = \sum_\ell \omega_\ell^{(j)} P_\ell

for Pauli words :math:`P_\ell` (e.g. :math:`P_\ell = X_0 Y_1` for some :math:`\ell`). With this decomposition, we can write the gradient of the cost function in terms of parameter-shift rules
that can be executed on a quantum computer:

.. math:: 
    \frac{\partial \mathcal{L}}{\partial \theta_j} = \langle 0 | \left[U(\theta)^\dagger H_\text{obj} U(\theta), -i\mathcal{H}_j \right] | 0 \rangle \\
    = \sum_\ell 2 \omega_\ell^{(j)} \langle 0 | \left[U(\theta)^\dagger H_\text{obj} U(\theta), -\frac{i}{2} P_\ell \right] |0\rangle \\
    = \sum_\ell 2 \omega_\ell^{(j)} \frac{d}{dx} \left[ \langle 0 | U(\theta)^\dagger e^{i\frac{x}{2} P_\ell} H_\text{obj} e^{-i\frac{x}{2} P_\ell} U(\theta)|0\rangle \right]_{x=0}.

In particular, we can identify

.. math:: L_\ell(x) = \langle 0 | U(\theta)^\dagger e^{i\frac{x}{2} P_\ell} H_\text{obj} e^{-i\frac{x}{2} P_\ell} U(\theta)|0\rangle

as an expectation value shifted by the dummy variable :math:`x`, whose derivative is given by the standard two-term parameter-shift rule (see e.g. `this derivation <https://pennylane.ai/qml/glossary/parameter_shift/>`_).
Overall, we have 

.. math:: \frac{\partial \mathcal{L}}{\partial \theta_j} = \sum_\ell \omega_\ell^{(j)} \left(L_\ell\left(\frac{\pi}{2}\right) - L_\ell\left(-\frac{\pi}{2}\right) \right).

The quantum resources for ODEgen are :math:`2` executions for each non-zero Pauli term :math:`\omega_\ell^{(j)} P_\ell` (i.e. non-zero for any :math:`j` for a particular :math:`\ell`) of the decomposition.
This number is at most :math:`4^n-1` for :math:`n` qubits. A better upper bound is given by the dimension of the dynamical Lie algebra (DLA) of the pulse Hamiltonian. That is, the number of linearly independent operators
that can be generated from nested commutators of the pulse generators and the drift term. An example would be a pulse Hamiltonian composed of terms :math:`X_0`, :math:`X_1` and :math:`Z_0 Z_1`. A basis for the DLA of
this pulse Hamiltonian is given by :math:`\{X_0, X_1, Z_0Z_1, Y_0Y_1, Y_0Z_0, Z_0Y_0\}`. This tells us that only those terms can be non-zero in a decomposition of any effective generator from gates generated by such a pulse Hamiltonian.

Overall, ODEgen is well-suited for pulse Hamiltonians that act effectively on few qubits - as is the case for superconducting qubit and ion trap qubit architectures - or yield a small DLA.




Numerical experiment
--------------------

We want to put ODEgen and SPS head to head in a variational quantum algorithm with the same available quantum resources.
For this, we are going to perform the variational quantum eigensolver (VQE) on the Heisenberg model Hamiltonian

.. math:: H_\text{obj} = X_0 X_1 + Y_0 Y_1 + Z_0 Z_1

for two qubits. The ground state of this Hamiltonian is the maximally entangled singlet state
:math:`|\phi^- \rangle = (|01\rangle - |10\rangle)/\sqrt{2}` with ground state energy :math:`-3`.
Let us define it in PennyLane and also import some libraries that we are going to need for this demo.

"""
import pennylane as qml
import numpy as np
import jax.numpy as jnp
import jax

import optax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import matplotlib.pyplot as plt

H_obj = qml.sum(qml.PauliX(0) @ qml.PauliX(1), qml.PauliY(0) @ qml.PauliY(1), qml.PauliZ(0) @ qml.PauliZ(1))
E_exact = -3.
wires = H_obj.wires

##############################################################################
# We are going to consider a system of transmon qubits described by the Hamiltonian
# 
# .. math:: H(\theta, t) = - \sum_q \frac{\omega_q}{2} Z_q + \sum_q \Omega_q(t) \sin(\nu_q t + \phi_q(t)) Y_q + \sum_{q, p \in \mathcal{C}} \frac{g_{qp}}{2} (X_q X_p + Y_q Y_p).
# 
# The first term describes the single qubits with frequencies :math:`\omega_q`. 
# The second term describes the driving with drive amplitudes :math:`\Omega_q`, drive frequencies :math:`\nu_q` and phases :math:`\phi_q`. 
# You can check out our :doc:`recent demo on driving qubits on OQC's Lucy </demos/oqc_pulse>` if 
# you want to learn more about the details of controlling transmon qubits.
# The third term describes the coupling between neighboring qubits. We only have two qubits and a simple topology of 
# :math:`\mathcal{C} = \{(0, 1)\}`.
# The coupling is necessary to generate entanglement, which is achieved with cross-resonant driving in fixed-coupling 
# transmon systems, as is the case here.
# 
# We will use realistic parameters for the transmons, taken from the `coaxmon design paper <https://arxiv.org/abs/1905.05670>`_ [#Patterson]_ 
# (this is the blue-print for the transmon qubits in OQC's Lucy that you can :doc:`access on a pulse level in PennyLane </demos/oqc_pulse>`).
# In order to prepare the singlet ground state, we will perform a cross-resonance pulse, i.e. driving one qubit at its coupled neighbor's 
# frequency for entanglement generation (see [#Patterson]_ or [#Krantz]_) while simultaneously driving the other qubit on resonance.
# We choose a gate time of :math:`100 \text{ ns}`. We will use a piecewise constant function :func:`~pennylane.pulse.pwc` to parametrize both
# the amplitude :math:`\Omega_q(t)` and the phase :math:`\phi_q(t)` in time, with ``t_bins = 10`` time bins to allow for enough flexibility in the evolution.

T_CR = 100.   # gate time for two qubit drive (cross resonance)
qubit_freq = 2*np.pi*np.array([6.509, 5.963])

def drive_field(T, wdrive):
    """Set the evolution time ``T`` and drive frequency ``wdrive``"""
    def wrapped(p, t):
        # The first len(p) values of the trainable params p characterize the pwc function
        amp = qml.pulse.pwc(T)(p[:len(p)//2], t)
        phi = qml.pulse.pwc(T)(p[len(p)//2:], t)
        return amp * jnp.sin(wdrive * t + phi)

    return wrapped

H_pulse = qml.dot(-0.5*qubit_freq, [qml.PauliZ(i) for i in wires])
H_pulse += 2 * np.pi * 0.0123 * (qml.PauliX(wires[0]) @ qml.PauliX(wires[1]) + qml.PauliY(wires[0]) @ qml.PauliY(wires[1]))

H_pulse += drive_field(T_CR, qubit_freq[0]) * qml.PauliY(wires[0]) # on-resonance driving of qubit 0
H_pulse += drive_field(T_CR, qubit_freq[0]) * qml.PauliY(wires[1]) # off-resonance driving of qubit 1

##############################################################################
# We can now define the cost function that computes the expectation value of 
# the Heisenberg Hamiltonian after evolving the state with the parametrized pulse Hamiltonian.
# We then define the two separate qnodes with ODEgen and SPS as their differentiation methods, respectively.

def qnode0(params):
    qml.evolve(H_pulse)((params[0], params[1]), t=T_CR)
    return qml.expval(H_obj)

dev = qml.device("default.qubit", wires=range(2))

qnode_jax = qml.QNode(qnode0, dev, interface="jax")
value_and_grad_jax = jax.jit(jax.value_and_grad(qnode_jax))

num_split_times = 8
qnode_sps = qml.QNode(qnode0, dev, interface="jax", diff_method=qml.gradients.stoch_pulse_grad, use_broadcasting=True, num_split_times=num_split_times)
value_and_grad_sps = jax.value_and_grad(qnode_sps)

qnode_odegen = qml.QNode(qnode0, dev, interface="jax", diff_method=qml.gradients.pulse_odegen)
value_and_grad_odegen = jax.value_and_grad(qnode_odegen)

##############################################################################
# We note that for as long as we are in simulation, there is naturally no difference between the gradients obtained
# from direct backpropagation and using ODEgen.

tbins = 10            # number of time bins per pulse
n_param_batch = 2     # number of parameter batches

x = jnp.ones((n_param_batch, tbins * 2))

res0, grad0 = value_and_grad_jax(x)
res1, grad1 = value_and_grad_odegen(x)
np.allclose(res0, res1, atol=1e-3), np.allclose(grad0, grad1, atol=1e-3)

##############################################################################
# This allows us to use direct backpropagation in this demo, which is always faster in simulation.
# We now have all ingredients to run VQE with ODEgen and SPS. We define the following standard
# optimization loop and run it from the same random initial values
# with ODEgen and SPS gradients.

def run_opt(value_and_grad, theta, n_epochs=120, lr=0.1, b1=0.9, b2=0.999):

    optimizer = optax.adam(learning_rate=lr, b1=b1, b2=b2)
    opt_state = optimizer.init(theta)

    energy = np.zeros(n_epochs)
    gradients = []
    thetas = []

    @jax.jit
    def partial_step(grad_circuit, opt_state, theta):
        # SPS gradients don't allow for full jitting of the update step
        updates, opt_state = optimizer.update(grad_circuit, opt_state)
        theta = optax.apply_updates(theta, updates)

        return opt_state, theta

    ## Optimization loop
    for n in range(n_epochs):
        val, grad_circuit = value_and_grad(theta)
        opt_state, theta = partial_step(grad_circuit, opt_state, theta)

        energy[n] = val
        gradients.append(grad_circuit)
        thetas.append(theta)

    return thetas, energy

key = jax.random.PRNGKey(0)
theta0 = jax.random.normal(key, shape=(n_param_batch, tbins * 2))

thetaf_odegen, energy_odegen = run_opt(value_and_grad_jax, theta0)
thetaf_sps, energy_sps = run_opt(value_and_grad_sps, theta0)

plt.plot(np.array(energy_sps) - E_exact, label="SPS")
plt.plot(np.array(energy_odegen) - E_exact, label="ODEgen")
plt.legend()
plt.yscale("log")
plt.ylabel("$E(\\theta) - E_{{FCI}}$")
plt.xlabel("epochs")
plt.show()


##############################################################################
# We see that with analytic gradients (ODEgen), we can reach the ground state energy within 100 epochs, whereas with SPS gradients we cannot find the path 
# towards the minimum due to the stochasticity of the gradient estimates. Note that both optimizations start from the same (random) initial point.
# This picture solidifies when repeating this procedure for multiple runs from different random initializations, as was demonstrated in [#Kottmann]_.
#
# We also want to make sure that this is a fair comparison in terms of quantum resources. In the case of ODEgen we maximally have :math:`\mathcal{R}_\text{ODEgen} = 2 (4^n - 1) = 30` expectation values.
# For SPS we have :math:`N_s 4 = 32` (due to :math:`N_s=8` time samples per gradient that we chose in ``num_split_times`` above). Thus, overall, we require fewer 
# quantum resources for ODEgen gradients while achieving better performance.
#
# Conclusion
# ----------
#
# We introduced ODEgen for computing analytic gradients of pulse gates and showcased its advantages in simulation for a VQE example.
# The method is particularly well-suited for quantum computing architectures that build complexity from few-qubit gates, as is the case
# for superconducting qubit architectures.
# We invite you to play with ODEgen yourself. Note that this feature is amenable to hardware and you can compute gradients on OQC's Lucy via PennyLane.
# We show you how to connect to Lucy and run pulse gates in our :doc:`recent demo </demos/oqc_pulse>`.
# Running VQE using ODEgen on hardware has recently been demonstrated in [#Kottmann]_ and you can directly find `the code here <https://github.com/XanaduAI/Analytic_Pulse_Gradients/tree/main/VQE_OQC>`_.
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
# ..  [#Patterson]
#
#     A. D. Patterson, J. Rahamim, T. Tsunoda, P. Spring, S. Jebari, K. Ratter, M. Mergenthaler, G. Tancredi, B. Vlastakis, M. Esposito, P. J. Leek
#     "Calibration of the cross-resonance two-qubit gate between directly-coupled transmons"
#     `arXiv:1905.05670 <https://arxiv.org/abs/1905.05670>`__, 2019
#
##############################################################################
# About the author
# ----------------
# .. include:: ../_static/authors/korbinian_kottmann.txt
