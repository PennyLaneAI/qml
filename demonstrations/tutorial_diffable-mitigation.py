r"""Differentiating quantum error mitigation transforms
=======================================================

.. meta::
    :property="og:description": Differentiable error mitigation
    :property="og:image": https://pennylane.ai/qml/_images/diffable_mitigation_thumb.png

.. related::

    tutorial_error_mitigation Error mitigation with Mitiq and PennyLane

*Author: KK,.. Posted: 27 July 2022*

Differentiating quantum error mitigation transforms
---------------------------------------------------

Most variational quantum algorithms (VQAs) are concerned with optimizing a `quantum function` 

.. math:: f(\theta) = \langle 0 | U^\dagger(\theta) H U(\theta) | 0 \rangle

for some Ansatz unitary :math:`U` with variational parameters :math:`\theta` and observable :math:`H`. These algorithms are specifically designed
to be executed on noisy hardware. This means that naturally we do not have direct access to :math:`f`, but rather a noisy version
:math:`f^{⚡}(\theta) := \text{tr}\left[H \Phi(\rho(\theta)) \right]` where :math:`\Phi(\bullet)` is a noise channel and
:math:`\rho(\theta) = U^\dagger(\theta)|0\rangle \langle 0|U(\theta)` the variational state.

To be able to get the most out of these devices, it is advisable to use quantum error mitigation --- a method of
altering and/or post-processing the quantum function :math:`f^{⚡}(\theta)` to improve the result and be closer to the ideal scenario of an error free execution, :math:`f(\theta)`.

Formally, we can treat error mitigation as yet another transform that maps the noisy quantum function :math:`f^{⚡}` to a new, mitigated, quantum function :math:`\tilde{f}`,

.. math:: \text{mitigate}: f^{⚡} \mapsto \tilde{f}.

In order to run our VQA with our mitigated quantum function, we need to ensure that :math:`\tilde{f}` is differentiable --- both formally and practically in our implementation.
PennyLane now provides one such differentiable quantum error mitigation technique with `zero noise extrapolation` (ZNE), which can be used and differentiated in simulation and on hardware.
We will briefly introduce these new functionalities and afterwards go more in depth to explore what happens under the hood.

Differentiable transform
------------------------

We start by initializing a noisy device under the ``qml.DepolarizingChannel``:
"""

import pennylane as qml
import pennylane.numpy as np
from pennylane.transforms import mitigate_with_zne, fold_global, richardson_extrapolate

from matplotlib import pyplot as plt

n_wires = 4
np.random.seed(1234)

# Describe noise
noise_gate = qml.DepolarizingChannel
noise_strength = 0.05

# Load devices
dev_ideal = qml.device("default.mixed", wires=n_wires)
dev_noisy = qml.transforms.insert(noise_gate, noise_strength, position="all")(dev_ideal)

##############################################################################
# We are going to use the transverse field Ising model Hamiltonian :math:`H = - \sum_i X_i X_{i+1} + 0.5 \sum_i Z_i` as our observable.

coeffs = [1.0] * (n_wires - 1) + [0.5] * n_wires
observables = [qml.PauliX(i) @ qml.PauliX(i + 1) for i in range(n_wires - 1)]
observables += [qml.PauliZ(i) for i in range(n_wires)]

H = qml.Hamiltonian(coeffs, observables)


##############################################################################
# As well as a simplified unitary 2-design as our Ansatz. The qfunc can then be executed on the noisy or ideal device
# by creating individual ``qml.QNode`` functions.

n_layers = 2

w1 = np.ones((n_wires), requires_grad=True)
w2 = np.ones((n_layers, n_wires - 1, 2), requires_grad=True)


def qfunc(w1, w2):
    qml.SimplifiedTwoDesign(w1, w2, wires=range(n_wires))
    return qml.expval(H)


qnode_noisy = qml.QNode(qfunc, dev_noisy)
qnode_ideal = qml.QNode(qfunc, dev_ideal)

##############################################################################
# We can then simply transform the noisy qnode :math:`f^{⚡}` with ``qml.transforms.mitigate_with_zne`` to generate :math:`\tilde{f}`.
# If everything goes as planned, executing the mitigated ``qnode`` is then closer to the ideal result:

scale_factors = [1.0, 2.0, 3.0]

qnode_mitigated = mitigate_with_zne(scale_factors, fold_global, richardson_extrapolate)(qnode_noisy)

print("Ideal qnode: ", qnode_ideal(w1, w2))
print("Mitigated qnode: ", qnode_mitigated(w1, w2))
print("Noisy qnode: ", qnode_noisy(w1, w2))

##############################################################################
# The cool thing about this new mitigated qnode is that it is still differentiable! I.e. we can compute its gradient as usual:

grad = qml.grad(qnode_mitigated)
print(grad(w1, w2))


##############################################################################
# Under the hood of Zero Noise Extrapolation
# ------------------------------------------
# What is happening here under the hood? The basic idea of ZNE is to artificially increase the noise in a circuit,
# controlled by a parameter :math:`\lambda` that is called the ``scale_factor``, to be able to then extrapolate in :math:`\lambda` to zero noise.
#
# Notice how the two circuits :math:`U` and :math:`U U^\dagger U` are logically equivalent, but we can expect the latter to have more noise due its larger gate count.
# This is the underlying concept of unitary folding, which is used to artificially increase the noise of a quantum function. Given a unitary circuit :math:`U = L_d .. L_1`,
# where :math:`L_i` can be either a gate or layer, we use ``qml.transforms.fold_global`` to construct
#
# .. math:: \text{fold_global}(U) = U (U^\dagger U)^n (L^\dagger_d L^\dagger_{d-1} .. L^\dagger_s) (L_s .. L_d),
#
# where :math:`n = \lfloor (\lambda - 1)/2 \rfloor` and :math:`s = \lfloor \left((\lambda -1) \mod 2 \right) (d/2) \rfloor` are determined via the ``scale_factor`` :math:`=\lambda`.
#
# The version of ZNE that we are showcasing is simply executing the noisy quantum function :math:`f^{⚡}` for different scale factors,
# and then extrapolate to :math:`\lambda \rightarrow 0`. Note that ``scale_factor=1`` corresponds to the original circuit, i.e. the noisy execution.

scale_factors = [1.0, 2.0, 3.0]
folded_res = [qml.transforms.fold_global(qnode_noisy, lambda_)(w1, w2) for lambda_ in scale_factors]

ideal_res = qnode_ideal(w1, w2)

# coefficients are ordered like coeffs[0] * x**2 + coeffs[1] * x + coeffs[0], i.e. fitted_func(0)=coeff[-1]
coeffs = np.polyfit(scale_factors, folded_res, 2)
zne_res = coeffs[-1]

x_fit = np.linspace(0, scale_factors[-1], 20)
y_fit = np.poly1d(coeffs)(x_fit)

plt.figure(figsize=(8, 5))
plt.plot(scale_factors, folded_res, "x--", label="folded")
plt.plot(0, ideal_res, "X", label="ideal res")
plt.plot(0, zne_res, "X", label="ZNE res", color="tab:red")
plt.plot(x_fit, y_fit, label="fit", color="tab:red", alpha=0.5)
plt.legend()
plt.show()

##############################################################################
# We see that the mitigated result comes close to the ideal result, whereas the noisy result is further off (see value at ``scale_factor=1``).
#
# Note that this folding scheme is relatively simple and only really is sensible for integer values of ``scale_factor``. At the same time, ``scale_factor`` is
# limited from above by the noise as the noisy quantum function quickly decoheres under this folding. Therefore, one typically only uses ``scale_factors = [1, 2, 3]``, but you
# can in principle think of more fine grained folding schemes and test them by providing custom folding operations, see details in :func:`~.pennylane.transforms.mitigate_with_zne`.
#
#
# Differentiable mitigation in a variational quantum algorithm
# ------------------------------------------------------------
#
# We now use mitigation while we optimize the parameters our variational circuit to obtain the ground state of the Hamiltonian. We compare VQE optimization runs for
# the ideal, noisy and mitigated qnodes and see that the mitigated one comes close to the ideal results, whereas the noisy execution is further off.


def VQE_run(cost_fn, max_iter, stepsize=0.1):
    """VQE Optimization loop"""
    opt = qml.AdamOptimizer(stepsize=stepsize)

    # fixed initial guess
    w1 = np.ones((n_wires), requires_grad=True)
    w2 = np.ones((n_layers, n_wires - 1, 2), requires_grad=True)

    energy = []

    # Optimization loop
    for _ in range(max_iter):
        (w1, w2), prev_energy = opt.step_and_cost(cost_fn, w1, w2)

        energy.append(prev_energy)

    energy.append(cost_fn(w1, w2))

    return energy


max_iter = 70

energy_ideal = VQE_run(qnode_ideal, max_iter)
energy_noisy = VQE_run(qnode_noisy, max_iter)
energy_mitigated = VQE_run(qnode_mitigated, max_iter)

energy_exact = np.min(np.linalg.eigvalsh(qml.matrix(H)))

plt.figure(figsize=(8, 5))
plt.plot(energy_mitigated, ".--", label="VQE E_mitigated")
plt.plot(energy_noisy, ".--", label="VQE E_noisy")
plt.plot(energy_ideal, ".--", label="VQE E_ideal")
plt.plot([1, max_iter + 1], [energy_exact] * 2, "--", label="E_exact")
plt.legend(fontsize=14)
plt.xlabel("epoch", fontsize=18)
plt.ylabel("energy", fontsize=18)
plt.show()

##############################################################################
# Note that the discrepancies between the ideal simulation and exact result are due to the limited expressivity of our Ansatz,
# which on the other hand is due to limitations in depth by noise.
#
# So far we have been using PennyLane gradient methods that use ``autograd`` for simulation and ``parameter-shift`` rules for real device
# executions. We can also use the other interfaces that are supported by PennyLane, ``jax``, ``torch`` and ``tensorflow``, in the usual way 
# as described in the :doc:`interfaces section of the documentation <../introduction/interfaces>`.
#
##############################################################################
# Differentiating the mitigation transform itself
# ------------------------------------------------
#
# So far we have been concerned with differentiating `through` the mitigation transform. An interesting direction for future work
# is differentiating the transform itself [#DiffableTransforms]_. In particular, the authors in [#VAQEM]_ make the interesting observation
# that for some error mitigation schemes, the cost function is smooth in some of the mitigation parameters. Here, we show one of their
# examples, which is a time-sensitive dynamical decoupling scheme:
#
# .. figure:: /demonstrations/diffable-mitigation/Mitigate_real_vs_sim3.png
#     :width: 50%
#     :align: center
#
#     Time-sensitive dynamical decoupling scheme.
#
# In this mitigation technique, the single qubit state is put into equal superposition
# :math:`|+\rangle = (|0\rangle + |1\rangle)/\sqrt{2}`. During the first idle time :math:`t_1`, the state is altered due to noise.
# Applying :math:`X` reverses the roles of each computational basis state. The idea is that the noise in the second idle time
# :math:`T-t_1` is canceling out the effect of the first time window. We see that the output fidelity with respect to the noise-free
# execution is a smooth function of :math:`t_1`. This was executed on ``ibm_perth``, and we note that simple noise models,
# like the simulated IBM device, do not suffice to reproduce the behavior of the real device.
#
# Obtaining the gradient with respect to this parameter is diffult. There are formal and practical obstacles:
# Formally, writing down the derivative of this transform with respect to the idle time in order to derive its parameter-shift 
# rules would require access to the noise model.
# This, on the other hand, is very difficult for a realistic scenario. Further, most mitigation parameters are integers and would have
# to be smoothed in a differentiable way. A simple but effective strategy is using finite differences for the gradient with respect to mitigation parameters.
# 
# Overall, this is a nice example of a mitigation scheme where varying the mitigation parameter has direct impact to the simulation result.
# It is therefore desirable to be able to optimize this parameter at the same time as we perform a variational quantum algorithm.
#
#
# References
# ----------
#
# .. [#DiffableTransforms]
#
#     Olivia Di Matteo, Josh Izaac, Tom Bromley, Anthony Hayes, Christina Lee, Maria Schuld, Antal Száva, Chase Roberts, Nathan Killoran.
#     "Quantum computing with differentiable quantum transforms."
#     `arXiv:2202.13414 <https://arxiv.org/abs/2202.13414>`__, 2021.
#
# .. [#VAQEM]
#
#     Gokul Subramanian Ravi, Kaitlin N. Smith, Pranav Gokhale, Andrea Mari, Nathan Earnest, Ali Javadi-Abhari, Frederic T. Chong.
#     "VAQEM: A Variational Approach to Quantum Error Mitigation."
#     `arXiv:2112.05821 <https://arxiv.org/abs/2112.05821>`__, 2021.
