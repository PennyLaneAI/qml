r""".. _spsa:

Optimization using SPSA
=======================

.. meta::
    :property="og:description": Use the simultaneous perturbation stochastic
        approximation algorithm to optimize variational circuits in PennyLane.
    :property="og:image": https://pennylane.ai/qml/_images/spsa_mntn.png

.. related::

   tutorial_vqe A brief overview of VQE
   tutorial_vqe_qng Accelerating VQE with the QNG

*Author: PennyLane dev team. Posted: 19 Mar 2021. Last updated: 8 Apr 2021.*

In this tutorial, we investigate using a gradient-free optimizer called
the Simultaneous Perturbation Stochastic Approximation (SPSA) algorithm to optimize quantum
circuits. SPSA is a technique that involves approximating the gradient of a
quantum circuit without having to compute it.

This demonstration shows how the SPSA optimizer performs:

1. A simple task on a sampling device,
2. The variational quantum eigensolver on a simulated hardware device.

Throughout the demo, we show results obtained with SPSA and with gradient
descent and also compare the number of device executions required to complete
each optimization.

Background
----------

In PennyLane, quantum gradients on hardware are commonly computed using
`parameter-shift rules
<https://pennylane.ai/qml/glossary/parameter_shift.html>`_. Computing quantum
gradients involves evaluating the partial derivative of the quantum function
with respect to every free parameter. The partial derivatives are used to apply
the product rule to compute the gradient of the quantum circuit. For qubit
operations that are generated by one of the Pauli matrices, each partial
derivative computation will involve two quantum circuit evaluations with a
positive and a negative shift in the parameter values.

As there are two circuit evaluations for each free parameter, the number of
overall quantum circuit executions for computing a quantum gradient is
:math:`O(p)` as it scales linearly with the number of free parameters
:math:`p`. This scaling can be very costly for optimization tasks with many
free parameters.  For the overall optimization this scaling means we need
:math:`O(pn)` quantum circuit evaluations, where :math:`n` is the number of
optimization steps taken.

Fortunately, there are certain optimization techniques that offer an
alternative to computing the gradients of quantum circuits. One such technique
is called the Simultaneous Perturbation Stochastic Approximation (SPSA)
algorithm [#spall_overview]_. SPSA is an optimization method that involves
*approximating* the gradient of the cost function at each iteration step. This
technique requires only two quantum circuit executions per iteration step,
regardless of the number of free parameters. Therefore the overall number of
circuit executions would be :math:`O(n')` where :math:`n'` is the number of
optimization steps taken when using SPSA. This technique is also considered
robust against noise, making it a great optimization method in the NISQ era.

In this demo, you'll learn how the SPSA algorithm works, and how to apply it in
PennyLane to compute gradients of quantum circuits. You'll also see it in action
using noisy quantum data!

Simultaneous perturbation stochastic approximation (SPSA)
---------------------------------------------------------

SPSA is a general method for minimizing differentiable multivariate functions.
It is particularly useful for functions for which evaluating the gradient is not
possible, or too resource intensive. SPSA provides a stochastic method for
approximating the gradient of a multivariate differentiable cost function. To
accomplish this the cost function is evaluated twice using perturbed parameter
vectors: every component of the original parameter vector is simultaneously
shifted with a randomly generated value. This is in contrast to
finite-differences methods where for each evaluation only one component of the
parameter vector is shifted at a time.

Similar to gradient-based approaches such as gradient descent, SPSA is an
iterative optimization algorithm. Let's consider a differentiable cost function
:math:`L(\theta)` where :math:`\theta` is a :math:`p`-dimensional vector and
where the optimization problem can be translated into finding a :math:`\theta^*`
at which :math:`\frac{\partial L}{\partial \theta} = 0`.  It is assumed that
measurements of :math:`L(\theta)` are available at various values of
:math:`\theta`---this is exactly the problem that we'd consider when optimizing
quantum functions!

Just like with gradient-based methods, SPSA starts with an initial parameter
vector :math:`\hat{\theta}_{0}`. After :math:`k` iterations, the :math:`(k+1)` th
parameter iterates can be obtained as

.. math:: \hat{\theta}_{k+1} = \hat{\theta}_{k} - a_{k}\hat{g}_{k}(\hat{\theta}_{k}),

where :math:`\hat{g}_{k}` is the estimate of the gradient :math:`g(\theta) =
\frac{ \partial L}{\partial \theta}` at the iterate :math:`\hat{\theta}_{k}`
based on prior measurements of the cost function, and :math:`a_{k}` is a
positive number [#spall_overview]_.

One of the advantages of SPSA is that it is robust to any noise that may occur
when measuring the function :math:`L`. Therefore, let's consider the function
:math:`y(\theta)=L(\theta) + \varepsilon`, where :math:`\varepsilon` is some
perturbation of the output. In SPSA, the estimated gradient at each iteration
step is expressed as

.. math:: \hat{g}_{ki} (\hat{\theta}_{k}) = \frac{y(\hat{\theta}_{k} +c_{k}\Delta_{k})
    - y(\hat{\theta}_{k} -c_{k}\Delta_{k})}{2c_{k}\Delta_{ki}},

where :math:`c_{k}` is a positive number and :math:`\Delta_{k} = (\Delta_{k_1},
\Delta_{k_2}, ..., \Delta_{k_p})^{T}` is a perturbation vector. The
stochasticity of the technique comes from the fact that for each iteration step
:math:`k` the components of the :math:`\Delta_{k}` perturbation vector are
randomly generated using a zero-mean distribution. In most cases, the Bernoulli
distribution is used, meaning each parameter is simultaneously perturbed by
either :math:`\pm c_k`.

It is this perturbation that makes SPSA robust to noise --- since every
parameter is already being shifted, additional shifts due to noise are less
likely to hinder the optimization process. In a sense, noise gets "absorbed"
into the already-stochastic process. This is highlighted in the figure below,
which portrays an example of the type of path SPSA takes through the space of
the function, compared to a standard gradient-based optimizer.

.. figure:: ../demonstrations/spsa/spsa_mntn.png
   :align: center
   :width: 60%

   ..

   A schematic of the search paths used by gradient descent with
   parameter-shift and SPSA.

Now that we have explored how SPSA works, let's see how it performs in practice!

Optimization on a sampling device
---------------------------------

.. important::

    To run this demo locally, you'll need to install the `noisyopt
    <https://github.com/andim/noisyopt>`_ library. This library contains a
    straightforward implementation of SPSA that can be used in the same way as the
    optimizers available in `SciPy's minimize method
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`_.

First, let's consider a simple quantum circuit on a sampling device. For this,
we'll be using a device from the `PennyLane-Qiskit plugin
<https://pennylaneqiskit.readthedocs.io/en/latest/>`_ that samples quantum
circuits to get measurement outcomes and later post-processes these outcomes to
compute statistics like expectation values.

.. note::

    Just as with other PennyLane devices, the number of samples taken for a device
    execution can be specified using the ``shots`` keyword argument of the
    device.

Once we have a device selected, we just need a couple of other ingredients for
the pieces of an example optimization to come together:

* a circuit ansatz: :func:`~.pennylane.templates.layers.StronglyEntanglingLayers`,
* initial parameters: the correct shape can be computed by :func:`~.pennylane.templates.layers.StronglyEntanglingLayers.shape`,
* an observable: :math:`\bigotimes_{i=0}^{N-1}\sigma_z^i`, where :math:`N` stands
  for the number of qubits.

"""
import pennylane as qml
import numpy as np

num_wires = 4
num_layers = 5

dev_sampler_spsa = qml.device("qiskit.aer", wires=num_wires, shots=1000)

##############################################################################
# We seed so that we can simulate the same circuit every time.
np.random.seed(50)

all_pauliz_tensor_prod = qml.operation.Tensor(*[qml.PauliZ(i) for i in range(num_wires)])


def circuit(params):
    qml.templates.StronglyEntanglingLayers(params, wires=list(range(num_wires)))
    return qml.expval(all_pauliz_tensor_prod)


##############################################################################
# After this, we'll initialize the parameters in a way that is compatible with
# the ``noisyopt`` package. The ``noisyopt`` package requires the trainable parameters
# be a flattened array. As a result, our cost function must accept a flat array of parameters
# to be optimized.
flat_shape = num_layers * num_wires * 3
param_shape = qml.templates.StronglyEntanglingLayers.shape(n_wires=num_wires, n_layers=num_layers)
init_params = np.random.normal(size=param_shape)

init_params_spsa = init_params.reshape(flat_shape)

qnode_spsa = qml.QNode(circuit, dev_sampler_spsa)


def cost_spsa(params):
    return qnode_spsa(params.reshape(num_layers, num_wires, 3))


##############################################################################
# Once we have defined each piece of the optimization, there's only one
# remaining component required: the *SPSA optimizer*.
# We'll use the SPSA optimizer provided by the ``noisyopt`` package. Once
# imported, we can initialize parts of the optimization such as the number of
# iterations, a collection to store the cost values, and a callback function.
# Once the optimization has concluded, we save the number of device executions
# required for completion using the callback function. This will be an
# interesting quantity!
from noisyopt import minimizeSPSA

niter_spsa = 200

# Evaluate the initial cost
cost_store_spsa = [cost_spsa(init_params_spsa)]
device_execs_spsa = [0]


def callback_fn(xk):
    cost_val = cost_spsa(xk)
    cost_store_spsa.append(cost_val)

    # We've evaluated the cost function, let's make up for that
    num_executions = int(dev_sampler_spsa.num_executions / 2)
    device_execs_spsa.append(num_executions)

    iteration_num = len(cost_store_spsa)
    if iteration_num % 10 == 0:
        print(
            f"Iteration = {iteration_num}, "
            f"Number of device executions = {num_executions}, "
            f"Cost = {cost_val}"
        )


##############################################################################
# Choosing the hyperparameters
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The ``noisyopt`` package allows us to choose the initial value of two
# hyperparameters for SPSA: the :math:`c` and :math:`a` coefficients. Recall
# from above that the :math:`c` values control the amount of random shift when
# evaluating the cost function, while the :math:`a` coefficients are analogous to a learning
# rate and affect the degree to which the parameters change at each update
# step.
#
# With stochastic approximation, specifying such hyperparameters significantly
# influences the convergence of the optimization for a given problem. Although
# there is no universal recipe for selecting these values (as they depend
# strongly on the specific problem), [#spall_implementation]_ includes
# guidelines for the selection. In our case, the initial values for :math:`c`
# and :math:`a` were selected as a result of a grid search to ensure a fast
# convergence.  We further note that apart from :math:`c` and :math:`a`, there
# are further coefficients that are initialized in the ``noisyopt`` package
# using the
# previously mentioned guidelines.
#
# Our cost function does not take a seed as a keyword argument (which would be
# the default behaviour for ``minimizeSPSA``), so we set ``paired=False``.
#
res = minimizeSPSA(
    cost_spsa,
    x0=init_params_spsa.copy(),
    niter=niter_spsa,
    paired=False,
    c=0.15,
    a=0.2,
    callback=callback_fn,
)

##############################################################################
# .. rst-class:: sphx-glr-script-out
#
#  Out:
#
#  .. code-block:: none
#
#     Iteration = 10, Number of device executions = 14, Cost = 0.09
#     Iteration = 20, Number of device executions = 29, Cost = -0.638
#     Iteration = 30, Number of device executions = 44, Cost = -0.842
#     Iteration = 40, Number of device executions = 59, Cost = -0.926
#     Iteration = 50, Number of device executions = 74, Cost = -0.938
#     Iteration = 60, Number of device executions = 89, Cost = -0.94
#     Iteration = 70, Number of device executions = 104, Cost = -0.962
#     Iteration = 80, Number of device executions = 119, Cost = -0.938
#     Iteration = 90, Number of device executions = 134, Cost = -0.946
#     Iteration = 100, Number of device executions = 149, Cost = -0.966
#     Iteration = 110, Number of device executions = 164, Cost = -0.954
#     Iteration = 120, Number of device executions = 179, Cost = -0.964
#     Iteration = 130, Number of device executions = 194, Cost = -0.952
#     Iteration = 140, Number of device executions = 209, Cost = -0.958
#     Iteration = 150, Number of device executions = 224, Cost = -0.968
#     Iteration = 160, Number of device executions = 239, Cost = -0.948
#     Iteration = 170, Number of device executions = 254, Cost = -0.974
#     Iteration = 180, Number of device executions = 269, Cost = -0.962
#     Iteration = 190, Number of device executions = 284, Cost = -0.988
#     Iteration = 200, Number of device executions = 299, Cost = -0.964

##############################################################################
#
# Now let's perform the same optimization using gradient descent. We set the
# step size according to a favourable value found after grid search for fast
# convergence. Note that we also create a new device in order to reset the execution count to 0. 

opt = qml.GradientDescentOptimizer(stepsize=0.3)

# Create a device, qnode and cost function specific to gradient descent
dev_sampler_gd = qml.device("qiskit.aer", wires=num_wires, shots=1000)
qnode_gd = qml.QNode(circuit, dev_sampler_gd)


def cost_gd(params):
    return qnode_gd(params)


steps = 20
params = init_params.copy()

device_execs_grad = [0]
cost_store_grad = []

for k in range(steps):
    params, val = opt.step_and_cost(cost_gd, params)
    device_execs_grad.append(dev_sampler_gd.num_executions)
    cost_store_grad.append(val)
    print(
        f"Iteration = {k}, "
        f"Number of device executions = {dev_sampler_gd.num_executions}, "
        f"Cost = {val}"
    )

# The step_and_cost function gives us the cost at the previous step, so to find
# the cost at the final parameter values we have to compute it manually
cost_store_grad.append(cost_gd(params))

##############################################################################
# .. rst-class:: sphx-glr-script-out
#
#  Out:
#
#  .. code-block:: none
#
#     Iteration = 0, Number of device executions = 121, Cost = 0.904
#     Iteration = 1, Number of device executions = 242, Cost = 0.758
#     Iteration = 2, Number of device executions = 363, Cost = 0.284
#     Iteration = 3, Number of device executions = 484, Cost = -0.416
#     Iteration = 4, Number of device executions = 605, Cost = -0.836
#     Iteration = 5, Number of device executions = 726, Cost = -0.964
#     Iteration = 6, Number of device executions = 847, Cost = -0.992
#     Iteration = 7, Number of device executions = 968, Cost = -0.994
#     Iteration = 8, Number of device executions = 1089, Cost = -0.992
#     Iteration = 9, Number of device executions = 1210, Cost = -0.994
#     Iteration = 10, Number of device executions = 1331, Cost = -0.998
#     Iteration = 11, Number of device executions = 1452, Cost = -0.992
#     Iteration = 12, Number of device executions = 1573, Cost = -0.994
#     Iteration = 13, Number of device executions = 1694, Cost = -1.0
#     Iteration = 14, Number of device executions = 1815, Cost = -0.996
#     Iteration = 15, Number of device executions = 1936, Cost = -0.996
#     Iteration = 16, Number of device executions = 2057, Cost = -0.998
#     Iteration = 17, Number of device executions = 2178, Cost = -0.996
#     Iteration = 18, Number of device executions = 2299, Cost = -0.996
#     Iteration = 19, Number of device executions = 2420, Cost = -0.996
#

##############################################################################
# SPSA and gradient descent comparison
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# At this point, nothing else remains but to check which of these approaches did
# better!
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

plt.plot(device_execs_grad, cost_store_grad, label="Gradient descent")
plt.plot(device_execs_spsa, cost_store_spsa, label="SPSA")

plt.xlabel("Number of device executions", fontsize=14)
plt.ylabel("Cost function value", fontsize=14)
plt.grid()

plt.title("Gradient descent vs. SPSA for simple optimization", fontsize=16)
plt.legend(fontsize=14)
plt.show()

##############################################################################
#
# .. figure:: ../demonstrations/spsa/first_comparison.png
#     :align: center
#     :width: 75%
#
# It seems that SPSA performs great and it does so with a significant savings when
# compared to gradient descent!
#
# Let's take a deeper dive to see how much better it actually is.
#
grad_desc_exec_min = device_execs_grad[np.argmin(cost_store_grad)]
spsa_exec_min = device_execs_spsa[np.argmin(cost_store_spsa)]
print(f"Device execution ratio: {grad_desc_exec_min/spsa_exec_min}.")

##############################################################################
# .. rst-class:: sphx-glr-script-out
#
#  Out:
#
#  .. code-block:: none
#
#     Device execution ratio: 4.161375661375661
#
# This means that SPSA can potentially find the minimum of a cost function by
# using over 4 times fewer device executions than gradient descent! That's a huge
# saving, especially in cases such as running on actual quantum hardware.

##############################################################################
# SPSA and the variational quantum eigensolver
# --------------------------------------------
#
# Now that we've explored the theoretical underpinnings of SPSA, let's use it
# to optimize a real chemical system, that of the hydrogen molecule :math:`H_2`.
# This molecule was studied previously in the `introductory variational quantum
# eigensolver (VQE) demo </demos/tutorial_vqe>`_, and so we will reuse some of
# that machinery below to set up the problem.
#

from pennylane import qchem

symbols = ["H", "H"]
coordinates = np.array([0.0, 0.0, -0.6614, 0.0, 0.0, 0.6614])
h2_ham, num_qubits = qchem.molecular_hamiltonian(symbols, coordinates)

# Variational ansatz for H_2 - see Intro VQE demo for more details
def circuit(params, wires):
    qml.BasisState(np.array([1, 1, 0, 0]), wires=wires)
    for i in wires:
        qml.Rot(*params[i], wires=i)
    qml.CNOT(wires=[2, 3])
    qml.CNOT(wires=[2, 0])
    qml.CNOT(wires=[3, 1])


##############################################################################
#
# The :math:`H_2` Hamiltonian uses 4 qubits, contains 15 terms, and has a ground
# state energy of :math:`-1.136189454088` Hartree.
#
# Since SPSA is robust to noise, let's see how it fares compared to gradient
# descent when run on noisy hardware. For this, we will set up and use a simulated
# version of IBM Q's Melbourne hardware.
#

from qiskit import IBMQ
from qiskit.providers.aer import noise

# Note: you will need to be authenticated to IBMQ to run the following code.
# Do not run the simulation on this device, as it will send it to real hardware
# For access to IBMQ, the following statements will be useful:
# IBMQ.save_account(TOKEN)
# IBMQ.load_account() # Load account from disk
# List the providers to pick an available backend:
# IBMQ.providers()    # List all available providers

dev_melbourne = qml.device(
    "qiskit.ibmq", wires=num_qubits, backend="ibmq_16_melbourne"
)
noise_model = noise.NoiseModel.from_backend(dev_melbourne.backend.properties())
dev_noisy = qml.device(
    "qiskit.aer", wires=dev_melbourne.num_wires, shots=1000, noise_model=noise_model
)

def exp_val_circuit(params):
    circuit(params, range(dev_melbourne.num_wires))
    return qml.expval(h2_ham)

# Initialize the optimizer - optimal step size was found through a grid search
opt = qml.GradientDescentOptimizer(stepsize=2.2)
cost = qml.QNode(exp_val_circuit, dev_noisy)

# This random seed was used in the original VQE demo and is known to allow the
# algorithm to converge to the global minimum.
np.random.seed(0)
init_params = np.random.normal(0, np.pi, (num_qubits, 3))
params = init_params.copy()

h2_grad_device_executions_melbourne = [0]
h2_grad_energies_melbourne = []

max_iterations = 20

# Run the gradient descent algorithm
for n in range(max_iterations):
    params, energy = opt.step_and_cost(cost, params)
    h2_grad_device_executions_melbourne.append(dev_noisy.num_executions)
    h2_grad_energies_melbourne.append(energy)

    if n % 5 == 0:
        print(
            f"Iteration = {n}, "
            f"Number of device executions = {dev_noisy.num_executions},  "
            f"Energy = {energy:.8f} Ha"
        )

h2_grad_energies_melbourne.append(cost(params))

true_energy = -1.136189454088

print()
print(f"Final estimated value of the ground-state energy = {energy:.8f} Ha")
print(
    f"Accuracy with respect to the true energy: {np.abs(energy - true_energy):.8f} Ha"
)

##############################################################################
# .. rst-class:: sphx-glr-script-out
#
#  Out:
#
#  .. code-block:: none
#
#     Iteration = 0, Number of device executions = 333,  Energy = -0.66345346 Ha
#     Iteration = 5, Number of device executions = 1998,  Energy = -0.99124272 Ha
#     Iteration = 10, Number of device executions = 3663,  Energy = -1.00105536 Ha
#     Iteration = 15, Number of device executions = 5328,  Energy = -0.99592924 Ha
#
#     Final estimated value of the ground-state energy = -0.98134253 Ha
#     Accuracy with respect to the true energy: 0.15484692 Ha
#


plt.figure(figsize=(10, 6))

plt.plot(
    h2_grad_device_executions_melbourne,
    h2_grad_energies_melbourne,
    label="Gradient descent, Melbourne sim.",
)

plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.xlabel("Device executions", fontsize=14)
plt.ylabel("Energy (Ha)", fontsize=14)
plt.grid()

plt.axhline(y=true_energy, color="black", linestyle="dashed", label="True energy")

plt.legend(fontsize=14)

plt.title("H2 energy from the VQE using gradient descent", fontsize=16)

##############################################################################
#
# .. figure:: ../demonstrations/spsa/h2_vqe_noisy_shots_melbourne.png
#     :align: center
#     :width: 90%
#
# On noisy hardware, the energy never quite reaches its true value, no matter
# how many iterations are used. In order to reach the true value, we would have
# to incorporate error mitigation techniques.
#
# VQE with SPSA
# ^^^^^^^^^^^^^
#
# Now let's perform the same experiment using SPSA instead of the VQE.
# SPSA should use only 2 device executions per term in the expectation value.
# Since there are 15 terms, and 200 iterations, we expect 6000 total device
# executions.
dev_noisy_spsa = qml.device(
    "qiskit.aer", wires=dev_melbourne.num_wires, shots=1000, noise_model=noise_model
)
cost_spsa = qml.QNode(exp_val_circuit, dev_noisy_spsa)

# Wrapping the cost function and flattening the parameters to be compatible
# with noisyopt which assumes a flat array of input parameters
def wrapped_cost(params):
    return cost_spsa(params.reshape(num_qubits, num_params))


num_qubits = 4
num_params = 3

params = init_params.copy().reshape(num_qubits * num_params)

h2_spsa_device_executions_melbourne = [0]
h2_spsa_energies_melbourne = [wrapped_cost(params)]


def callback_fn(xk):
    cost_val = wrapped_cost(xk)
    h2_spsa_energies_melbourne.append(cost_val)

    # We have evaluated every term twice, so we need to make up for this
    num_executions = int(dev_noisy_spsa.num_executions / 2)
    h2_spsa_device_executions_melbourne.append(num_executions)

    iteration_num = len(h2_spsa_energies_melbourne)
    if iteration_num % 10 == 0:
        print(
            f"Iteration = {iteration_num}, "
            f"Number of device executions = {num_executions},  "
            f"Energy = {cost_val:.8f} Ha"
        )


res = minimizeSPSA(
    # Hyperparameters chosen based on grid search
    wrapped_cost,
    x0=params,
    niter=niter_spsa,
    paired=False,
    c=0.3,
    a=1.5,
    callback=callback_fn,
)

print()
print(f"Final estimated value of the ground-state energy = {energy:.8f} Ha")
print(
    f"Accuracy with respect to the true energy: {np.abs(energy - true_energy):.8f} Ha"
)

##############################################################################
# .. rst-class:: sphx-glr-script-out
#
#  Out:
#
#  .. code-block:: none
#
#     Iteration = 10, Number of device executions = 210,  Energy = -0.93065488 Ha
#     Iteration = 20, Number of device executions = 435,  Energy = -0.97890496 Ha
#     Iteration = 30, Number of device executions = 660,  Energy = -0.96639933 Ha
#     Iteration = 40, Number of device executions = 885,  Energy = -0.96915750 Ha
#     Iteration = 50, Number of device executions = 1110,  Energy = -0.96290227 Ha
#     Iteration = 60, Number of device executions = 1335,  Energy = -0.98274165 Ha
#     Iteration = 70, Number of device executions = 1560,  Energy = -0.98002812 Ha
#     Iteration = 80, Number of device executions = 1785,  Energy = -0.98027459 Ha
#     Iteration = 90, Number of device executions = 2010,  Energy = -0.99295116 Ha
#     Iteration = 100, Number of device executions = 2235,  Energy = -0.96745352 Ha
#     Iteration = 110, Number of device executions = 2460,  Energy = -0.96522842 Ha
#     Iteration = 120, Number of device executions = 2685,  Energy = -0.98482781 Ha
#     Iteration = 130, Number of device executions = 2910,  Energy = -0.98701641 Ha
#     Iteration = 140, Number of device executions = 3135,  Energy = -0.97656477 Ha
#     Iteration = 150, Number of device executions = 3360,  Energy = -0.98735587 Ha
#     Iteration = 160, Number of device executions = 3585,  Energy = -0.98969587 Ha
#     Iteration = 170, Number of device executions = 3810,  Energy = -0.96972110 Ha
#     Iteration = 180, Number of device executions = 4035,  Energy = -0.98354804 Ha
#     Iteration = 190, Number of device executions = 4260,  Energy = -0.96640637 Ha
#     Iteration = 200, Number of device executions = 4485,  Energy = -0.98526135 Ha
#
#     Final estimated value of the ground-state energy = -0.98134253 Ha
#     Accuracy with respect to the true energy: 0.15484692 Ha
#

plt.figure(figsize=(10, 6))

plt.plot(
    h2_grad_device_executions_melbourne,
    h2_grad_energies_melbourne,
    label="Gradient descent, Melbourne sim.",
)

plt.plot(
    h2_spsa_device_executions_melbourne,
    h2_spsa_energies_melbourne,
    label="SPSA, Melbourne sim.",
)

plt.title("H2 energy from the VQE using gradient descent vs. SPSA", fontsize=16)
plt.xlabel("Number of device executions", fontsize=14)
plt.ylabel("Energy (Ha)", fontsize=14)
plt.grid()

plt.legend(fontsize=14)
plt.show()

##############################################################################
#
# .. figure:: ../demonstrations/spsa/h2_vqe_noisy_spsa.png
#     :align: center
#     :width: 90%
#
# We observe here that the SPSA optimizer again converges in fewer device
# executions than the gradient descent optimizer. 🎉
#
# Due to the (simulated) hardware noise, however, the obtained energies are
# higher than the true energy, and the output still bounces around (in SPSA
# this is expected due to the inherently stochastic nature of the algorithm).
#
#

##############################################################################
# Conclusion
# ----------
#
# SPSA is a useful optimization technique that may be particularly beneficial on
# near-term quantum hardware. It uses significantly fewer iterations to achieve
# comparable result quality as gradient-based methods, giving it the potential
# to save time and resources. It can be a good alternative to
# gradient-based methods when the optimization problem involves executing
# quantum circuits with many free parameters.
#
# There are also extensions to SPSA that could be interesting to explore in
# this context. One, in particular, uses an adaptive technique to approximate
# the *Hessian* matrix during optimization to effectively increase the
# convergence rate of SPSA [#spall_overview]_. The proposed technique can also
# be applied in cases where there is direct access to the gradient of the cost
# function.
#
#

##############################################################################
# References
# ----------
#
# .. [#spall_overview]
#
#    James C. Spall, "`An Overview of the Simultaneous Perturbation Method
#    for Efficient Optimization <https://www.jhuapl.edu/SPSA/PDF-SPSA/Spall_An_Overview.PDF>`__", 1998
#
# .. [#spall_implementation]
#
#    J. C. Spall, "Implementation of the simultaneous perturbation algorithm
#    for stochastic optimization," in IEEE Transactions on Aerospace and
#    Electronic Systems, vol. 34, no. 3, pp. 817-823, July 1998, doi:
#    10.1109/7.705889.
#
# .. [#spall_hessian]
#
#    J. C. Spall, "Adaptive stochastic approximation by the simultaneous
#    perturbation method," in IEEE Transactions on Automatic Control,
#    vol. 45, no. 10, pp. 1839-1853, Oct 2020, doi:
#    10.1109/TAC.2000.880982.
#
#
#
