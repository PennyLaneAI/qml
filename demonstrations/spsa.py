r""".. _spsa:

Optimization using SPSA
=======================

.. meta::
    :property="og:description": Use the simultaneous perturbation stochastic
        approximation algorithm to optimize variational circuits in PennyLane.
    :property="og:image": https://pennylane.ai/qml/_images/pes_h2.png

.. related::

   tutorial_vqe A brief overview of VQE
   tutorial_vqe_qng Accelerating VQE with the QNG

*Author: PennyLane dev team. Posted: 19 Mar 2021. Last updated: 19 Mar 2021.*

In this tutorial, we investigate using a gradient-free optimizer called
Simultaneous Perturbation Stochastic Approximation (SPSA) to optimize quantum
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
<https://pennylane.ai/qml/glossary/parameter_shift.html>`_.  For quantum
circuits that have multiple free parameters, using these parameter-shift rules
to compute quantum gradients involves computing the partial derivatives of the
quantum function with respect to every free parameter. These partial
derivatives are then used to apply the product rule when computing the quantum
gradient. For qubit operations that are generated by one of the Pauli matrices,
each partial derivative computation will involve two quantum circuit
evaluations with a positive and a negative shift in the parameter values.

As there are two circuit evaluations for each free parameter, the number of
overall quantum circuit executions for computing a quantum gradient scales
linearly with the number of free parameters, i.e., :math:`O(p)` with :math:`p` being
the number of free parameters. This scaling can be very costly for optimization
tasks with many free parameters. For the overall optimization this scaling means
we need :math:`O(pn)` quantum circuit evaluations, where :math:`n` is the number of
optimization steps taken.

Fortunately, there are certain optimization techniques that offer an
alternative to computing the gradients of quantum circuits. One such technique
is called the Simultaneous Perturbation Stochastic Approximation (SPSA)
algorithm. SPSA is an optimization method that involves *approximating* the
gradient of the cost function at each iteration step. This technique requires
only two quantum circuit executions per iteration step, regardless of the
number of free parameters. Therefore the overall number of circuit executions
would be :math:`O(n')` where :math:`n'` is the number of optimization steps
taken when using SPSA. This technique is also considered robust against noise,
making it a great optimization method in the NISQ era.

In this demo, you'll learn how the SPSA algorithm works, and how to apply it in
PennyLane to compute gradients of quantum circuits. You'll also see it in action
using noisy quantum data!

Simultaneous perturbation stochastic approximation (SPSA)
---------------------------------------------------------

SPSA is a general method for minimizing differentiable multivariate functions.
It is particularly useful for functions for which evaluating the gradient is not
possible, or too resource intensive. SPSA provides a stochastic method for approximating the gradient of a
multivariate differentiable cost function. To accomplish this the cost function
is evaluated twice using perturbed parameter vectors: every component of the
original parameter vector is simultaneously shifted with a randomly generated
value. This is in contrast to finite-differences methods where for each
evaluation only one component of the parameter vector is shifted at a time.

Similar to gradient-based approaches such as gradient descent, SPSA is an
iterative optimization algorithm. Let's consider a differentiable cost function
:math:`L(\theta)` where :math:`\theta` is a :math:`p`-dimensional vector and
where the optimization problem can be translated into finding a :math:`\theta^*`
at which :math:`\frac{\partial L}{\partial \theta} = 0`.  It is assumed that
measurements of :math:`L(\theta)` are available at various values of
:math:`\theta` --- this is exactly the problem that we'd consider when optimizing
quantum functions!

Just like with gradient-based methods, SPSA starts with an initial parameter
vector :math:`\hat{\theta}_{0}`. After :math:`k` iterations, the :math:`(k+1)` th
parameter iterates can be obtained as

.. math:: \hat{\theta}_{k+1} = \hat{\theta}_{k} - a_{k}\hat{g}_{k}(\hat{\theta}_{k}),

where :math:`\hat{g}_{k}` is the estimate of the gradient :math:`g(u) = \frac{
\partial L}{\partial \theta}` at the iterate :math:`\hat{\theta}_{k}` based on
prior measurements of the cost function, and :math:`a_{k}` is a positive number.

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

.. figure:: ../demonstrations/spsa/spsa_opt.png
   :align: center
   :width: 60%

   ..

   A schematic of the search paths used by gradient descent with
   parameter-shift and SPSA in a low-noise setting.
   Image source: [#spall_overview]_

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

    Just with other PennyLane devices, the number of samples taken for a device
    execution can be specified using the ``shots`` keyword argument of the
    device.

Once we have a device selected, we just need a couple of other ingredients for
the pieces of an example optimization to come together:

* a circuit ansatz: :func:`~.pennylane.templates.layers.StronglyEntanglingLayers`,
* initial parameters: conveniently generated using :func:`~.pennylane.init.strong_ent_layers_normal`,
* an observable: :math:`\bigotimes_{i=0}^{N-1}\sigma_z^i`, where :math:`N` stands
  for the number of qubits.

"""
import pennylane as qml
import numpy as np

num_wires = 4
num_layers = 5

dev_sampler = qml.device("qiskit.aer", wires=num_wires, shots=1000)

##############################################################################
# We seed so that we can simulate the same circuit every time.
np.random.seed(50)

all_pauliz_tensor_prod = qml.operation.Tensor(*[qml.PauliZ(i) for i in range(num_wires)])

@qml.qnode(dev_sampler)
def circuit(params):
    qml.templates.StronglyEntanglingLayers(params, wires=list(range(num_wires)))
    return qml.expval(all_pauliz_tensor_prod)

##############################################################################
# After this, we'll initialize the parameters in a way that is compatible with
# the ``noisyopt`` package. The ``noisyopt`` package requires the trainable parameters
# be a flattened array. As a result, our cost function must accept a flat array of parameters
# to be optimized.
flat_shape = num_layers * num_wires * 3
init_params = qml.init.strong_ent_layers_normal(
    n_wires=num_wires, n_layers=num_layers
).reshape(flat_shape)

def cost(params):
    return circuit(params.reshape(num_layers, num_wires, 3))

##############################################################################
# Once we have defined each piece of the optimization, there's only one
# remaining component required for the optimization: the *SPSA optimizer*.
# We'll use the SPSA optimizer provided by the ``noisyopt`` package. Once
# imported, we can initialize parts of the optimization such as the number of
# iterations, a collection to store the cost values, and a callback function.
# Once the optimization has concluded, we save the number of device executions
# required for completion using the callback function. This will be an
# interesting quantity!
from noisyopt import minimizeSPSA

niter_spsa = 200

# Evaluate the initial cost
cost_store_spsa = [cost(init_params)]
device_execs_spsa = [0]
dev_sampler._num_executions = 0

def callback_fn(xk):
    cost_val = cost(xk)
    cost_store_spsa.append(cost_val)

    # We've evaluated the cost function, let's make up for that
    dev_sampler._num_executions -= 1
    device_execs_spsa.append(dev_sampler.num_executions)

    iteration_num = len(cost_store_spsa)
    if iteration_num % 10 == 0:
        print(f"Iteration = {iteration_num}, Cost = {cost_val}")

##############################################################################
# Choosing the hyperparameters
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The ``noisyopt`` package allows us to choose the initial value of two
# hyperparameters for SPSA: the :math:`c` and :math:`a` coefficients. Recall
# from above that the :math:`c` values control the amount of random shift when
# evaluating the cost function, while the :math:`a` coefficients are analogous to a learning
# rate and affects the degree to which the parameters change at each update
# step.
#
# With stochastic approximation, specifying such hyperparameters significantly
# influences the convergence of the optimization for a given problem. Although
# there is no universal recipe for selecting these values (as they are greatly
# dependent on the specific problem), [#spall_implementation]_ includes
# guidelines for the selection. In our case, the initial values for :math:`c`
# and :math:`a` were selected as a result of a grid search to ensure a fast
# convergence.
# We further note that apart from :math:`c` and :math:`a`, there are further
# coefficients that are initialized in the noisyopt package using the
# previously mentioned guidelines.
#
# Our cost function does not take a seed as a keyword argument (which would be
# the default behaviour for ``minimizeSPSA``), so we set ``paired=False``.
#
res = minimizeSPSA(
    cost,
    x0=init_params.copy(),
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
#     Iteration = 10, Cost = 0.09
#     Iteration = 20, Cost = -0.638
#     Iteration = 30, Cost = -0.842
#     Iteration = 40, Cost = -0.926
#     Iteration = 50, Cost = -0.938
#     Iteration = 60, Cost = -0.94
#     Iteration = 70, Cost = -0.962
#     Iteration = 80, Cost = -0.938
#     Iteration = 90, Cost = -0.946
#     Iteration = 100, Cost = -0.966
#     Iteration = 110, Cost = -0.954
#     Iteration = 120, Cost = -0.964
#     Iteration = 130, Cost = -0.952
#     Iteration = 140, Cost = -0.958
#     Iteration = 150, Cost = -0.968
#     Iteration = 160, Cost = -0.948
#     Iteration = 170, Cost = -0.974
#     Iteration = 180, Cost = -0.962
#     Iteration = 190, Cost = -0.988
#     Iteration = 200, Cost = -0.964

##############################################################################
#
# Now let's perform the same optimization using gradient descent. We set the
# step size according to a favourable value found after grid search for fast
# convergence. Note that we also reset the number of executions of the device.

opt = qml.GradientDescentOptimizer(stepsize=0.3)

steps = 20
params = init_params.copy()

device_execs_grad = [0]
cost_store_grad = []

# Reset the number of executions of the device
dev_sampler._num_executions = 0

for k in range(steps):
    params, val = opt.step_and_cost(cost, params)
    device_execs_grad.append(dev_sampler.num_executions)
    cost_store_grad.append(val)
    print(f"Iteration = {k}, Cost = {val}")

# The step_and_cost function gives us the cost at the previous step, so to find
# the cost at the final parameter values we have to compute it manually
cost_store_grad.append(cost(params))

##############################################################################
# .. rst-class:: sphx-glr-script-out
#
#  Out:
#
#  .. code-block:: none
#
#     Iteration = 0, Cost = 0.904
#     Iteration = 1, Cost = 0.758
#     Iteration = 2, Cost = 0.284
#     Iteration = 3, Cost = -0.416
#     Iteration = 4, Cost = -0.836
#     Iteration = 5, Cost = -0.964
#     Iteration = 6, Cost = -0.992
#     Iteration = 7, Cost = -0.994
#     Iteration = 8, Cost = -0.992
#     Iteration = 9, Cost = -0.994
#     Iteration = 10, Cost = -0.998
#     Iteration = 11, Cost = -0.992
#     Iteration = 12, Cost = -0.994
#     Iteration = 13, Cost = -1.0
#     Iteration = 14, Cost = -0.996
#     Iteration = 15, Cost = -0.996
#     Iteration = 16, Cost = -0.998
#     Iteration = 17, Cost = -0.996
#     Iteration = 18, Cost = -0.996
#     Iteration = 19, Cost = -0.996
#

##############################################################################
# SPSA and gradient descent comparison
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# At this point, nothing else remains, but to check which of these approaches did
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
# that machinery here to set up the problem.
#
# We'll start by loading up the :math:`H_2` Hamiltonian and taking a look.
#

from pennylane import qchem

geometry = "h2.xyz"
charge = 0
multiplicity = 1
basis_set = "sto-3g"
name = "h2"

h2_ham, num_qubits = qchem.molecular_hamiltonian(
    name,
    geometry,
    charge=charge,
    mult=multiplicity,
    basis=basis_set,
    active_electrons=2,
    active_orbitals=2,
    mapping="jordan_wigner",
)

##############################################################################
#
# This Hamiltonian uses 4 qubits and contains 15 terms. The ground state energy
# of :math:`H_2` is :math:`-1.136189454088` Hartree. As per the original demo, this
# ground state energy can be found through the VQE procedure using the following
# variational ansatz:
#
# .. figure:: ../demonstrations/spsa/h2_ansatz.svg
#     :align: center
#     :width: 40%

def circuit(params, wires):
    qml.BasisState(np.array([1, 1, 0, 0]), wires=wires)
    for i in wires:
        qml.Rot(*params[i], wires=i)
    qml.CNOT(wires=[2, 3])
    qml.CNOT(wires=[2, 0])
    qml.CNOT(wires=[3, 1])

##############################################################################
#
# Let's first approach this problem using gradient descent. We'll run the VQE
# with a cost function that computes the expectation value of each Hamiltonian
# term after running the ansatz circuit. We'll also set the maximum number of
# gradient descent iterations to 20.
#
# To set the hyperparameters, we refer back to the `intro VQE demo
# </demos/tutorial_vqe>`_, which specifies a random seed to choose variational
# parameters that will converge well. It is common to have to run the VQE
# multiple times, with different sets of initial parameters, to ensure
# convergence. If you try out the code yourself, you'll find that there is a
# local minimum around :math:`-0.47` Hartree that the gradient descent
# optimizer might get stuck in. The initial parameters used below are known to
# converge to the true minimum. Furthermore, a grid search was performed to
# find the step size which yielded the most accurate result, with the fewest
# iterations.

# Initialize the optimizer - optimal step size was found through a grid search
opt = qml.GradientDescentOptimizer(stepsize=2.2)
max_iterations = 20

# Cost function for VQE
cost = qml.ExpvalCost(circuit, h2_ham, dev_sampler)

# Initialize parameters and compute initial energy
np.random.seed(0)
init_params = np.random.normal(0, np.pi, (num_qubits, 3))
params = init_params.copy()

h2_grad_device_executions = [0]
h2_grad_energies = []

dev_sampler._num_executions = 0

for n in range(max_iterations):
    params, energy = opt.step_and_cost(cost, params)

    if n % 5 == 0:
        print(f"Iteration = {n},  Energy = {energy:.8f} Ha")

    h2_grad_device_executions.append(dev_sampler.num_executions)
    h2_grad_energies.append(energy)

# Append the final cost
h2_grad_energies.append(cost(params))

true_energy = -1.136189454088

print()
print(f"Final estimated value of the ground-state energy = {energy:.8f} Ha")
print(f"Accuracy with respect to the true energy: {np.abs(energy - true_energy):.8f} Ha")

##############################################################################
# .. rst-class:: sphx-glr-script-out
#
#  Out:
#
#  .. code-block:: none
#
#     Iteration = 0,  Energy = -0.80509768 Ha
#     Iteration = 5,  Energy = -1.12506107 Ha
#     Iteration = 10,  Energy = -1.13597945 Ha
#     Iteration = 15,  Energy = -1.13459302 Ha
#
#     Final estimated value of the ground-state energy = -1.13259650 Ha
#     Accuracy with respect to the true energy: 0.00359295 Ha
#

##############################################################################
#
# Let's plot how our optimizer fares over time.

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(h2_grad_device_executions, h2_grad_energies, label="Gradient descent")

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
# .. figure:: ../demonstrations/spsa/h2_vqe_noisy_shots.png
#     :align: center
#     :width: 90%
#

##############################################################################
#
# We can see that as the optimization progresses, we approach the true value.
# However, due to shot noise it bounces around the optimum.
#
# Of interest now are the number of device executions. Gradients need
# to be computed for each of the 4 qubit rotations with 3 parameters. The
# gradient of each parameter requires 2 evaluations, and this must be done over
# all 15 terms. Factoring in the number of iteration steps (20), we estimate the
# total number of device executions as follows:
#
max_dev_execs = params.size * len(h2_ham.terms[0]) * 2 * max_iterations
print(f"Expected device executions = {max_dev_execs}")

##############################################################################
# .. rst-class:: sphx-glr-script-out
#
#  Out:
#
#  .. code-block:: none
#
#     Expected device executions = 7200

##############################################################################
#
# Note from the plot that the total number of executions is actually less than
# this. This is because PennyLane is clever about taking the gradient, and will
# not do so in cases where there is no dependence on the parameters. For example,
# no gradients need to be computed for the Hamiltonian term that is simply `I`,
# and there may be shortcuts for other Pauli terms as well.
#
# Gradient descent on simulated hardware
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Our next step will be to run the same VQE simulation on a simulated version
# of the Melbourne hardware. The entire process and cost function remain the
# same; we'll simply swap out the device.
#

from qiskit import IBMQ
from qiskit.providers.aer import noise

# Note: you will need to be authenticated to IBMQ to run the following code.
# Do not run the simulation on this device, as it will send it to a real hardware
dev_melbourne = qml.device("qiskit.ibmq", wires=num_qubits, backend="ibmq_16_melbourne")
noise_model = noise.NoiseModel.from_backend(dev_melbourne.backend.properties())
dev_noisy = qml.device(
    "qiskit.aer", wires=dev_melbourne.num_wires, shots=1000, noise_model=noise_model
)

# Initialize the optimizer - optimal step size was found through a grid search
opt = qml.GradientDescentOptimizer(stepsize=2.2)
cost = qml.ExpvalCost(circuit, h2_ham, dev_noisy)

params = init_params.copy()

h2_grad_device_executions_melbourne = [0]
h2_grad_energies_melbourne = []

for n in range(max_iterations):
    params, energy = opt.step_and_cost(cost, params)

    if n % 5 == 0:
        print(f"Iteration = {n},  Energy = {energy:.8f} Ha")

    h2_grad_device_executions_melbourne.append(dev_noisy.num_executions)
    h2_grad_energies_melbourne.append(energy)

h2_grad_energies_melbourne.append(cost(params))

print()
print(f"Final estimated value of the ground-state energy = {energy:.8f} Ha")
print(f"Accuracy with respect to the true energy: {np.abs(energy - true_energy):.8f} Ha")

##############################################################################
# .. rst-class:: sphx-glr-script-out
#
#  Out:
#
#  .. code-block:: none
#
#     Iteration = 0,  Energy = -0.63437409 Ha
#     Iteration = 5,  Energy = -0.89866763 Ha
#     Iteration = 10,  Energy = -0.90846615 Ha
#     Iteration = 15,  Energy = -0.90582867 Ha
#
#     Final estimated value of the ground-state energy = -0.93208561 Ha
#     Accuracy with respect to the true energy: 0.20410384 Ha
#

plt.figure(figsize=(10, 6))

plt.scatter(h2_grad_device_executions, h2_grad_energies, label="Gradient descent")
plt.scatter(
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
# We see a similar trend, however on the noisy hardware, the energy never quite
# reaches its true value, no matter how many iterations are used. In order to
# reach the true value, we would have to incorporate error mitigation techniques.
#

##############################################################################
# VQE with SPSA
# ^^^^^^^^^^^^^
#
# Finally, we will perform the same experiment using SPSA instead of the VQE.
# SPSA should use only 2 device executions per term in the expectation value.
# Since there are 15 terms, and 200 iterations, we expect 6000 total device
# executions.

params = init_params.copy()

h2_spsa_device_executions_melbourne = [0]
h2_spsa_energies_melbourne = [cost(params)]

dev_noisy._num_executions = 0

num_qubits = 4
num_params = 3
params = params.reshape(num_qubits * num_params)

# Wrapping the cost function to be compatible with noisyopt which assumes a
# flat array of parameters
def wrapped_cost(params):
    return cost(params.reshape(num_qubits, num_params))

def callback_fn(xk):
    cost_val = wrapped_cost(xk)
    h2_spsa_energies_melbourne.append(cost_val)

    # For this case, every term in the Hamiltonian counts towards evaluating the
    # cost function, so to take this into account we need to subtract the number of terms
    dev_noisy._num_executions -= len(h2_ham.terms[0])
    h2_spsa_device_executions_melbourne.append(dev_noisy.num_executions)

    iteration_num = len(h2_spsa_energies_melbourne)
    if iteration_num % 10 == 0:
        print(f"Iteration = {iteration_num},  Energy = {cost_val:.8f} Ha")

res = minimizeSPSA(
    # Hyperparameters chosen based on grid search
    wrapped_cost, x0=params, niter=niter_spsa, paired=False, c=0.3, a=1.5, callback=callback_fn
)

print()
print(f"Final estimated value of the ground-state energy = {energy:.8f} Ha")
print(f"Accuracy with respect to the true energy: {np.abs(energy - true_energy):.8f} Ha")

##############################################################################
# .. rst-class:: sphx-glr-script-out
#
#  Out:
#
#  .. code-block:: none
#
#     Iteration = 10,  Energy = -0.81263091 Ha
#     Iteration = 20,  Energy = -0.87451933 Ha
#     Iteration = 30,  Energy = -0.91126159 Ha
#     Iteration = 40,  Energy = -0.92543558 Ha
#     Iteration = 50,  Energy = -0.91074332 Ha
#     Iteration = 60,  Energy = -0.88884624 Ha
#     Iteration = 70,  Energy = -0.89771656 Ha
#     Iteration = 80,  Energy = -0.88451027 Ha
#     Iteration = 90,  Energy = -0.90159613 Ha
#     Iteration = 100,  Energy = -0.87074395 Ha
#     Iteration = 110,  Energy = -0.88314750 Ha
#     Iteration = 120,  Energy = -0.90602021 Ha
#     Iteration = 130,  Energy = -0.91128931 Ha
#     Iteration = 140,  Energy = -0.92292835 Ha
#     Iteration = 150,  Energy = -0.92499790 Ha
#     Iteration = 160,  Energy = -0.91601173 Ha
#     Iteration = 170,  Energy = -0.89362510 Ha
#     Iteration = 180,  Energy = -0.92450527 Ha
#     Iteration = 190,  Energy = -0.89094628 Ha
#     Iteration = 200,  Energy = -0.88564296 Ha
#
#     Final estimated value of the ground-state energy = -0.93208561 Ha
#     Accuracy with respect to the true energy: 0.20410384 Ha
#

plt.figure(figsize=(10, 6))

plt.plot(h2_grad_device_executions, h2_grad_energies, label="Gradient descent")
plt.plot(h2_grad_device_executions_melbourne, h2_grad_energies_melbourne, label="Gradient descent, Melbourne sim.")
plt.plot(h2_spsa_device_executions_melbourne, h2_spsa_energies_melbourne, label="SPSA, Melbourne sim.")

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
#     :width: 70%
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
