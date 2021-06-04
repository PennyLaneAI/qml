r"""

.. _quantum_analytic_descent:

Quantum analytic descent
============================

.. meta::
    :property="og:description": Implement the Quantum analytic descent algorithm for VQE.
    :property="og:image": https://pennylane.ai/qml/_images/flowchart.png

.. related::

   tutorial_vqe A brief overview of VQE
   tutorial_quantum_natural_gradient Quantum Natural Gradient (QNG)
   tutorial_rotoselect Leveraging trigonometry with Rotoselect
   tutorial_stochastic_parameter_shift Obtaining gradients stochastically


*Authors: Elies Gil-Fuster, David Wierichs. Posted: ?? May 2021.*

One of the main problems of many-body physics is that of finding the ground
state and ground state energy of a given Hamiltonian.
The Variational Quantum Eigensolver (VQE) combines a smart circuit
design with a gradient-based optimization to solve this task
(take a look at the `overview demo <https://pennylane.ai/qml/demos/tutorial_vqe.html>`_ for more details).
Several practical demonstrations have pointed out how near-term quantum
devices may be well-suited platforms for VQE and other variational quantum algorithms.
One issue for such an approach is, though, that the optimization landscape is
non-convex, so reaching a good enough local minimum quickly requires hundreds or
Using near-term devices for VQE becomes a non-convex optimization problem, which requires computing gradients at hundreds or thousands of update steps.
Computing gradients on a quantum computer carries the caveat of inefficiency when it comes to circuits with many parameters.

At the same time, though, we do have a good understanding of the local shape
of the cost landscape around any reference point.
Cashing in on this, the authors of the `Quantum Analytic Descent paper <https://arxiv.org/pdf/2008.13774.pdf>`_ propose and algorithm to construct a
classical model to approximate the landscape, so that the gradients can be calculated on a classical computer, which is much
cheaper.
In order to build the classical model, we need to use the quantum device to estimate the cost values on (a) a reference point :math:`\theta_0` and (b) a number of points in the vicinity of the reference one (more on that later).
With the cost values at these points, we can build the classical model which
approximates the landscape.

Here we demonstrate how to implement Quantum Analytic Descent on a quantum
computer using PennyLane.
Inbetween we will look at the constructed model and the optimization steps
carried out by the algorithm.
So: sit down, relax, and enjoy your optimization!

|

.. figure:: ../demonstrations/quantum_analytic_descent/xkcd.png
    :align: center
    :width: 50%
    :target: javascript:void(0)



VQEs give rise to trigonometric cost functions
----------------------------------------------

In a few words, what we have in mind when we talk about VQEs is a quantum circuit.
Typically, this :math:`n`-qubit quantum circuit is initialized in the :math:`|0\rangle^{\otimes n}` state (in an abuse of notation, we name it simply :math:`|0\rangle` in the rest of the demo).
The body of the circuit is populated with a *variational form* :math:`V(\boldsymbol{\theta})` -- a fixed architecture of quantum gates parametrized by an array of real-numbers :math:`\boldsymbol{\theta}\in\mathbb{R}^m`.
After the variational form, the circuit ends with a measurement of an observable :math:`\mathcal{M}`, which is also fixed at the start.

With these, our goal is to minimize the cost function :math:`E(\boldsymbol{\theta})`:

.. math:: E(\boldsymbol{\theta}) = \langle 0|V^\dagger(\boldsymbol{\theta})\mathcal{M}V(\boldsymbol{\theta})|0\rangle,

or, in short

.. math:: E(\boldsymbol{\theta}) = \operatorname{tr}[\rho(\boldsymbol{\theta})\mathcal{M}],

where :math:`\rho(\boldsymbol{\theta})=V(\boldsymbol{\theta})|0\rangle\!\langle0|V^\dagger(\boldsymbol{\theta})` is the density matrix of the quantum state after applying the variational form on :math:`|0\rangle`.

It can be seen that if the gates in the variational form which take the parameters :math:`\theta` 
are restricted to be Pauli rotations, then the cost function is a sum of multilinear
trigonometric terms in each of the parameters.
That's a scary sequence of words!
What that means is that if we look at :math:`E(\boldsymbol{\theta})` but we focus on one of the parameter values only, say :math:`\theta_i`, then we can write the functional dependence as a linear combination of three terms: :math:`1`, :math:`\sin(\theta_i)`, and :math:`\cos(\theta_i)`.

That is, for some coefficients :math:`a_i`, :math:`b_i`, and :math:`c_i` depending on all parameters but one (which we could write for instance as :math:`a_i = a_i(\theta_1, \ldots, \hat{\theta}_i, \ldots, \theta_m)`, but we don't do it everywhere for the sake of notation ease), we can write :math:`E(\boldsymbol{\theta})` as

.. math:: E(\boldsymbol{\theta}) = a_i + b_i\sin(\theta_i) + c_i\cos(\theta_i).

All parameters but :math:`\theta_i` are absorbed in the coefficients :math:`a_i`, :math:`b_i` and :math:`c_i`.
Another technique using this structure of :math:`E(\boldsymbol{\theta})` are the
Rotosolve/Rotoselect algorithms [#Rotosolve]_ for which there also is `a PennyLane demo <https://pennylane.ai/qml/demos/tutorial_rotoselect.html>`__.

Let's look at a toy example to illustrate this structure.
"""

import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# Create a device with 2 qubits.
dev = qml.device("default.qubit", wires=2)

# Define the variational form V and observable M and combine them into a QNode.
@qml.qnode(dev)
def circuit(parameters):
    qml.RX(parameters[0], wires=0)
    qml.RX(parameters[1], wires=1)
    return qml.expval(qml.PauliZ(0)@qml.PauliZ(1))

# Fix a parameter position.
parameters = np.array([3.3, .5])
# Evaluate the circuit for these parameters.
print(f"At the parameters {np.round(parameters,4)} the energy is {circuit(parameters)}")
###############################################################################
# It is this simple in PennyLane to obtain the energy of the given state
# for a specific Hamiltonian!
# Let us now look at how the energy value depends on each of the two parameters:

# Create 1D sweeps through parameter space with the other parameter fixed.
num_samples = 50
theta_func = np.linspace(0, 2*np.pi, num_samples)
C1 = [circuit(np.array([theta, .5])) for theta in theta_func]
C2 = [circuit(np.array([3.3, theta])) for theta in theta_func]

# Show the sweeps.
fig, ax = plt.subplots(1, 1, figsize=(4, 3));
ax.plot(theta_func, C1, label="$E(\\theta, 0.5)$", color='r');
ax.plot(theta_func, C2, label="$E(3.3
, \\vartheta)$", color='orange');
ax.set_xlabel("$\\vartheta$");
ax.set_ylabel("$E$");
ax.legend();
plt.tight_layout();

# Create a 2D grid and evaluate the energy on the grid points.
# We cut out a part of the landscape to increase clarity.
X, Y = np.meshgrid(theta_func, theta_func);
Z = np.zeros_like(X)
for i, t1 in enumerate(theta_func):
    for j, t2 in enumerate(theta_func):
        if (2*np.pi-t2)**2+t1**2>4:
            Z[i,j] = circuit([t1, t2])
        else:
            X[i,j] = np.nan
            Y[i,j] = np.nan
            Z[i,j] = np.nan

# Show the energy landscape on the grid.
fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"}, figsize=(4, 4));
surf = ax.plot_surface(X, Y, Z, label="$E(\\theta_1, \\theta_2)$", alpha=0.7, color='#209494');
line1 = ax.plot([parameters[1]]*num_samples, theta_func, C1,
        label="$E(\\theta_1, \\theta_2^{(0)})$", color='r', zorder=100);
line2 = ax.plot(theta_func, [parameters[0]]*num_samples, C2,
        label="$E(\\theta_1^{(0)}, \\theta_2)$", color='orange', zorder=100);
plt.tight_layout();

###############################################################################
# Of course this is an overly simplified example, but the key take-home message so far is:
# *if the variational parameters feed into Pauli rotations, the energy landscape is a 
# multilinear combination of trigonometric functions*.
# What is a good thing about trigonometric functions?
# That's right!
# We have studied them since high-school and know how their graphs look!
#
# In our overly simplified example we had to query a quantum computer for every point on the surface.
# Querying the quantum computer in this example means calling the ``circuit`` function.
# Could we have spared some computational resources?
# Well, since we know the ultimate shape the landscape is supposed to have, we should in principle be able to construct it only from a few points (much the same way two points already uniquely specify a line, or three non-aligned points specify a circle, there should be a certain fixed number of points that completely specify the loss landscape).
#
# Computing a classical model
# ---------------------------
#
# One could aim at reconstructing the cost function over the entire parameter space.
# In practice, that would be *very* expensive in number of required quantum circuit evaluations, so the cure would be roughly as bad as the sickness.
# What can we do, then?
# Well, the authors of QAD claim that building a model that is only accurate in the region close to a given reference point is already good enough!
# This makes *all* the difference.
# If we wanted to reconstruct the entire landscape, we would need to estimate around :math:`3^m` independent parameters, which would require a comparable amount of circuit evaluations.
# If, on the contrary, we are satisfied with an approximation that is cheaper to construct (polynomial in :math:`m` instead of exponential), we can borrow a page from Taylor's book.
# In the paper, the authors state that a *trigonometric expansion* up to second order is already a sound model candidate.
#
# In spirit, a trigonometric expansion and a Taylor expansion are not that different: both are linear combinations of the functions in some basis, where the coefficients of the sum take very specific values usually related to the derivatives of the function we want to approximate.
# The difference between Taylor's and a trigonometric expansion is mainly what basis of functions we take.
# In Calculus I we learnt a Taylor series in one variable :math:`x` uses the integer powers of the variable :math:`\{1, x, x^2, x^3, \ldots\}`, in short :math:`\{x^n\}_{n\in\mathbb{N}}`.
# Conversely, a trigonometric expansion uses a different base, also for one variable: :math:`\{1, \sin(x), \cos(x), \sin(2x), \cos(2x), \ldots\}`, which we could call the set of trigonometric monomials with integer frequency, or in short :math:`\{\sin(nx),\cos(nx)\}_{n\in\mathbb{N}}`.
# For higher-dimensional variables we have to take products of the trigonometric functions of each coordinate.
# Yet, since we only want to expand up to second order, the series will not get too much out of hand.
# We can group the terms we will have to deal with by orders.
# 
# #. The :math:`0`th order term (the constant in a Taylor series) is :math:`A(\boldsymbol{\theta})= \prod_{i=1}^m \cos\left(\frac{\theta_i}{2}\right)^2.
# #. The :math:`1`st order term (would be :math:`x_k` for each component :math:`k`) is :math:`B_k(\boldsymbol{\theta}) = \cos\left(\frac{\theta_k}{2}\right)\sin\left(\frac{\theta_k}{2}\right)\prod_{i\neq k} \cos\left(\frac{\theta_i}{2}\right)^2`.
# #. We distinguish the :math:`2`nd order terms in two groups, the squares (:math:`x_k^2` for each component :math:`k`), and the non-squares (:math:`x_kx_l` for each pair of *different* components :math:`(k,l)`):
#    #. For the square :math:`2`nd order terms we use the letter :math:`C` and, for every component :math:`k`, they look like :math:`C_k(\boldsymbol{\theta}) &= \sin\left(\frac{\theta_k}{2}\right)^2\prod_{i\neq k} \cos\left(\frac{\theta_i}{2}\right)^2`.
#    #. The non-square :math:`2`nd order terms depend on two indices, referring to each pair of different components :math;`(k,l)` and they have the form :math:`E^{(D)}_{kl} &= E\left(\boldsymbol{\theta}+\frac{\pi}{2}\boldsymbol{v}_k+\frac{\pi}{2}\boldsymbol{v}_l\right) + E\left(\boldsymbol{\theta}-\frac{\pi}{2}\boldsymbol{v}_k-\frac{\pi}{2}\boldsymbol{v}_l\right) - E\left(\boldsymbol{\theta}+\frac{\pi}{2}\boldsymbol{v}_k-\frac{\pi}{2}\boldsymbol{v}_l\right) - E\left(\boldsymbol{\theta}-\frac{\pi}{2}\boldsymbol{v}_k+\frac{\pi}{2}\boldsymbol{v}_l\right)`.
#
# You may have noticed all of those terms have quite some parts in common.
# Indeed, we can rewrite the longer ones in a shorter way which is a bit more decent to look at:
#
# .. math::
#
#   B_k(\boldsymbol{\theta}) &= \tan\left(\frac{\theta_k}{2}\right)A(\boldsymbol{\theta})\\
#   C_k(\boldsymbol{\theta}) &= \tan\left(\frac{\theta_k}{2}\right)^2 A(\boldsymbol{\theta})\\
#   D_{kl}(\boldsymbol{\theta}) &= \tan\left(\frac{\theta_k}{2}\right)\tan\left(\frac{\theta_l}{2}\right)A(\boldsymbol{\theta})
#
# And with that we know what type of terms we should expect to encounter in our local classical model.
# They look somewhat ugly at first, but if you stare at them long enough, with a bit of luck you will see they make some sense given their trigonometric nature.
# In any case: we know the classical model we want to construct is a linear combination of the functions :math:`\{A(\boldsymbol{theta}),B_k(\boldsymbol{theta}), C_k(\boldsymbol{theta}), D_{kl}(\boldsymbol{theta})\}` for every pair of different variable components :math:`(\theta_k,\theta_l)`.
#
# When it comes to the coefficients of the linear combination, we need to compute the derivatives of the function we are approximating with respect to each variable :math:`\partial E(\boldsymbol{theta})/\partial \theta_k`, :math:`\partial^2 E(\boldsymbol{theta})/\partial\theta_k^2`, and :math:`\partial^2 E(\boldsymbol{theta})/\partial \theta_k\partial\theta_l` just as in a Taylor expansion.
# We can name the different derivatives (including the :math:`0`th order derivative, i.e. the function itself) accordingly to how we named the terms in the series:
#
# .. math::
#
#   E^{(A)} &= E(\boldsymbol{\theta}) \\
#   E^{(B)}_k &= \frac{\partial E(\boldsymbol{\theta})}{\partial\theta_k} \\
#   E^{(C)}_k &= \frac{\partial^2 E(\boldsymbol{\theta})}{\partial\theta_k^2} \\
#   E^{(D)}_{kl} &= \frac{\partial^2 E(\boldsymbol{\theta})}{\partial\theta_k\partial\theta_l}
#
# In PennyLane, computing the gradient of a cost function with respect to an array of parameters can be easily done with a parameter-shift rule (SHOULD WE CITE ANOTHER DEMO/PAPER?).
# (DO WE WANT THE FOLLOWING LINES?)
# Under the hood, what PennyLane will compute is
#
# .. math::
#
#   E^{(A)} &= E(\boldsymbol{\theta})\\
#   E^{(B)}_k &= E\left(\boldsymbol{\theta}+\frac{\pi}{2}\boldsymbol{v}_k\right)-E\left(\boldsymbol{\theta}-\frac{\pi}{2}\boldsymbol{v}_k\right)\\
#   E^{(C)}_k &= E\left(\boldsymbol{\theta}+\pi\boldsymbol{v}_k\right)\\
#   E^{(D)}_{kl} &= E\left(\boldsymbol{\theta}+\frac{\pi}{2}\boldsymbol{v}_k+\frac{\pi}{2}\boldsymbol{v}_l\right) + E\left(\boldsymbol{\theta}-\frac{\pi}{2}\boldsymbol{v}_k-\frac{\pi}{2}\boldsymbol{v}_l\right) - E\left(\boldsymbol{\theta}+\frac{\pi}{2}\boldsymbol{v}_k-\frac{\pi}{2}\boldsymbol{v}_l\right) - E\left(\boldsymbol{\theta}-\frac{\pi}{2}\boldsymbol{v}_k+\frac{\pi}{2}\boldsymbol{v}_l\right)
#
# which is an analytical expression for the gradient based on the finite differences method.
# 
# Now, recall the true cost function, for which we want to build a local classical model:
#
# .. math:: E(\boldsymbol{\theta}) := \operatorname{tr}\left[\mathcal{M}V(\boldsymbol{\theta})|0\rangle\!\langle 0| V(\boldsymbol{\theta})^\dagger\right]
#
# which should be accurate in the vicinity of a reference point :math:`\boldsymbol{\theta}_0`.
# Bringing all the ingredients we mentioned together, we have the following gorgeous trigonometric expansion up to second order:
#
# .. math:: \hat{E}(\boldsymbol{\theta}_0+\boldsymbol{\theta}) := A(\boldsymbol{\theta}) E^{(A)} + \sum_{k=1}^m\left[B_k(\boldsymbol{\theta})E_k^{(B)} + C_k(\boldsymbol{\theta}) E_k^{(C)}\right] + \sum_{l>k}^m\left[D_{kl}(\boldsymbol{\theta}) E_{kl}^{(D)}\right].
#
# Let us now take a few moments to breath deeply and admire the entirety of it.
# On the one hand, we have the :math:`A`, :math:`B_k`, :math:`C_k`, and :math:`D_{kl}` functions, which we said would play the role of the monomials :math:`1`, :math:`x_k`, :math:`x_k^2`, and :math:`x_kx_l` in a regular Taylor expansion.
# At the same time, on the other hand we have the real-valued coefficients :math:`E_{(k(l))}^{(A/B/C/D)}`, which multiply the previous functions and are nothing but the gradients in the corresponding input components.
#
# Now we can also appreciate how many parameters our model needs.
# For an :math:`m`-dimensional input variable :math:`\boldsymbol{\theta}=(\theta_1,\ldots,\theta_m)`:
#
# #. We have :math:`1` :math:`E^{(A)}`.
# #. We have :math:`m` many :math:`E^{(B)}`'s, one for each component.
# #. We have :math:`m` many :math:`E^{(C)}`'s, also one for each component.
# #. We have :math:`(m-1)m/2` many :math:`E^{(D)}`'s, because we only need to check every pair once.
#
# So, in total, there are :math:`m^2/2 + 3m/2 + 1` parameters.
# In practice, not every parameter needs the same amount of circuit evaluations, though!
# If we look at the formulae for each of the :math:`E`'s from above, we can see that we need :math:`1\times1 + 2\times m + 1\times m + 4\times (m-1)m/2` many circuit evaluations, which amounts to a total of :math:`2m^2+m+1` circuit evaluations.
# This is already much better than the :math:`3^m` we would need if we naively tried to approximate the cost landscape exactly, without chopping after second order.
#
# Quantum Analytic Descent
# ------------------------
#
# The underlying idea we are trying to exploit here is the following:
# If we can model the cost around the reference point well enough, we will be able to find a rough estimate of where the global minimum of the landscape is.
# Granted, our model represents the true landscape less accurately the further we go away from the reference point :math:`\theta_0`, but nonetheless the global minimum *of the model* will bring us much closer to the global minimum *of the true cost* than a random walk.
# And hereby, the complete strategy:
#
# #. Set an initial reference point :math:`\boldsymbol{\theta}_0`.
# #. Build the model :math:`\hat{E}(\boldsymbol{\theta})\approx E(\boldsymbol{\theta}_0+\boldsymbol{\theta})` at this point.
# #. Find the global minimum :math:`\boldsymbol{\theta}_\text{min}` of the model.
# #. Set :math:`\boldsymbol{\theta}_\text{min}` as the new reference point :math:`\boldsymbol{\theta}_0`, go back to step 2.
# #. After convergence or a fixed number of iterations, output the last global minimum :math:`\boldsymbol{\theta}_\text{opt}=\boldsymbol{\theta}_\text{min}`.
#
# This provides an iterative strategy which will take us to a good enough solution much faster (in number of iterations) than for example regular stochastic gradient descent (SGD).
# The procedure of Quantum Analytic Descent is shown in this flowchart:
#
# .. figure:: ../demonstrations/quantum_analytic_descent/flowchart.png
#    :align: center
#    :width: 80%
#    :target: javascript:void(0)
#
# (I LEFT THIS CODE BLOCK AS IT WAS, BUT IT DOESN'T BELONG HERE.)
# Let us create a function that will take care of evaluating :math:`E` at all these shifted parameter points and combines the results to obtain the above coefficients:

from pennylane import numpy as np

def get_model_data(fun, params, *args):
    """Computes the coefficients for the classical model landscape, E_A, E_B, E_C, and E_D.
    Args:
        fun (callable): Cost function.
        params (array[float]): Parameter position at which to build the model.
        args: Additional arguments passed to the cost function.
    Returns:
        E_A (float): Coefficients E^(A)
        E_B (array[float]): Coefficients E^(B)
        E_C (array[float]): Coefficients E^(C)
        E_D (array[float]): Coefficients E^(D) (upper right triangular entries)
    """
    num_params = len(params)
    E_A = fun(params, *args)
    E_B = np.zeros(num_params)
    E_C = np.zeros(num_params)
    E_D = np.zeros((num_params, num_params))

    shifts = np.eye(num_params) * 0.5 * np.pi
    for i in range(num_params):
        E_B[i] = fun(params + shifts[i], *args) - fun(params - shifts[i], *args)
        E_C[i] = fun(params + 2 * shifts[i], *args)
        for j in range(i+1, num_params):
            E_D_tmp = [
                fun(params + shifts[i] + shifts[j], *args),
                fun(params - shifts[i] - shifts[j], *args),
                -fun(params + shifts[i] - shifts[j], *args),
                -fun(params - shifts[i] + shifts[j], *args),
            ]
            E_D[i,j] = sum(E_D_tmp)

    return E_A, E_B, E_C, E_D

###############################################################################
# (I LEFT THIS CODE BLOCK AS IT WAS, BUT IT DOESN'T BELONG HERE.)
# Let's test our brand-new function at a random parameter position!

parameters = np.random.random(2) * 2 * np.pi
print(f"Random parameters (params): {parameters}")
coeffs = get_model_data(circuit, parameters)
print(f"Coefficients at params:",
      f" E_A = {coeffs[0]}",
      f" E_B = {coeffs[1]}",
      f" E_C = {coeffs[2]}",
      f" E_D = {coeffs[3]}",
      sep='\n')

###############################################################################
# (I LEFT THIS CODE BLOCK AS IT WAS, BUT IT DOESN'T BELONG HERE.)
# Let's implement this model:

def model_cost(params, E_A, E_B, E_C, E_D):
    """Compute the model cost for given function data and relative parameters.
    Args:
        params (array[float]): Relative parameters at which to evaluate the model.
        E_A (float): Coefficients E^(A) that define the model.
        E_B (array[float]): Coefficients E^(B) that define the model.
        E_C (array[float]): Coefficients E^(C) that define the model.
        E_D (array[float]): Coefficients E^(D) that define the model.
            The lower left triangular part must be 0.
    Returns:
        cost (float): The model cost at the relative parameters.
    """
    A = np.prod(0.5 * (1+np.cos(params)))
    B_over_A = np.tan(0.5 * params)
    C_over_A = B_over_A**2
    D_over_A = np.outer(B_over_A, B_over_A)
    terms_without_A = [
        E_A,
        np.dot(E_B, B_over_A),
        np.dot(E_C, C_over_A),
        np.dot(B_over_A, E_D @ B_over_A),
    ]
    cost = A * np.sum(terms_without_A)

    return cost

###############################################################################
# Note that the signature of this function does not include the function parameters ``*args`` any longer, because they were fixed when we obtained the coefficients :math:`E^{(A)}` etc.
# In addition, the parameter input to ``model_cost`` is *relative* to the parameters at which the Quantum Analytic Descent is positioned currently.
# Let's test the classical energy function:

# Compute the circuit at parameters (This value is also stored in E_A=coeffs[0])
E_original = circuit(parameters)
# Compute the model at parameters by plugging in relative parameters 0.
E_model = model_cost(np.zeros_like(parameters), *coeffs)
print(f"The cost function at parameters:",
      f"  Model:    {E_model}",
      f"  Original: {E_original}", sep='\n')
# Check that coeffs[0] indeed is the original energy and that the model is correct at 0.
print(f"E_A and E_original are the same: {coeffs[0]==E_original}")
print(f"E_model and E_original are the same: {E_model==E_original}")

###############################################################################
# As we can see, the model works fine and reproduces the correct energy at the parameter position :math:`\boldsymbol{\theta}_0` at which we constructed it.
# When we move away from :math:`\boldsymbol{\theta}_0`, the model starts to deviate:

# Obtain a random shift away from parameters
shift = 0.1 * np.random.random(2)
print(f"Shift parameters by the vector {np.round(shift, 4)}.")
new_parameters = parameters + shift
# Compute the cost function and the model at the shifted position.
E_original = circuit(new_parameters)
E_model = model_cost(shift, *coeffs)
print(f"The cost function at parameters:",
      f"  Model:    {E_model}",
      f"  Original: {E_original}", sep='\n')
print(f"E_model and E_original are the same: {E_model==E_original}")

###############################################################################
# Let's try to visualize how sensible our approximate model is.
# We let our toy model to be 2-dimensional so we could make the following 3D-plot.

from mpl_toolkits.mplot3d import Axes3D
from itertools import chain

# We actually make the plotting a function because we will reuse it below.
def plot_cost_and_model(fun, model, parameters, shift_radius=5*np.pi/8, num_points=20):
    """
    Args:
        fun (callable): Original cost function.
        model (callable): Model cost function.
        parameters (array[float]): Parameters at which the model was built.
        shift_radius (float): Maximal shift value for each parameter.
        num_points (int): Number of points to create grid.
    """
    coords = np.linspace(-shift_radius, shift_radius, num_points)
    X, Y = np.meshgrid(coords+parameters[0], coords+parameters[1])
    # Compute the original cost function and the model on the grid.
    Z_original = np.array(
            [[fun(parameters+np.array([t1, t2])) for t2 in coords] for t1 in coords]
            )
    Z_model = np.array(
            [[model(np.array([t1, t2])) for t2 in coords] for t1 in coords]
            )
    # Prepare sampled points for plotting.
    shifts = [-np.pi/2, 0, np.pi/2]
    samples = chain.from_iterable(
        [
            [
                [parameters[0]+s2, parameters[1]+s1, fun(parameters+np.array([s1, s2]))]
            for s2 in shifts]
        for s1 in shifts]
    )
    # Display landscapes incl. sampled points and deviation.
    # Transparency parameter for landscapes.
    alpha = 0.6
    fig, ax = plt.subplots(1, 2, subplot_kw={"projection": "3d"}, figsize=(9, 4));
    green = '#209494'
    orange = '#ED7D31'
    surf = ax[0].plot_surface(X, Y, Z_original, label="Original energy",
            color=green, alpha=alpha);
    surf._facecolors2d = surf._facecolor3d
    surf._edgecolors2d = surf._edgecolor3d
    surf = ax[0].plot_surface(X, Y, Z_model, label="Model energy", color=orange, alpha=alpha);
    surf._facecolors2d = surf._facecolor3d
    surf._edgecolors2d = surf._edgecolor3d
    for sample in samples:
        ax[0].scatter(*sample, marker='d', color='r');
        ax[0].plot([sample[0]]*2, [sample[1]]*2, [np.min(Z_original), sample[2]], color='k')
    surf = ax[1].plot_surface(X, Y, Z_original-Z_model, label="Deviation",
            color=green, alpha=alpha);
    surf._facecolors2d = surf._facecolor3d
    surf._edgecolors2d = surf._edgecolor3d
    ax[0].legend();
    ax[1].legend();

# Define a mapped model that has the model coefficients fixed.
mapped_model = lambda parameters: model_cost(parameters, *coeffs)
plot_cost_and_model(circuit, mapped_model, parameters)

###############################################################################
# Here, in orange we see the true landscape, and in blue the approximate model.
# The vertical rod marks the reference point, from where we computed the trigonometric expansion.
#
# In the second plot, we see the difference between model and true landscape.
# It is noteworthy that this deviation is an order of magnitude smaller than the cost function even for the rather large radius we chose to display.
# This already hints at the value of the model.
#
# We are now in a position from where we can implement the loop between Steps 2 and 4 of our strategy.
# Indeed, for a relatively small number of iterations we should already find a low enough value.
# If we look back at the circuit we defined, we notice that we are measuring the observable
#
# .. math ::
#
#   Z\otimes X=\begin{pmatrix}
#   0 & 1 & 0 & 0 \\
#   1 & 0 & 0 & 0 \\
#   0 & 0 & 0 & -1 \\
#   0 & 0 & -1 & 0 \end{pmatrix},
#
# which has the eigenvalues :math:`1` and :math:`-1`.
# This means our function is bounded and takes values in the range :math:`[-1,1]`, so that the global minimum should be around :math:`-1` if our circuit is expressive enough.
# Let's try it and apply the full optimization:

import copy
# Set the number of iterations of steps 2 to 4
N_iter_outer = 3
# Set the number of iterations for the model optimization in step 3.
N_iter_inner = 50
# Let's get some fresh random parameters.
parameters = np.random.random(2) * 2 * np.pi
# Store the coefficients for the model and parameter positions.
past_coeffs = []
past_parameters = []
circuit_log = [circuit(parameters)]
model_logs = []
for iter_outer in range(N_iter_outer):
    # Model building phase of outer iteration - step 2.
    coeffs = get_model_data(circuit, parameters)
    # Map the function to be only depending on the parameters, not the coefficients.
    mapped_model = lambda par: model_cost(par, *coeffs)
    # Store the coefficients and parameters.
    past_coeffs.append(copy.deepcopy(coeffs))
    past_parameters.append(parameters.copy())

    # Let's see at which cost we start off
    if iter_outer==0:
        print(f"True energy at initial parameters: {np.round(coeffs[0],decimals=4)}\n")

    # Create the optimizer instance for step 3, we here choose ADAM.
    opt = qml.AdamOptimizer(0.05)
    # Recall that the parameters of the model are relative coordinates.
    # Correspondingly, we initialize at 0, not at parameters.
    relative_parameters = np.zeros_like(parameters)
    model_log = [mapped_model(relative_parameters)]
    # Some pretty-printing
    print(f"-Step {iter_outer+1}-")
    # Run the optimizer for N_iter_inner epochs - step 3.
    for iter_inner in range(N_iter_inner):
        relative_parameters = opt.step(mapped_model, relative_parameters)
        circuit_log.append(circuit(parameters+relative_parameters))
        model_log.append(mapped_model(relative_parameters))
        if (iter_inner+1)%50==0:
            E_model = mapped_model(relative_parameters)
            print(f"Epoch {iter_inner+1:4d}: Model cost = {np.round(E_model, 4)}",
                f" at relative parameters {np.round(relative_parameters, 4)}")

    # Store the relative parameters that minimize the model by adding the shift - step 4.
    parameters += relative_parameters
    # Let's check what the original cost at the updated parameters is.
    E_original = circuit(parameters)
    print(f"True energy at the minimum of the model: {E_original}")
    print(f"New reference parameters: {np.round(parameters, 4)}\n")
    model_logs.append(model_log)

###############################################################################
# This looks great! Quantum Analytic Descent found the minimum.
# Let us take a look at the intermediate models it built:

mapped_model = lambda par: model_cost(par, *past_coeffs[0])
plot_cost_and_model(circuit, mapped_model, past_parameters[0])

###############################################################################
# **Iteration 1:** We see the cost function around our (new) starting point and the model similar to the inpection above.

mapped_model = lambda par: model_cost(par, *past_coeffs[1])
plot_cost_and_model(circuit, mapped_model, past_parameters[1])

###############################################################################
# **Iteration 2:** Now we observe the model to stay closer to the original landscape. In addition, the minimum of the model is within the displayed range.

mapped_model = lambda par: model_cost(par, *past_coeffs[2])
plot_cost_and_model(circuit, mapped_model, past_parameters[2])

###############################################################################
# **Iteration 3:** Both, the model and the original cost function now show a minimum close to our parameter position, Quantum Analytic Descent converged.
# Note how the large deviations of the model close to the boundaries are not a problem at all because we only use the model in the central area,
# in which the deviation plateaus at zero.
#
# If we pay close attention to the values printed during the optimization, we can identify a curious phenomenon.
# At the last epoch within each iteration, sometimes the *model cost* goes beyond :math:`-1`.
# Could we visualize this behavior more clearly, please?

fig, ax = plt.subplots(1, 1, figsize=(6,4))
ax.plot(range(N_iter_outer*N_iter_inner+1), circuit_log, color="#209494", label="True");
for i in range(N_iter_outer):
    line, = ax.plot(range(i*N_iter_inner,(i+1)*N_iter_inner+1), model_logs[i],
             ls='--', color="#ED7D31")
    if i==0:
        line.set_label('Model')
ax.set_xlabel("epochs")
ax.set_ylabel("cost")
leg = ax.legend()

###############################################################################
# Each of the orange lines corresponds to minimizing the model constructed at a
# different reference point.
# We can now more easily appreciate the phenomenon we just described:
# towards the end of each "outer" optimization step, the model cost
# can be seen to be significantly lower than the true cost.
# Once the true cost itself approaches the absolute minimum, this means the
# model cost can overstep the allowed range.
# *Wasn't this forbidden? You guys told us the function could only take values in* :math:`[-1,1]` *(nostalgic* emoticon *time >:@)!*
# Yes, but careful!
# We said the *true cost* values are bounded, yet, that does not mean the ones of the *model* are!
# Notice also how this only happens at the first stages of analytic descent.
#
# Bringing together a few chords we have touched so far: once we fix a reference value, the further we go from it, the less accurate our model becomes.
# Thus, if we start far off from the true minimum, it can happen that our model exaggerates how steep the landscape is and then the model minimum lies lower than that of the true cost.
# We see how values exiting the allowed range of the true cost function does not have an
# impact on the overall optimization.
#
# Once we have found the model global minimum, we use the corresponding parameters as our new reference ones.
# This includes, at the first step, computing the true cost at those parameters, which we print as the "original energy at the minimum of the model" at the end of each iteration.
#
# In this demo we've seen how to implement the Quantum Analytic Descent algorithm
# using PennyLane for a toy-example of a Variational Quantum Eigensolver.
# By making extensive use of 3D-plots we have also tried to illustrate exactly
# what is going on in both the true cost landscape and the trigonometric expansion
# approximation.
# Recall we wanted to avoid working on the true landscape itself because we can
# only access it via very costly quantum computations.
# Instead, a few runs on the quantum device allowed us to construct a classical
# model on which we performed classical (cheap) optimization.
#
# And that was it! Thanks for coming to our show.
# Don't forget to fasten your seat belts on your way home! it was a pleasure
# having you here today.
#
# References
# ----------
#
# .. [#QAG]
#
#     Balint Koczor, Simon Benjamin. "Quantum Analytic Descent".
#     `arXiv preprint arXiv:2008.13774 <https://arxiv.org/abs/2008.13774>`__.
#
# .. [#Rotosolve]
#
#     Mateusz Ostaszewski, Edward Grant, Marcello Benedetti.
#     "Structure optimization for parameterized quantum circuits".
#     `arXiv preprint arXiv:1905.09692 <https://arxiv.org/abs/1905.09692>`__.
