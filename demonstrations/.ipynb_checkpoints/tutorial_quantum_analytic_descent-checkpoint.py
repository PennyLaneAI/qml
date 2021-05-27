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
   tutorial_vqe_qng Accelerating VQE with the QNG
   tutorial_rotoselect Leveraging trigonometry with Rotoselect
   tutorial_stochastic_parameter_shift Obtaining gradients stochastically


*Authors: David Wierichs, Elies Gil-Fuster (Xanadu Residents) Posted: ?? May 2021. Last updated: ?? May 2021.*

One of the main problems of Many-Body Physics is that of finding the ground 
state and ground state energy of a given Hamiltonian.
The Variational Quantum Eigensolver (VQE) algorithm combines a smart circuit 
design with a gradient based optimization to solve the ground state energy 
problem.
One issue for such an approach is, though, that the optimization landscape is 
non-convex, so reaching a good enough local minimum easily requires hundreds or 
thousands of update steps.

At the same time, though, we do have a good understanding of the local shape
of the cost landscape around any reference point.
Cashing in on this, the Quantum Analytic Descent algorithm proposes using a
local approximation to the landscape where gradient based optimisation is much
cheaper.
Indeed, this approach avoids using the quantum computer to estimate the gradient
at every single optimization step.
Instead, we can use the quantum device to estimate the cost values at a few
points close to the reference one.
With the cost values at these points, we can build a classical model which 
approximates the landscape locally, and then perform the optimization routine
solely on a classical device.

Here we demonstrate how to implement Quantum Analytic Descent on a quantum
computer using PennyLane.
Next to the code snippets we provide with the theory required to understand
what is going on at the different steps, too.
So: sit down, relax, and enjoy your optimization!

|

.. figure:: ../demonstrations/quantum_analytic_descent/xkcd_plot.png
    :align: center
    :width: 50%
    :target: javascript:void(0)


What is VQE?
------------

One of the prominent examples of Quantum Machine Learning (QML) algorithms is the so-called Variational Quantum Eigensolver (VQE).
The origin of VQE is in trying to find the ground state energy of molecules or the eigenstates of a given Hamiltonian.
Devoid of context, though, VQE is nothing but another instance of unsupervised learning for which we use a quantum device.
Here the goal is to find a configuration of parameters that minimizes a cost function; no data set, no labels.

Several practical demonstrations have pointed out how near-term quantum devices may be well-suited platforms for VQE and other variational quantum algorithms.

VQEs give rise to trigonometric cost functions
----------------------------------------------

In a few words, what we have in mind when we talk about VQEs is a quantum circuit.
Typically, this :math:`n`-qubit quantum circuit is initialized in the :math:`|0\rangle^{\otimes n}` state (in an abuse of notation, we name it simply :math:`|0\rangle` in the rest of the demo).
The body of the circuit is populated with a *variational form* :math:`V(\boldsymbol{\theta})` -- a fixed architecture of quantum gates parametrized by an array of real-numbers :math:`\boldsymbol{\theta}\in\mathbb{R}^m`.
After the variational form, the circuit ends with a measurement of an observable :math:`\mathcal{M}`, which is also fixed at the start.

With these, the common choice of cost function :math:`E(\boldsymbol{\theta})` is

.. math:: E(\boldsymbol{\theta}) = |\langle0|V^\dagger(\boldsymbol{\theta})\mathcal{M}V(\boldsymbol{\theta})|0\rangle|^2,

or, in short

.. math:: E(\boldsymbol{\theta}) = \operatorname{tr}[\rho(\boldsymbol{\theta})\mathcal{M}],

where :math:`\rho(\boldsymbol{\theta})=V(\boldsymbol{\theta})|0\rangle\!\langle0|V^\dagger(\boldsymbol{\theta})` is the density matrix of the quantum state after applying the variational form on the initial state.

It can be seen that if the variational form is composed only of Pauli gates, then the cost function is a sum of multilinear trigonometric terms in each of the parameters.
That's a scary sequence of words!
What that means is that if we look at :math:`E(\boldsymbol{\theta})` but we focus on one of the parameter values only, say :math:`\boldsymbol{\theta}_i`, then we can write the functional dependence as a linear combination of three terms: :math:`1`, :math:`\sin(\boldsymbol{\theta}_i)`, and :math:`\cos(\boldsymbol{\theta}_i)` assuming Pauli rotation gates in the circuit.

That is, for some coefficients :math:`a_i`, :math:`b_i`, and :math:`c_i` depending on all parameters but one (which we could write for instance as :math:`a_i = a_i(\boldsymbol{\theta}_1, \ldots, \hat\boldsymbol{\theta}_i, \ldots, \boldsymbol{\theta}_m)`, but we don't do it everywhere for the sake of notation ease), we can write :math:`E(\boldsymbol{\theta})` as

.. math:: E(\boldsymbol{\theta}) = a_i + b_i\sin(\boldsymbol{\theta}_i) + c_i\cos(\boldsymbol{\theta}_i).

Let's look at a toy example to illustrate this!
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
plt.plot(theta_func, C1, label="$E(\\vartheta, \\theta_2^{(0)})$", color='r');
plt.plot(theta_func, C2, label="$E(\\theta_1^{(0)}, \\vartheta)$", color='orange');
plt.xlabel("$\\vartheta$");
plt.ylabel("$E$");
plt.legend();

# Create a 2D grid and evaluate the energy on the grid points.
X, Y = np.meshgrid(theta_func, theta_func);
Z = np.array([[circuit([t1, t2]) for t2 in theta_func] for t1 in theta_func]);

# Show the energy landscape on the grid.
fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"}, figsize=(9, 4));
surf = ax.plot_surface(X, Y, Z, label="$E(\\theta_1, \\theta_2)$", alpha=0.7, color='#209494');
line1 = ax.plot([parameters[1]]*num_samples, theta_func, C1, label="$E(\\theta_1, \\theta_2^{(0)})$", color='r', zorder=100);
line2 = ax.plot(theta_func, [parameters[0]]*num_samples, C2, label="$E(\\theta_1^{(0)}, \\theta_2)$", color='orange', zorder=100);

###############################################################################
# Of course this is an overly simplified example, but the key take-home message so far is: *the parameter landscape is a multilinear combination of trigonometric functions*.
# What is a good thing about trigonometric functions?
# That's right!
# We have studied them since high-school and know how to find their minima!
#
# In our overly simplified example we had to query a quantum computer for every point on the surface.
# Could we have spared some computational resources?
# Well, since we know the ultimate shape the landscape is supposed to have, we should in principle be able to construct it only from a few points (much the same way two points already uniquely specify a line, or three non-aligned points specify a circle, there should be a certain fixed number of points that completely specify the loss landscape).
#
# Quantum Analytic Descent
# ------------------------
#
# Although in principle we should be able to reconstruct the cost function over the entire parameter space, in practice we are mostly interested in what is happening in the vicinity of a given reference point.
# This makes *all* the difference!
# If we wanted to reconstruct the entire landscape, we would need to estimate around :math:`3^m` independent parameters, which would require the same number of function evaluations!
# If, on the contrary, we are satisfied with an approximation that is cheaper to construct (polynomial in :math:`m` instead of exponential), we can borrow a page from Taylor's book!
#
# As explained in the paper, an approximation via trigonometric series up to second order is already a sound candidate!
# In particular, we want to approximate
#
# .. math:: E(\boldsymbol{\theta}) := \operatorname{tr}[\mathcal{M}\Phi(\boldsymbol{\theta})\rho_0]
#
# in the vicinity of a reference point :math:`\boldsymbol{\theta}_0`.
# Here :math:`\rho_0` is the density matrix of the initial state, and :math:`\Phi(\boldsymbol{\theta})` is the quantum channel that implements the variational form.
# We then have:
#
# .. math:: \hat{E}(\boldsymbol{\theta}_0+\boldsymbol{\theta}) := A(\boldsymbol{\theta}) E^{(A)} + \sum_{k=1}^m[B_k(\boldsymbol{\theta})E_k^{(B)} + C_k(\boldsymbol{\theta}) E_k^{(C)}] + \sum_{l>k}^m[D_{kl}(\boldsymbol{\theta}) E_{kl}^{(D)}].
#
# We have introduced a number of :math:`E`\ 's, in order to build each of these we need to sample some points in the landscape with a quantum computer.
# Before eyeing the formulas for each of them, though, it is important we now only need to estimate :math:`2m^2+m+1` points (as we will see in the model formulas further below).
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
#
# .. figure:: ../demonstrations/quantum_analytic_descent/flowchart.png
#    :align: center
#    :width: 80%
#    :target: javascript:void(0)
#
# Computing the model
# -------------------
# In order to construct the model landscape, we still need to compute all those :math:`E^{(A/B/C/D)}`\ 's from the previous equation.
# In order to do so, we evaluate the original cost function on the quantum computer at specific shifted positions in parameter space.
# (If you are familiar with the parameter shift rule for analytic quantum gradient computations, you might recognize some of the following computations.)
# We combine the function evaluations according to Eqs. (B1) and the following in [#QAG]_ and obtain
#
# .. math::
#
#   E^{(A)} &= E(\boldsymbol{\theta})\\
#   E^{(B)}_k &= E(\boldsymbol{\theta}+\frac{\pi}{2}\boldsymbol{v}_k)-E(\boldsymbol{\theta}-\frac{\pi}{2}\boldsymbol{v}_k)\\
#   E^{(C)}_k &= E(\boldsymbol{\theta}+\pi\boldsymbol{v}_k)\\
#   E^{(D)}_{kl} &= E(\boldsymbol{\theta}+\frac{\pi}{2}\boldsymbol{v}_k+\frac{\pi}{2}\boldsymbol{v}_l) + E(\boldsymbol{\theta}-\frac{\pi}{2}\boldsymbol{v}_k-\frac{\pi}{2}\boldsymbol{v}_l) - E(\boldsymbol{\theta}+\frac{\pi}{2}\boldsymbol{v}_k-\frac{\pi}{2}\boldsymbol{v}_l) - E(\boldsymbol{\theta}-\frac{\pi}{2}\boldsymbol{v}_k+\frac{\pi}{2}\boldsymbol{v}_l)
#
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
# As we can see, only the upper right triangular part of :math:`E^{(D)}` -- in this case a single entry -- was computed.
#
# Next, we want to construct our classical model that locally represents the original cost function.
# For this we use the coefficients computed by ``get_model_data`` and the definition in Eq. (A13, B1) and the following equations in [#QAG]_ respectively.
# There are 4 trigonometric functions to be combined:
#
# .. math::
#
#   A(\boldsymbol{\theta}) &= \prod_{i=1}^m \cos\left(\frac{\theta_i}{2}\right)^2\\
#   B_k(\boldsymbol{\theta}) &= \cos\left(\frac{\theta_k}{2}\right)\sin\left(\frac{\theta_k}{2}\right)\prod_{i\neq k} \cos\left(\frac{\theta_i}{2}\right)^2\\
#   C_k(\boldsymbol{\theta}) &= \sin\left(\frac{\theta_k}{2}\right)^2\prod_{i\neq k} \cos\left(\frac{\theta_i}{2}\right)^2\\
#   D_{kl}(\boldsymbol{\theta}) &= \cos\left(\frac{\theta_k}{2}\right)\sin\left(\frac{\theta_k}{2}\right)\cos\left(\frac{\theta_l}{2}\right)\sin\left(\frac{\theta_l}{2}\right)\prod_{k\neq i\neq l} \cos\left(\frac{\theta_i}{2}\right)^2
#
# While this expression for these functions seems quite long, it shows us a nice relation between the four sets of trigonometric polynomials :math:`A`, :math:`B`, :math:`C`, and :math:`D`:
#
# .. math::
#
#   B_k(\boldsymbol{\theta}) &= \tan\left(\frac{\theta_k}{2}\right)A(\boldsymbol{\theta})\\
#   C_k(\boldsymbol{\theta}) &= \tan\left(\frac{\theta_k}{2}\right)^2 A(\boldsymbol{\theta})\\
#   D_{kl}(\boldsymbol{\theta}) &= \tan\left(\frac{\theta_k}{2}\right)\tan\left(\frac{\theta_l}{2}\right)A(\boldsymbol{\theta})
#
# Using these, we can compute the classical surrogate model :math:`\tilde{E}(\boldsymbol{\theta})`:
#
# .. math::
#
#   \tilde{E}(\boldsymbol{\theta}) &= A(\theta) E^{(A)} + \sum_k B_k(\boldsymbol{\theta}) E^{(B)}_k + C_k(\boldsymbol{\theta}) E^{(C)}_k + \sum_{l>k} D_{kl}(\boldsymbol{\theta}) E^{(D)}_{kl}\\
#   \phantom{\tilde{E}(\boldsymbol{\theta})}&=A(\boldsymbol{\theta})\left[E^{(A)}+\sum_k \tan\left(\frac{\theta_k}{2}\right)E^{(B)}_k + \tan\left(\frac{\theta_k}{2}\right)^2 E^{(C)}_k + \sum_{l>k} \tan\left(\frac{\theta_k}{2}\right)\tan\left(\frac{\theta_l}{2}\right)E^{(D)}_{kl}\right]
#
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
# Here, in orange we see the true landscape, and in blue the approximate model.
# The vertical rod marks the reference point, from where we computed the trigonometric expansion.
#
# In the second plot, we see the difference between model and true landscape.
# It is noteworthy that this deviation is an order of magnitude smaller than the cost function even for the rather large radius we chose to display.
# This already hints at the value of the model.

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
    surf = ax[0].plot_surface(X, Y, Z_original, label="Original energy", color=green, alpha=alpha);
    surf._facecolors2d = surf._facecolor3d
    surf._edgecolors2d = surf._edgecolor3d
    surf = ax[0].plot_surface(X, Y, Z_model, label="Model energy", color=orange, alpha=alpha);
    surf._facecolors2d = surf._facecolor3d
    surf._edgecolors2d = surf._edgecolor3d
    for sample in samples:
        ax[0].scatter(*sample, marker='d', color='r');
        ax[0].plot([sample[0]]*2, [sample[1]]*2, [np.min(Z_original), sample[2]], color='k')
    surf = ax[1].plot_surface(X, Y, Z_original-Z_model, label="Deviation", color=green, alpha=alpha);
    surf._facecolors2d = surf._facecolor3d
    surf._edgecolors2d = surf._edgecolor3d
    ax[0].legend();
    ax[1].legend();

# Define a mapped model that has the model coefficients fixed.
mapped_model = lambda parameters: model_cost(parameters, *coeffs)
plot_cost_and_model(circuit, mapped_model, parameters)

###############################################################################
# We are now in a position from where we can implement the loop between steps 2. and 4. of our strategy.
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
    # Some pretty-printing
    print(f"-Step {iter_outer+1}-")
    # Run the optimizer for N_iter_inner epochs - step 3.
    for iter_inner in range(N_iter_inner):
        relative_parameters = opt.step(mapped_model, relative_parameters)
        if (iter_inner+1)%50==0:
            E_model = mapped_model(relative_parameters)
            print(f"Epoch {iter_inner+1:4d}: Model cost = {np.round(E_model, 4)} at relative parameters {np.round(relative_parameters, 4)}")

    # Store the relative parameters that minimize the model by adding the shift - step 4.
    parameters += relative_parameters
    # Let's check what the original cost at the updated parameters is.
    E_original = circuit(parameters)
    print(f"True energy at the minimum of the model: {E_original}")
    print(f"New reference parameters: {np.round(parameters, 4)}\n")

###############################################################################
# This looks great! Quantum Analytic Descent found the minimum.
# Let us take a look at the intermediate models it built:

mapped_model = lambda par: model_cost(par, *past_coeffs[0])
plot_cost_and_model(circuit, mapped_model, past_parameters[0])

###############################################################################
# Step 1: We again see the cost function around our starting point and the model we already inspected above.

mapped_model = lambda par: model_cost(par, *past_coeffs[1])
plot_cost_and_model(circuit, mapped_model, past_parameters[1])

###############################################################################
# Step 2: Now we observe the model to stay closer to the original landscape. In addition, the minimum of the model is within the displayed range.

mapped_model = lambda par: model_cost(par, *past_coeffs[2])
plot_cost_and_model(circuit, mapped_model, past_parameters[2])

###############################################################################
# Step 3: Both, the model and the original cost function show a minimum close to our parameter position, Quantum Analytic Descent converged.
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
# Don't forget to fasten your seat belts on your way home, it was a pleasure
# having you here today.
#
# References
# ----------
#
# .. [#QAG]
#
#     Balint Koczor, Simon Benjamin. "Quantum Analytic Descent".
#     `arXiv preprint arXiv:2008.13774 <https://arxiv.org/abs/2008.13774>`__.
