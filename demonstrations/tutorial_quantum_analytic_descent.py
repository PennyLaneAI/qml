r"""
.. _quantum_analytic_descent:

Quantum analytic descent
============================

.. meta::
    :property="og:description": Implementing the Quantum analytic descent algorithm for VQE.
    :property="og:image": https://pennylane.ai/qml/_images/margin_2.png

.. related::

   tutorial_variational_classifier Variational quantum classifier
   tutorial_data_reuploading_classifier Data-reuploading classifier

*Author: David Wierichs, Elies Gil-Fuster (Xanadu Residents) Posted: 8 May 2021. Last updated: 8 May 2021.*

In this tutorial, we show how to use the PyTorch interface for PennyLane
to implement a multiclass variational classifier. We consider the iris database
from UCI, which has 4 features and 3 classes. We use multiple one-vs-all
classifiers with a margin loss (see `Multiclass Linear SVM
<http://cs231n.github.io/linear-classify/>`__) to classify data. Each classifier is implemented
on an individual variational circuit, whose architecture is inspired by
`Farhi and Neven (2018) <https://arxiv.org/abs/1802.06002>`__ as well as
`Schuld et al. (2018) <https://arxiv.org/abs/1804.00633>`__.

|

.. figure:: ../demonstrations/multiclass_classification/margin_2.png
    :align: center
    :width: 50%
    :target: javascript:void(0)

|


What is VQE?
~~~~~~~~~~~~

One of the prominent examples of Quantum Machine Learning (QML) algorithms is the so-called Variational Quantum Eigensolver (VQE).
The origin of VQE is in trying to find the ground state energy of molecules or the eigenstates of a given Hamiltonian.
Devoid of context, though, VQE is nothing but another instance of unsupervised learning for which we use a quantum device.
Here the goal is to find a configuration of parameters that minimizes a cost function; no data set, no labels.

Several practical demonstrations have pointed out how near-term quantum devices may be well-suited platforms for VQE and VQE*-ish* algorithms.

VQEs give rise to trigonometric cost functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In a few words, what we have in mind when we talk about VQEs is a quantum circuit.
Typically, this $n$-qubit quantum circuit is initialized in the :math:`|0\rangle^{\otimes n}` state (in an abuse of notation, we name it simply :math:`|0\rangle` in the rest of the demo).
The body of the circuit is populated with a *variational form* :math:`V(\theta)` -- a fixed architecture of quantum gates parametrized by an array of real-numbers :math:`\theta\in\mathbb{R}^m`.
After the variational form, the circuit ends with a measurement of an observable :math:`\mathcal{M}`, which is also fixed at the start.

With these, the common choice of cost function :math:`E(\theta)` is

.. math:: C(\theta) = |\langle0|V^\dagger(\theta)\mathcal{M}V(\theta)|0\rangle|^2,

or, in short

.. math:: C(\theta) = \operatorname{tr}[\rho(\theta)\mathcal{M}],

where :math:`\rho(\theta)=V(\theta)|0\rangle\!\langle0|V^\dagger(\theta)` is the density matrix of the quantum state after applying the variational form on the initial state.

It can be seen that if the variational form is composed only of Pauli gates, then the cost function is a sum of multilinear trigonometric terms in each of the parameters.
That's a scary sequence of words!
What that means is, that if we look at :math:`C` but we focus on one of the parameter values only, say :math:`\theta_i`, then we can write the functional dependence as a linear combination of three terms: :math:`1`, :math:`\sin(\theta_i)`, and :math:`\cos(\theta_i)`.

That is, for some coefficients :math:`a_i`, :math:`b_i`, and :math:`c_i` depending on all parameters but one (which we could write for instance as :math:`a_i = a_i(\theta_1, \ldots, \hat\theta_i, \ldots, \theta_m)`, but we don't do it everywhere for the sake of notation ease), we can write :math:`C(\theta)` as

.. math:: C(\theta) = a_i + b_i\sin(\theta_i) + c_i\cos(\theta_i).

Let's look at a toy example to illustrate this!
"""

import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

dev = qml.device("default.qubit", wires=2)

def circuit(parameters, wires=2):
    qml.RX(parameters[0], wires=0)
    qml.RX(parameters[1], wires=1)
    return qml.expval(qml.PauliZ(0)@qml.PauliZ(1))

parameters = np.array([3.3, .5])

print(circuit([parameters]))

theta_func = np.linspace(0,2*np.pi,50)
C1 = [circuit(np.array([theta, .5])) for theta in theta_func]
C2 = [circuit(np.array([3.3, theta])) for theta in theta_func]

plt.plot(theta_func, C1);
plt.plot(theta_func, C2);

X, Y = np.meshgrid(theta_func, theta_func)
Z = np.zeros_like(X)
for i, t1 in enumerate(theta_func):
    for j, t2 in enumerate(theta_func):
        Z[i,j] = cost(np.array([t1, t2]))
fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"})
surf = ax.plot_surface(X, Y, Z)

print("The choice of parameters",theta_func[int(np.argmin(Z)/len(Z))],theta_func[np.argmin(Z)%len(Z)],"yields an energy of",np.min(Z))


#################################################################################
# Of course this is an overly simplified example, but the key take-home message so far is: *the parameter landscape is a multilinear combination of trigonometric functions*.
# What is a good thing about trigonometric functions?
# That's right!
# We have studied them since high-school and know how to find their minima!
#
# In our overly simplified example we had to query a quantum computer for every point on the surface.
# Could we have spared some computational resources?
# Well, since we know the ultimate shape the landscape is supposed to have, we should in principle be able to construct it only from a few points (much the same way two points already uniquely specify a line, or three non-aligned points specify a circle, there should be a certain fixed number of points that completely specify the loss landscape).
# 
# LINK TO PAPER AND INTUITION FOR HOW MANY POINTS EXACTLY WE NEED?
#
# Quantum Analytic Descent
# ~~~~~~~~~~~~~~~~~~~~~~~~
#
# Although in principle we should be able to reconstruct the cost function over the entire parameter space, in practice we are interested in mostly what is happening in the vecinity of a given reference point.
# This makes *all* the difference!
# If we wanted to reconstruct the entire landscape, we would need to estimate around :math:`3^m` independent parameters, which would require about that many points!
# If, on the contrary, we are satisfied with an approximation that is cheaper to construct (polynomial in :math:`m` instead of exponential), we can borrow a page from Taylor's book!
#
# As explained in the paper, an approximation via trigonometric series up to second order is already a sound candidate!
# In particular, we want to approximate
#
# .. math:: E(\theta) := \operatorname{tr}[\mathcal{M}\Phi(\theta)\rho_0]
#
# in the vecinity of a reference point :math:`\theta_0`.
# Here :math:`\rho_0` is the density matrix of the initial state, and :math:`\Phi(\theta)` is the quantum channel that implements the variational form.
# We then have: 
# 
# .. math:: \hat{E}(\theta_0+\theta) & := A(\theta) E^{(A)} + \sum_{i=1}^m[B_i(\theta)E_i^{(B)} + C_i(\theta) E_i^{(C)}] + \sum_{j>i}^m[D_{ij}(\theta) E_{ij}^{(D)}].
# 
# We have introduced a number of :math:`E`'s, we build each of these by sampling some points in the landscape with a quantum computer.
# DAVID ALREADY WROTE THE FORMULAS, RIGHT?
# Important is we only need to estimate :math:`2m^2 - 2m +1` many parameters, and thus a comparable amount of points.
# 
# The underlying idea we are trying to exploit here is the following.
# If we can model the cost around the reference point good enough, we will be able to find a rough estimate of where the global minimum *of the model* is.
# Granted, our model represents the true landscape less accurately the further we go from the reference point, BUT even then, the global minimum *of the model* will bring us much closer to the global minimum *of the true cost* than a random walk.
# Maybe by now it's already clear what the complete strategy is, but just in case: of course, once we have found the global minimum of the model, we could then use that as our new reference point, build a new model around it and find the global minimum of that new model.
# This provides an iterative strategy which will take us to a good enough solution much faster (in number of steps) than for example regular SGD.
#
# How to build the classical model using a quantum computer
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# In order to construct the model landscape, we evaluate the original cost function on the quantum computer at specific shifted positions in parameter space.
# (If you are familiar with the parameter shift rule for analytic quantum gradient computations, you might recognize some of the following computations.)
# We combine the function evaluations according to Eqs. (B1) and the following in [1][QAG paper].
# 
# ..math:: 
#
#   E^{(A)} = E(\boldsymbol{\theta})\\
#   E^{(B)}_k = E(\boldsymbol{\theta}+\frac{\pi}{2}\boldsymbol{v}_k)-E(\boldsymbol{\theta}-\frac{\pi}{2}\boldsymbol{v}_k)\\
#   E^{(C)}_k = E(\boldsymbol{\theta}+\pi\boldsymbol{v}_k)\\
#   E^{(D)}_{kl} = E(\boldsymbol{\theta}+\frac{\pi}{2}\boldsymbol{v}_k+\frac{\pi}{2}\boldsymbol{v}_l) + E(\boldsymbol{\theta}-\frac{\pi}{2}\boldsymbol{v}_k-\frac{\pi}{2}\boldsymbol{v}_l) - E(\boldsymbol{\theta}+\frac{\pi}{2}\boldsymbol{v}_k-\frac{\pi}{2}\boldsymbol{v}_l) - E(\boldsymbol{\theta}-\frac{\pi}{2}\boldsymbol{v}_k+\frac{\pi}{2}\boldsymbol{v}_l)
# 
# Let us create a function that will take care of evaluating :math:`E` at all these shifted parameter points and combines the results to obtain the above coefficients: 
#
# `QAG paper <https://arxiv.org/pdf/2008.13774.pdf>`_

try:
    from pennylane import numpy as np
except ModuleNotFoundError:
    !pip install pennylane
    from pennylane import numpy as np

def get_data_for_model(fun, params, *args):
    """Computes the coefficients for the classical model landscape, E_A, E_B, E_C, and E_D."""
    num_params = len(params)
    E_A = fun(params, *args)

    E_B = np.zeros(num_params)
    E_C = np.zeros(num_params)
    E_D = np.zeros((num_params, num_params))

    shifts = np.eye(num_params) * 0.5 * np.pi
    for k in range(num_params):
        E_B[k] = fun(params + shifts[k], *args) - fun(params - shifts[k], *args)
        E_C[k] = fun(params + 2 * shifts[k], *args)
        for l in range(k+1, num_params):
            E_D_tmp = [
                fun(params + shifts[k] + shifts[l], *args),
                fun(params - shifts[k] - shifts[l], *args),
                -fun(params + shifts[k] - shifts[l], *args),
                -fun(params - shifts[k] + shifts[l], *args),  
            ]
            E_D[k,l] = sum(E_D_tmp)

    return E_A, E_B, E_C, E_D


#################################################################################
# Next, we want to construct our classical model that locally represents the original cost function.
# For this we use the coefficients computed by `get_data_for_model` and the definition in Eq. (A13, B1) and following equations respectively.
# There are 4 trigonometric functions to be combined:
# 
# ..math::
#
#   A(\boldsymbol{\theta}) = \prod_m \frac{1+\cos(\theta_m)}{2} = \prod_m \cos\left(\frac{\theta_m}{2}\right)^2\\
#   B_k(\boldsymbol{\theta}) = \frac{\sin(\theta_k)}{2}\prod_{m\neq k} \frac{1+\cos(\theta_m)}{2} = \cos\left(\frac{\theta_k}{2}\right)\sin\left(\frac{\theta_k}{2}\right)\prod_{m\neq k} \cos\left(\frac{\theta_m}{2}\right)^2\\
#   C_k(\boldsymbol{\theta}) = \frac{1-\cos(\theta_k)}{2}\prod_{m\neq k} \frac{1+\cos(\theta_m)}{2} = \sin\left(\frac{\theta_k}{2}\right)^2\prod_{m\neq k} \cos\left(\frac{\theta_m}{2}\right)^2\\
#   D_{kl}(\boldsymbol{\theta}) = \frac{\sin(\theta_k)}{2}\frac{\sin(\theta_l)}{2}\prod_{m\neq k,l} \frac{1+\cos(\theta_m)}{2} = \cos\left(\frac{\theta_k}{2}\right)\sin\left(\frac{\theta_k}{2}\right)\cos\left(\frac{\theta_l}{2}\right)\sin\left(\frac{\theta_l}{2}\right)\prod_{m\neq k} \cos\left(\frac{\theta_m}{2}\right)^2
#
#
# While the latter formulation of these functions seems quite long, it shows us a nice relation between the four sets of trigonometric polynomials :math:`A`, :math:`B`, :math:`C`, and :math:`D`:
#
# ..math::
#
#   B_k(\boldsymbol{\theta}) = \tan\left(\frac{\theta_k}{2}\right)A(\boldsymbol{\theta})\\
#   C_k(\boldsymbol{\theta}) = \tan\left(\frac{\theta_k}{2}\right)^2 A(\boldsymbol{\theta})\\
#   D_{kl}(\boldsymbol{\theta}) = \tan\left(\frac{\theta_k}{2}\right)\tan\left(\frac{\theta_l}{2}\right)A(\boldsymbol{\theta})\\
#
#
# Using these, we can compute the classical surrogate model :math:`\tilde{E}(\boldsymbol{\theta})`:
#
# ..math::
#
#   \tilde{E}(\boldsymbol{\theta}) = A(\theta) E^{(A)} + \sum_k B_k(\boldsymbol{\theta}) E^{(B)}_k + C_k(\boldsymbol{\theta}) E^{(C)}_k + \sum_{k<l} D_{kl}(\boldsymbol{\theta}) E^{(D)}_{kl}\\
#   \phantom{\tilde{E}(\boldsymbol{\theta})}=A(\boldsymbol{\theta})\left[E^{(A)}+\sum_k \tan\left(\frac{\theta_k}{2}\right)E^{(B)}_k + \tan\left(\frac{\theta_k}{2}\right)^2 E^{(C)}_k + \sum_{k<l} \tan\left(\frac{\theta_k}{2}\right)\tan\left(\frac{\theta_l}{2}\right)E^{(D)}_{kl}\right]
#     


def model_cost(params, E_A, E_B, E_C, E_D):
    A = np.prod(0.5 * (1+np.cos(params)))
    B_over_A = np.tan(0.5 * params)
    C_over_A = B_over_A**2
    D_over_A = np.outer(B_over_A, B_over_A)
    terms_without_A = [
        E_A,
        np.dot(E_B, B_over_A),
        np.dot(E_C, C_over_A),
        np.trace(E_D @ D_over_A),                 
    ]
    cost = A * np.sum(terms_without_A)

    return cost


#################################################################################
# Note that the signature of this function does not include the function parameters `*args` any longer, because they were fixed when we obtained the coefficients $E^{(A)}$ etc.
# In addition, the parameter input to `model_cost` is _relative_ to the parameters at which Quantum Analytic Gradient descent is positioned currently.
# Let's try to use the classical energy function:


dev = qml.device('default.qubit', wires=2, shots=None)
@qml.qnode(dev)
def circuit(params):
    qml.RY(params[0], wires=[0])
    qml.CNOT(wires=[0,1])
    qml.RY(params[1], wires=[0])
    # qml.CNOT(wires=[0,1])
    return qml.expval(qml.PauliZ(0)@qml.PauliX(1))
  
num_params = 2

params = np.random.random(num_params) * 2 * np.pi - np.pi
print(f"Random parameters (params): {params}")
coeffs = get_data_for_model(circuit, params)
print(f"Coefficients at params:", 
      f" E_A = {coeffs[0]}",
      f" E_B = {coeffs[1]}",
      f" E_C = {coeffs[2]}",
      f" E_D = {coeffs[3]}",
      sep='\n')

original = circuit(params)
model = model_cost(params-params, *coeffs)
print(f"The cost function at params:", f"  Model:    {model}", f"  Original: {original}", sep='\n')

new_params = params + 0.1 * np.random.random(num_params)
print(f"New random parameters close to params: {new_params}")
original = circuit(new_params)
model = model_cost(new_params-params, *coeffs)
print(f"The cost function at new_params:", f"  Model:    {model}", f"  Original: {original}", sep='\n')


##############################################################################
# We may even take a look at the model and the original cost function as landscapes:

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define some shift values for the plotting grid, maximal shift is larger 
#  than typical parameter updates in optimizations.
shift_radius = np.pi/4
x_rel = np.linspace(-shift_radius, shift_radius, 20)
y_rel = np.linspace(-shift_radius, shift_radius, 20)
X, Y = np.meshgrid(x_rel+params[0], y_rel+params[1])

Z_model = np.zeros_like(X)
Z_orig = np.zeros_like(X)
for i, x in enumerate(x_rel):
    for j, y in enumerate(y_rel):
        Z_model[i,j] = model_cost(np.array([x, y]), *coeffs)
        Z_orig[i,j] = circuit(params+np.array([x, y]))

fig, ax = plt.subplots(2, 1, subplot_kw={"projection": "3d"}, figsize=(9, 12))
surf = ax[0].plot_surface(X, Y, Z_model, label="Model energy", alpha=0.7)
surf._facecolors2d = surf._facecolors3d
surf._edgecolors2d = surf._edgecolors3d
surf = ax[0].plot_surface(X, Y, Z_orig, label="Original energy", alpha=0.7)
surf._facecolors2d = surf._facecolors3d
surf._edgecolors2d = surf._edgecolors3d
ax[0].plot([params[0]]*2, [params[1]]*2, [np.min(Z_orig), np.max(Z_orig)], color='k')
surf = ax[1].plot_surface(X, Y, Z_orig-Z_model, label="Deviation", alpha=0.7)
surf._facecolors2d = surf._facecolors3d
surf._edgecolors2d = surf._edgecolors3d
ax[0].legend()
ax[1].legend()

###################################################################
# This looks great, we have constructed a purely classical surrogate model that locally has a small deviation from the original cost function.
# The error becomes bigger for large distances from the parameter position at which we constructed the model, but those distances are rather big compared to typical update steps in the optimization procedure.
#
# Let us now apply an optimization to the model cost function:


# Create the optimizer instance, we here choose ADAM.
opt = qml.AdamOptimizer(0.05)
# Recall that the parameters of the model are relative coordinates. Correspondingly, we initialize at 0, not at params.
trained_params = np.zeros_like(params)
print(f"Original energy at the minimum of the model: {coeffs[0]}")
# Map the function to be only depending on the parameters
mapped_model = lambda par: model_cost(par, *coeffs)
# Run the optimizer for 100 epochs
for i in range(100):
    trained_params = opt.step(mapped_model, trained_params)
    if (i+1)%10==0:
        cost = mapped_model(trained_params)
        print(f"Epoch {i+1:4d}: {cost} at (relative) parameters {trained_params}")

trained_cost_orig = circuit(params+trained_params)
print(f"Original energy at the minimum of the model: {trained_cost_orig}")