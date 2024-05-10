r"""
Quantum Variational Circuit for Multidimensional Regression
==========================================

In this tutorial, we show how to use a quantum variational circuit to fit a simple multivariate function,
:math:`f(x_1, x_2) = \frac{1}{2} \left( x_1^2 + x_2^2 \right)`. [#schuld]_
shows that, under some conditions, there exist quantum variational circuits that are expressive enough to realize any possible set
of Fourier coefficients. We will use a simple two-qubit parameterized quantum circuit to construct a partial Fourier series for fitting
the target function.
The main outline of the process is as follows:
1. Build a circuit consisting of layers of alternating data-encoding and parameterized training blocks
2. Optimize the expectation value of the circuit output against a target function that is the function that
we want to fit
3. Obtain a partial Fourier series for the target function; since the function is not periodic, this partial
Fourier series will only approximate the function in the region we will use for training.
4. Plot the optimized circuit expectation value against the exact function to compare the two.
What is a quantum model?
------------
A quantum model :math:`g_{\vec{\theta}}(\vec{x})` is the expectation value of some observable : math : `M` with
respect to a state prepared by a parameterized circuit :math:`U(\vec{x}, \vec{\theta})`:
.. math:: g_{\vec{\theta}}(\vec{x}) = \langle 0 | U^\dagger (\vec{x}, \vec{\theta}) M U(\vec{x}, \vec{\theta}) | 0 \rangle.
By repeatedly running the circuit with a set of parameters :math:`\vec{\theta}` and set of data points :math:`\vec{x}`, we can
approximately find the expectation value with respect to the observable :math:`M`. Then, the expectation value can be
optimized with respect to some loss function by adjusting :math:`\vec{\theta}`.
What are we using the variational circuit for?
------------
In this example, we will use a quantum neural network to find the Fourier series that
approximates the function :math:`f(x_1, x_2) = \frac{1}{2} \left( x_1^2 + x_2^2 \right)`. The quantum neural
network we are working with is a circuit made up of :math:`L` layers. Each layer consists of a *data encoding block*
:math:`S(\vec{x})` and a *training block* :math:`W(\vec{\theta})`. The overall circuit is:
.. math:: U(x, \vec{\theta}) = W^{(L+1)}(\vec{\theta}) S(\vec{x}) W^{(L)} (\vec{\theta}) \ldots W^{(2)}(\vec{\theta}) S(\vec{x}) W^{(1)}(\vec{\theta}).
The training blocks :math:`W(\vec{\theta})` depend on a vector of parameters :math:`\vec{\theta}` that can be optimized classically.
.. figure:: ../_static/demonstration_assets/qnn_multivariate_regression/qnn_circuit.png
    :align: center
    :width: 90%
We will build a circuit such that the expectation value of the :math:`Z` observable is a partial Fourier series
that approximates :math:`f(\vec{x})`, i.e.,
.. math:: f(\vec{x}) \approx \sum_{\vec{\omega} \in \Omega} c_\vec{\omega} e^{i \vec{\omega} \vec{x}} = g_{\vec{\theta}}.
Then, we can directly plot the partial Fourier series. We can also apply a Fourier transform to
:math:`g_{\vec{\theta}}`, so we can obtain the Fourier coefficients :math:`c_\vec{\omega}`. To know more about how to obtain the 
Fourier series check out these two related tutorials [#demoschuld]_,[#qibodemo]_.
How do we actually construct the quantum circuit?
------------
"""
##############################################################################
# First, let's import the necessary libraries and seed the random number generator.
# We will also define the device, which has two qubits, using qml.device.
import matplotlib.pyplot as plt
import pennylane as qml
from pennylane import numpy as np
import jax
from jax import numpy as jnp
import optax

np.random.seed(42)

dev = qml.device('default.qubit', wires=2)

######################################################################
# Now we will define the data-encoding circuit block, :math:`S(\vec{x})`, where :math:`\vec{x} = (x_1, x_2)`. We can encode the data in the operator
# .. math::
#    e^{-i H x_1} \otimes e^{-i H x_2}
# where :math:`H` is the *encoding Hamiltonian*. In this example, we will use :math:`\frac{1}{2} Z`, where :math:`Z` is the Pauli-X operator, as our encoding Hamiltonian. This gives us a product of :math:`R_z` rotations:
# .. math:: S(\vec{x}) = e^{-i \frac{1}{2} Z x_1} \otimes e^{i \frac{1}{2} Z x_2} = R_z(x_1) \otimes R_z(x_2).

# Specifically, we define the :math:`S(\vec{x})` operator using the :class:`~.pennylane.templates.AngleEmbedding` function

def S(x):
    #Data-encoding circuit block
    qml.AngleEmbedding( x, wires=[0,1],rotation='Z')

######################################################################
# For the :math:`W(\vec{\theta})` operator, we will use an ansatz that is available in PennyLane, called :class:`~pennylane..templates.StronglyEntanglingLayers`.

def W(params):
    # Trainable circuit block
    qml.StronglyEntanglingLayers(params, wires=[0,1])

######################################################################
# Now we will build the circuit in PennyLane by alternating layers of :math:`W(\vec{\theta})` and :math:`S(x)` layers. The output that we are getting from the circuit is the expectation value of the :math:`Z` operator on both qubits, which we will compute using PennyLane's expval function.

@qml.qnode(dev,interface="jax")
def quantum_neural_network(params, x):
    # The quantum model we will use for regression
    layers=len(params[:,0,0])
    for i in range(layers):
      W(params[i,:,:].reshape(1,len(params[0,:,0]),len(params[0,0,:])))
      S(x)
    # (L+1)'th unitary
    W(params[-1,:,:].reshape(1,len(params[0,:,0]),len(params[0,0,:])))

    return qml.expval(qml.PauliZ(wires=0)@qml.PauliZ(wires=1))

######################################################################
# The function we will be fitting is :math:`f(x_1, x_2) = \frac{1}{2} \left( x_1^2 + x_2^2 \right)`, which we will define as target_function:

def target_function(x):
    # Generate a target function that we want to fit
    f=1/2*(x[0]**2+x[1]**2)
    return f

######################################################################
# Now we will specify the range of :math:`x_1` and :math:`x_2` values and store those values in an input data vector. We are fitting the function for :math:`x_1, x_2 \in [-1, 1]`, using 30 evenly-spaced samples for each variable.

x1_min=-1
x1_max=1
x2_min=-1
x2_max=1
num_samples=30

######################################################################
# Now we build the training data with the exact target function :math:`f(x_1, x_2)`.

x1_train=np.linspace(x1_min,x1_max, num_samples)
x2_train=np.linspace(x2_min,x2_max, num_samples)
x1_mesh,x2_mesh=np.meshgrid(x1_train, x2_train)

# We define x_train, y_train
x_train=np.stack((x1_mesh.flatten(), x2_mesh.flatten()), axis=1)
y_train = np.array(np.real(target_function([x1_mesh,x2_mesh])).reshape(-1,1))

######################################################################
# What do we do with the output from the circuit?
# ------------------------------------------
#
# We want to optimize the circuit above so that the :math:`Z` expectation value 
# approximates the exact target function. This is done by minimizing the mean squared error between
# the circuit :math:`Z` expectation value and the exact target function. In particular, the optimization
# process to train the quantum neural network will be performed using JAX. Checkout [#demojax]_
# to learn more about
# how to use JAX to optimize your QML models.
# .. figure:: ../_static/demonstration_assets/qnn_multivariate_regression/qnn_diagram.jpg
#   :align: center
#   :width: 90%

@jax.jit
def mse(params,x,targets):
    # Squared error of the expectation value of the quantum model and the target function.
    return (quantum_neural_network(params,x)-jnp.array(targets))**2
@jax.jit
def loss_fn(params, x,targets):
    # Compute the loss function to feed our optimizer.
    mse_pred = jax.vmap(mse,in_axes=(None, 0,0))(params,x,targets)
    loss = jnp.mean(mse_pred)
    return loss

####################################################################### 

#Here, we are choosing a learning rate of 0.05 and 200 steps.

opt = optax.adam(learning_rate=0.05)
# Define number of steps
max_steps=200

@jax.jit
def update_step_jit(i, args):
    # Step we loop over to optimize the trainable parameters
    params, opt_state, data, targets, print_training = args
    loss_val, grads = jax.value_and_grad(loss_fn)(params, data, targets)
    updates, opt_state = opt.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

    def print_fn():
        jax.debug.print("Step: {i}  Loss: {loss_val}", i=i, loss_val=loss_val)
    # if print_training=True, print the loss every 10 steps
    jax.lax.cond((jnp.mod(i, 10) == 0 ) & print_training, print_fn, lambda: None)
    return (params, opt_state, data, targets, print_training)

@jax.jit
def optimization_jit(params, data, targets, print_training=False):
    #Main function that performs the optimization process.
    opt_state = opt.init(params)
    args = (params, opt_state, jnp.asarray(data), targets, print_training)
    # We loop over max_steps iterations to optimize the parameters
    (params, opt_state, _, _, _) = jax.lax.fori_loop(0, max_steps+1, update_step_jit, args)
    return params

######################################################################
# Now we will train the variational circuit with four layers and obtain a vector :math:`\vec{\theta}` with the optimized parameters. At every 10 steps, the code prints out the value of the loss function.

wires=2
layers=4
params_shape = qml.StronglyEntanglingLayers.shape(n_layers=layers,n_wires=wires)
params=np.random.default_rng().random(size=params_shape)
best_params=optimization_jit(params, x_train, jnp.array(y_train), print_training=True)

######################################################################
# Once the optimized :math:`\vec{\theta}` has been obtained, we can use those parameters to build our fitted version of the function.

def evaluate(params, data):
    y_pred = jax.vmap(quantum_neural_network, in_axes=(None, 0))(params, data)
    return y_pred
y_predictions=evaluate(best_params,x_train)

######################################################################
# To compare the fitted function to the exact target function, let's take a look at the :math:`R^2` score.

# Calculate R^2 score
from sklearn.metrics import r2_score
r2 = round(float(r2_score(y_train, y_predictions)),3)
# Calculate mean squared error (MSE)
print("R^2 Score:", r2)

######################################################################
# We can also plot the results to check visually the goodness of the fit:

fig = plt.figure()

# Target function
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot_surface(x1_mesh,x2_mesh, y_train.reshape(x1_mesh.shape), cmap='viridis')
ax1.set_zlim(0,1)
# Add labels and title
ax1.set_xlabel('$x$',fontsize=10)
ax1.set_ylabel('$y$',fontsize=10)
ax1.set_zlabel('$f(x,y)$',fontsize=10)
ax1.set_title('Target ')

# Predictions
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.plot_surface(x1_mesh,x2_mesh, y_predictions.reshape(x1_mesh.shape), cmap='viridis')
ax2.set_zlim(0,1)
# Add labels and title
ax2.set_xlabel('$x$',fontsize=10)
ax2.set_ylabel('$y$',fontsize=10)
ax2.set_zlabel('$f(x,y)$',fontsize=10)
ax2.set_title(f' Predicted \nAccuracy: {round(r2*100,3)}%')

# Show the plot
plt.tick_params(axis='both', which='major', labelsize=8)
plt.tight_layout(pad=4.0)

######################################################################
# Conclusions

# In this demo we have shown how to leverage the Fourier series representation of a quantum variational circuit to solve a regression problem for a two dimensional function. In particular we used JAX, an auto differentiable machine learning framework to accelerate the classicaloptimization of the parameters. The results show a good agreement with the target function and the model can be trainedfurther, increasing number of iterations in the training to maximize the accuracy. It also paves the way for addressing a regression problem for a N-dimensional function, as everything presented here can be easily generalized. A final check that could be done is to obtain the Fourier coefficients of the trained circuit and compare it with the Fourier series we obtained when Fourier Transform is applied directly to the circuit.


# References
# ----------
#
# .. [#schuld]
#
#     Maria Schuld, Ryan Sweke, and Johannes Jakob Meyer
#     "The effect of data encoding on the expressive power of variational quantum machine learning models.",
#     `arXiv:2008.0865 <https://arxiv.org/pdf/2008.08605>`__, 2021.
#
# .. [#qibodemo]
#
#     Jorge J. Martinez de Lejarza
#     "Tutorial: Quantum Fourier Iterative Amplitude Estimation",
#     `Qibo: Quantum Fourier Iterative Amplitude Estimation <https://qibo.science/qibo/stable/code-examples/tutorials/qfiae/qfiae_demo.html>`__, 2023.
#
# .. [#demoschuld]
#
#    Johannes Jakob Meyer, Maria Schuld
#    “Tutorial: Quantum models as Fourier series”,
#    `Pennylane: Quantum models as Fourier series <https://pennylane.ai/qml/demos/tutorial_expressivity_fourier_series/>`__, 2021.
#
# .. [#demojax]
#
#    Josh Izaac, Maria Schuld
#    "How to optimize a QML model using JAX and Optax",
#    `Pennylane: How to optimize a QML model using JAX and Optax  <https://pennylane.ai/qml/demos/tutorial_How_to_optimize_QML_model_using_JAX_and_Optax/>`__, 2024
#
# About the author
# ----------------

