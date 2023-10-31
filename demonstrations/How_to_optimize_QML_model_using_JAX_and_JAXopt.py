r"""How to optimize a quantum machine learning model using JAX and JAXopt
=====================================================================
"""

######################################################################
# Once you have set up your quantum machine learning model (which typically includes deciding on your
# circuit architecture/ansatz, determining how you embed or integrate your data, and creating your
# cost function to minimize a quantity of interest), the next step is **optimization**. That is,
# setting up a classical optimization loop to find a minimal value of your cost function.
# 
# In this example, we’ll show you how to use `JAX <https://jax.readthedocs.io>`__, an
# autodifferentiable machine learning framework, and `JAXopt <https://jaxopt.github.io/>`__, a suite
# of JAX-compatible gradient-based optimizers, to optimize a PennyLane quantum machine learning model.
# 

######################################################################
# Set up your model, data, and cost
# ---------------------------------
# 

######################################################################
# Here, we will create a simple QML model for our optimization. In particular:
# 
# -  We will embed our data through a series of rotation gates.
# -  We will then have an ansatz of trainable rotation gates with parameters ``weights``; it is these
#    values we will train to minimize our cost function.
# -  We will train the QML model on ``data``, a ``(5, 4)`` array, and optimize the model to match
#    target predictions given by ``target``.
# 

import pennylane as qml
import jax
from jax import numpy as jnp
import jaxopt

n_wires = 5
data = jnp.sin(jnp.mgrid[-2:2:0.2].reshape(n_wires, -1)) ** 3
targets = jnp.array([-0.2, 0.4, 0.35, 0.2])

dev = qml.device("default.qubit", wires=n_wires)

@qml.qnode(dev)
def circuit(data, weights):
    """Quantum circuit ansatz"""

    # data embedding
    for i in range(n_wires):
        # data[i] will be of shape (4,); we are
        # taking advantage of operation vectorization here
        qml.RY(data[i], wires=i)

    # trainable ansatz
    for i in range(n_wires):
        qml.RX(weights[i, 0], wires=i)
        qml.RY(weights[i, 1], wires=i)
        qml.RX(weights[i, 2], wires=i)
        qml.CNOT(wires=[i, (i + 1) % n_wires])

    # we use a sum of local Z's as an observable since a
    # local Z would only be afffected by params on that qubit.
    return qml.expval(qml.sum(*[qml.PauliZ(i) for i in range(n_wires)]))

def my_model(data, weights, bias):
    return circuit(data, weights) + bias

######################################################################
# We will define a simple cost function that computes the overlap between model output and target
# data:
# 

@jax.jit
def loss_fn(params, data, targets):
    predictions = my_model(data, params["weights"], params["bias"])
    loss = jnp.sum((targets - predictions) ** 2 / len(data))
    return loss

######################################################################
# Note that the model above is just an example for demonstration – there are important considerations
# that must be taken into account when performing QML research, including methods for data embedding,
# circuit architecture, and cost function, in order to build models that may have use. This is still
# an active area of research; see our `demonstrations <https://pennylane.ai/qml/demonstrations>`__ for
# details.
# 

######################################################################
# Initialize your parameters
# --------------------------
# 

######################################################################
# Now, we can generate our trainable parameters ``weights`` and ``bias`` that will be used to train
# our QML model.
# 

weights = jnp.ones([n_wires, 3])
bias = jnp.array(0.)
params = {"weights": weights, "bias": bias}

######################################################################
# Plugging the trainable parameters, data, and target labels into our cost function, we can see the
# current loss as well as the parameter gradients:
# 

loss_fn(params, data, targets)

######################################################################
# .. rst-class :: sphx-glr-script-out
# 
# .. code-block: none
# 
#    Array(0.29232618, dtype=float32)

jax.grad(loss_fn)(params, data, targets)

######################################################################
# .. rst-class :: sphx-glr-script-out
# 
# .. code-block: none
# 
#    {'bias': Array(-0.754321, dtype=float32, weak_type=True),
#     'weights': Array([[-1.9507733e-01,  5.2854650e-02, -4.8925212e-01],
#            [-1.9968867e-02, -5.3287148e-02,  9.2290469e-02],
#            [-2.7175695e-03, -9.6455216e-05, -4.7958046e-03],
#            [-6.3544422e-02,  3.6111072e-02, -2.0519713e-01],
#            [-9.0263695e-02,  1.6375928e-01, -5.6426275e-01]], dtype=float32)}

######################################################################
# Create the optimizer
# --------------------
# 

######################################################################
# We can now use JAXopt to create a gradient descent optimizer, and train our circuit.
# 
# To do so, we first create a function that returns the loss value *and* the gradient value during
# training; this allows us to track and print out the loss during training within JAXopt’s internal
# optimization loop.
# 

def loss_and_grad(params, data, targets, print_training, i):
    loss_val, grad_val = jax.value_and_grad(loss_fn)(params, data, targets)

    def print_fn():
        jax.debug.print("Step: {i}  Loss: {loss_val}", i=i, loss_val=loss_val)

    # if print_training=True, print the loss every 5 steps
    jax.lax.cond((jnp.mod(i, 5) == 0) & print_training, print_fn, lambda: None)

    return loss_val, grad_val

######################################################################
# Note that we use a couple of JAX specific functions here:
# 
# -  ``jax.lax.cond`` instead of a Python ``if`` statement
# -  ``jax.debug.print`` instead of a Python ``print`` function
# 
# These JAX compatible functions are needed because JAXopt will automatically JIT compile the
# optimizer update step.
# 

opt = jaxopt.GradientDescent(loss_and_grad, stepsize=0.3, value_and_grad=True)
opt_state = opt.init_state(params)

for i in range(100):
    params, opt_state = opt.update(params, opt_state, data, targets, True, i)

######################################################################
# .. rst-class :: sphx-glr-script-out
# 
# .. code-block: none
# 
#    Step: 0  Loss: 0.004011586308479309
#    Step: 5  Loss: 0.004008766263723373
#    Step: 10  Loss: 0.004004144575446844
#    Step: 15  Loss: 0.003998623229563236
#    Step: 20  Loss: 0.003992810845375061
#    Step: 25  Loss: 0.0039870040491223335
#    Step: 30  Loss: 0.003981292713433504
#    Step: 35  Loss: 0.003975625149905682
#    Step: 40  Loss: 0.003969968296587467
#    Step: 45  Loss: 0.003964546136558056
#    Step: 50  Loss: 0.003959367983043194
#    Step: 55  Loss: 0.0039545586332678795
#    Step: 60  Loss: 0.003950078040361404
#    Step: 65  Loss: 0.003945896402001381
#    Step: 70  Loss: 0.003942088223993778
#    Step: 75  Loss: 0.003938441630452871
#    Step: 80  Loss: 0.003935104236006737
#    Step: 85  Loss: 0.003931929357349873
#    Step: 90  Loss: 0.003929028287529945
#    Step: 95  Loss: 0.003926197066903114

######################################################################
# Jitting the optimization loop
# -----------------------------
# 

######################################################################
# In the above example, we just-in-time (JIT) compiled our cost function ``loss_fn``. However, we can
# also JIT compile the entire optimization loop; this means that the for-loop around optimization is
# not happening in Python, but is compiled and executed native. This avoids (potentially costly) data
# transfer between Python and our JIT compiled cost function with each update step.
# 

@jax.jit
def optimization_jit(params, data, targets, print_training=False):
    opt = jaxopt.GradientDescent(loss_and_grad, stepsize=0.3, value_and_grad=True)
    opt_state = opt.init_state(params)

    def update(i, args):
        params, opt_state = opt.update(*args, i)
        return (params, *args[1:])

    args = (params, opt_state, data, targets, print_training)
    (params, opt_state, _, _, _) = jax.lax.fori_loop(0, 100, update, args)

    return params

######################################################################
# Note that we use ``jax.lax.fori_loop`` and ``jax.lax.cond``, rather than a standard Python for loop
# and if statement, to allow the control flow to be JIT compatible.
# 

optimization_jit(params, data, targets, print_training=True)

######################################################################
# .. rst-class :: sphx-glr-script-out
# 
# .. code-block: none
# 
#    Step: 0  Loss: 0.003924083895981312
#    Step: 5  Loss: 0.003924083895981312
#    Step: 10  Loss: 0.003924083895981312
#    Step: 15  Loss: 0.003924083895981312
#    Step: 20  Loss: 0.003924083895981312
#    Step: 25  Loss: 0.003924083895981312
#    Step: 30  Loss: 0.003924083895981312
#    Step: 35  Loss: 0.003924083895981312
#    Step: 40  Loss: 0.003924083895981312
#    Step: 45  Loss: 0.003924083895981312
#    Step: 50  Loss: 0.003924083895981312
#    Step: 55  Loss: 0.003924083895981312
#    Step: 60  Loss: 0.003924083895981312
#    Step: 65  Loss: 0.003924083895981312
#    Step: 70  Loss: 0.003924083895981312
#    Step: 75  Loss: 0.003924083895981312
#    Step: 80  Loss: 0.003924083895981312
#    Step: 85  Loss: 0.003924083895981312
#    Step: 90  Loss: 0.003924083895981312
#    Step: 95  Loss: 0.003924083895981312# .. rst-class :: sphx-glr-script-out
# 
# .. code-block: none
# 

######################################################################
# Appendix: Timing the two approaches
# -----------------------------------
# 
# We can time the two approaches (JIT compiling just the cost function, vs JIT compiling the entire
# optimization loop) to explore the differences in performance:
# 

def optimization(params, data, targets):
    opt = jaxopt.GradientDescent(loss_and_grad, stepsize=0.3, value_and_grad=True)
    opt_state = opt.init_state(params)

    for i in range(100):
        params, opt_state = opt.update(params, opt_state, data, targets, False, i)

    return params

# %%timeit
optimization(params, data, targets)

######################################################################
# .. rst-class :: sphx-glr-script-out
# 
# .. code-block: none
# 
#    1.62 s ± 322 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

# %%timeit
optimization_jit(params, data, targets)

######################################################################
# .. rst-class :: sphx-glr-script-out
# 
# .. code-block: none
# 
#    860 µs ± 98.7 µs per loop (mean ± std. dev. of 7 runs, 1 loop each)

######################################################################
# About the author
# ----------------
# # .. include:: ../_static/authors/josh_izaac.txt

# # .. include:: ../_static/authors/maria_schuld.txt
