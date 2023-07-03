r"""
Optimal control for gate compilation
====================================

.. meta::
    :property="og:description": Optimize a pulse program to obtain digital gates.
    :property="og:image": https://pennylane.ai/qml/_images/thumbnail_tutorial_optimal_control.png

.. related::

    tutorial_pulse_programming101 Introduction to pulse programming in PennyLane
    tutorial_neutral_atoms Introduction to neutral-atom quantum computers
    ahs_aquila Pulse programming on Rydberg atom hardware

*Author: David Wierichs. Posted: xx July, 2023.*


Today, quantum computations largely are phrased as quantum circuits--or gate
sequences more generally--that are composed of digital quantum gates.
However, most quantum hardware does not come with such digital
gates as elementary native operations.
Instead, the hardware allows us to play sequences of analogue electromagnetic pulses,
for example by shining laser pulses on trapped ions or Rydberg atoms, or by sending microwave
pulses onto superconducting qubit cavities.
These pulses need to be calibrated to yield the desired digital gates, and in
this tutorial we will be concerned with exactly that step.
For this, we parametrize a pulse sequence, which leads to a whole *space*
of possible sequences. Then we optimize the pulse parameters in order to
find a configuration in this space that behaves as closely to the gate
of interest as possible.
This training of control parameters to achieve a specific time
evolution is a standard task in the field of *quantum optimal control*.

More concretely, we will optimize simple pulse programs on two and three qubits to
obtain a CNOT and a Toffoli gate.

For an introduction see
:doc:`the demo on differentiable pulse programming </demos/tutorial_pulse_programming101>`
in PennyLane.
Instead of optimizing pulses to yield digital gates that are used in quantum circuits,
we may use them directly to solve minimization problems, as is also showcased in the demo.
If you are interested in specific hardware pulses, take a look at
:doc:`an introduction to neutral-atom quantum computing </demos/tutorial_neutral_atoms>`
or :doc:`the tutorial on the QuEra Aquila device </demos/ahs_aquila>`, which treat pulse
programming with Rydberg atoms.

Quantum optimal control
-----------------------

The overarching goal of quantum optimal control is to find the best way to steer
a microscopical physical system such that its dynamics matches a desired behaviour.
The meaning of "best" and "desired behaviour" will depend on the specific
task, and it is important to specify underlying assumptions and constraints on
the system controls in order to make the problem statement well-defined.
Once we specified all these details, optimal control theory is concerned with
questions like
"How close can the system get to modelling the desired behaviour?",
"How can we find the best (sequence of) control parameters to obtain the desired behaviour?",
or
"What is the shortest time in which the system can reach a specific state, given some
initial state?".

In this tutorial, we consider the control of few-qubit systems through pulse sequences,
with the goal to produce a given target, namely a digital gate, to the highest-possible
precision.
To do this, we will choose an ansatz for the pulse sequence that contains
free parameters and define a cost function that quantifies the deviation of the qubit
operation from the target gate.
Then we minimize the cost function by optimizing the pulse parameters until we
find the desired gate to a sufficient precision--or can no longer improve on the
approximation we found.
For the training phase we will make use of fully-differentiable classical simulations
of the qubit dynamics, allowing us to make use of backpropagation, the efficient
differentiation technique widely used in machine learning, and of gradient-based
optimization.
At the same time we attempt to find pulse shapes and control parameters that are
(to some degree) realistic to be implemented, including bounded
pulse amplitudes and rates of change of the amplitudes.

Gate calibration via pulse programming
--------------------------------------

Here we briefly discuss the general setup of pulse programs that we will use for our
optimal control application. For more details also consider the related
tutorials focusing on pulse programming.

Consider a quantum system comprised of :math:`N` two-level systems, or qubits, described
by a Hamiltonian

.. math::

    H(\boldsymbol{p}, t) = H_d + \sum_{i=1}^K f_i(\boldsymbol{p_i}, t) H_i.

As we can see, :math:`H` depends on the time :math:`t` and on a set of control parameters
:math:`\boldsymbol{p}`. Both feed into functions :math:`f_i` that return scalar coefficients
for (constant) Hamiltonian terms :math:`H_i`. In addition, there is a constant drift
Hamiltonian :math:`H_d`.
We will assume that the Hamiltonian :math:`H` fully describes the system of interest and
in particular we do not consider sources of noise in the system, such as leakage, dephasing
or crosstalk, i.e. the accidental interaction with other parts of a larger, surrounding system.

The time evolution of the state of our quantum system will be described
by the Schrödinger equation associated with :math:`H`.
However, for our purposes it will be more useful to consider the full unitary evolution that
the Hamiltonian causes, independently of the initial state. This way, we can compare it to
the digital target gate without iterating over different input and output states.
The Schrödinger equation dictates the behaviour of the evolution operator :math:`U` to be

.. math::

    \frac{d}{dt} U(\boldsymbol{p}, t) = -i H(\boldsymbol{p}, t) U(\boldsymbol{p}, t),

where we implicitly fixed the initial time of the evolution to :math:`t_0=0`.
It is possible to simulate the dynamics of sufficiently small quantum systems on
a classical computer by solving the ordinary differential equation (ODE) above numerically.
For a fixed pulse duration :math:`T` and given control parameters :math:`\boldsymbol{p}`,
a numerical ODE solver computes the matrix :math:`U(\boldsymbol{p}, T)`.

How can we tell whether the evolution of the qubit system is close to the digital gate
we aim to produce? We will need a distance measure!

|

.. figure:: ../demonstrations/optimal_control/distance.png
    :align: center
    :width: 80%
    :alt: Illustration of landscape with a path drawn between markers for a pulse unitary and a CNOT gate
    :target: javascript:void(0);

|

In this tutorial we will describe the distance of two unitary matrices :math:`U` and
:math:`V` based on the (average) gate fidelity:

.. math::

    d(U,V) = 1 - \frac{1}{2^N}\big|\operatorname{tr}[U^\dagger V]\big|.

It is similar to the distance measure obtained from the
`Frobenius norm <https://en.wikipedia.org/wiki/Matrix_norm#Frobenius_norm>`__
but allows us to ignore differences in the global phase.

With a distance measure in our hands, we can write out the cost function that we want to
minimize by training the pulse parameters:

.. math::

    C(\boldsymbol{p}) = d(U_\text{target}, U(\boldsymbol{p}, T)).

Here :math:`U_\text{target}` is the unitary matrix of the gate that we want to compile
and we exclude the total duration :math:`T` from :math:`C` of the pulse sequence because we will
consider it as a constraint to the optimization problem, rather than a free variable.

We can then minimize the cost function :math:`C`, for example using gradient-based
optimization algorithms like adam [#KingmaBa14]_.
But how do we obtain the gradient of a cost function that requires us to run an ODE solver
to obtain its value? We are in luck! The implementation of pulse programming in PennyLane is
fully differentiable via backpropagation thanks to its backend based on the machine
learning library `JAX <https://jax.readthedocs.io/en/latest/>`__.
This enables us to optimize the gate sequences using efficiently computed gradients
(provided the target gate is not too large).

Before we dive into the task of calibrating a gate, let's briefly talk about
the pulse shape that we will use:

Smooth rectangle pulses
-----------------------

|

.. figure:: ../demonstrations/optimal_control/rect_to_smooth.png
    :align: center
    :width: 60%
    :alt: Sketch of converting a rectangular pulse shape into a smoothened rectangular pulse shape
    :target: javascript:void(0);

|

Let's look at a building block that we will use a lot: smoothened rectangular pulses.
We start with a simple rectangular pulse

.. math::

    R_\infty(t, (\Omega, t_0, t_1)) = \Omega \Theta(t-t_0) \Theta(t_1-t)

where :math:`\Omega` is the amplitude, :math:`t_{0,1}` are the start and end
times of the pulse, and :math:`\Theta(t)` is the
`Heaviside step function <https://en.wikipedia.org/wiki/Heaviside_step_function>`__
which is one for :math:`t\geq 0` and zero otherwise.
The trainable parameters of this pulse are the amplitude and the start/end times.

There are two main issues with :math:`R_\infty` for our purposes:

#. The Heaviside step function is not differentiable with respect
   to the times :math:`t_0` and :math:`t_1` in the conventional sense (but
   only if we were to consider distributions in addition to functions), and in
   particular we cannot differentiate the resulting :math:`U(\boldsymbol{p},T)`
   within the automatic differentiation framework provided by JAX.

#. The instantaneous change in the amplitude will not be realizable in practice.
   In reality, the pulses describe some electromagnetic control field that can only
   be changed at a bounded rate and in a smooth manner. :math:`R_\infty` is not
   only not smooth, it is not even continuous. So we should consider smooth
   pulses with a bounded rate of change instead.

We can solve both these issues by smoothening the rectangular pulse:
We simply replace the step functions above by a smooth variant, namely by sigmoid functions:

.. math::

    R_k(t, (\Omega, t_0, t_1)) &= \Omega S(t-t_0, k) S(t_1-t, k)\\
    S(t, k) &= (1+\exp(-k t))^{-1}.

We introduced an additional parameter, :math:`k`, that controls the steepness of the sigmoid
functions and can be adapted to the constraints posed by hardware on the maximal rate of change.
In contrast to :math:`R_\infty`, its sister :math:`R_k` is smooth in all three arguments
:math:`\Omega`, :math:`t_0` and :math:`t_1`, and training these three parameters with
automatic differentiation will not be a problem.

Let's implement the smooth-rectangle function using JAX's ``numpy``. We
directly implement the product of the two sigmoids in the function ``sigmoid_rectangle``:

.. math::

    R_k(t, (\Omega, t_0, t_1), k)=
    \Omega [1+\exp(-k (t-t_0))+\exp(-k (t_1-t))+\exp(-k(t_1-t_0))]^{-1}.
"""
import jax
from jax import numpy as jnp

jax.config.update("jax_enable_x64", True)  # Use float64 precision
jax.config.update("jax_platform_name", "cpu")  # Disables a warning regarding device choice

def sigmoid_rectangle(t, t_0, t_1, k=1.0):
    """Smooth-rectangle pulse with unit amplitude."""
    return 1 / (
        1 + jnp.exp(-k * (t - t_0)) + jnp.exp(-k * (t_1 - t)) + jnp.exp(-k * (t_1 - t_0))
    )


#############################################################################
# Let's look at a rectangular pulse and its smoothened sister, for a number of
# different smoothness parameters:

import matplotlib.pyplot as plt

t = jnp.linspace(0, 6, 1000)
t_0, t_1 = (1.3, 5.4)
amplitude = 2.3
ks = [5, 10, 50]
rect = amplitude * jnp.heaviside(t - t_0, 1.0) * jnp.heaviside(t_1 - t, 1.0)
smooths = [amplitude * sigmoid_rectangle(t, t_0, t_1, k) for k in ks]

plt.plot(t, rect, label="Rectangle $R_{\\infty}$, $k\\to\\infty$")
for k, sm in zip(ks, smooths):
    plt.plot(t, sm, label=f"Smooth rectangle $R_k$, $k={k}$")
plt.legend(bbox_to_anchor=(0.6, 0.05), loc="lower center")
plt.xlabel("time $t$")
plt.ylabel("Pulse function")
plt.show()

#############################################################################
# We see that for very large :math:`k`, the smooth rectangle becomes practically
# indistinguishable from the original rectangle function :math:`R_\infty`. This means
# that we can consider the smooth :math:`R_k` a *generalization* of the pulse shape,
# rather than a restriction.
#
# With this nicely trainable pulse shape in our hands, we now turn to the first gate
# calibration task.
#
# Pulse ansatz for CNOT calibration
# ---------------------------------
#
# In this first example we will tune a two-qubit pulse to produce a standard CNOT gate.
#
# We start by choosing a system Hamiltonian.
# It contains the drift term :math:`H_d = Z_0 + Z_1`, i.e. a Pauli :math:`Z` operator
# acting on each qubit, with a constant unit amplitude.
# The parametrized part uses five generating terms: Pauli :math:`Z` acting on the
# first qubit (:math:`Z_0`), all three Pauli operators acting on the second qubit
# (:math:`X_1, Y_1, Z_1`) and a single interaction term :math:`Z_0X_1`, resembling an
# abstract cross resonance driving term. Due to this choice, the :math:`Z_0` term
# commutes with all other terms, including the drift term, and can be considered a
# correction of the drive term to obtain the correct action on the first qubit.
#
# The five coefficient functions :math:`f_i` are sums of multiple smooth-rectangle
# pulse shapes :math:`R_k`
# (see the section above) each, using distinct parameters for each generating term.
# Each coefficient function sums :math:`P` smooth rectangles :math:`R_k` with individual
# amplitudes and start and end times. Overall, this leads to
# :math:`n=5\cdot 3\cdot P=15P` parameters in :math:`H`.
#
# Before we define this Hamiltonian, we implement the sum over multiple
# ``sigmoid_rectangle`` functions, including two normalization steps:
# First, we normalize the start and end times of the rectangles to the interval
# :math:`[\epsilon, T-\epsilon]`, which makes sure that the pulse amplitudes are
# close to zero at :math:`t=0` and :math:`t=T`. Without this step, we might be
# tuning the pulses to be turned on (off) instantaneously at the beginning (end) of the
# sequence, negating our effort on the pulse shape itself not to vary too quickly.
# Second, we normalize the final output value to the interval
# :math:`(-\Omega_\text{max}, \Omega_\text{max})`, which
# allows us to bound the maximal amplitudes of the pulses to a realizable range while
# maintaining differentiability.
#
# For the normalization steps we define a ``sigmoid`` and a ``normalize`` function.
# The first is a straight-forward implementation of :math:`R_k` whereas the second
# uses the ``sigmoid`` function to normalize real numbers to the interval :math:`(-1, 1)`.


def sigmoid(t, k=1.0):
    """Sigmoid function with steepness parameter ``k``."""
    return 1 / (1 + jnp.exp(-k * t))


def normalize(t, k=1.0):
    """Smoothly normalize a real input value to the interval (-1, 1) using ``sigmoid``
    with steepness parameter ``k``."""
    return 2 * sigmoid(t, k) - 1.0


def smooth_rectangles(params, t, k=2.0, max_amp=1.0, eps=0.0, T=1.0):
    """Compute the sum of :math:`P` smooth-rectangle pulses and normalize their
    starting and ending times, as well as the total output amplitude.

    Args:
        params (tensor_like): Amplitudes and start and end times for the rectangles,
            in the order ``[amp_1, ... amp_P, t_{1, 0}, t_{1, 1}, ... t_{P, 0}, t_{P, 1}]``.
        t (float): Time at which to evaluate the pulse function.
        k (float): Steepness of the sigmoid functions that delimit the rectangles
        max_amp (float): Maximal amplitude of the rectangles. The output will be normalized
            to the interval ``(-max_amp, max_amp)``.
        eps (float): Margin to beginning and end of the pulse sequence within which the
            start and end times of the individual rectangles need to lie.
        T (float): Total duration of the pulse.

    Returns:
        float: Value of sum of smooth-rectangle pulses at ``t`` for the given parameters.
    """
    P = len(params) // 3
    # Split amplitudes from times
    amps, times = jnp.split(params, [P])
    # Normalize times to be sufficiently far away from 0 and T
    times = sigmoid(times - T / 2, k=1.0) * (T - 2 * eps) + eps
    # Extract start and end times of single rectangles
    t_0, t_1 = jnp.reshape(times, (-1, 2)).T
    # Contract amplitudes with products of sigmoids (unit rectangles)
    value = jnp.dot(amps, sigmoid_rectangle(t, t_0, t_1, k))
    # Normalize the output value to be in [-max_amp, max_amp] with standard steepness
    return max_amp * normalize(value, k=1.0)


#############################################################################
# Let's look at this function for some example parameters, with the same steepness
# parameter :math:`k=20` for all rectangles in the sum:

from functools import partial

T = 2 * jnp.pi  # Total pulse sequence time
k = 20.0  # Steepness parameter
max_amp = 1.0  # Maximal amplitude \Omega_{max}
eps = 0.1 * T  # Margin for the start/end times of the rectangles
# Bind hyperparameters to the smooth_rectangles function
f = partial(smooth_rectangles, k=k, max_amp=max_amp, eps=eps, T=T)

# Set some arbitrary amplitudes and times
amps = jnp.array([0.4, -0.2, 1.9, -2.0])  # Four amplitudes
times = jnp.array([0.2, 0.6, 1.2, 1.8, 2.1, 3.7, 4.9, 5.9])  # Four pairs of start/end times
params = jnp.hstack([amps, times])  # Amplitudes and times constitute the trainable params

plot_times = jnp.linspace(0, T, 300)
plot_f = [f(params, t) for t in plot_times]

plt.plot(plot_times, plot_f)
ax = plt.gca()
ax.set(xlabel="Time t", ylabel=r"Pulse function $f(p, t)$")
plt.show()

#############################################################################
# Note that the rectangles are rather round for these generic parameters.
# The optimized parameters in the training workflows below will lead to more
# sharply defined pulses that resemble rectangles more closely. The amplitude normalization
# step in ``smooth_rectangles`` enables us to produce them in a differentiable manner,
# as was our goal with introducing :math:`R_k`.
# Also note that the normalization of the final output value is not a simple clipping
# step, but again a smooth function. As a consequence, the values ``1.9`` and ``-2.``
# in the example above, which are not in the interval ``[-1, 1]``,
# are not set to ``1`` and ``-1`` but take smaller absolute values.
#
# Using this function, we now may build the parametrized pulse Hamiltonian and the
# cost function discussed above. We make use of just-in-time (JIT) compilation,
# which will make the first execution of ``cost`` and ``grad`` slower, but speed
# up the subsequent executions a lot. For optimization workflows of small-scale
# functions, this almost always pays off.

import pennylane as qml

X, Y, Z = qml.PauliX, qml.PauliY, qml.PauliZ

num_wires = 2
# Hamiltonian terms of the drift and parametrized parts of H
ops_H_d = [Z(0), Z(1)]
ops_param = [Z(0), X(1), Y(1), Z(1), Z(0) @ X(1)]
# Coefficients: 1 for drift Hamiltonian and smooth rectangles for parametrized part
coeffs = [1.0, 1.0] + [f for op in ops_param]
# Build H
H = qml.dot(coeffs, ops_H_d + ops_param)
# Set tolerances for the ODE solver
atol = rtol = 1e-10

# Target unitary is CNOT. We get its matrix and note that we do not need the dagger
# because CNOT is Hermitian.
target = qml.CNOT([0, 1]).matrix()
target_name = "CNOT"
print(f"Our target unitary is a {target_name} gate, with matrix\n{target.astype('int')}")


def pulse_matrix(params):
    """Compute the unitary time evolution matrix of the pulse for given parameters."""
    return qml.evolve(H, atol=atol, rtol=rtol)(params, T).matrix()


@jax.jit
def cost(params):
    """Compute the infidelity cost function for given parameters."""
    # Compute the unitary time evolution of the pulse Hamiltonian
    op_mat = pulse_matrix(params)
    # Compute the infidelity between the target and the pulse evolution
    return 1 - jnp.abs(jnp.trace(target @ op_mat)) / 2**num_wires


grad = jax.jit(jax.grad(cost))

#############################################################################
# For the arbitrary parameters from above, of course we get a rather arbitrary unitary
# time evolution, which does not match the CNOT at all:

params = [params] * len(ops_param)
arb_mat = jnp.round(pulse_matrix(params), 4)
arb_cost = cost(params)
print(
    f"The arbitrarily chosen parameters yield the unitary\n{arb_mat}\n"
    f"which has an infidelity of {arb_cost:.6f}."
)

#############################################################################
# Before we can start the optimization, we require initial parameters.
# We set small alternating amplitudes and evenly distributed start and end times
# for :math:`P=3` smoothened rectangles. This choice leads to
# a total of :math:`15P=45` parameters in the pulse sequence.

P = 3  # Number of rectangles P
# Initial parameters for the start and end times of the rectangles
times = [jnp.linspace(eps, T - eps, P * 2) for op in ops_param]
# All initial parameters: small alternating amplitudes and times
params = [jnp.hstack([[0.1 * (-1) ** i for i in range(P)], time]) for time in times]

#############################################################################
# Now we are all set up to train the parameters of the pulse sequence to produce
# our target gate, the CNOT. We will use the adam optimizer [#KingmaBa14]_, implemented in the
# `optax <https://optax.readthedocs.io/en/latest/>`__
# library to our convenience. We keep track of the optimization via a list that contains
# the parameters and cost function values. Then we can plot the cost across the optimization.
# As we will run a second optimization later on, we code up the optimizer run as a function.

import optax


def run_adam(cost_fn, grad_fn, params, learning_rate, num_steps, target_name):
    # Initialize the adam optimizer
    optimizer = optax.adam(learning_rate, b1=0.97)
    opt_state = optimizer.init(params)
    # Initialize a memory buffer for the optimization
    hist = [(params.copy(), cost_fn(params))]
    for step in range(num_steps):
        g = grad_fn(params)
        updates, opt_state = optimizer.update(g, opt_state, params)

        params = optax.apply_updates(params, updates)
        hist.append([params, c := cost_fn(params)])
        if (step + 1) % (num_steps // 10) == 0:
            print(f"Step {step+1:4d}: {c:.6f}")
    _, cost_hist = list(zip(*hist))
    plt.plot(list(range(num_steps + 1)), cost_hist)
    ax = plt.gca()
    ax.set(
        xlabel="Iteration", ylabel=f"Infidelity $d(U_{{{target_name}}}, U(p))$", yscale="log"
    )
    plt.show()
    return hist


learning_rate = 0.2
num_steps = 500
hist = run_adam(cost, grad, params, learning_rate, num_steps, target_name)

#############################################################################
# As we can see, adam steadily reduces the cost function, bringing the pulse program
# closer and closer to the target unitary. On its way, the optimizer produces a mild
# oscillating behaviour. The precision to which the optimization can produce the
# target unitary depends on the expressivity of the pulses we use,
# but also on the precision with which we run the ODE solver and the hyperparameters
# of the optimizer.
#
# Let's pick those parameters with the smallest cost function we observed during
# the training and take a look at the pulses we found. We again prepare a function
# that plots the pulse sequence which we can reuse later on.
# For the single-qubit terms we encode their qubit in the color and the type of Pauli
# operator in the line style of the plotted line.

colors = {0: "#70CEFF", 1: "#C756B2", 2: "#FDC357"}
dashes = {"X": [10, 0], "Y": [2, 2, 10, 2], "Z": [6, 2]}


def plot_optimal_pulses(hist, pulse_fn, ops, T, target_name):
    _, cost_hist = list(zip(*hist))
    fig, axs = plt.subplots(2, 1, figsize=(10, 9), gridspec_kw={"hspace": 0.0}, sharex=True)

    # Pick optimal parameters from the buffer of all observed cost values
    min_params, min_cost = hist[jnp.argmin(jnp.array(cost_hist))]
    plot_times = jnp.linspace(0, T, 300)
    # Iterate over pulse parameters and parametrized operators
    for p, op in zip(min_params, ops):
        # Create label, and pick correct axis
        label = op.name
        ax = axs[0] if isinstance(label, str) else axs[1]
        # Convert the label into a concise string. This differs depending on
        # whether the operator has a single or multiple Paulis. Pick dashes
        if isinstance(label, str):
            label = f"${label[-1]}_{op.wires[0]}$"
            dash = dashes[label[1]]
        else:
            label = "$" + " ".join([f"{n[-1]}_{w}" for w, n in zip(op.wires, label)]) + "$"
            dash = [10, 0]

        # Set color according to qubit the term acts on
        col = colors[op.wires[0]]
        # Plot the pulse
        values = [pulse_fn(p, t) for t in plot_times]
        ax.plot(plot_times, values, label=label, dashes=dash, color=col)
    ax.legend()
    # Set legends and axis descriptions
    axs[0].legend(title="Single-qubit terms", ncol=int(jnp.sqrt(len(ops))))
    axs[1].legend(title="Two-qubit terms")
    title = f"{target_name}, Fidelity={1-min_cost:.6f}"
    axs[0].set(ylabel=r"Pulse function $f(p, t)$", title=title)
    axs[1].set(xlabel="Time $t$", ylabel=r"Pulse function $f(p, t)$")
    plt.show()


plot_optimal_pulses(hist, f, ops_param, T, target_name)

#############################################################################
# We observe that a single rectangular pulse is sufficient for most of the
# generating terms in the Hamiltonian, and we see that their shape is closer to
# actual rectangles now, in particular for those with a saturated amplitude.
#
# The final fidelity tells us that we achieved our goal of finding a pulse
# sequence that implements a unitary close to a CNOT gate.
# It could be optimized further, for example by running the optimization for more
# training iterations, by tuning the optimizer further to avoid oscillations,
# or by increasing the precision with which we run the ODE solver.
#
# Pulse sequence for Toffoli
# --------------------------
#
# The second example we consider is the compilation of a Toffoli--or CCNOT--gate.
# We reuse most of the workflow from above and only change the pulse Hamiltonian as
# well as a few hyperparameters.
#
# In particular, the Hamiltonian uses the drift term :math:`H_d=Z_0+Z_1+Z_2`
# and the generators are all single-qubit Paulis on all three qubits, together
# with the interaction generators :math:`Z_0X_1, Z_1X_2, Z_2X_0`. Again,
# all parametrized terms use the coefficient function ``smooth_rectangles``.
# We allow for a longer pulse duration of :math:`3\pi` and five smooth rectangles in
# each pulse shape.
#
# In summary, we use nine single-qubit generators and three two-qubit generators, with
# five rectangles in each pulse shape and each rectangle being given by an amplitude and
# a start and end time. The pulse sequence thus has :math:`(9+3)\cdot 5\cdot 3=180`
# parameters.

num_wires = 3
# New pulse hyperparameters
T = 3 * jnp.pi  # Longer total duration
eps = 0.1 * T
P = 5  # More rectangles in sum: P=5
f = partial(smooth_rectangles, k=k, max_amp=max_amp, eps=eps, T=T)

# Hamiltonian terms of the drift and parametrized parts of H
ops_H_d = [Z(0), Z(1), Z(2)]
ops_param = [P(w) for P in [X, Y, Z] for w in range(num_wires)]
ops_param += [Z(0) @ X(1), Z(1) @ X(2), Z(2) @ X(0)]

# Coefficients: 1. for drift Hamiltonian and smooth rectangles for parametrized part
coeffs = [1.0, 1.0, 1.0] + [f for op in ops_param]
# Build H
H = qml.dot(coeffs, ops_H_d + ops_param)
# Set tolerances for the ODE solver
atol = rtol = 1e-10

# Target unitary is Toffoli. We get its matrix and note that we do not need the dagger
# because Toffoli is Hermitian and unitary.
target = qml.Toffoli([0, 1, 2]).matrix()
target_name = "Toffoli"
print(f"Our target unitary is a {target_name} gate, with matrix\n{target.astype('int')}")


def pulse_matrix(params):
    """Compute the unitary time evolution matrix of the pulse for given parameters."""
    return qml.evolve(H, atol=atol, rtol=rtol)(params, T).matrix()


@jax.jit
def cost(params):
    """Compute the infidelity cost function for given parameters."""
    # Compute the unitary time evolution of the pulse Hamiltonian
    op_mat = pulse_matrix(params)
    # Compute the infidelity between the target and the pulse evolution
    return 1 - jnp.abs(jnp.trace(target @ op_mat)) / 2**num_wires


grad = jax.jit(jax.grad(cost))

#############################################################################
# We create initial parameters similar to above but allow for a larger number
# of :math:`1200` optimization steps and use a reduced learning rate
# in the optimization with adam. Our ``run_adam`` function from above comes
# in handy and also provides an overview of the optimization process in the
# produced plot.

times = [jnp.linspace(eps, T - eps, P * 2) for op in ops_param]
params = [jnp.hstack([[0.2 * (-1) ** i for i in range(P)], time]) for time in times]

num_steps = 1200
learning_rate = 2e-3
hist = run_adam(cost, grad, params, learning_rate, num_steps, target_name)

params_hist, cost_hist = list(zip(*hist))
min_params = params_hist[jnp.argmin(jnp.array(cost_hist))]

#############################################################################
# This looks promising: adam minimized the cost function successfully and we thus compiled
# a pulse sequence that implements a Toffoli gate! Let's look at the pulse
# sequence itself:

plot_optimal_pulses(hist, f, ops_param, T, target_name)

#############################################################################
# As we can see, the pulse sequence does not make use of all the single-qubit
# Pauli terms, and the smooth rectangles do not fill out the time at maximal
# amplitudes. This means that we can skip some of the terms, and that we
# probably can find shorter pulse sequences that produce a Toffoli with the
# same fidelity. If you are interested, take a shot at it and try to
# optimize the sequence regarding the number of generators and pulse duration!
#
# Conclusion
# ----------
#
# In this tutorial we calibrated a two-qubit and a three-qubit pulse sequence
# to obtain a CNOT and a Toffoli gate, respectively. For this, we used smooth
# rectangular pulse shapes together with toy pulse Hamiltonians, and obtained
# very good approximations to the target gates.
# Thanks to JAX, just-in-time (JIT) compiling and the PennyLane ``pulse``
# module, training the pulse sequences was simple to implement and fast to run.
#
# There are many different techniques in quantum optimal control that can be
# used to calibrate pulse sequences, some of which include gradient-based
# training. A widely-used technique called GRAPE [#KhanejaReiss05]_
# makes use of discretized pulses, which leads to a large number of free parameters
# to be optimized with gradient ascent.
# The technique shown here reduces the parameter count signficantly
# and provides smooth, bounded shapes by definition.
#
# While setting up the application examples, we accomodated for
# some requirements of realistic hardware, like smooth pulse shapes with bounded
# maximal amplitudes and bounded rates of change, and we tried to use only few
# interaction terms between qubits. However, it is important to note
# that the shown optimization remains a toy model for calibration of
# quantum hardware: We did not take into account the interaction terms
# or pulse shapes available on realistic devices and their control electronics.
# We also did not consider a unit system tied to real devices, and we
# ignored noise, which plays a very important role in today's quantum devices
# and in quantum optimal control.
# We leave the extension to real-world pulse Hamiltonians and noisy systems
# to a future tutorial--or maybe your work?
#
# Bibliography
# -------------
#
# .. [#KingmaBa14]
#
#     D. Kingma and J. Ba
#     "Adam: A method for Stochastic Optimization"
#     `arxiv:1412.6980 <https://arxiv.org/abs/1412.6980>`__, 2014
#
# .. [#KhanejaReiss05]
#
#     N. Khaneja, T. Reiss, C. Kehlet, T. Schulte-Herbrüggen, S.J. Glaser
#     "Optimal Control of Coupled Spin Dynamics:
#     Design of NMR Pulse Sequences by Gradient Ascent Algorithms"
#     `J. Magn. Reson. 172, 296-305 <https://www.ch.nat.tum.de/fileadmin/w00bzu/ocnmr/pdf/94_GRAPE_JMR_05_.pdf>`__,
#     2005
#
#
# About the author
# ----------------
# .. include:: ../_static/authors/david_wierichs.txt
