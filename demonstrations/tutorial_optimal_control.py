r"""
Optimal control for gate compilation
====================================

.. meta::
    :property="og:description": Optimize a pulse program to obtain a digital CNOT gate.
    :property="og:image": https://pennylane.ai/qml/_images/thumbnail_tutorial_optimal_control.png

.. related::

    tutorial_pulse_programming101 Introduction to pulse programming in PennyLane
    ahs_aquila

*Author: David Wierichs. Posted: xx June, 2023.*


Quantum computations largely are phrased in the picture of quantum circuits that consist
of digital quantum gates. However, most quantum hardware does not provide such digital
gates as native operations. Instead, they play sequences of analogue pulses, for example
laser pulses that interact with trapped ions or Rydberg atoms, or microwave pulses acting
on superconducting qubits.
The internal state of the quantum device then follows the Schrödinger equation dictated
by the time-dependent Hamiltonian created with the pulse sequence.
The full pulse sequence will effect a change of the quantum state depending on the
types, amplitudes and durations of the pulses.
If we parametrize the pulses, for example via their time-dependent amplitudes, we
get a whole *space* of pulse programs, and in this tutorial we are interested in finding
a specific digital gate in this space, which is an example for *quantum optimal control*.

More concretely, we will optimize simple pulse programs on two and three qubits to
obtain CNOT and Toffoli gates. 


For an introduction see
[the demo on differentiable pulse programming in PennyLane].
Instead of optimizing pulses to yield digital gates that are used in quantum circuits,
we may use them directly to solve minimization problems, as showcased in the
[ctrl-VQE demo].
If you are interested in specific hardware pulses, take a look at
[our demo on the QuEra Aquila device].


Quantum optimal control
-----------------------

The overarching goal of quantum optimal control is to find the best way to steer
a microscopical physical system such that its dynamics matches a desired behaviour.
The meaning of "best" and "desired behaviour" will depend on the specific
task, and it is important to specify underlying assumptions and constraints on
how the system may be controlled in order to make the problem statement well-defined.
If we specified all these details, optimal control theory is concerned with
questions like
"How close can the system get to modelling the desired behaviour?",
"How can we find the best (sequence of) control parameters to obtain the desired behaviour?",
or
"What is the shortest time in which the system can reach a specific state, given some
initial state?".

In this tutorial, we consider the control of few-qubit systems through pulse sequences,
with the goal to produce a given target, a digital gate, to the highest-possible precision.
At the same time we attempt to obtain pulse shapes and control parameters that are
realistic to be implemented (to some degree).
To tackle this problem, we will choose an ansatz for the pulse sequence that contains
free parameters and define a cost function that measures the deviation of the qubit
system from the target gate.
Then we minimize the cost function by optimizing the pulse parameters until we
find the desired gate to a sufficient precision--or can no longer improve on the
approximation we found.

Pulse programming
-----------------

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
or crosstalk, the accidental interaction with other parts of a larger surrounding system.

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
In this tutorial we will describe the distance between two unitary matrices :math:`U` and
:math:`V` with the commonly used function

.. math::

    d(U,V) = \frac{1}{2^N}\big|\operatorname{tr}[U^\dagger V]\big|.

It is similar to the distance measure obtained from the
[Frobenius norm]
but allows us to ignore differences in the global phase.
With the distance measure in our hands, we can construct the cost function to be minimized 
while training the pulse parameters:

.. math::

    C(\boldsymbol{p}) = 1 - d(U_\text{target}, U(\boldsymbol{p}, T)).

Here :math:`U_\text{target}` is the unitary matrix of the gate that we want to compile
and we exclude the total duration :math:`T` of the pulse sequence because we will
consider it as a constraint to the optimization problem, rather than a free variable.


Smooth-rectangle pulses
-----------------------

Before diving into the applications themselves, let's look at a small building block
that we will use a lot: rectangular pulses.
We start with a simple rectangular pulse

.. math::

    R_0(t, (\Omega, t_0, t_1)) = \Omega \Theta(t-t_0) \Theta(t_1-t)

where :math:`\Omega` is the amplitude, :math:`t_{0,1}` are the start and end
times of the pulse, and :math:`\Theta(t)` is the
[Heaviside step function]
that is one for :math:`t\geq 0` and zero otherwise.

There are two main issues with :math:`R_0` for our purposes:

  #. The function :math:`\mathbb{1}_{[t_0, t_1]}` is not differentiable with respect
     to the times :math:`t_0` and :math:`t_1` in the conventional sense, and in
     particular we cannot differentiate the resulting :math:`U(\boldsymbol{p},T)`
     within the automatic differentiation framework of JAX.

  #. The instantaneous change in the amplitude will not be realizable in practice.
     In reality, the pulses describe some electromagnetic control field that only
     can be changed at a bounded rate and in a smooth manner. :math:`R_0` is not
     only not smooth, it is not even continuous. So we should consider smooth
     pulses with a bounded rate of change instead.

We can solve both these issues by smoothening the rectangular pulse:
We simply replace the step functions above by a smooth variant, namely sigmoid functions:

.. math::

    R(t, (\Omega, t_0, t_1), k) &= \Omega S(t-t_0, k) S(t_1-t, k)\\
    S(t, k) &= (1+\exp(-k t))^{-1}.

We introduced an additional parameter, :math:`k`, that controls the steepness of the sigmoid
functions and can be adapted to the constraints posed by hardware on the maximal rate of change.
In contrast to :math:`R_0`, its sister :math:`R` is smooth in all three arguments
:math:`\Omega`, :math:`t_0` and :math:`t_1`.
Let's implement this function using JAX's ``numpy``. We immediately implement the product
of the two sigmoids in the function ``sigmoid_rectangle``.
In addition, we define a sigmoid alone, as well as a ``normalize`` function based 
on the sigmoid curve that will come in handy later on.
"""
from functools import partial
import numpy as np
import matplotlib.pyplot as plt

import pennylane as qml
import jax
from jax import numpy as jnp
jax.config.update("jax_enable_x64", True)
import optax

def sigmoid_rectangle(t, t_0, t_1, k=1.):
    """Smoothened unit rectangle pulse."""
    return 1/(1 + jnp.exp(-k * (t-t_0)) + jnp.exp(-k * (t_1-t)) + jnp.exp(-k * (t_1-t_0)))

def sigmoid(t, k=1.):
    """Differentiable sigmoid function to smoothen step functions."""
    return 1/(1 + jnp.exp(-k * t))

def normalize(t, k=1.):
    """Smoothly normalize a real input value to the interval [-1, 1]."""
    return 2 * sigmoid(t, k) - 1.

"""
Pulse ansatz for CNOT
---------------------

In this first example we will tune a two-qubit pulse to produce a standard CNOT gate.
We start by choosing a system Hamiltonian.
It contains the drift term :math:`H_d = Z_0 + Z_1`, i.e. a Pauli :math:`Z` operator
acting on each qubit, with a constant unit amplitude.
The parametrized part uses seven generating terms: all three Pauli operators acting
on either of the qubits (:math:`X_0, Y_0, Z_0, X_1, Y_1, Z_1`) and a single interaction
term :math:`X_0X_1`.
The seven coefficient functions :math:`f_i` are all identical to a smooth-rectangle
pulse shape (see the box below) but use distinct parameters for each term.
Each coefficient function sums :math:`P` smooth rectangles with individual amplitudes
and start and end times. Overall, this leads to :math:`n=21P` parameters in :math:`H`.

We implement the sum over multiple ``sigmoid_rectangle`` functions, including two
normalization steps:
First, a normalization of the final output value to the interval
:math:`[-\Omega_\text{max}, \Omega_\text{max}]`, which
allows us to bound the maximal amplitudes of the pulses to a realizable range.
Second, a normalization of the start and end times of the single retangles to the interval
:math:`[\epsilon, T-\epsilon]`, which makes sure that the pulses start and end close to zero.
"""

def smooth_rectangles(params, t, k=2., max_amp=1., eps=0., T=1.):
    """Compute the sum of :math:`P` smooth-rectangle pulses.

    Args:
        params (tensor_like): Amplitudes and start and end times for the rectangles,
            in the order ``[amp_1, ... amp_P, t_{1, 0}, t_{1, 1}, ... t_{P, 0}, t_{P, 1}]``.
        t (float): Time at which to evaluate the pulse function.
        k (float): Steepness of the sigmoid functions that delimit the rectangles
        max_amp (float): Maximal amplitude of the rectangles. The output will be normalized to
            the interval ``[-max_amp, max_amp]``.
        eps (float): Margin to beginning and end of the pulse sequence within which the 
            start and end times of the individual rectangles need to lie.
        T (float): Total duration of the pulse.
    Returns:
        float: Value of sum of smooth-rectangle pulses at ``t`` for the given parameters.
    """
    num_pulses = len(params) // 3
    # Split amplitudes from times
    amps, times = jnp.split(params, [num_pulses])
    # Normalize times to be sufficiently far away from 0 and T
    times = sigmoid(times - T / 2, k=1.) * (T - 2 * eps) + eps
    # Extract start and end times of single rectangles
    t_0, t_1 = jnp.reshape(times, (-1, 2)).T
    # Contract amplitudes with products of sigmoids (unit rectangles)
    value = jnp.dot(amps, sigmoid_rectangle(t, t_0, t_1, k))
    # Normalize the output value to be in [-max_amp, max_amp] with standard steepness
    return max_amp * normalize(value, k=1.)


"""
Let's look at this function for some example parameters:
"""

# Bind hyperparameters to the smooth_rectangles function
T = 2 * jnp.pi
k = 20.
max_amp = 1.
eps = 0.1 * T
f = partial(smooth_rectangles, k=k, max_amp=max_amp, eps=eps, T=T)

# Set some arbitrary amplitudes and times
amps = jnp.array([0.4, -0.2, 1.9, -2.])
times = jnp.array([0.2, 0.6, 1.2, 1.8, 2.1, 3.7, 4.9, 5.9])
params = jnp.hstack([amps, times])

plot_times = jnp.linspace(0, T, 300)
plot_f = [f(params, t) for t in plot_times]

plt.plot(plot_times, plot_f)
ax = plt.gca()
ax.set(xlabel="Time t", ylabel=r"Pulse function $f(p, t)$")
#plt.show()
plt.close()
        
"""
Note that the rectangles are barely visible as such with these generic parameters.
They will be optimized to be closer to rectangles in the training workflows below.
Also note that the normalization of the final output value is not a simple clipping
step, but again a smooth function. As a consequence, the values ``1.9` and ``-2.``,
which are not in the interval ``[-1, 1]`` given by the ``max_amp`` parameter,
are not set to ``+-1`` but take smaller absolute values.

Using this function, we may build the parametrized pulse Hamiltonian and the
cost function discussed above.
"""
X, Y, Z = qml.PauliX, qml.PauliY, qml.PauliZ

num_wires = 2
# Hamiltonian terms of the drift and parametrized parts of H
ops_H_d = [Z(0), Z(1)]
ops_param = [X(0), Y(0), Z(0), X(1), Y(1), Z(1), X(0) @ X(1)]
# Coefficients: 1. for drift Hamiltonian and smooth rectangles for parametrized part
coeffs = [1., 1.] + [f for op in ops_param]
# Build H
H = qml.dot(coeffs, ops_H_d + ops_param)
# Set tolerances for the ODE solver
atol = rtol = 1e-10

# Target unitary is CNOT. We get its matrix and note that we do not need the dagger
# because CNOT is Hermitian and unitary.
target = qml.CNOT([0, 1]).matrix()

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

"""
For the arbitrary parameters from above, we naturally get a rather arbitrary unitary
time evolution, which does not match the CNOT at all:
"""

params = [params] * 7
print(cost(params))
print(jnp.round(pulse_matrix(params), 4))

"""
Before we can start the optimization, we need to set the number of rectangles
we want to use in each pulse, and we require initial parameters. For the latter,
we start with small alternating amplitudes and regularly distributed start and end times
for the smoothened rectangles.
"""
num_pulses = 3
# Initial parameters for the start and end times of the rectangles
times = [jnp.linspace(eps, T-eps, num_pulses * 2) for op in ops_param]
# All initial parameters: small alternating amplitudes and times
params = [jnp.hstack([[0.1 * (-1)**i for i in range(num_pulses)], time]) for time in times]

"""
Now we are all set up to train the parameters of the pulse sequence to produce
our target gate, the CNOT. We will use the 
[adam]
optimizer, conveniently
implemented in the
[``optax``]
library. We keep track of the optimization in a list with the parameters and cost
function values and plot the cost across the optimization.
"""
learning_rate = 0.5

# Initialize the adam optimizer
optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(params)
# Initialize a memory buffer for the optimization
hist = [(params.copy(), cost(params))]
num_steps = 5
for step in range(num_steps):
    g = grad(params)
    updates, opt_state = optimizer.update(g, opt_state, params)
    
    params = optax.apply_updates(params, updates)
    hist.append([params, c:=cost(params)])
    if (step+1) % 50 == 0:
        print(f"Step {step+1:4d}: {c:.6f}")

params_hist, cost_hist = list(zip(*hist))
plt.plot(list(range(num_steps+1)), cost_hist)
ax = plt.gca()
ax.set(xlabel="Iteration", ylabel="Infidelity $1-d(U_{CNOT}, U(p))$", yscale="log")
plt.show()

"""
As we can see, adam runs into an oscillating behaviour during the optimization. The
precision with which it can minimize the cost function depends on the expressivity
of the pulses we use, but also on the precision with which we run the ODE solver.

Let's pick the parameters with the smallest cost function we observed and take
a look at the pulses we found:
"""
fig, ax = plt.subplots(1, 1, figsize=(10, 7))
min_params, min_cost = hist[np.argmin(cost_hist)]
for p, op in zip(min_params, ops_param):
    label = op.name
    if isinstance(label, str):
        label = [label]
    label = "$" + " ".join([f"{n[-1]}_{w}" for w, n in zip(op.wires, label)]) + "$"
    ax.plot(plot_times, [f(p, t) for t in plot_times], label=label)
ax.legend()
title = f"CNOT, Fidelity={1-min_cost:.6f}"
ax.set(xlabel="Time $t$", ylabel=r"Pulse function $f(p, t)$", title=title)
plt.show()

"""
We observe that a single rectangular pulse is sufficient for most of the 
generating terms in the Hamiltonian, and we see that their shape is closer to
actual rectangles now, in particular for those with a saturated amplitude.

The final fidelity tells us that we achieved our goal of finding a pulse
sequence that implements a CNOT gate on the two qubit system.
It could be optimized further for example by running the optimization for more
training iterations, by tuning the optimizer to avoid oscillations, or by
increasing the precision with which we run the ODE solver.

Pulse sequence for Toffoli
--------------------------

"""
from itertools import product, combinations
num_wires = 3
# Hamiltonian terms of the drift and parametrized parts of H
ops_H_d = [Z(0), Z(1), Z(2)]
ops_param = [P(w) for P in [X, Y, Z] for w in range(num_wires)]
ops_param += [P(w)@Q(v) for P, Q in product([X, Y, Z],repeat=2) for w, v in combinations(range(num_wires), r=2)]
#ops_param += [X(0) @ X(1), Y(1) @ Y(2), Z(0) @ Z(2)]
# Coefficients: 1. for drift Hamiltonian and smooth rectangles for parametrized part
coeffs = [1., 1., 1.] + [f for op in ops_param]
# Build H
H = qml.dot(coeffs, ops_H_d + ops_param)
# Set tolerances for the ODE solver
atol = rtol = 1e-10

# Target unitary is Toffoli. We get its matrix and note that we do not need the dagger
# because Toffoli is Hermitian and unitary.
target = qml.Toffoli([0, 1, 2]).matrix()

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

num_pulses = 4
jitter = lambda *_: 0.1 * (
    jnp.array(np.random.random(num_pulses * 2) * (1 - (jnp.linspace(0, 1, num_pulses * 2) - 0.5)**2))
)
# Initial parameters for the start and end times of the rectangles
times = [jnp.linspace(eps, T-eps, num_pulses * 2) + jitter() for op in ops_param]
# All initial parameters: small alternating amplitudes and times
params = [jnp.hstack([[0.1 * (-1)**i for i in range(num_pulses)], time]) for time in times]

learning_rate = 0.5

# Initialize the adam optimizer
optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(params)
# Initialize a memory buffer for the optimization
hist = [(params.copy(), cost(params))]
num_steps = 1000
for step in range(num_steps):
    g = grad(params)
    updates, opt_state = optimizer.update(g, opt_state, params)
    
    params = optax.apply_updates(params, updates)
    hist.append([params, c:=cost(params)])
    if (step+1) % 50 == 0:
        print(f"Step {step+1:4d}: {c:.6f}")

params_hist, cost_hist = list(zip(*hist))
plt.plot(list(range(num_steps+1)), cost_hist)
ax = plt.gca()
ax.set(xlabel="Iteration", ylabel="Infidelity $1-d(U_{Toffoli}, U(p))$", yscale="log")
plt.show()

fig, axs = plt.subplots(2, 1, figsize=(10, 14), gridspec_kwargs={"hspace": 0.}, sharex=True)
min_params, min_cost = hist[np.argmin(cost_hist)]
for p, op in zip(min_params, ops_param):
    label = op.name
    if isinstance(label, str):
        ax = axs[0]
        label = [label]
    else:
        ax = axs[1]
    label = "$" + " ".join([f"{n[-1]}_{w}" for w, n in zip(op.wires, label)]) + "$"
    ax.plot(plot_times, [f(p, t) for t in plot_times], label=label)
axs[0].legend(title="Single-qubit terms")
axs[0].legend(title="Two-qubit terms")
title = f"Toffoli, Fidelity={1-min_cost:.6f}"
axs[0].set(ylabel=r"Pulse function $f(p, t)$", title=title)
axs[1].set(xlabel="Time $t$", ylabel=r"Pulse function $f(p, t)$")
plt.show()

"""

#############################################################################
"""
