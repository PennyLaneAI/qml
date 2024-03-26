r"""
Gate calibration with reinforcement learning
============================================

Gate-based quantum circuits are the most common representation of quantum computations. These
provide an abstraction layer that enables the development of quantum algorithms without considering
the hardware in charge of the execution. However, every quantum platform offers a different set of
interactions and controls that define the natural operations that can be performed in the hardware.
These are known as the *native gates* of the device, and they constitute the fundamental
building blocks of any quantum algorithm executed in it. Therefore, it is essential that such
operations are performed as accurately as possible, which requires the careful tunning of the
hardware's controls. In this demo, we will learn how to use reinforcement learning to find the
optimal control parameters to accurately execute quantum gates using superconducting quantum
devices as an example.

Gate calibration
----------------

Calibrating quantum gates consists in finding the best possible control parameters of the device
that yield the most accurate gate execution. For instance, the gates in superconducting quantum
devices are performed by targetting the qubits with microwave pulses of the form

.. math:: \Omega(t)\sin(\phi(t) + \omega t)\,,

\ where :math:`\Omega(t)` is a time-dependent amplitude, :math:`\phi(t)` is a time-dependent phase,
and :math:`\omega` is the pulse's frequency. Hence, the proper execution of any gate relies on the
careful selection of these parameters in combination with the pulse duration, which we collectively
refer to as a *pulse program*. However, each qubit in the device has distinct properties, such as
the frequency and the connectivity to other qubits. These differences cause the same pulse programs
to produce different operations for every qubit. Consequently, every gate must be carefully
calibrated for each individual qubit in the hardware. For further details about superconducting
quantum computers and their control see `this demo <https://pennylane.ai/qml/demos/oqc_pulse/>`__.

A common strategy to calibrate quantum gates involves the detailed modelling of the quantum
computer, enabling the gate optimization through analytical and numerical techniques. Nevertheless,
developing such models requires an exhaustive characterization of the hardware and, even then, it
is very challenging to account for all the possible relevant factors. Therefore, it is unfeasible
to develop highly accurate models already for moderately sized quantum computers, in practice.

A promissing approach to obtain the optimal pulse programs of quantum computers is through the
direct interaction with the device, refraining from deriving any explicit model of the system.
Here, we frame qubit calibration as a reinforcement learning problem, drawing inspiration from the
experimentally-friendly method proposed in reference [#BaumPRXQ21]_. In this setting, a
reinforcement learning agent learns to calibrate the gates by tuning the control parameters and
directly observing the response of the qubits. Through this process, the agent implicitly learns an
effective model of the device, as it faces all the experimental nuances associated with the process
of executing the gates, such as the effect of the most relevant noise sources. This makes the
resulting agent an excellent calibrator that is robust to these phenomena.

Quantum gates in superconducting quantum computers
--------------------------------------------------

In superconducting quantum computers, the qubits are coupled to wave guides that allow us to target
them with microwave pulses. These pulses change the state of the qubits depending on their
properties, allowing us to perform operations like single qubit rotations. When we do so, we say
that we *drive* the qubit. Furthermore, some qubits are connected to others with coupling buses that
allow them to interact and build entanglement. The most common strategy to entangle two connected
qubits is with a cross-resonance operation, which consists on driving one qubit (target) through the
coupling with another one (control). Here, we will focus on designing a single-qubit gate, although
all the principles apply to two-qubit gates.

In general, we can consider the microwave pulse to have the form

.. math:: \Omega(t)\sin(\phi(t) + \omega t)\,,

\ where :math:`\Omega(t)` is a time-dependent amplitude, :math:`\phi(t)` is a time-dependent phase,
and :math:`\omega` is the pulse's frequency. Thus, we can engineer any single-qubit rotation by
tuning these parameters in combination with the pulse duration, which we call a *pulse program* all
together. Let's see how to simulate this with PennyLane!

The PennyLane :mod:`~pennylane.pulse` module provides the tools to simulate quantum systems
through time. This allows us to easily simulate the pulse-level control of quantum computers by
implementing an effective time-dependent Hamiltonian. For the case of superconducting quantum
computers, we have pre-defined functions to build such Hamiltonian. We often distinguish between two
main components: a constant drift term that describes the interaction between the qubits in our
system (see :func:`~pennylane.pulse.transmon_interaction`), and a time-dependent drive term that
represents the pulse (see :func:`~pennylane.pulse.transmon_drive`).

For simplicity, we will simulate a single-qubit device, but the same code will work with additional
qubits.
"""

import pennylane as qml

# Define the single-qubit device
qubit_freqs = [4.81]  # GHz
connections = []  # No connections
couplings = []  # No couplings
wires = [0]

H_int = qml.pulse.transmon_interaction(qubit_freqs, connections, couplings, wires)

######################################################################
# Now let's define the drive Hamiltonian associated to the pulse. We will consider the pulse to always
# be resonant with the qubit, and take constant amplitude and phase through time.
#

freq = qubit_freqs[0]  # Resonant with the qubit
amplitude = qml.pulse.constant
phase = qml.pulse.constant

H_drive = qml.pulse.transmon_drive(amplitude, phase, freq, wires)

######################################################################
# The resulting ``H_drive`` targets the qubit in wire 0. In case that we wanted to individually
# control more qubits, we would have to create a drive Hamiltonian for each of them with their
# corresponding wire.
#
# To simulate the time evolution of our system, we can use the function :func:`~pennylane.evolve`.
# This also allows us to extract the unitary matrix associated to the time evolution and visualize the
# resulting rotation axis in the Bloch sphere. Let's see the effect that the amplitude and phase have
# in the resulting single-qubit rotation.
#

import jax.numpy as jnp
import warnings

warnings.simplefilter("ignore")  # Silence JAX's warnings (nothing to worry about)

X, Y, Z = qml.PauliX(0).matrix(), qml.PauliY(0).matrix(), qml.PauliZ(0).matrix()


def get_rotation_axis(H, params, time):
    """Compute the rotation axis in the Bloch sphere associated the time evolution of H."""
    matrix = get_pulse_matrix(H, params, time)
    _, evecs = jnp.linalg.eig(matrix)
    return vector_to_bloch(evecs[:, 0])


def get_pulse_matrix(H, params, time):
    """Compute the unitary matrix associated to the time evolution of H."""
    return qml.evolve(H)(params, time, atol=1e-5).matrix()


def vector_to_bloch(vector):
    """Transform a vector into Bloch sphere coordinates."""
    rho = jnp.outer(vector, vector.conj())
    x, y, z = (
        jnp.trace(rho @ X).real.item(),
        jnp.trace(rho @ Y).real.item(),
        jnp.trace(rho @ Z).real.item(),
    )
    return [x, y, z]


# Complete Hamiltonian and pulse duration
H = H_int + H_drive
pulse_duration = 22.4  # ns

# Rotation axes at different amplitudes and constant phase
rot_axes_ampl = []
amplitudes = jnp.linspace(0, 0.04, 9)
for ampl in amplitudes:
    params = [ampl, 1.5]  # Constant phase
    rot_axes_ampl.append(get_rotation_axis(H, params, pulse_duration))

# Rotation axes at different phases and constant amplitude
rot_axes_phase = []
phases = jnp.linspace(0, 2 * jnp.pi, 9)
for phi in phases:
    params = [0.01, phi]  # Constant amplitude
    rot_axes_phase.append(get_rotation_axis(H, params, pulse_duration))

######################################################################
# Let's plot the results!
#

import qutip
import matplotlib.pyplot as plt


def plot_rotation_axes(rotation_axes, color=["#70CEFF"], fig=None, ax=None):
    """Plot the rotation axes in the Bloch sphere."""
    bloch = qutip.Bloch(fig=fig, axes=ax)
    bloch.sphere_alpha = 0.05
    bloch.vector_color = color
    bloch.add_vectors(rotation_axes)
    bloch.render()


fig = plt.figure(figsize=(10, 4))

ax_ampl = fig.add_subplot(1, 2, 1, projection="3d")
plot_rotation_axes(rot_axes_ampl, fig=fig, ax=ax_ampl)
ax_ampl.set_title("Varying amplitude")

ax_phase = fig.add_subplot(1, 2, 2, projection="3d")
plot_rotation_axes(rot_axes_phase, color=["#FFE096"], fig=fig, ax=ax_phase)
ax_phase.set_title("Varying phase")
plt.show()

######################################################################
# Playing with both parameters and the pulse duration, we can create any single-qubit rotation.
# However, every qubit in the quantum device has different properties, such as its intrinsic
# frequency, coupling to other qubits, etc. These differences cause the same pulse program (defined by
# :math:`\Omega(t),\,\phi(t),\,\omega,` and the duration) to result in different operations for every
# qubit, even in the absence of noise (e. g., try changing the qubit frequency above to 4.85 GHz).
# This is why we need to find the appropriate pulse program, i.e., calibrate, every gate for each
# individual qubit (or qubit pair) in our device.
#
# One way to do it is by simulating the quantum computer and optimizing the pulse program classically
# to execute a certain gate, similar to what we introduced in `this
# demo <https://pennylane.ai/qml/demos/tutorial_optimal_control/>`__. However, simulating our quantum
# devices is hard in practice (we would not need them otherwise!). Even for the smallest ones, we
# would still need a thorough characterization of the device to account for all the possible
# interactions and noise sources, which is unfeasible in most cases. Here, we focus on an alternative
# approach. The goal is to design a calibrator that can tailor pulse programs for each qubit in our
# quantum computer by directly observing its reaction to the pulses and adjusting their parameters
# accordingly, similar to what was proposed in [#BaumPRXQ21]_.
#

######################################################################
# Reinforcement learning basics
# -----------------------------
#
# In the typical reinforcement learning setting, we find two main entities: an agent and an
# environment. The environment contains the relevant information about the problem and defines the
# “rules of the game”. The main goal of the agent is to find the optimal strategy to perform a given
# task through the interaction with the environment.
#
# In order to complete the desired task, the agent can observe the environment and perform *actions*,
# which can affect the environment and change its *state*. This way, the interaction between them is
# cyclic, as depicted in the figure below. At a given point in time, the agent observes the
# environment's state. With this information, it can choose to perform a certain action. Doing so
# affects the environment, which provides the agent with the new state it is found in and a *reward*.
# The reward is a measure of how well the agent is performing the task given the last interaction.
#
# .. figure:: ../demonstrations/rl_pulse/sketch_rl.png
#    :align: center
#    :width: 75%
#
# The agent chooses its actions according to a *policy*, and **the ultimate goal is to learn the
# optimal policy that maximizes the obtained rewards**. In general, the policy can take any form, and
# its nature is usually related to the reinforcement learning algorithm that we implement to learn it.
# We will see how to learn the optimal policy later. For now, let's see how all these concepts apply
# to our task.
#

######################################################################
# Framing qubit calibration as a reinforcement learning problem
# -------------------------------------------------------------
#
# Our main objective is to accurately execute a desired quantum gate in our computer's qubits. To do
# so, we need a way to find the correct pulse program for each qubit. In reinforcement learning terms,
# our agent needs to learn the optimal policy to obtain the pulse program for every qubit. In this
# case, the environment is the quantum computer itself, and the agent's actions correspond to
# adjusting the different “control knobs” we can turn to modulate the microwave pulse.
#
# Naively, we could define the time-dependent pulse parameters beforehand, execute the pulse, evaluate
# the results, and modify the parameters for the next try based on the outcome. However, this
# black-box approach suffers from major limitations. For instance, we would need to perform a separate
# optimization for every single qubit every time we need to calibrate the quantum computer.
#
# With this approach, we would be neglecting the interaction loop between the agent and the
# environment, which is perhaps the most important aspect of the reinforcement learning pipeline. It
# is best if the agent receives information about the qubit's evolution during the pulse execution.
# This allows the agent to react to the qubit's peculiarities and adapt the pulse parameters
# accordingly to execute the target gate. This way, we can train a single agent to calibrate all the
# qubits in our computer.
#

######################################################################
# The hardest part of this approach is extracting the information about the qubit's evolution under
# the pulse, since observing it destroys the quantum state. Here, we follow a similar
# experimentally-friendly approach to the one introduced in [#BaumPRXQ21]_. The main idea is to
# split the pulse program into segments and evaluate the intermediate states:
#
# 1. First, we fix the pulse duration and split it into segments with constant parameters. This is
#    commonly known as a piece-wise constant (PWC) pulse.
# 2. We reset the qubit.
# 3. We perform quantum tomography to determine the qubit's state.
# 4. With this information, the agent fixes the parameters for the next pulse segment.
# 5. We reset the qubit and execute the pulse up to the last segment with fixed parameters.
# 6. We repeat steps 3-5 until we reach the end of the pulse and evaluate the average gate fidelity.
#
# 
# In the image below, we see an example of the tomography-parameters-reset&execute loop.
#
# .. figure:: ../demonstrations/rl_pulse/sketch_protocol.png
#    :align: center
#    :width: 90%
#
# With this protocol, the agent iteratively builds a PWC pulse according to the qubit's evolution.
# Even though this involves multiple executions to perform the intermediate tomography steps, the
# overall cost is rather low provided that it is only for the qubit(s) involved in the gate.
#

######################################################################
# Building an :math:`R_X(\pi/2)` calibrator
# -----------------------------------------
#
# Let's take all these concepts and apply them to train a calibrator for the :math:`R_X(\pi/2)` gate
# (a.k.a. :math:`\sqrt{X}`), which is a common native gate in superconducting quantum computers. To do
# so, we need to define: 
# 
# - The environment (hardware, actions, and rewards)
# - The agent (the policy and how to act)
# - The learning algorithm
#
# Then, we'll put all the pieces together and train our agent. Let's introduce these concepts one by
# one and implement them from scratch with PennyLane and JAX.
#

######################################################################
# The environment
# ```````````````
#
# As we mentioned earlier, the environment contains all the information about the problem. In an
# experimental setting, the actual superconducting quantum computer and how we interact with it would
# constitute the environment. Here, we will simulate it with PennyLane.
#
# We start by defining the quantum hardware. We will define an effective model for both the qubits and
# the pulse. Again, in order to keep the implementation as simple as possible, we will work with a
# single-qubit device. At the end of the tutorial, we provide insights of how to implement what we
# show here to multi-qubit devices and gates.
#

# Quantum computer
qubit_freqs = [4.81]  # GHz
connections = []  # No connections
couplings = []  # No couplings
wires = [0]

H_int = qml.pulse.transmon_interaction(qubit_freqs, connections, couplings, wires)

# Microwave pulse
pulse_duration = 22.4  # ns
n_segments = 8
segment_duration = pulse_duration / n_segments

freq = qubit_freqs[0]  # Resonant with the qubit
amplitude = qml.pulse.pwc(pulse_duration)
phase = qml.pulse.pwc(pulse_duration)

H_drive = qml.pulse.transmon_drive(amplitude, phase, freq, wires)

# Full time-dependent parametrized Hamiltonian
H = H_int + H_drive

######################################################################
# Now that we have the effective model of our system, we need to simulate its time evolution. Since we
# are simulating the whole process, we will simplify the qubit reset, evolution and tomography steps.
# We can simply stop the simulation, look at the qubit's state, and then continue after the agent
# chooses the parameters of the following segment. Hence, the environment's state will be the the
# qubit's state.
#
# We will do it with a :class:`~pennylane.QNode` that can evolve several states in parallel
# following different pulse programs to speed up the process.
#

import jax
from functools import partial

device = qml.device("default.qubit", wires=1)


@partial(jax.jit, static_argnames="H")
@partial(jax.vmap, in_axes=(0, None, 0, None))
@qml.qnode(device=device, interface="jax")
def evolve_states(state, H, params, t):
    qml.QubitStateVector(state, wires=wires)
    qml.evolve(H)(params, t, atol=1e-5)
    return qml.state()


state_size = 2 ** len(wires)

######################################################################
# Now that we have a model of the quantum computer and we have defined the environment's states, we
# can proceed to define the actions. As we mentioned before, the actions will adjust the knobs we can
# turn to generate the microwave pulse. We have four parameters to play with in our pulse program:
# amplitude :math:`\Omega(t)`, phase :math:`\phi(t)`, frequency :math:`\omega`, and duration. Out of
# those, we fix the duration beforehand (point 1 in the protocol), and we will always work with
# resonant pulses with the qubit, thus fixing the frequency.
#
# Hence, we will let the agent change the amplitude and the phase for every segment in our pulse
# program. To keep the pipeline as simple as possible, we will discretize their values within an
# experimentally feasible range, and associate every action to a combination of amplitude and phase
# values.
#

values_phase = jnp.linspace(-jnp.pi, jnp.pi, 9)[1:]
values_ampl = jnp.linspace(0.0, 0.2, 11)
ctrl_values = jnp.stack(
    (jnp.repeat(values_ampl, len(values_phase)), jnp.tile(values_phase, len(values_ampl))), axis=1
)
n_actions = len(ctrl_values)

######################################################################
# Finally, we need to define the reward function. In the typical reinforcement learning setting, there
# are rewards after every action is performed, as in the schematic depiction above. However, in some
# cases, those rewards are zero until the task is finished. For example, if we are training an agent
# to play chess, we cannot evaluate every single move on its own, and we need to wait until the game
# is resolved in order to provide the agent with a reward.
#
# Our case is similar to the chess example. The intermediate states visited along the time evolution
# do not necessarily provide a clear indication of how well the target gate is being executed.
# Therefore, we need to evaluate the pulse program as a whole after we have gone through all the pulse
# segments. The reward will be the average gate fidelity of our pulse program with respect to the
# target gate.
#
# In order to evaluate it, we sample several random initial states and apply the pulse program a few
# consecutive times. Then, we compute the average fidelity between the resulting intermediate and
# final states from our pulse, and the expected states from the target gate. Finally, we take the
# weighted average of all the fidelities, giving more relevance to the initial gate applications. This
# process of repeatedly applying the pulse programs accumulates the errors, making our reward function
# more sensitive to them. This allows the agent to better refine the pulse programs.
#
# The number of initial states and gate repetitions are hyper parameters that will add up really
# quickly with others such as the pulse duration, the segment duration, and so on. In order to keep
# the code as clean as possible, we will put all of them in a ``config`` container (a jit-friendly
# named tuple) and the code below will assume this container is being passed.
#

target = jnp.array(qml.RX(jnp.pi / 2, 0).matrix())  # RX(pi/2) gate


@partial(jax.jit, static_argnames=["H", "config"])
@partial(jax.vmap, in_axes=(0, None, None, None, None))
def compute_rewards(pulse_params, H, target, config, subkey):
    """Compute the reward for the pulse program based on the average gate fidelity."""
    n_gate_reps = config.n_gate_reps
    # Sample the random initial states
    states = jnp.zeros((config.n_eval_states, n_gate_reps + 1, state_size), dtype=complex)
    states = states.at[:, 0, :].set(sample_random_states(subkey, config.n_eval_states, state_size))
    target_states = states.copy()

    # Repeatedly apply the gates and store the intermediate states
    matrix = get_pulse_matrix(H, pulse_params, config.pulse_duration)
    for s in range(n_gate_reps):
        states = states.at[:, s + 1].set(apply_gate(matrix, states[:, s]))
        target_states = target_states.at[:, s + 1].set(apply_gate(target, target_states[:, s]))

    # Compute all the state fidelities (excluding the initial states)
    overlaps = jnp.einsum("abc,abc->ab", target_states[:, 1:], jnp.conj(states[:, 1:]))
    fidelities = jnp.abs(overlaps) ** 2

    # Compute the weighted average gate fidelities
    weights = 2 * jnp.arange(n_gate_reps, 0, -1) / (n_gate_reps * (n_gate_reps + 1))
    rewards = jnp.einsum("ab,b->a", fidelities, weights)
    return rewards.mean()


@partial(jax.jit, static_argnames=["n_states", "dim"])
def sample_random_states(subkey, n_states, dim):
    """Sample random states from the Haar measure."""
    subkey0, subkey1 = jax.random.split(subkey, 2)

    s = jax.random.uniform(subkey0, (n_states, dim))
    s = -jnp.log(jnp.where(s == 0, 1.0, s))
    norm = jnp.sum(s, axis=-1, keepdims=True)
    phases = jax.random.uniform(subkey1, s.shape) * 2.0 * jnp.pi
    random_states = jnp.sqrt(s / norm) * jnp.exp(1j * phases)
    return random_states


@jax.jit
def apply_gate(matrix, states):
    """Apply the unitary matrix of the gate to a batch of states."""
    return jnp.einsum("ab,cb->ca", matrix, states)


######################################################################
# The agent
# `````````
#
# The agent is a rather simple entity: it observes a state from the environment and selects an action
# among the ones we have defined above. The action selection is performed according to a policy, which
# is typically denoted by :math:`\pi`. In this case, we will use a stochastic policy
# :math:`\pi_{\mathbf{\theta}}(a_i|s_t)` that provides the probability of choosing the action
# :math:`a_i` given an observed state :math:`s_t` at a given time :math:`t`, according to some
# parameters :math:`\mathbf{\theta}`. Learning the optimal policy :math:`\pi^*` will consist on
# learning the parameters :math:`\mathbf{\theta}^*` that best approximate it
# :math:`\pi_{\mathbf{\theta}^*}\approx\pi^*`.
#
# We parametrize the policy with a small feed-forward neural network that takes the state :math:`s_t`
# as input, and provides the probability to select every action :math:`a_i` at the end. Therefore, the
# input and output layers have ``state_size=2`` and ``n_actions=88`` neurons, respectively. We include
# a hidden layer in between with a hyperbolic tangent activation function. Let's implement this with
# `Flax <https://flax.readthedocs.io/en/latest/>`__.
#

from flax import linen as nn


# Define the architecture
class MLP(nn.Module):
    """Multi layer perceptron (MLP) with a single hidden layer."""

    hidden_size: int
    out_size: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_size)(x)
        x = nn.tanh(x)
        x = nn.Dense(self.out_size)(x)
        return nn.softmax(jnp.sqrt((x * x.conj()).real))


policy_model = MLP(hidden_size=30, out_size=n_actions)

# Initialize the parameters passing a mock sample
key = jax.random.PRNGKey(7)
key, subkey = jax.random.split(key)

mock_state = jnp.empty((1, state_size))
policy_params = policy_model.init(subkey, mock_state)

######################################################################
# The policy parameters are stored in a nested dictionary structure. This kind of tree-like structure
# is known as a `pytree <https://jax.readthedocs.io/en/latest/pytrees.html>`__ in JAX, and we can see
# its structure.
#

jax.tree_util.tree_structure(policy_params)

######################################################################
# For every dense layer, we have a ``bias`` (vector) and a ``kernel`` (matrix) and they perform an
# operation of the form ``x @ kernel + bias``. The parameters are leaves in the tree and we can see
# their shapes by mapping the ``jnp.sape`` operation to the tree.
#

jax.tree_map(jnp.shape, policy_params)

######################################################################
# To act in the environment, we simply need to pass the state through the network and sample an action
# according to the discrete probability distribution provided by the output layer. However, we will
# see how to implement it below to act and compute the policy gradient at once.
#

######################################################################
# The learning algorithm
# ``````````````````````
#
# Among the plethora of reinforcement leraning algorithms, we use the REINFORCE [Williams 1987, 1992]
# algorithm. It belongs to the policy gradient algorithms family, which propose a parametrization of
# the policy (as we have conveniently done in the previous section) and directly optimize its
# parameters. The main principle of REINFORCE is to directly modify the policy to favour series of
# actions within the agent's experience that have lead to high rewards. This way, past beneficial
# actions are more likely to happen again.
#
# We achieve this by maximizing the expected *return* of the policy. To keep it simple, we can
# understand the return (typically :math:`G`) as the weighted sum of rewards along an *episode*, which
# are full executions of our reinforcement learning “game”. In our case, an episode would be the full
# execution of a pulse program, which is comprised by several interactions between the agent and the
# environment. Since the reward will only be given at the end, we will take the return to be the final
# reward. For those who are familiar with the reinforcement learning terminology, we're setting the
# discount factor :math:`\gamma=1`.
#
# We perform the maximization of the expected return by gradient ascent over the policy parameters. We
# can compute the gradient of the expected return as follows
#
# .. math:: \nabla_{\mathbf{\theta}} \mathbb{E}\left[G\right] = \mathbb{E}\left[\sum_{t=1}^{T}G_t\nabla_{\mathbf{\theta}}\log\pi_{\mathbf{\theta}}(a_t|s_t)\right],
#
# where the expectation values are over episodes sampled following the policy
# :math:`\pi_{\mathbf{\theta}}`. The sum goes over the episode time steps :math:`t`, where the agent
# observes the state :math:`s_t` and chooses to perform the action :math:`a_t`. The term
# :math:`\nabla_{\mathbf{\theta}}\log\pi_{\mathbf{\theta}}(a_t|s_t)` is known as the *score function*
# and it is the gradient of the logarithm of the probability with which the action is taken. Finally,
# :math:`G_t` is the return associated to the episode from time :math:`t` onwards, which is always the
# final reward of the episode, as we mentioned.
#
# To learn the optimal policy, we can estimate the gradient
# :math:`\nabla_{\mathbf{\theta}} \mathbb{E}\left[G\right]` by sampling a bunch of episodes following
# the current policy :math:`\pi_{\mathbf{\theta}}`. Then, we can perform a gradient ascent update to
# the policy parameters :math:`\mathbf{\theta}`, sample some episodes with the new parameters, and
# repeat until we converge to the optimal policy.
#
# Let's implement these ideas one by one, starting by the episode sampling. The episodes start with
# the qubit in the :math:`|0\rangle` state. Then, we start the observe-act loop, where the agent
# observes the qubit's state and chooses the parameters for the next pulse segment according to its
# policy. We can sample the action and compute the score function at the same time to make the code as
# efficient as possible. Finally, when we reach the end of the pulse program, we compute its reward.
# We like it when things go fast, so we'll sample all the episodes in parallel!
#


@partial(jax.jit, static_argnames=["H", "config"])
def play_episodes(policy_params, H, ctrl_values, target, config, key):
    """Play episodes in parallel."""
    n_episodes, n_segments = config.n_episodes, config.n_segments

    # Initialize the qubits on the |0> state
    states = jnp.zeros((n_episodes, n_segments + 1, target.shape[0]), dtype=complex)
    states = states.at[:, 0, 0].set(1.0)

    # Perform the PWC evolution of the pulse program
    pulse_params = jnp.zeros((n_episodes, 2, n_segments))
    actions = jnp.zeros((n_episodes, n_segments), dtype=int)
    score_functions = []
    for s in range(config.n_segments):
        # Observe the current state and select the parameters for the next pulse segment
        sf, (a, key) = act(states[:, s], policy_params, key)
        pulse_params = pulse_params.at[..., s].set(ctrl_values[a])

        # Evolve the states with the next pulse segment
        time_window = (
            s * config.segment_duration,  # Start time
            (s + 1) * config.segment_duration,  # End time
        )
        states = states.at[:, s + 1].set(evolve_states(states[:, s], H, pulse_params, time_window))

        # Save the experience for posterior learning
        actions = actions.at[:, s].set(a)
        score_functions.append(sf)

    # Compute the final reward
    key, subkey = jax.random.split(key)
    rewards = compute_rewards(pulse_params, H, target, config, subkey)
    return states, actions, score_functions, rewards, key


@jax.jit
def act(states, params, key):
    """Act on states with the current policy params."""
    keys = jax.random.split(key, states.shape[0] + 1)
    score_funs, actions = score_function_and_action(params, states, keys[1:])
    return score_funs, (actions, keys[0])


@jax.jit
@partial(jax.vmap, in_axes=(None, 0, 0))
@partial(jax.grad, argnums=0, has_aux=True)
def score_function_and_action(params, state, subkey):
    """Sample an action and compute the associated score function."""
    probs = policy_model.apply(params, state)
    action = jax.random.choice(subkey, policy_model.out_size, p=probs)
    return jnp.log(probs[action]), action


######################################################################
# Now that we have a way to sample the episodes, we need to process the collected experience to
# compute the gradient of the expected return
# :math:`\nabla_{\mathbf{\theta}} \mathbb{E}\left[G\right]`. First, however, we will define two
# utility functions: one to add a list of pytrees together (helps with the temporal sum), and one to
# add extra dimensions to arrays to enable broadcasting. These solve a couple of technicalities that
# will make the following code much more readable.
#


@jax.jit
def sum_pytrees(pytrees):
    """Sum a list of pytrees."""
    return jax.tree_map(lambda *x: sum(x), *pytrees)


@jax.jit
def adapt_shape(array, reference):
    """Adapts the shape of an array to match the reference (either a batched vector or matrix).
    Example:
    >>> a = jnp.ones(3)
    >>> b = jnp.ones((3, 2))
    >>> adapt_shape(a, b).shape
    (3, 1)
    >>> adapt_shape(a, b) + b
    Array([[2., 2.],
           [2., 2.],
           [2., 2.]], dtype=float32)
    """
    n_dims = len(reference.shape)
    if n_dims == 2:
        return array.reshape(-1, 1)
    return array.reshape(-1, 1, 1)


######################################################################
# In order to compute the gradient, we need to compute the sum within the expectation value for every
# episode. However, we will give this expression a twist and bring our reinforcement learning skills a
# step further. We will subtract a baseline :math:`b` to the return such that
#
# .. math:: \nabla_{\mathbf{\theta}} \mathbb{E}\left[G\right] = \mathbb{E}\left[\sum_{t=1}^{T}(G_t - b(s_t))\nabla_{\mathbf{\theta}}\log\pi_{\mathbf{\theta}}(a_t|s_t)\right].
#
# Intuitively, the magnitude of the reward is arbitrary and depends on our function of choice. Hence,
# it provides the same information if we shift it. For example, our reward based on the average gate
# fidelity is bound between zero and one, but everything would conceptually be the same if we added
# one. In a sense, we could consider the original expression to have a baseline of zero, and this new
# one to be a generalization. Any baseline is valid provided that it does not depend on the action
# :math:`a_t` because this ensures it has a null expectation value, leaving the gradient unafected.
#
# While these baselines leave the expectation value of the gradient unchanged, they do have an impact
# in its variance. Here, we will implement the optimal state-independent baseline that minimizes the
# variance of the gradient. The optimal baseline for the :math:`k`-th component of the gradient is:
#
# .. math:: b_k = \frac{\mathbb{E}\left[G_t\left(\partial_{\theta_k}\log\pi_{\mathbf{\theta}}(a|s)\right)^2\right]}{\mathbb{E}\left[\left(\partial_{\theta_k}\log\pi_{\mathbf{\theta}}(a|s)\right)^2\right]}\,,
#
# where :math:`\partial_{\theta_k}\log\pi_{\mathbf{\theta}}(a|s)` is the :math:`k`-th component of the
# score function. Thus, this baseline has the same shape as the gradient. You can find a proof in
# Chapter 6.3.1 in [#Dawid22]_. Reducing the variance of our
# estimates significantly speeds up the training process by providing better gradient updates with
# less samples.
#


@jax.jit
def reinforce_gradient_with_baseline(episodes):
    """Estimates the parameter gradient from the episodes with a state-independent baseline."""
    _, _, score_functions, returns = episodes
    ret_episodes = returns.sum()  # Sum of episode returns to normalize the final value
    # b
    baseline = compute_baseline(episodes)
    # G - b
    ret_minus_baseline = jax.tree_map(lambda b: adapt_shape(returns, b) - b, baseline)
    # sum((G - b) * sf)
    sf_sum = sum_pytrees(
        [jax.tree_map(lambda r, s: r * s, ret_minus_baseline, sf) for sf in score_functions]
    )
    # E[sum((G - b) * sf)]
    return jax.tree_map(lambda x: x.sum(0) / ret_episodes, sf_sum)


@jax.jit
def compute_baseline(episodes):
    """Computes the optimal state-independent baseline to minimize the gradient variance."""
    _, _, score_functions, returns = episodes
    n_episodes = returns.shape[0]
    n_segments = len(score_functions)
    total_actions = n_episodes * n_segments
    # Square of the score function: sf**2
    sq_sfs = jax.tree_map(lambda sf: sf**2, score_functions)
    # Expected value: E[sf**2]
    exp_sq_sfs = jax.tree_map(
        lambda sqsf: sqsf.sum(0, keepdims=True) / total_actions, sum_pytrees(sq_sfs)
    )
    # Return times score function squared: G*sf**2
    r_sq_sf = sum_pytrees(
        [jax.tree_map(lambda sqsf: adapt_shape(returns, sqsf) * sqsf, sq_sf) for sq_sf in sq_sfs]
    )
    # Expected product: E[G_t*sf**2]
    exp_r_sq_sf = jax.tree_map(lambda rsqsf: rsqsf.sum(0, keepdims=True) / total_actions, r_sq_sf)
    # Ratio of espectation values: E[G_t*sf**2] / E[sf**2]  (avoid dividing by zero)
    return jax.tree_map(lambda ersq, esq: ersq / jnp.where(esq, esq, 1.0), exp_r_sq_sf, exp_sq_sfs)


######################################################################
# Finally, we can choose any optimizer that we like to perform the policy parameter updates with the
# gradient information. We will use the Adam optimizer [#kingma14]_ provided by
# `Optax <https://optax.readthedocs.io/en/latest/api.html#optax.adam>`__.
#

import optax


def get_optimizer(params, learning_rate):
    """Create and initialize an Adam optimizer for the parameters."""
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)
    return optimizer, opt_state


######################################################################
# And we will define our parameter update function to perform the gradient ascent step. These
# optimizers default to gradient descent, which is the most common in machine learning problems.
# Hence, we'll have to subtract the parameter udpate they provide to go in the opposite direction.
#


def update_params(params, gradients, optimizer, opt_state):
    """Update model parameters with gradient ascent."""
    updates, opt_state = optimizer.update(gradients, opt_state, params)
    new_params = jax.tree_map(lambda p, u: p - u, params, updates)  # Negative update
    return new_params, opt_state


######################################################################
# The training
# ------------
#
# We have all the building blocks that we need to train an :math:`R_X(\pi/2)` calibrator for our
# superconducting quantum computer. We just need to ensemble the pieces together and choose the right
# parameters for the task.
#
# Let's define the config object that will contain all the hyper parameters of the training process.
#

from collections import namedtuple

hyperparams = [
    "pulse_duration",  # Total pulse duration
    "segment_duration",  # Duration of every pulse segment
    "n_segments",  # Number of pulse segments
    "n_episodes",  # Episodes to estimate the gradient
    "n_epochs",  # Training iterations
    "n_eval_states",  # Random states to evaluate the fidelity
    "n_gate_reps",  # Gate repetitions for the evaluation
    "learning_rate",  # Step size of the parameter update
]
Config = namedtuple("Config", hyperparams, defaults=[None] * len(hyperparams))

config = Config(
    pulse_duration=pulse_duration,
    segment_duration=segment_duration,
    n_segments=8,
    n_episodes=200,
    n_epochs=320,
    n_eval_states=200,
    n_gate_reps=2,
    learning_rate=5e-3,
)

######################################################################
# Finally, the training loop:
# 1. Sample episodes
# 2. Compute the gradient
# 3. Update the policy parameters
#

optimizer, opt_state = get_optimizer(policy_params, config.learning_rate)

learning_rewards = []
for epoch in range(config.n_epochs):
    *episodes, key = play_episodes(policy_params, H, ctrl_values, target, config, key)
    grads = reinforce_gradient_with_baseline(episodes)
    policy_params, opt_state = update_params(policy_params, grads, optimizer, opt_state)

    learning_rewards.append(episodes[3].mean())
    if (epoch % 40 == 0) or (epoch == config.n_epochs - 1):
        print(f"Iteration {epoch}: reward {learning_rewards[-1]:.4f}")

plt.plot(learning_rewards)
plt.xlabel("Training iteration")
plt.ylabel("Average reward")
plt.grid(alpha=0.3)

######################################################################
# The algorithm has converged to a policy with a very high average reward!! Let's see what this agent
# is capable of.
#

######################################################################
# Calibrating the qubits
# ----------------------
#
# After the training, we have a pulse calibrator for the :math:`R_X(\pi/2)` gate with a high average
# gate fidelity. The next step is to actually use it to calibrate the qubits in our device.
#
# Notice that, during the whole training process, the actions of our agent have been stochastic. At
# every step, the policy provides the probability to choose every action and we sample one
# accordingly. When we deploy our calibrator, we want it to consistently provide good pulse programs
# on a single pass through the qubit. Therefore, rather than sampling the actions, we take the one
# with the highest probability.
#
# Let's define a function to extract the pulse program from the policy.
#


def get_pulse_program(policy_params, H, ctrl_values, config):
    """Extract the pulse program from the trained policy."""
    state = jnp.zeros((1, state_size), dtype=complex).at[:, 0].set(1.0)
    pulse_params = jnp.zeros((1, ctrl_values.shape[-1], config.n_segments))
    for s in range(config.n_segments):
        probs = policy_model.apply(policy_params, state)
        action = probs.argmax()
        pulse_params = pulse_params.at[..., s].set(ctrl_values[action])
        time_window = (s * config.segment_duration, (s + 1) * config.segment_duration)
        state = evolve_states(state, H, pulse_params, time_window)
    return pulse_params[0]


######################################################################
# We can evaluate the resulting average gate fidelity over several random initial states. Remember
# that the reward is a combination of the gate fidelity over several gate repetitions, which is a
# lower bound to the actual gate fidelity.
#


def evaluate_program(pulse_program, H, target, config, subkey):
    """Compute the average gate fidelity over 1000 random initial states."""
    states = sample_random_states(subkey, 1000, state_size)
    target_states = jnp.einsum("ab,cb->ca", target, states)
    pulse_matrix = qml.matrix(qml.evolve(H)(pulse_program, config.pulse_duration))
    final_states = jnp.einsum("ab,cb->ca", pulse_matrix, states)
    fidelities = qml.math.fidelity_statevector(final_states, target_states)
    return fidelities


######################################################################
# Let's extract the pulse program for our qubit, the rotation axis of the resulting gate, and the
# average gate fidelity.
#

pulse_program = get_pulse_program(policy_params, H, ctrl_values, config)

matrix = get_pulse_matrix(H, pulse_program, config.pulse_duration)
_, evecs = jnp.linalg.eigh(matrix)
rot_axis = vector_to_bloch(evecs[:, 1])

fidelities = evaluate_program(pulse_program, H, target, config, key)
avg_gate_fidelity = fidelities.mean()

######################################################################
# We can plot the amplitude and phase over time to get a better idea of what's going on. Furthermore,
# we can visualize the rotation axis in the Bloch sphere to see its alignment with the :math:`X` axis.
#

ts = jnp.linspace(0, pulse_duration - 1e-3, 100)
fig, axs = plt.subplots(ncols=3, figsize=(14, 4), constrained_layout=True)

axs[0].plot(ts, qml.pulse.pwc(pulse_duration)(pulse_program[0], ts), color="#70CEFF", linewidth=3)
axs[0].set_ylabel("Amplitude (GHz)", fontsize=14)
axs[0].set_yticks(values_ampl)
axs[0].set_ylim([values_ampl[0], values_ampl[-1]])

axs[1].plot(ts, qml.pulse.pwc(pulse_duration)(pulse_program[1], ts), color="#FFE096", linewidth=3)
axs[1].set_ylabel("Phase (rad)", fontsize=14)
axs[1].set_yticks(
    values_phase,
    ["$-3\pi/4$", "$-\pi/2$", "$-\pi/4$", "0", "$\pi/4$", "$\pi/2$", "$3\pi/4$", "$\pi$"],
)
axs[1].set_ylim([values_phase[0], values_phase[-1]])

for ax in axs[:2]:
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=12)
    ax.set_xlabel("Time (ns)", fontsize=14)

axs[2].axis("off")
ax2 = fig.add_subplot(1, 3, 3, projection="3d")
plot_rotation_axes(rot_axis, fig=fig, ax=ax2)
ax2.set_title(f"Average gate fidelity {avg_gate_fidelity:.3f}", fontsize=14)
plt.show()

######################################################################
# Beyond single-qubit quantum computers and gates
# -----------------------------------------------
#
# All the concepts that we have learned throughout the demo are directly applicable to quantum
# computers with more qubits, two-qubit gates, and even other platforms beyond superconducting
# circuits. Perhaps the most straightforward extension of the code presented here would be calibrating
# an entangling gate, such as a CNOT. We will provide an overview of how to adapt this tutorial to do
# it, but they take significantly longer to train (mainly because the pulses need to be much longer).
#
# The first thing we need to build an entangling gate is a second qubit in our device with a different
# frequency coupled to the first one.
#

# Quantum computer
qubit_freqs = [4.81, 4.88]  # GHz
connections = [[0, 1]]
couplings = [0.02]
wires = [0, 1]

H_int = qml.pulse.transmon_interaction(qubit_freqs, connections, couplings, wires)

######################################################################
# The CNOT gate is typically constituted by a series of single-qubit pulses and cross-resonant (CR)
# pulses. Every single-qubit pulse is resonant with the qubit it targets, i.e., it has the same
# frequency, whereas CR pulses are shone on the control qubit at the target's frequency. This way, the
# target is indirectly driven through the coupling with the control qubit, which entangles them.
#
# We will make the CR pulse take the first qubit as control and the second as target. We will
# implement what's known as an echoed CR pulse, which consists of flipping the state of the control
# qubit (applying an :math:`X` gate) in the middle of the CR pulse. The second half of the pulse is
# the negative of the previous one. This “echoes out” the rapid interactions between the qubits, while
# preserving the entangling slower interaction that we are interestd in, as introduced in [#SheldonPRA16]_.
#
# Our full pulse ansatz will consist of sandwitching the echoed CR gate between single qubit pulses.
# Therefore, we will have a total of six PWC pulses: two single qubit ones, two CR, and two single
# qubit.
#

# Microwave pulse
pulse_duration_sq = 22.4  # Single qubit pulse duration
pulse_duration_cr = 100.2  # CR pulse duration


def get_drive(timespan, freq, wire):
    """Parametrized Hamiltonian driving the qubit in wire with a fixed frequency."""
    amplitude = qml.pulse.pwc(timespan)
    phase = qml.pulse.pwc(timespan)
    return qml.pulse.transmon_drive(amplitude, phase, freq, wire)


pulse_durations = jnp.array(
    [
        0,
        pulse_duration_sq,
        pulse_duration_sq,
        pulse_duration_cr,
        pulse_duration_cr,
        pulse_duration_sq,
        pulse_duration_sq,
    ]
)
change_times = jnp.cumsum(pulse_durations)
timespans = [(t0.item(), t1.item()) for t0, t1 in zip(change_times[:-1], change_times[1:])]

H_sq_0_ini = get_drive(timespans[0], qubit_freqs[0], wires[0])
H_sq_1_ini = get_drive(timespans[1], qubit_freqs[1], wires[1])
H_cr_pos = get_drive(timespans[2], qubit_freqs[1], wires[0])  # Target qubit 0 with freq from 1
H_cr_neg = get_drive(timespans[3], qubit_freqs[1], wires[0])
H_sq_0_end = get_drive(timespans[4], qubit_freqs[0], wires[0])
H_sq_1_end = get_drive(timespans[5], qubit_freqs[1], wires[1])

######################################################################
# Notice that the CR pulses are an order of magnitude longer than the single-qubit ones. This is
# because the drive on the target qubit is dampened by the coupling constant, thus requiring much
# longer times to observe a comparable effect compared to driving it directly.
#
# We now have to put everything together in the :class:`~pennylane.QNode` that will be in charge of
# the evolution. In general, we could add all the Hamiltonians together and we would get a PWC drive
# without a problem. However, we need to flip the control qubit between the CR pulses and revert it at
# the end, as well as fix the second CR pulse to be the negative of the previous one. Notice that both
# CR pulses share their parameters.
#
# Hence, we will add all the single qubit pulses together in a single Hamiltonian, and treat the CR
# ones separately. We will need to carefully handle the parameters for this, but the code can look
# similar to the following.
#

H_sq = H_sq_0_ini + H_sq_1_ini + H_sq_0_end + H_sq_1_end


@jax.jit
@partial(jax.vmap, in_axes=(0, None, 0, None))
@qml.qnode(device=device, interface="jax")
def evolve_states(state, params, t):
    params_sq, params_cr = params
    qml.QubitStateVector(state, wires=wires)
    # Single qubit pulses
    qml.evolve(H_int + H_sq)(params_sq, t, atol=1e-5)

    # Echoed CR
    qml.evolve(H_int + H_cr_pos)(params_cr, t, atol=1e-5)
    qml.PauliX(0)  # Flip control qubit
    qml.evolve(H_int - H_cr_neg)(params_cr, t, atol=1e-5)  # Negative CR
    qml.PauliX(0)  # Recover control qubit

    # Single qubit pulses
    qml.evolve(H_int + H_sq)(params_sq, t, atol=1e-5)

    return qml.state()


######################################################################
# The state of the environment would be the 2-qubit state, and we would treat the qubit pair as our
# previous single qubit. We would need to decide how many segments we wish to split each pulse into,
# and define the time windows within ``play_episodes`` accordingly.
#
# Finally, in order to train a calibrator for a quantum computer with several qubits (or qubit pairs),
# we need to define a separate drive Hamiltonian for each indidividual qubit. Then, every episode
# samples one of these at random to perform the evolution. This way, if we are calibrating the
# :math:`R_X(\pi/2)` gate for a device with multiple qubits, we would have a separate ``H_drive`` with
# the control pulse for each qubit and every episode would carry the time evolution of
# ``H_int + H_drive`` for a randomly selected qubit. This is a bit more involved to adapt. Mainly, we
# would have to randomly sample which qubits to evolve in every episode inside ``play_episodes``.
# Notice that, since we evolve the qubits in parallel, ``evolve_states`` should be parallelized over
# the Hamiltonian too.
#
# Don't forget that gates involving multiple qubits can go in both directions!
#

######################################################################
# Conclusions
# -----------
#
# In this demo, we have learned how quantum gates are implemented in superconducting quantum circuits,
# and how to design an experimentally-friendly calibrator with reinforcement leraning. To do so, we
# have learned the fundamental principles of reinforcement learning, the REINFORCE algorithm, and how
# to frame the calibration of quantum computers within this framework. Then, we have put everything
# together to calibrate our qubit, and we have learned how to apply the principles explored here to
# gates involving multiple qubits and larger devices.
#
# To continue learning about this topic, try implementing one of the two extensions we explain above.
# To learn more about reinforcement learning, we recommend [#SuttonBarto18]_ for an introduction to
# the topic, and [#Dawid22]_ for an introduction of machine
# learning for physics (reinforcement learning in chapter 6). To learn more about how superconducting
# quantum computers work, see [#KrantzAPR19]_ for an extensive review, and the related `PennyLane
# documentation <https://docs.pennylane.ai/en/stable/code/qml_pulse.html>`__.
#
# Finally, check out these other demos for alternative ways to tune pulse programs.
#
# References
# ----------
#
# .. [#BaumPRXQ21]
#
#     Y. Baum, et. al. (2019)
#     "Experimental Deep Reinforcement Learning for Error-Robust Gate-Set Design on a Superconducting Quantum Computer."
#     `PRX Quantum 2(4), 040324 <https://link.aps.org/doi/10.1103/PRXQuantum.2.040324>`__.
#
# .. [#Dawid22]
#
#     A. Dawid, et. al. (2022)
#     "Modern applications of machine learning in quantum sciences."
#     `arXiv:2204.04198 <https://arxiv.org/abs/2204.04198>`__.
#
# .. [#kingma14]
#
#     D. Kingma and J. Ba. (2014)
#     "Adam: A method for Stochastic Optimization."
#     `arXiv:1412.6980 <https://arxiv.org/abs/1412.6980>`__.
# 
# .. [#SheldonPRA16]
#
#     S. Sheldon, E. Magesan, J. M. Chow and J. M. Gambetta. (2016)
#     "Procedure for systematically tuning up cross-talk in the cross-resonance gate."
#     `Phys. Rev. A, 93(6), 060302 <https://link.aps.org/doi/10.1103/PhysRevA.93.060302>`__.
#
# .. [#SuttonBarto18]
#
#     R. S. Sutton and A. G. Barto. (2018)
#     "Reinforcement learning: An introduction."
#     `MIT Press <https://mitpress.mit.edu/9780262039246/reinforcement-learning/>`__.
#
# .. [#KrantzAPR19]
#
#     P. Krantz, M. Kjaergaard, F. Yan, T. P. Orlando, S. Gustavsson and W. D. Oliver. (2019)
#     "A quantum engineer's guide to superconducting qubits."
#     `Applied physics reviews 6(2) <https://pubs.aip.org/aip/apr/article/6/2/021318/570326/A-quantum-engineer-s-guide-to-superconducting>`__.
#

######################################################################
# About the author
# ----------------
# 
# .. include:: ../_static/authors/borja_requena.txt
