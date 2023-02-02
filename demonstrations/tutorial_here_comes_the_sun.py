r"""

Here comes the SU(N): multivariate quantum gates and gradients
==============================================================

.. meta::
    :property="og:description": Learn about multivariate quantum gates for optimization

    :property="og:image": https://pennylane.ai/qml/_images/SpecialUnitary.png

.. related::
    TODO


*Author: David Wierichs — Posted: xx.*

Variational quantum algorithms have been promoted to be useful for many applications.
When designing such an algorithm, a central task is to choose the quantum circuit ansatz,
which provides a parametrization of quantum states. In the course of the algorithm,
the corresponding parameters are then optimized in order to reduce some cost function.
The ansatz can have a big impact on the quantum states that can be found (expressivity)
within the algorithm and on the optimization behaviour. It also strongly affects the
cost of executing the algorithm on quantum hardware, and the strength of the noise
which enters the computation. Finally, the application itself often influences, or
even fixes, the choice of ansatz.

.. figure:: ../demonstrations/here_comes_the_sun/pennylane_guy_scratching_head.png
    :align: center
    :width: 50%

While a number of best practices for ansatz design have been developed,
a lot is still unknown about the connection between circuit structures and the
resulting properties. Therefore circuit design often is based on intuition or heuristics
as well, and an ansatz reported in the literature might just have turned out
to work particularly well for a given problem, or might fall into a "standard"
category of circuits.
One approach to make an ansatz generic and avoid arbitrary choices in
the circuit design is to perform the most general operations *locally* on a few qubits
and to reduce the design question to choosing the sets of qubits such operations are
applied to and their order.

.. figure:: ../demonstrations/here_comes_the_sun/sun_fabric.png
    :align: center
    :width: 50%

In this tutorial, you will learn about a particular quantum gate which can act like
*any* gate on the corresponding qubits by chosing the parameters accordingly.
We will then look at a custom differentiation rule for this gate, construct 
a simple ansatz, and use it in a toy minimization problem.
Let's start with a brief math intro (no really, just a little bit).

The special unitary group SU(N) and its Lie algebra
---------------------------------------------------

The gate we will look at is given by a specific parametrization of the special
unitary group :math:`SU(N)`, where :math:`N=2^n` is the Hilbert space dimension of the gate
for :math:`n` qubits. Mathematically, the group can be defined as the set of operators
(or matrices) that can be inverted by taking their adjoint and that have 
determinant :math:`1`. All gates acting on :math:`n` qubits are elements of :math:`SU(N)`
up to a global phase.

The group :math:`SU(N)` is a Lie group, and its associated Lie algebra 
is :math:`\mathfrak{su}(N)`. For our purposes it will be sufficient to look at a matrix
representation of the algebra and we may define it as

.. math::

    \mathfrak{su}(N) = \{\Omega \in \mathbb{C}^{N\times N}| \Omega^\dagger=-\Omega, \Tr{\Omega}=0\}.

We will use so-called canonical coordinates for the algebra which are simply the coefficients
of an algebra element :math:`\Omega` in the Pauli basis:
    
.. math::

    \Omega &= \sum_{m=1}^d \theta_m G_m\\
    \theta_m &\in \mathbb{R}\\
    G_m &\in \mathcal{P}^{(n)} = i \left\{I,X,Y,Z\right\}^n \setminus \{i I^n\}.

As you can see, we actually use the Pauli basis words with a prefactor :math:`i`, and
we skip the identity, because it does not have a vanishing trace.
We can use the canonical coordinates of the algebra to express a group element
:math:`SU(N)` as well and the ``qml.SpecialUnitary`` gate we will use is defined as

.. math::

    U(\bm{\theta}) = \exp\left\{\sum_{m=1}^d \theta_m G_m \right\}.

The number of coordinates and of Pauli words in :math:`\mathcal{P}^{(n)}` is :math:`d=4^n-1`.
Therefore, this will be the number of parameters a single ``qml.SpecialUnitary`` gate acting on
:math:`n` qubits will take. For example, it takes just three parameters for a single qubit and 
a moderate number of 15 parameters for two qubits, but already requires 63 parameters for
three qubits.

Obtaining the gradient
----------------------

In variational quantum algorithms, we typically use the circuit to prepare a quantum state and
then we measure some observable :math:`H`. The resulting real-valued output is considered to be the
cost function :math:`C` that should be minimized. If we want to use gradient-based optimization for
this task, we need a method to compute the gradient :math:`\nabla C` in addition to the cost
function itself. As derived in the publication [#wiersema]_, this is possible on quantum hardware
as long as the :math:`SU(N)` gate itself can be implemented.
We will not go through the entire derivation, but note the following key points:

    #. The gradient with respect to all :math:`d` parameters of an :math:`SU(N)` gate can be
       computed using :math:`2d` auxiliary circuits. Each of the circuits contains one additional
       operation compared to the original circuit, namely a ``qml.PauliRot`` gate with rotation
       angles :math:`\pm\frac{\pi}{2}. Not that these Pauli rotations act on up to :math:`n` 
       qubits.
    #. This differentiation uses automatic differentiation during compilation and postprocessing,
       which becomes expensive for large :math:`n`, but the differentiation is quantum 
       hardware-compatible.
    #. The computed gradient is not an approximative technique but allows for an exact computation 
       of the gradient on simulators. On quantum hardware, this leads to unbiased gradient
       estimators.
    #. With ``qml.SpecialUnitary`` and its gradient we effectively implement a so-called
       Riemannian gradient flow on the qubits the gate acts on.

The implementation in PennyLane takes care of creating the additional circuits and evaluating
them, together with adequate post-processing into the gradient :math:`\nabla C`.

.. figure:: ../demonstrations/here_comes_the_sun/mathy_riemannian_flow_pic.png
    :align: center
    :width: 50%

Comparing ansatz structures
---------------------------

We discussed above that there are many circuit architectures available and that choosing
a suitable ansatz is important but can be difficult. Here we will compare a simple ansatz
based on the ``qml.SpecialUnitary`` gate discussed above to other approaches that fully
parametrize the special unitary group for the respective number of qubits.
In particular, we will compare ``qml.SpecialUnitary`` to standard decompositions from the
literature that parametrize :math:`SU(N)` with elementary gates, as well as to a sequence
of Pauli rotation gates that also allows to create any special unitary.
Let us start by defining the decomposition of a two-qubit unitary.
We choose the decomposition, which is optimal but not unique, from [#vatan]_.
The Pauli rotation sequence is available in PennyLane
via ``qml.ArbitraryUnitary`` and we will not need to implement it ourselves.

- introduce one or two other ansatze with actually fewer parameters. 

"""

import pennylane as qml
import numpy as np

import time

start = time.process_time()


def two_qubit_decomp(params, wires):
    """Implement an arbitrary SU(4) gate on two qubits
    using the decomposition from Theorem 5 in
    https://arxiv.org/pdf/quant-ph/0308006.pdf"""
    i, j = wires
    # Single U(2) parameterization on qubit 1
    qml.Rot(*params[:3], wires=i)
    # Single U(2) parameterization on qubit 2
    qml.Rot(*params[3:6], wires=j)
    # CNOT with control on qubit 2
    qml.CNOT(wires=[j, i])
    # Rz and Ry gate
    qml.RZ(params[6], wires=i)
    qml.RY(params[7], wires=j)
    # CNOT with control on qubit 1
    qml.CNOT(wires=[i, j])
    # Ry gate on qubit 2
    qml.RY(params[8], wires=j)
    # CNOT with control on qubit 1
    qml.CNOT(wires=[j, i])
    # Single U(2) parameterization on qubit 1
    qml.Rot(*params[9:12], wires=i)
    # Single U(2) parameterization on qubit 2
    qml.Rot(*params[12:15], wires=j)


def decomp(params, wires):
    """Implement an arbitrary SU(2**n) gate on n qubits."""
    if len(wires) == 1:
        one_qubit_decomp(params, wires)
    elif len(wires) == 2:
        two_qubit_decomp(params, wires)
    else:
        raise ValueError("Not implemented for more than 2 wires")


operations = {
    "Decomposition": two_qubit_decomp,
    "PauliRot sequence": qml.ArbitraryUnitary,
    "SU(N) gate": qml.SpecialUnitary,
}

##############################################################################
# Now that we have the templates for the composition approach in place, we construct a toy
# problem to solve using the ansatze. We will sample a random Hamiltonian in the Pauli basis
# (this time without the prefactor :math:`i`)
# with independent coefficients that follow a normal distribution:
#
# .. math::
#
#   H = \sum_{m=1}^d h_m G_m,\quad h_m\sim \mathcal{N}(0,1)
#
# We will work with six qubits.

num_wires = 6
wires = list(range(num_wires))
np.random.seed(62213)

coefficients = np.random.randn(4**num_wires - 1)
basis = qml.ops.qubit.special_unitary.pauli_basis_matrices(num_wires)  # Create the Pauli basis
H = qml.math.tensordot(coefficients, basis, axes=[[0], [0]])
E_min = np.linalg.eigvalsh(H).min()  # Compute the ground state energy
print(E_min)
H = qml.Hermitian(H, wires=wires)

##############################################################################
# Using the toy problem Hamiltonian and the three ansatze for :math:`SU(N)` operations
# from above, we create a circuit template that applies these operations in a brickwall-like
# architecture with ``num_blocks=2`` blocks and each operation acting on ``num_wires_op=2`` qubits.
# For this we define a ``QNode``:

num_blocks = 2
num_wires_op = 2
d = num_wires_op**4 - 1  # d = 15 for two-qubit operations
dev = qml.device("default.qubit", wires=num_wires)
# two blocks with two layers each. Each layer contains three operations with d parameters each
param_shape = (num_blocks, num_wires_op, num_wires // num_wires_op, d)
init_params = np.zeros(param_shape)


def circuit(params, operation=None):
    """Apply an operation in a brickwall-like pattern to a qubit register and measure H.
    Parameters are assumed to have the dimensions (number of blocks, number of
    wires per operation, number of operations per layer, and number of parameters
    per operation), in that order.
    """
    for params_block in params:
        for i, params_layer in enumerate(params_block):
            for j, params_op in enumerate(params_layer):
                wires_op = [
                    w % num_wires for w in range(num_wires_op * j + i, num_wires_op * (j + 1) + i)
                ]
                operation(params_op, wires_op)
    return qml.expval(H)


qnode = qml.QNode(circuit, dev, interface="jax")
print(qml.draw(qnode)(init_params, qml.SpecialUnitary))

##############################################################################
# We can now proceed to preparing the optimization task using this circuit
# and an optimization routine of our choice. For simplicity, we run a vanilla gradient
# descent optimization with fixed learning rate for 200 steps. For auto-differentiation
# we make use of JAX.

import jax

jax.config.update("jax_enable_x64", True)

learning_rate = 5e-4
num_steps = 200
init_params = jax.numpy.array(init_params)
grad_fn = jax.jit(jax.jacobian(qnode), static_argnums=1)
qnode = jax.jit(qnode, static_argnums=1)

##############################################################################
# With this configuration, let's run the optimization!

energies = {}
for name, operation in operations.items():
    params = init_params.copy()
    energy = []
    for step in range(num_steps):
        cost = qnode(params, operation)
        params = params - learning_rate * grad_fn(params, operation)
        energy.append(cost)  # Store energy value
        if step % 10 == 0:  # Report current energy
            print(cost)

    energy.append(qnode(params, operation))  # Final energy value
    energies[name] = energy

##############################################################################
# So, did it work? Judging from the intermediate energy values it seems that the optimization
# outcomes differ notably. But let's take a look at the relative error in energy across the
# optimization process.

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1)
for name in operations.keys():
    error = (energies[name] - E_min) / abs(E_min)
    ax.plot(list(range(len(error))), error, label=name)

ax.set(
    xlabel="Iteration",
    ylabel="Relative error",
    yscale="log",
)
ax.legend()
plt.show()

##############################################################################
# We find that the optimization indeed performs significantly better for ``qml.SpecialUnitary``
# than for the other two general unitaries, while using the same number of parameters. This
# means that we found a particularly well-trainable parametrization of the unitaries that
# allows to reduce the energy of the prepared quantum state more easily.
#
# Finally, let's look at the optimization behaviour with a shot-based device. After all,
# PennyLane supports the differentiation of all three two-qubit operations we are using
# by applying the parameter-shift rule, enabling training on quantum hardware
# and shot-based simulators.
#
# TODO: REVERT num_wires reduction once we can JIT the decomposition of SpecialUnitary?

num_wires = 2
wires = list(range(num_wires))
np.random.seed(62213)

coefficients = np.random.randn(4**num_wires - 1)
basis = qml.ops.qubit.special_unitary.pauli_basis_matrices(num_wires)  # Create the Pauli basis
H = qml.math.tensordot(coefficients, basis, axes=[[0], [0]])
E_min = np.linalg.eigvalsh(H).min()  # Compute the ground state energy
H = qml.Hermitian(H, wires=wires)
param_shape = (num_blocks, num_wires_op, num_wires // num_wires_op, d)
init_params = jax.numpy.zeros(param_shape)

dev_shots = qml.device("default.qubit", wires=num_wires, shots=2000)
qnode_shots = qml.QNode(circuit, dev_shots, interface="jax")
grad_fn = jax.jacobian(qnode_shots)

energies_shots = {}
for name, operation in operations.items():
    params = init_params.copy()
    energy = []
    for step in range(num_steps):
        cost = qnode(params, operation)
        params = params - learning_rate * grad_fn(params, operation)
        energy.append(cost)  # Store energy value
        if step % 10 == 0:  # Report current energy
            print(cost)

    energy.append(qnode_shots(params, operation))  # Final energy value
    energies_shots[name] = energy

end = time.process_time()
print(end - start)
fig, ax = plt.subplots(1, 1)
for name in operations.keys():
    error = (energies_shots[name] - E_min) / abs(E_min)
    ax.plot(list(range(len(error))), error, label=name)

ax.set(
    xlabel="Iteration",
    ylabel="Relative error",
    yscale="log",
)
ax.legend()
plt.show()

#
# References
# ----------
#
# .. [#vatan]
#
#     Farrokh Vatan and Colin Williams,
#     "Optimal Quantum Circuits for General Two-Qubit Gates", arXiv:quant-ph/0308006, (2003).
#
# .. [#wiersema]
#
#     R. Wiersema, D. Lewis, J. F. Carrasquilla, and N. Killoran,
#     in preparation, arXiv:23xx.xxxxx, (2023)
#
#
# About the author
# ----------------
# .. include:: ../_static/authors/david_wierichs.txt
