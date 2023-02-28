r"""

Here comes the SU(N): multivariate quantum gates and gradients
==============================================================

.. meta::
    :property="og:description": Learn about multivariate quantum gates for optimization

    :property="og:image": https://pennylane.ai/qml/_images/SpecialUnitary.png

.. related::

   tutorial_vqe A brief overview of VQE
   tutorial_general_parshift General parameter-shift rules for quantum gradients
   tutorial_unitary_designs Unitary designs and their uses in quantum computing


*Author: David Wierichs — Posted: xx March 2023.*

Variational quantum algorithms have been promoted to be useful for many applications.
When designing these algorithms, a central task is to choose the quantum circuit ansatz,
which provides a parametrization of quantum states. In the course of the variational algorithm,
the circuit parameters are then optimized in order to minimize some cost function.
The choice of the ansatz can have a big impact on the quantum states that can be found
by the algorithm (expressivity), and on the optimization behaviour (trainability).
It also typically affects the
cost of executing the algorithm on quantum hardware, and the strength of the noise
that enters the computation. Finally, the application itself often influences, or
even fixes, the choice of ansatz, which can lead to constraints in the ansatz design.

.. figure:: ../demonstrations/here_comes_the_sun/scratching_head_in_front_of_circuits.jpeg
    :align: center
    :width: 50%

While a number of best practices for ansatz design have been developed,
a lot is still unknown about the connection between circuit structures and the
resulting properties. Therefore, circuit design often also is based on intuition or heuristics,
and an ansatz reported in the literature might just have turned out
to work particularly well for a given problem, or might fall into a "standard"
category of circuits.

It is therefore interesting to construct circuit ansätze that are rather generic and
avoid arbitrary choices in the design that might bias the optimization landscape.
One such approach is to perform the most general operations *locally* on a few qubits
and to reduce the design question to choosing the subsets of qubits these operations are
applied to (as well as their order). In particular, we will consider a general local operation
that comes without any preferred optimization direction.

.. figure:: ../demonstrations/here_comes_the_sun/sun_fabric.jpeg
    :align: center
    :width: 50%

In this tutorial, you will learn about ``qml.SpecialUnitary``, a particular quantum gate which
can act like *any* gate on the corresponding qubits by chosing the parameters accordingly.
Then we will look at a custom derivative rule for this gate and compare it to two
alternative differentiation strategies. Finally, we will compare the performance of
``qml.SpecialUnitary`` for a toy minimization problem to the one of two alternative general
local gates.

Let's start with a brief math intro (no really, just a little bit).

The special unitary group SU(N) and its Lie algebra
---------------------------------------------------

The gate we will look at is given by a specific parametrization of the special
unitary group :math:`\mathrm{SU}(N)`, where :math:`N=2^n` is the Hilbert space dimension of the gate
for :math:`n` qubits. Mathematically, the group can be defined as the set of operators
(or matrices) that can be inverted by taking their adjoint and that have
determinant :math:`1`. All gates acting on :math:`n` qubits are elements of :math:`\mathrm{SU}(N)`
up to a global phase.

The group :math:`\mathrm{SU}(N)` is a Lie group, and its associated Lie algebra
is :math:`\mathfrak{su}(N)`. For our purposes it will be sufficient to look at a matrix
representation of the algebra and we may define it as

.. math::

    \mathfrak{su}(N) =
    \{\Omega \in \mathbb{C}^{N\times N}| \Omega^\dagger=-\Omega, \operatorname{Tr}[\Omega]=0\}.

We will use so-called canonical coordinates for the algebra which are simply the coefficients
of an algebra element :math:`\Omega` in the Pauli basis:

.. math::

    \Omega &= \sum_{m=1}^d \theta_m G_m\\
    \theta_m &\in \mathbb{R}\\
    G_m &\in \mathcal{P}^{(n)} = i \left\{I,X,Y,Z\right\}^n \setminus \{i I^n\}.

As you can see, we actually use the Pauli basis words with a prefactor :math:`i`, and
we skip the identity, because it does not have a vanishing trace.
We can use the canonical coordinates of the algebra to express a group element
:math:`\mathrm{SU}(N)` as well and the ``qml.SpecialUnitary`` gate we will use is defined as

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
for :math:`\mathrm{SU}(N)` gates as long as the gates themselves can be implemented.
The implementation in PennyLane follows the decomposition idea described in App. F3, the
main text of [#wiersema]_ additionally proposes a method that scales better in some scenarios
but requires additional gates to be available on the quantum hardware. Here we will focus on the
former method.

We will not go through the entire derivation, but note the following key points:

    #. The gradient with respect to all :math:`d` parameters of an :math:`\mathrm{SU}(N)` gate can be
       computed using :math:`2d` auxiliary circuits. Each of the circuits contains one additional
       operation compared to the original circuit, namely a ``qml.PauliRot`` gate with rotation
       angles :math:`\pm\frac{\pi}{2}`. Note that these Pauli rotations act on up to :math:`n`
       qubits.
    #. This differentiation method uses automatic differentiation during compilation and
       classical coprocessing steps, but is compatible with quantum hardware. For large :math:`n`,
       the classical processing steps quickly may become prohibitively expensive.
    #. The computed gradient is not an approximative technique but allows for an exact computation
       of the gradient on simulators. On quantum hardware, this leads to unbiased gradient
       estimators.
    #. With ``qml.SpecialUnitary`` and its gradient we effectively implement a so-called
       Riemannian gradient flow *locally*, i.e. on the qubits the gate acts on.

The implementation in PennyLane takes care of creating the additional circuits and evaluating
them, together with adequate post-processing into the gradient :math:`\nabla C`.

Comparing gradient methods
--------------------------

Before we dive into using ``qml.SpecialUnitary`` in an optimization task, let's compare
a few methods to compute the gradient with respect to the parameters of such a gate.
In particular, we will look at a finite difference approach, the stochastic parameter-shift
rule, and the custom gradient method we described above.
We will use the standard central difference recipe given by

.. math::

    \partial_{\text{FD},\theta_j}C(\bm{\theta})
    =\left[C\left(\bm{\theta}+\frac{\delta}{2}\bm{e}_j\right)
    -C\left(\bm{\theta}-\frac{\delta}{2}\bm{e}_j\right)\right] / \delta.

Here, :math:`\delta` is a shift parameter we need to choose and :math:`\bm{e}_j` is the
:math:`j`-th canonical basis vector, i.e. the all-zeros vector with a one in the
:math:`j`-th entry.
The stochastic parameter-shift rule is a differentiation recipe developed for multi-parameter
gates like the :math:`\mathrm{SU}(N)` gates [#banchi]_. It involves the approximate
evaluation of an integration by sampling *splitting times* :math:`\tau` and evaluating an
expression close to the non-stochastic parameter-shift rule for each sample. For more details,
also consider the
:doc:`demo on the stochastic parameter-shift rule </demos/tutorial_stochastic_parameter_shift>`.

So let's dive into a toy example and explore the three gradient methods!
We start by defining a simple one-qubit circuit that contains a single :math:`\mathrm{SU}(2)`
gate and measures the expectation value of :math:`H=\frac{3}{5} Z - \frac{4}{5} Y`.
As ``qml.SpecialUnitary`` requires automatic differentiation subroutines even for the
hardware-ready derivative recipe, we will make use of JAX.
"""

import pennylane as qml
import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
jnp = jax.numpy

dev = qml.device("default.qubit", wires=1)
H = 0.6 * qml.PauliZ(0) - 0.8 * qml.PauliY(0)


def qfunc(theta):
    qml.SpecialUnitary(theta, wires=0)
    return qml.expval(H)


circuit = qml.QNode(qfunc, dev, interface="jax", diff_method="parameter-shift")

theta = jnp.array([0.4, 0.2, -0.5])

##############################################################################
# Now we need to set up the differentiation methods. For this demonstration, we will
# keep the first and last entry of ``theta`` fixed and only compute the gradient for the
# second parameter.
#
# We start with the central difference
# recipe, using a shift scale of :math:`\delta=0.75`. This choice of :math:`\delta`,
# which is much larger than usual for numerical differentiation on classical computers,
# is adapted to the scenario of shot-based gradients, as demonstrated further below.

# We compute the derivative with respect to the second entry of theta, so we need e_2:
unit_vector = np.array([0.0, 1.0, 0.0])


def central_diff_grad(theta, delta):
    plus_eval = circuit(theta + delta / 2 * unit_vector)
    minus_eval = circuit(theta - delta / 2 * unit_vector)
    return (plus_eval - minus_eval) / delta


delta = 0.75
print(f"Central difference: {central_diff_grad(theta, delta):.5f}")

##############################################################################
# Next up, we implement the stochastic parameter-shift rule. Of course we do not do
# so in full generality but for the particular circuit in this example. We will
# sample ten splitting times to obtain the gradient entry. For each splitting time,
# we need to insert a Pauli-:math:`Y` rotation (because :math:`\theta_2` belongs to
# the Pauli-:math:`Y` component of :math:`A(\bm{\theta})`). For this we define
# an auxiliary circuit.


@jax.jit
@qml.qnode(dev, interface="jax")
def aux_circuit(theta, tau, sign):
    qml.SpecialUnitary(tau * theta, wires=0)
    # This corresponds to the parameter-shift evaluations of RY at 0
    qml.RY(-sign * np.pi / 2, wires=0)
    qml.SpecialUnitary((1 - tau) * theta, wires=0)
    return qml.expval(H)


def stochastic_parshift_grad(theta, num_samples):
    grad = 0
    splitting_times = np.random.random(size=num_samples)
    for tau in splitting_times:
        # Evaluate the two-term parameter-shift rule of the auxiliar circuit
        grad += aux_circuit(theta, tau, 1.0) - aux_circuit(theta, tau, -1.0)
    return grad / num_samples


num_samples = 10
print(f"Stochastic parameter-shift: {stochastic_parshift_grad(theta, num_samples):.5f}")

##############################################################################
# Finally, we can make use of the custom parameter-shift rule introduced in
# [#wiersema]_, which is readily available in PennyLane. Due to the implementation
# chosen internally, the full gradient is returned, so that we need to pick the
# gradient entry for the second entry manually. For this small toy problem, this is
# not an issue.

sun_grad = jax.grad(circuit)
print(f"Custom SU(N) gradient: {sun_grad(theta)[1]:.5f}")

##############################################################################
# We obtained three values for the gradient of interest, and they do not agree.
# So what is going on here? First, let's use automatic differentiation to compute
# the exact value and see which method agrees with it (We again need to extract the
# corresponding entry from the full gradient).

autodiff_circuit = qml.QNode(qfunc, dev, interface="jax", diff_method="parameter-shift")
exact_grad = jax.grad(autodiff_circuit)(theta)[1]
print(f"Exact gradient: {exact_grad:.5f}")

##############################################################################
# As we can see, the custom differentiation method gave us the correct result,
# confirmed by automatic differentiation. Why do the other methods disagree?
# This is because the central difference recipe is an *approximate* gradient
# method. This means it has an error even if all circuit evaluations are
# made exact (up to numerical precision) like in the example above.
# As for the stochastic parameter-shift rule, you may already guess why there is
# a deviation: indeed, the *stochastic* nature of this method leads to derivative
# values that are scattered around the true value. It is an unbiased estimator,
# so that the average will approach the exact value with increasinlgy many evaluations.
# To demonstrate this, let's compute the same derivative many times and plot
# a histogram of what we get. We'll do so for ``num_samples=2``, ``10`` and ``100``.

import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 12})

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
colors = ["#90CCFF", "#DF74AF", "#EED0B0"]
for num_samples, color in zip([2, 10, 100], colors):
    grads = [stochastic_parshift_grad(theta, num_samples) for _ in range(1000)]
    ax.hist(grads, label=f"{num_samples} samples", alpha=0.8, color=color)
ylim = ax.get_ylim()
ax.plot([exact_grad] * 2, ylim, ls="--", c="k", label="Exact")
ax.set(xlabel=r"$\partial_{SPS,\theta_2}C(\theta)$", ylabel="Frequency", ylim=ylim)
ax.legend(loc="upper left")
plt.tight_layout()
plt.show()

##############################################################################
# As we can see, the stochastic parameter-shift rule comes with a variance
# that can be reduced at the additional cost of evaluating the auxiliary circuit
# for more splitting times.
#
# On quantum hardware, all measurement results are statistical by nature anyways.
# So how does this stochasticity combine with the exact and stochastic nature
# of the three differentiation methods? We will not go into detail here but refer
# to [#wiersema]_ which shows that the custom differentiation rule proposed in the
# main text leads to the lowest mean squared error. For a single-qubit circuit
# similar to the one above, the derivative and expected variance are shown in the
# following plot from the manuscript:
#
# .. figure:: ../demonstrations/here_comes_the_sun/sampled_grad.png
#    :align: center
#    :width: 50%
#
# As we can see, the custom :math:`\mathrm{SU}(N)` parameter-shift rule produces the
# gradient estimates with the smallest variance.
# The central difference gradient shown here was obtained using the shift
# scale :math:`\delta=0.75` as well. As we can see, this suppresses the variance down to
# a level comparable to the one for the shift rule derivatives. As shown in App. F3 of
# [#wiersema]_, this scale indeed is close to the optimal choice if we were to compute
# the gradient with 100 shots per circuit.
#
# Comparing ansatz structures
# ---------------------------
#
# We discussed above that there are many circuit architectures available and that choosing
# a suitable ansatz is important but can be difficult. Here we will compare a simple ansatz
# based on the ``qml.SpecialUnitary`` gate discussed above to other approaches that fully
# parametrize the special unitary group for the respective number of qubits.
# In particular, we will compare ``qml.SpecialUnitary`` to standard decompositions from the
# literature that parametrize :math:`\mathrm{SU}(N)` with elementary gates, as well as to a sequence
# of Pauli rotation gates that also allows to create any special unitary.
# Let us start by defining the decomposition of a two-qubit unitary.
# We choose the decomposition, which is optimal but not unique, from [#vatan]_.
# The Pauli rotation sequence is available in PennyLane
# via ``qml.ArbitraryUnitary`` and we will not need to implement it ourselves.


def two_qubit_decomp(params, wires):
    """Implement an arbitrary SU(4) gate on two qubits
    using the decomposition from Theorem 5 in
    https://arxiv.org/pdf/quant-ph/0308006.pdf"""
    i, j = wires
    # Single U(2) parameterization on both qubits separately
    qml.Rot(*params[:3], wires=i)
    qml.Rot(*params[3:6], wires=j)
    qml.CNOT(wires=[j, i])  # First CNOT
    qml.RZ(params[6], wires=i)
    qml.RY(params[7], wires=j)
    qml.CNOT(wires=[i, j])  # Second CNOT
    qml.RY(params[8], wires=j)
    qml.CNOT(wires=[j, i])  # Third CNOT
    # Single U(2) parameterization on both qubits separately
    qml.Rot(*params[9:12], wires=i)
    qml.Rot(*params[12:15], wires=j)


# The three building blocks on two qubits we will compare:
operations = {
    "Decomposition": two_qubit_decomp,
    "PauliRot sequence": qml.ArbitraryUnitary,
    "$\mathrm{SU}(N)$ gate": qml.SpecialUnitary,
}

##############################################################################
# Now that we have the template for the composition approach in place, we construct a toy
# problem to solve using the ansätze. We will sample a random Hamiltonian in the Pauli basis
# (this time without the prefactor :math:`i`, as we want to construct a Hermitian operator)
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
# Create the matrices for the entire Pauli basis
basis = qml.ops.qubit.special_unitary.pauli_basis_matrices(num_wires)
# Construct the Hamiltonian from the normal random coefficients and the basis
H = qml.math.tensordot(coefficients, basis, axes=[[0], [0]])
# Compute the ground state energy
E_min = np.linalg.eigvalsh(H).min()
print(f"Ground state energy: {E_min:.5f}")
H = qml.Hermitian(H, wires=wires)

##############################################################################
# Using the toy problem Hamiltonian and the three ansätze for :math:`\mathrm{SU}(N)` operations
# from above, we create a circuit template that applies these operations in a brick-layer
# architecture with two blocks and each operation acting on ``locality=2`` qubits.
# For this we define a ``QNode``:

locality = 2
d = locality**4 - 1  # d = 15 for two-qubit operations
dev = qml.device("default.qubit", wires=num_wires)
# two blocks with two layers. Each layer contains three operations with d parameters
param_shape = (2, 2, 3, d)
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
                    w % num_wires for w in range(locality * j + i, locality * (j + 1) + i)
                ]
                operation(params_op, wires_op)
    return qml.expval(H)


qnode = qml.QNode(circuit, dev, interface="jax")
print(qml.draw(qnode)(init_params, qml.SpecialUnitary))

##############################################################################
# We can now proceed to preparing the optimization task using this circuit
# and an optimization routine of our choice. For simplicity, we run a vanilla gradient
# descent optimization with fixed learning rate for 200 steps. Again, we use JAX
# for auto-differentiation.

learning_rate = 5e-4
num_steps = 500
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

fig, ax = plt.subplots(1, 1)
for (name, energy), c in zip(energies.items(), colors):
    error = (energy - E_min) / abs(E_min)
    ax.plot(list(range(len(error))), error, label=name, c=c)

ax.set(xlabel="Iteration", ylabel="Relative error")
ax.legend()
plt.show()

##############################################################################
# We find that the optimization indeed performs significantly better for ``qml.SpecialUnitary``
# than for the other two general unitaries, while using the same number of parameters. This
# means that we found a particularly well-trainable parametrization of the local unitaries which
# allows to reduce the energy of the prepared quantum state more easily, while using the same
# number of parameters.
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
#     R. Wiersema, D. Lewis, D. Wierichs, J. F. Carrasquilla, and N. Killoran,
#     in preparation, arXiv:23xx.xxxxx, (2023)
#
# .. [#banchi]
#
#     Leonardo Banchi and Gavin E. Crooks. "Measuring Analytic Gradients of
#     General Quantum Evolution with the Stochastic Parameter Shift Rule."
#     `Quantum **5** 386 <https://quantum-journal.org/papers/q-2021-01-25-386/>`__ (2021).
#
# About the author
# ----------------
# .. include:: ../_static/authors/david_wierichs.txt
