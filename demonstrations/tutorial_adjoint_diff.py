r"""

.. _adjoint_differentiation:

Adjoint Differentiation
=======================

.. meta::
    :property="og:description": Learn how the adjoint differentation method works.
    :property="og:image": https://pennylane.ai/qml/_static/thumbs/code.png

"""

##############################################################################
""" Classical autodifferentiation has two methods of calculation: forward and reverse.
 In forward mode, the computer calculates the function value and derivative at the same time.
 In reverse mode, the computer stores the derivatives of components during evaluation then
 calculates the total derivative later during a "backward pass". Which method someone should
 use depends on the structure of the problem; is the function many-to-one or one-to-many? We 
 use properties of the problem to optimize how we calculate derivatives.

Standard methods to calculate the derivatives for quantum circuits either are direct applications
of classical gradient methods to quantum simulations, or quantum hardware methods like parameter-shift
where we can only extract restricted pieces of information.

Adjoint differentiation straddles these two strategies taking benefits from each.
 On simulators, we can see the complete statevector and any point and apply any operation to it, 
 but we know our quantum circuit has a limited structure.  Namely, a circuit is initialization, 
 unitary operators, and an expectation value of a Hermitian operator:

.. math::
   |\Psi\rangle = U_{n} U_{n-1} \dots U_0 |0\rangle

.. math::
    \langle M \rangle = \langle \Psi | M | \Psi \rangle
    = \langle 0 | U_0^{\dagger} \dots U_{n-1}^{\dagger} U_{n}^{\dagger} M U_{n} U_{n-1} \dots U_0 |0\rangle

Since all our operators are unitary, we can easily "undo" or "erase" them by applying their adjoint.
The "adjoint" differentiation method takes advantage of this fact to create a time and memory  efficient
method for computing observables. """


##############################################################################
# Start with a quantum state
#
# .. math::
#   |\Psi\rangle = U_{n} U_{n-1} \dots U_0 |0\rangle
#
# We can then take an expectation value:
#
# .. math::
#   \langle M \rangle = \langle \Psi | M | \Psi \rangle
#   = \langle 0 | U_0^{\dagger} \dots U_{n-1}^{\dagger} U_{n}^{\dagger} M U_{n} U_{n-1} \dots U_0 |0\rangle
#
# created by a sequence of unitary operations.  I leave out parameters here for
# simplicity.
# We usually think of this expectation value as an inner product of a left vector
# and a right vector:
#
# .. math::
#   | b \rangle = M | \Psi \rangle = M U_n U_{n-1} \dots U_0 |0\rangle
#
# .. math::
#   | k \rangle =  |\Psi \rangle = U_n U_{n-1} \dots U_0 |0\rangle
#
# .. math::
#   \langle M \rangle = \langle b | k \rangle
#
# I use :math:`|b\rangle` and :math:`|k\rangle` to be the "bra" and "ket" vectors
# respectively.
#
# I could have attached :math:`M`, a Hermitian observable :math:`M^{\dagger}=M`, to
# either the bra or the ket, but attaching it to the bra side will be useful later.
#
# We will be using some hidden methods of ``'default.qubit'``. So we will start
# by creating a ``'default.qubit'`` instance with two wires:

import pennylane as qml
from pennylane import numpy as np

dev = qml.device("default.qubit", wires=2)

##############################################################################
# And then, we can apply operations to get any arbitrary state :math:`|\Psi\rangle`.  We
# will be using the a simple vector of operations ``ops`` and an observable ``M`` in our
#  operations, but we can compare our problem to that of the QNode ``circuit``.

x = np.array([0.1, 0.2, 0.3])

ops = [qml.RX(x[0], wires=0), qml.CNOT(wires=(0, 1)), qml.RY(x[1], wires=1), qml.RZ(x[2], wires=1)]
M = qml.PauliX(wires=1)

n = len(ops)
n_params = 3


@qml.qnode(dev)
def circuit(a):
    qml.RX(a[0], wires=0)
    qml.CNOT(wires=(0, 1))
    qml.RY(a[1], wires=1)
    qml.RZ(a[2], wires=1)
    return qml.expval(qml.PauliX(wires=1))


##############################################################################
# We create our state by using the ``"default.qubit"`` methods ``dev._create_basis_state`` and
# ``dev._apply_operation``.

state = dev._create_basis_state(0)

for op in ops:
    state = dev._apply_operation(state, op)

print(state)

##############################################################################
# Using this state, we can then create our :math:`|b\rangle` and :math:`|k\rangle` vectors from above:

bra = dev._apply_operation(state, M)
ket = state

#########################################################################################
# Now we use ``np.vdot`` take the inner product.  ``np.vdot`` sums over all dimensions and
# takes the complex conjugate of the first input.

M_expval = np.vdot(bra, ket)
print(M_expval)

##############################################################################
# We can compare this to the QNode calculation:

print(circuit(x))

##############################################################################
# But the dividing line between what makes the "bra" and "ket" vector is much more
# arbitrary than I stated above.  We can divide the two vectors at any point from
# one :math:`\langle 0 |` to the other :math:`|0\rangle`. For example, we could have used:
#
# .. math::
#
#   |b_n \rangle = U_n^{\dagger} M U_n U_{n-1} \dots U_1 |0\rangle
#
# .. math::
#
#   |k_n \rangle = U_{n-1} \dots U_1 |0\rangle
#
# And gotten the exact same results.  I use :math:`n` to indicate that :math:`U_n` was moved
# to the other side.  Let's calculate that instead:

bra_n = dev._create_basis_state(0)

for op in ops:
    bra_n = dev._apply_operation(bra_n, op)
bra_n = dev._apply_operation(bra_n, M)
bra_n = dev._apply_operation(bra_n, ops[-1].inv())

ops[-1].inv()  # returning the operation to an uninverted state


ket_n = dev._create_basis_state(0)

for op in ops[0:-1]:  # don't apply last operation
    ket_n = dev._apply_operation(ket_n, op)

M_expval_n = np.vdot(bra_n, ket_n)
print(M_expval_n)

##############################################################################
# Yay! Same answer!
#
# But we could have done that in a much more elegant way.  We had our initial bra and
# ket vectors.  To shift our splitting point, we don't have to recalculate
# everything from scratch.
#
# .. math::
#
#   |b_n\rangle = U_n^{\dagger} |b\rangle
#
# .. math::
#
#    |k\rangle = U_n |k_n\rangle \rightarrow |k_n\rangle = U_n^{\dagger} |k\rangle
#
#
# For the right side vector, I think of :math:`U_n^{\dagger}` as "eating" it's corresponding
# Unitary on the right side, erasing it from the list.

bra = dev._apply_operation(state, M)
ket = state

ops[-1].inv()

bra_n_v2 = dev._apply_operation(bra, ops[-1])
ket_n_v2 = dev._apply_operation(ket, ops[-1])

ops[-1].inv()

M_expval_n_v2 = np.vdot(bra_n_v2, ket_n_v2)
print(M_expval_n_v2)

##############################################################################
# Much simpler!
#
# We can easily iterate over all the operations to show you will always get the same
# result no matter where you split the operations.  We have to reverse the order of
# operations, as we start by moving :math:`U_n` at the end of the list and finish by
# moving :math:`U_1` at the start of the list.

bra_loop = dev._apply_operation(state, M)
ket_loop = state

for op in reversed(ops):
    op.inv()
    bra_loop = dev._apply_operation(bra_loop, op)
    ket_loop = dev._apply_operation(ket_loop, op)
    op.inv()
    print(np.vdot(bra_loop, ket_loop))

##############################################################################
# Finally to Derivatives!
# ~~~~~~~~~~~~~~~~~~~~~~~
#
# We showed how to calculate the same thing a bunch of
# different ways. Why is this useful?
#
# Wherever we have that cut, we can stick additional things in the middle.  Let's
# take a look at the formula for a derivative of an expectation value.
#
# For simplicity sake, I assume each unitary operation :math:`U_i` is a function of a
# single parameter :math:`\theta_i`.  For non-parametrized gates like CNOT, we can then
# say its derivative is zero.  Multi-parametrized gates can be decomposed into
# single parameter gates usually.
#
# .. math::
#
#   \frac{\partial \langle M \rangle}{\partial \theta_i} =
#   \langle 0 | U_1^{\dagger} \dots \frac{\text{d} U_i^{\dagger}}{\text{d} \theta_i} \dots M
#   \dots U_i \dots U_1 | 0\rangle
#
# .. math::
#
#   + \langle 0 | U_1^{\dagger} \dots U_i^{\dagger} \dots M
#   \dots \frac{\text{d} U_i}{\text{d} \theta_i}  \dots U_1 |0\rangle
#
# We can now notice that those two components are complex conjugates,
# so we can further simplify.  Note that each term is not an expectation value of
# a Hermitian observable, only the sum is an expectation value. Each component is
# not guaranteed to be real.  Therefore we have to select out the real component;
# the imaginary part gets cancelled out.
#
# .. math::
#   = 2 \text{Real}\left( \langle 0 | U_1^{\dagger} \dots U_i^{\dagger} \dots M \dots
#   \frac{\text{d} U_i}{\text{d} \theta_i}  \dots U_1 |0\rangle \right)
#
# We can take that formula and break it into its "bra" and "ket" halves for a
# derivative at the :math:`i` th position:
#
# .. math::
#
#   |k_i \rangle = U_{i-1} \dots U_1 |0\rangle
#
# .. math::
#
#   = U_{i}^{\dagger} | k_{i+1}\rangle
#
# .. math::
#
#   |b_i\rangle = U_{i+1}^{\dagger} \dots U_n^{\dagger} M U_n \dots U_1 |0\rangle
#
# .. math::
#
#   = U_{i+1}^{\dagger} |b_{i+1} \rangle
#
# This :math:`|b_i\rangle` is different than the one defined above when we weren't
# taking the derivative. Now, on step :math:`i`, the :math:`U_i` operator is removed
# and substituted for something else, its derivative.  Only once the calculation is
# complete do we continue moving the operator over to the bra.
#
# For the actually expectation value calculation, we use a temporary version of the bra:
#
# .. math::
#
#   |\tilde{b}_i\rangle = \frac{\text{d} U^{\dagger}_i}{\text{d} \theta_i} |b_i \rangle
#
# And use these to get the derivative:
#
# .. math::
#
#   \frac{\partial \langle M \rangle}{\partial \theta_i} = 2 \text{Real}\left( \langle
#   \tilde{b}_i | k_i \rangle \right)
#
# Both the bra and the ket have simple recursive definitions.  We can iterate through
# the operations starting at :math:`n` and ending at :math:`1`. As this iteration is
# reversed going from high to low values, the recursion formulas go from :math:`i+1`
# to :math:`i`.
#
# We need to calculate the state first, a "forward" pass, but once we have that,
# we only have about the same amount of work to calculate all the derivatives,
# instead of quadratically more work.
#
# Derivative of an Operator
# ~~~~~~~~~~~~~~~~~~~~~~~~~
#
# One final thing before we get back to coding, how do we get the derivative of an operator?
#
# Most parametrized gates can be represented in terms of Pauli Rotations, and Pauli
# rotations can be written as:
#
# .. math::
#
#   U = e^{i c \hat{G} \theta}
#
# For a matrix :math:`\hat{G}`, a constant :math:`c`, and the parameter :math:`\theta`.
# Thus we can write easily calculate their derivatives:
#
# .. math::
#   \frac{\text{d} U}{\text{d} \theta} = i c \hat{G} e^{i c \hat{G} \theta} = i c \hat{G} U
#
#
# Luckily, we already have a built-in PennyLane function for calculating this.

d_op = qml.operation.operation_derivative(ops[0])
print(d_op)

##############################################################################
# Now for calculating the derivative!
#
# We loop over the reversed operations, just as before.
#
# But if the operation has a parameter, we calculate it's derivative and append it
# to a list before moving on.
#
# Since the ``operation_derivative`` function spits back out a matrix instead of an
# operation, we have to use ``dev._apply_unitary`` instead to create :math:`|\tilde{b}_i\rangle`.

bra = dev._apply_operation(state, M)
ket = state

grads = []

for op in reversed(ops):
    op.inv()
    ket = dev._apply_operation(ket, op)

    # Calculating the derivative
    if op.num_params != 0:
        dU = qml.operation.operation_derivative(op)

        bra_temp = dev._apply_unitary(bra, dU, op.wires)

        dM = 2 * np.real(np.vdot(bra_temp, ket))
        grads.append(dM)

    bra = dev._apply_operation(bra, op)
    op.inv()


# Finally, reverse the order of the gradients
# since we calculated them in reverse
grads = grads[::-1]

print("our calculation: ", grads)

pl_calc = qml.grad(circuit)(x)
print("Comparison: ", pl_calc)

##############################################################################
# It matches!!!
#
# If you want to use adjoint differentiation without having to code up your own
# method that can support arbitrary circuits, you can use adjoint diff in PennyLane
# with ``"default.qubit"``


@qml.qnode(dev, diff_method="adjoint")
def circuit_adjoint(a):
    qml.RX(a[0], wires=0)
    qml.CNOT(wires=(0, 1))
    qml.RY(a[1], wires=1)
    qml.RZ(a[2], wires=1)
    return qml.expval(M)


qml.grad(circuit_adjoint)(x)


##############################################################################
# Time Scaling
# ~~~~~~~~~~~~
#
# In this section, I've deliberately used the same timing framework as the
# `backpropagation tutorial <https://pennylane.ai/qml/demos/tutorial_backprop.html#time-comparison>`_ .
# Though I don't repeat that data here, you can compare the graphs.
#
# For this section, we need two additional packages: ``timeit`` and ``pyplot``.

import timeit
import matplotlib.pyplot as plt

n_wires = 4

dev = qml.device("default.qubit", wires=n_wires)


@qml.qnode(dev, diff_method="adjoint")
def circuit(params):
    qml.templates.StronglyEntanglingLayers(params, wires=range(n_wires))
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2) @ qml.PauliZ(3))


reps = 2
num = 3

n_layers = range(1, 21)

t_exec = []
t_grad = []
ratio = []
n_params = []

rng = np.random.default_rng(seed=42)

for i_layers in n_layers:

    # set up the parameters
    param_shape = qml.templates.StronglyEntanglingLayers.shape(n_wires=4, n_layers=i_layers)
    params = np.array(rng.standard_normal(param_shape))
    params.requires_grad = True
    n_params.append(params.size)

    ti_exec_set = timeit.repeat("circuit(params)", globals=globals(), number=num, repeat=reps)
    ti_exec = min(ti_exec_set) / num
    t_exec.append(ti_exec)

    ti_grad_set = timeit.repeat(
        "qml.grad(circuit)(params)", globals=globals(), number=num, repeat=reps
    )
    ti_grad = min(ti_grad_set) / num
    t_grad.append(ti_grad)

    ratio.append(ti_grad / ti_exec)

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.plot(n_params, t_exec, label="execution")
ax.plot(n_params, t_grad, label="gradient")

ax.legend()

ax.set_xlabel("Number of parameters")
ax.set_ylabel("Time")

fig.suptitle("Adjoint Time Scaling")
plt.show()

##############################################################################
# Just like backpropagation, adjoint is roughly a constant factor longer than straight
# execution times.  BUT, the adjoint method has a constant memory overhead.
# Backpropagation balloons in memory. Memory is already a limiting factor in quantum
# computation even before you start taking derivatives.  You can always run a
# simulation for twice as long; it is much harder to double your available RAM.

n_params = np.array(n_params)

m, b = np.polyfit(n_params, ratio, deg=1)
ratio_fit = lambda x: m * x + b

print(f"ratio fit: {m}*x + {b}")

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.plot(n_params, ratio, label="ratio")
ax.plot(n_params, ratio_fit(n_params), label=f"{m:.3f}*x + {b:.3f}")

fig.suptitle("Gradient time per execution time")
ax.set_xlabel("number of parameters")
ax.set_ylabel("Normalized Time")
ax.legend()
plt.show()


##############################################################################
# Bibliography
# ~~~~~~~~~~~~
#
# Jones and Gacon. Efficient calculation of gradients in classical simulations
#  of variational quantum algorithms.
# `https://arxiv.org/abs/2009.02823 <https://arxiv.org/abs/2009.02823>`_
