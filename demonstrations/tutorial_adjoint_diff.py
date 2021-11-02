r"""

.. _adjoint_differentiation:

Adjoint Differentiation
=======================

.. meta::
    :property="og:description": Learn how the adjoint differentation method works.
    :property="og:image": https://pennylane.ai/qml/_static/thumbs/code.png

"""

##############################################################################
# Introduction
# ------------
# 
# Classical automatic differentiation has two methods of calculation: forward and reverse.
# The optimal choice of method depends on the structure of the problem; is the function 
# many-to-one or one-to-many? We use the properties of the problem to optimize how we 
# calculate derivatives.
# 
# Most methods for calculating the derivatives of quantum circuits either are direct applications
# of classical gradient methods to quantum simulations, or quantum hardware methods like parameter-shift
# where we can only extract restricted pieces of information.
# 
# Adjoint differentiation straddles these two strategies taking benefits from each.
# On simulators, we can examine and modify the state vector at any point. At the same time, we know our
# quantum circuit holds specific properties not present in an arbitrary classical computation. 
#  
# Quantum circuits only involve:
#
# 1) initialization,
#
# .. math:: |0\rangle,
# 
# 2) application of unitary operators,
#
# .. math:: |\Psi\rangle = U_{n} U_{n-1} \dots U_0 |0\rangle,
#
# 3) taking an expectation value of a Hermitian operator,
#
# .. math:: \langle M \rangle = \langle \Psi | M | \Psi \rangle.
# 
# Since all our operators are unitary, we can easily "undo" or "erase" them by applying their adjoint:
# 
# .. math:: U^{\dagger} U | \phi \rangle = |\phi\rangle.
# 
# The "adjoint" differentiation method takes advantage of the ability to erase, creating a time and
# memory-efficient method for computing quantum gradients.
#
# In this demo, you will learn how adjoint differentiation works and how to request it
# for your PennyLane QNode. We will also look at the performance benefits and when you might want
# to use it.
# 
# Time for some code
# ------------------
#
# So how does it work? Instead of jumping straight to the algorithm, let's explore the above equations
# and their implementation in a bit more detail.
# 
# To start, we import PennyLane and PennyLane's numpy:

import pennylane as qml
from pennylane import numpy as np

##############################################################################
# We also need a circuit to simulate:

dev = qml.device('default.qubit', wires=2)

x = np.array([0.1, 0.2, 0.3])

@qml.qnode(dev, diff_method="adjoint")
def circuit(a):
    qml.RX(a[0], wires=0)
    qml.CNOT(wires=(0,1))
    qml.RY(a[1], wires=1)
    qml.RZ(a[2], wires=1)
    return qml.expval(qml.PauliX(wires=1))

##############################################################################
# The fast c++ simulator device ``"lightning.qubit"`` also supports adjoint differentiation,
# but here we want to quickly prototype a minimal version to illustrate how the algorithm works,
# not understand how to optimize the code for performance.
#
# We will use the ``circuit`` QNode just for comparison purposes.  Thoughout this
# demo, we will instead use a list of its operations ``ops`` and a single observable ``M``.

n_gates = 4
n_params = 3

ops = [
    qml.RX(x[0], wires=0),
    qml.CNOT(wires=(0,1)),
    qml.RY(x[1], wires=1),
    qml.RZ(x[2], wires=1)
]
M = qml.PauliX(wires=1)

##############################################################################
# We create our state by using the ``"default.qubit"`` methods ``_create_basis_state``
# and ``_apply_operation``.
# 
# These are private methods that you typically wouldn't need to know about,
# but we use them here to illustrate the algorithm.

state = dev._create_basis_state(0)

for op in ops:
    state = dev._apply_operation(state, op)
    
print(state)

##############################################################################
# We can think of the expectation :math:`\langle M \rangle` as an inner product between a bra and a ket:
#
# .. math:: \langle M \rangle = \langle b | k \rangle = \langle \Psi | M | \Psi \rangle
#
# where:
#
# .. math:: \langle b | = \langle \Psi| M = \langle 0 | U_0^{\dagger} \dots U_n^{\dagger} M
#
# .. math:: | k \rangle =  |\Psi \rangle = U_n U_{n-1} \dots U_0 |0\rangle
# 
# We could have attached :math:`M`, a Hermitian observable (:math:`M^{\dagger}=M`), to either the
# bra or the ket, but attaching it to the bra side will be useful later.
# 
# Using the ``state`` calculated above, we can create these :math:`|b\rangle` and :math:`|k\rangle`
# vectors.  In the code, ``bra`` indicates :math:`|b\rangle`, the conjugate transpose of :math:`\langle b|`. 

bra = dev._apply_operation(state, M)
ket = state

##############################################################################
# Now we use ``np.vdot`` to take their inner product.  ``np.vdot`` sums over all dimensions
# and takes the complex conjugate of the first input.

M_expval = np.vdot(bra, ket)
print("vdot  : ", M_expval)
print("QNode : ", circuit(x))

##############################################################################
# We got the same result via both methods! This validates our use of ``vdot`` and
# device methods.
#
# But the dividing line between what makes the "bra" and "ket" vector is actually
# fairly arbitrary.  We can divide the two vectors at any point from one :math:`\langle 0 |`
# to the other :math:`|0\rangle`. For example, we could have used:
# 
# .. math:: \langle b_n | = \langle 0 | U_1^{\dagger} \dots  U_n^{\dagger} M U_n
# 
# .. math:: |k_n \rangle = U_{n-1} \dots U_1 |0\rangle
#
# And gotten the exact same results.  I use :math:`n` to indicate that :math:`U_n`
# was moved to the bra side of the expression.  Let's calculate that instead:

bra_n = dev._create_basis_state(0)

for op in ops:
    bra_n = dev._apply_operation(bra_n, op)
bra_n = dev._apply_operation(bra_n, M)
bra_n = dev._apply_operation(bra_n, ops[-1].inv())

ops[-1].inv() # returning the operation to an uninverted state

ket_n = dev._create_basis_state(0)

for op in ops[:-1]: # don't apply last operation
    ket_n = dev._apply_operation(ket_n, op)

M_expval_n = np.vdot(bra_n, ket_n)
print(M_expval_n)

##############################################################################
# Same answer!
# 
# We can calculate this in a more efficient way if we already have the
# initial ``state`` :math:`| \Psi \rangle`. To shift the splitting point, we don't
# have to recalculate everything from scratch. We just remove the operation from
# the ket and add it to the bra.
# 
# .. math:: \langle b_n | = \langle b | U_n
#
# .. math:: |k_n\rangle = U_n^{\dagger} |k\rangle 
# 
# For the ket vector, you can think of :math:`U_n^{\dagger}` as "eating" it's
# corresponding unitary from the vector, erasing it from the list of operations.
# 
# Of course, we actually work with the conjugate transpose of :math:`\langle b_n |`:
#
# .. math:: |b_n\rangle = U_n^{\dagger} | b \rangle
#
# Once we write it in this form, we see that the adjoint of the operation :math:`U_n^{\dagger}`
# operates on both :math:`|k_n\rangle` and :math:`|b_n\rangle` to move the splitting point right.
# 
# Let's call this the "version 2" method.

bra_n_v2 = dev._apply_operation(state, M)
ket_n_v2 = state

ops[-1].inv()

bra_n_v2 = dev._apply_operation(bra_n_v2, ops[-1])
ket_n_v2 = dev._apply_operation(ket_n_v2, ops[-1])

ops[-1].inv()

M_expval_n_v2 = np.vdot(bra_n_v2, ket_n_v2)
print(M_expval_n_v2)

##############################################################################
# Much simpler!
# 
# We can easily iterate over all the operations to show that the same result occurs
# no matter where you split the operations.  
# 
# .. math:: \langle b_i | = \langle b_{i+1}| U_{i}
# 
# .. math:: |k_{i+1} \rangle = U_{i} |k_{i}\rangle
# 
# Rewritten, we have our iteration formulas:
# 
# .. math:: | b_i \rangle = U_i^{\dagger} |b_{i+1}\rangle
# 
# .. math:: | k_i \rangle  = U_i^{\dagger} |k_{i+1}\rangle
# 
# For each iteration, we move an operation from the ket side to the bra side.
# We start near the center at :math:`U_n` and reverse through the operations list till we reach :math:`U_0`.

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
# -----------------------
#
# We showed how to calculate the same thing a bunch of different ways. Why is this useful? 
# 
# Wherever we cut, we can stick additional things in the middle. What are we sticking in the middle?
# The derivative of a gate.
# 
# For simplicity's sake, assume each unitary operation :math:`U_i` is a function of a single
# parameter :math:`\theta_i`.  For non-parametrized gates like CNOT, we say its derivative is zero.
# We can also generalize the algorithm to multi-parameter gates, but we leave those out for now.
#
# Remember that each parameter occurs twice in :math:`\langle M \rangle`: once in the bra and once in
# the ket. Therefore we use the product rule to take the derivative with respect to both locations.
# 
# .. math::
#       \frac{\partial \langle M \rangle}{\partial \theta_i} = 
#       \langle 0 | U_1^{\dagger} \dots \frac{\text{d} U_i^{\dagger}}{\text{d} \theta_i} \dots M \dots U_i \dots U_1 | 0\rangle
#
# 
# .. math:: 
#     + \langle 0 | U_1^{\dagger} \dots U_i^{\dagger} \dots M \dots \frac{\text{d} U_i}{\text{d} \theta_i}  \dots U_1 |0\rangle
#
# We can now notice that those two components are complex conjugates of each other, so we can
# further simplify.  Note that each term is not an expectation value of a Hermitian observable,
# only the sum is an expectation value. Each component is not guaranteed to be real,
# but when we add them together, the imaginary part cancels out, we obtain twice the
# value of the real part.
# 
# .. math::
#       = 2 \cdot \text{Re}\left( \langle 0 | U_1^{\dagger} \dots U_i^{\dagger} \dots M \dots \frac{\text{d} U_i}{\text{d} \theta_i}  \dots U_1 |0\rangle \right)
#
# We can take that formula and break it into its "bra" and "ket" halves for a derivative at the :math:`i` th position:
# 
# .. math::
#    \frac{\partial \langle M \rangle }{\partial \theta_i } = 
#    2 \text{Real} \left( \langle b_i | \frac{\text{d} U_i }{\text{d} \theta_i} | k_i \rangle \right)
#
# where
# 
# .. math:: \langle b_i | = \langle 0 | U_1^{\dagger} \dots U_n^{\dagger} M U_n \dots U_{i+1}
# 
#
# .. math :: |k_i \rangle = U_{i-1} \dots U_1 |0\rangle
# 
# 
# This :math:`|b_i\rangle` is different than the one defined above when we weren't taking
# the derivative. Now, on step :math:`i`, the :math:`U_i` operator is removed and substituted
# for something else, it's derivative.  Only once the calculation is complete do we continue
# moving the operator over to the bra.
# 
# For the actual expectation value calculation, we use a temporary version of the bra:
# 
# .. math:: \langle \tilde{b}_i | = \langle b_i | \frac{\text{d} U_i}{\text{d} \theta_i}
# 
# And use these to get the derivative
# 
# .. math::
#       \frac{\partial \langle M \rangle}{\partial \theta_i} = 2 \text{Real}\left( \langle \tilde{b}_i | k_i \rangle \right)
# 
# Both the bra and the ket can be calculated recursively:
#
# .. math:: | b_{i} \rangle = U^{\dagger}_{i+1} |b_{i+1}\rangle
#
# .. math:: | k_{i} \rangle = U^{\dagger}_{i} |k_{i+1}\rangle
# 
# We can iterate through the operations starting at :math:`n` and ending at :math:`1`.
# 
# We do have to calculate the state first, a "forward" pass, but once we have that,
# we only have about the same amount of work to calculate all the derivatives, 
# instead of quadratically more work.
# 
# Derivative of an Operator
# -------------------------
#
# One final thing before we get back to coding, how do we get the derivative of a operator?
# 
# Most parametrized gates can be represented in terms of Pauli Rotations, and Pauli rotations can be written as:
# 
# .. math:: U = e^{i c \hat{G} \theta}
# 
# for a matrix :math:`\hat{G}`, a constant :math:`c`, and the parameter :math:`\theta`.
# Thus we can write easily calculate their derivatives:
#
# .. math:: \frac{\text{d} U}{\text{d} \theta} = i c \hat{G} e^{i c \hat{G} \theta} = i c \hat{G} U 
# 
# Luckily, PennyLane already has a built-in function for calculating this.

grad_op0 = qml.operation.operation_derivative(ops[0])
print(grad_op0)

##############################################################################
# Now for calculating the derivative!
# 
# We loop over the reversed operations, just as before.  But if the operation has a parameter,
# we calculate it's derivative and append it to a list before moving on. Since the ``operation_derivative``
# function spits back out a matrix instead of an operation,
# we have to use ``dev._apply_unitary`` instead to create :math:`|\tilde{b}_i\rangle`.

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


# Finally reverse the order of the gradients
# since we calculated them in reverse
grads = grads[::-1]

print("our calculation: ", grads)

grad_compare = qml.grad(circuit)(x)
print("comparison: ", grad_compare)

##############################################################################
# It matches!!!
# 
# If you want to use adjoint differentiation without having to code up your own
# method that can support arbitrary circuits, you can use ``diff_method=adjoint`` in PennyLane with 
# ``"default.qubit"`` or PennyLane's fast C++ simulator ``"lightning.qubit"``.


dev_lightning = qml.device('lightning.qubit', wires=2)

@qml.qnode(dev_lightning, diff_method="adjoint")
def circuit_adjoint(a):
    qml.RX(a[0], wires=0)
    qml.CNOT(wires=(0,1))
    qml.RY(a[1], wires=1)
    qml.RZ(a[2], wires=1)
    return qml.expval(M)

qml.grad(circuit_adjoint)(x)

##############################################################################
# Time Scaling
# --------------
#
# In this section, I've deliberately used the same timing framework as the
# `backpropogation tutorial <https://pennylane.ai/qml/demos/tutorial_backprop.html#time-comparison>`__.
# Though I don't repeat that data here, you can compare the graphs.
# 
# For this section, we need two additional packages: ``timeit`` and ``pyplot``.
#
# To determine scaling, we will instead use ``"lightning.qubit"```, our fast c++ simulator
# that natively supports adjoint differentiation. 
#
# .. code-block:: python
# 
#       import timeit
#       import matplotlib.pyplot as plt
#       import pennylane as qml
#       from pennylane import numpy as np
#       plt.style.use("bmh")
#
#       n_wires = 4
#
#       dev = qml.device("lightning.qubit", wires=n_wires)
#
#       @qml.qnode(dev, diff_method="adjoint")
#       def circuit(params):
#           qml.templates.StronglyEntanglingLayers(params, wires=range(n_wires))
#           return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2) @ qml.PauliZ(3))
#
#
#       reps = 2
#       num = 3
#
#       n_layers = range(1, 21)
#
#       t_exec = []
#       t_grad = []
#       ratio = []
#       n_params = []
#   
#       rng = np.random.default_rng(seed=42)
#   
#       for i_layers in n_layers:
#           
#           # set up the parameters
#           param_shape = qml.templates.StronglyEntanglingLayers.shape(n_wires=n_wires, n_layers=i_layers)
#           params = rng.standard_normal(param_shape)
#           params.requires_grad = True
#           n_params.append(params.size)
#           
#           
#           ti_exec_set = timeit.repeat("circuit(params)",
#               globals=globals(), number=num, repeat=reps)
#           ti_exec = min(ti_exec_set)/num
#           t_exec.append(ti_exec)
#           
#           ti_grad_set = timeit.repeat("qml.grad(circuit)(params)",
#               globals=globals(), number=num, repeat=reps)
#           ti_grad = min(ti_grad_set)/num
#           t_grad.append(ti_grad)
#           
#           ratio.append(ti_grad/ti_exec)
#   
#       fig, ax = plt.subplots(1, 1, figsize=(6, 4))
#   
#       ax.plot(n_params, t_exec, '.-', label="circuit execution")
#       ax.plot(n_params, t_grad, '.-', label="gradient")
#   
#       ax.legend()
#   
#       ax.set_xlabel("Number of parameters")
#       ax.set_ylabel("Time")
#       fig.suptitle("")
#   
#       plt.show()
#   
#       n_params = np.array(n_params)
#   
#       m, b = np.polyfit(n_params, ratio, deg=1)
#       ratio_fit = lambda x: m*x+b
#   
#       fig2, ax2 = plt.subplots(1, 1, figsize=(6, 4))
#   
#       ax2.plot(n_params, ratio, '.-', label="ratio")
#       ax2.plot(n_params, ratio_fit(n_params), label=f"{m:.3f}*x + {b:.2f}")
#   
#       fig2.suptitle("Gradient time per execution time")
#       ax2.set_xlabel("number of parameters")
#       ax2.set_ylabel("Normalized Time")
#       ax2.legend()
#   
#       plt.show()
#   
#
# .. figure:: ../demonstrations/adjoint_diff/adjoint_timing1.png
#     :width: 50%
#     :align: center
#
# .. figure:: ../demonstrations/adjoint_diff/adjoint_timing2.png
#     :width: 50%
#     :align: center
#
# Just like backpropagation, adjoint is roughly a constant factor longer than straight execution times.
# BUT, the adjoint method has a constant memory overhead. Backpropagation balloons in memory, which is
# already a limiting factor in quantum computation even before you start taking derivatives.  You can
# always run a simulation for twice as long; much harder to double your available RAM. 
# 
#
# Bibliography
# -------------
# 
# Jones and Gacon. Efficient calculation of gradients in classical simulations of variational quantum algorithms.
# `https://arxiv.org/abs/2009.02823 <https://arxiv.org/abs/2009.02823>`__





