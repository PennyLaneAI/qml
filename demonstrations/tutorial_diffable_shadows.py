r"""Classical shadows in the Pauli basis
========================================

.. meta::
    :property="og:description": Classical shadows in the Pauli basis
    :property="og:image": https://pennylane.ai/qml/_images/diffable_mitigation_thumb.png

.. related::

    classical_shadows
    ml_classical_shadows

*Author: Korbinian Kottmann, Posted: 22 September 2022*

abstract

Classical shadow theory
-----------------------

A `classical shadow` is a classical description of a quantum state that is capable of reproducing expectation values of local Pauli observables, see [#Huang2020]_.

The idea is to capture :math:`T` local snapshots (given by the ``shots`` set in the device) of the state by performing measurements in random Pauli bases at each qubit.
The measurement outcomes, denoted ``bits``, as well as the choices of measurement bases, ``recipes``, are recorded in two ``(T, n_qubits)`` integer tensors, respectively.
Recipes ``[0, 1, 2]`` corresponds to measurement bases ``[X, Y, Z]``, respectively. For each shot, we record output bit ``[0, 1]``, corresponding to eigenvalue ``[1, -1]``, respectrively.

From the :math:`t`-th measurement, we can reconstruct the ``local_snapshots``

.. math:: \rho^{(t)} = \bigotimes_{i=1}^{n} 3 U^\dagger_i |b_i \rangle \langle b_i | U_i - \mathbb{I},

where :math:`U_i` is the rotation corresponding to the measurement (e.g. :math:`U_i=H` for measurement in :math:`X`) of qubit :math:`i` at snapshot :math:`t` and
:math:`|b_i\rangle = (1 - b_i, b_i)` the corresponding computational basis state given the output bit :math:`b_i`.

From these local snapshots, one can compute expectation values of q-local Pauli strings, where locality refers to the number of non-Identity operators.

We show how to efficiently compute expectation values of Pauli strings without reconstructing any local density matrices using algebraic properties of Pauli operators.
Let us start by looking at individual snapshot expectation values $\braket{\bigotimes_i\tilde{P}_i}^{(t)} = \tr{\rho^{(t)} \bigotimes_i\tilde{P}_i}$ with $\tilde{P}_i \in \{X, Y, Z, \mathbb{1}\}$. For that, note that $\ket{b_i} = (1-b_i, b_i)$ and convince yourself of the identity

.. math:: U_i^\dagger |b_i\rangle \langle b_i| U_i = \frac{1}{2}\left((1-2b_i) P_i + \mathbb{1}\right),

where $P_i \in \{X, Y, Z\}$ is the Pauli operator corresponding to $U_i$ (Note that in this case $P_i$ is never the identity). 
The snapshot expectation value then reduces to


.. math:: \braket{\bigotimes_i\tilde{P}_i}^{(t)} = \prod_{i=1}^n \tr{\frac{3}{2}(1-2b_i)P_i \tilde{P}_i + \frac{1}{2}\tilde{P}_i}.

The cases where $\tilde{P}_i=1$ yield a trivial factor $1$ to the product. The full product is always zero if any of the non-trivial $\tilde{P}_i$ do not match $P_i$. So in total, in the case that all Pauli operators match, we find

.. math:: \braket{\bigotimes_i\tilde{P}_i}^{(t)} = 3^q \prod_{\text{i non-trivial}}(1-2b_i).

This implies that in order to compute the expectation value of a Pauli string

.. math:: \braket{\bigotimes_i\tilde{P}_i} = \frac{1}{\tilde{T}} \sum_{\tilde{t}} \braket{\bigotimes_i\tilde{P}_i}^{(t)}

we simply need to sum the result bits for those  $\tilde{T}$ snapshots where the measurement recipe matches the observable,
indicated by the special index $\tilde{t}$ for the mathing measurements.

This, on the other hand, implies that computing expectation values with classical shadows comes down to picking the specific subset of snapshots where those specific observables
were already measured. This was referred to as `derandomization` by the authors in a follow-up paper [#Huang2021]_.
Error bounds given by the number of measurements :math:`T = \mathcal{O}\left( \log(M) 4^\ell/\epsilon^2 \right)` guarantee that sufficiently many correct measurements
were performed to estimate :math:M different observables up to additive error :math:\varepsilon.

We will later compare this to directly measuring the desired observables and making use of simultaneously measuring qubit-wise-commuting observables. Before that, let us
demonstrate how to perform classical shadow measurements in a differentiable manner in PennyLane.

PennyLane implementation
------------------------

There are two ways of computing expectation values with classical shadows in PennyLane. The first is to return :func:`shadow_expval` directly from the qnode.
"""

import pennylane as qml
import pennylane.numpy as np
from matplotlib import pyplot as plt

H = qml.Hamiltonian([1., 1.], [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliX(0) @ qml.PauliX(1)])

dev = qml.device("default.qubit", wires=range(2), shots=10000)
@qml.qnode(dev)
def qnode(x):
    qml.Hadamard(0)
    qml.CNOT((0,1))
    qml.RX(x, wires=0)
    return shadow_expval(H)

x = np.array(0.5, requires_grad=True)

##############################################################################
# The big advantage of this way of computing expectation values is that it is differentiable.

print(qnode(x), qml.grad(qnode)(x))

##############################################################################
# Note that to avoid unnecessary device executions you can provide a list of observables to :func:`shadow_expval`.

Hs = [H, qml.PauliX(0), qml.PauliY(0), qml.PauliZ(0)]
print(qnode(x, Hs))
print(qml.jacobian(qnode)(x, Hs))

##############################################################################
# Alternatively, you can compute expectation values by first performing the shadow measurement and then perform classical post-processing using the :class:`ClassicalShadow`
# class methods.

dev = qml.device("default.qubit", wires=range(2), shots=1000)
@qml.qnode(dev)
def qnode(x):
    qml.Hadamard(0)
    qml.CNOT((0,1))
    qml.RX(x, wires=0)
    return classical_shadow(wires=range(2))

bits, recipes = qnode(0)
shadow = ClassicalShadow(bits, recipes)

##############################################################################
# After recording these ``T=1000`` quantum measurements, we can post-process the results to arbitrary local expectation values of Pauli strings.
# For example, we can compute the expectation value of a Pauli string

print(shadow.expval(qml.PauliX(0) @ qml.PauliX(1), k=1))

##############################################################################
# or of a Hamiltonian:

print(H = qml.Hamiltonian([1., 1.], [qml.PauliZ(0)@qml.PauliZ(1), qml.PauliX(0)@qml.PauliX(1)]))
print(shadow.expval(H, k=1))



##############################################################################
# Comparing quantum resources with conventional measurement methods 
# -----------------------------------------------------------------
# 
# Simultaneously measuring qubit-wise-commuting observables




##############################################################################
#
# Conclusion
# ----------
# 
# conclusion
#
#
# References
# ----------
#
# .. [#Huang2020]
#
#     Hsin-Yuan Huang, Richard Kueng, John Preskill
#     "Predicting Many Properties of a Quantum System from Very Few Measurements."
#     `arXiv:2002.08953 <https://arxiv.org/abs/2002.08953>`__, 2020.
#
# .. [#Huang2021]
#
#     Hsin-Yuan Huang, Richard Kueng, John Preskill
#     "Efficient estimation of Pauli observables by derandomization."
#     `arXiv:2103.07510 <https://arxiv.org/abs/2103.07510>`__, 2021.

##############################################################################
# .. bio:: Korbinian Kottmann
#    :photo: ../_static/authors/qottmann.jpg
#
#    Korbinian is a summer resident at Xanadu, interested in (quantum) software development, quantum computing and (quantum) machine learning.
