r"""The power of Iterative Quantum Phase Estimation
=============================================================

Halfway between the current NISQ era  with few noisy qubits, and fault-tolerant computers with many noiseless qubits,
we find what is called the ISQ era. This term refers to the situation in which we find ourselves with few qubits but
we can execute deep circuits since we have surpassed the noise barrier: goodbye to the "N".

One of the most important challenges in this ISQ era will be to adapt algorithms that depend on many qubits to others
that can be executed on these devices. From this need, the *Iterative Quantum Phase Estimation* algorithm was born,
which seeks to reduce the number of qubits of the *Quantum Phase Estimation* algorithm (QPE) to be able to execute it
on smaller devices. In this demo you will see the potential of this approach!

Quantum Phase Estimation
-----------------------------------------

QPE is an algorithm that is responsible for estimating the eigenvalue of an eigenvector associated with an operator.
We are not going to delve into this algorithm since we have already worked with it in this other :doc:`demo <tutorial_quantum_phase_estimation>` but it is worth
remembering that if :math:`U` is our operator and :math:`|\phi\rangle` our eigenstate, we have that:

.. math::
    U|\phi\rangle = e^{2\pi i\theta}|\phi\rangle,

where :math:`e^{2\pi i\theta}` is the eigenvalue and :math:`\theta` is what we will call the phase.
Said value $\theta$ can be represented in binary as

.. math::
    \theta = \overline{0.\theta_0\theta_1\theta_2}...

so each of these :math:`\theta_i` can take the value :math:`0` or :math:`1`.

.. note::
    We will use the horizontal line to refer to the fact that it is a binary representation and
    we are not multiplying the values of :math:`\theta_i`.

Well, with Quantum Phase Estimation we can obtain the precision of as many decimal bits as `estimation_wires`,
This this because we store each different bi in each qubit:

.. math::
    \text{QPE}|0\rangle^{\otimes 3} |\psi\rangle ≈ |\theta_0\theta_1\theta_2\rangle |\psi\rangle.

Let's look at an example where the phase is :math:`\theta = 0.375_{10}`, that is, :math:`\theta = 0.011` in binary.
"""

import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt

# We build a simple operator
def U(wires):
  return qml.PhaseShift(2 * np.pi * 0.375, wires = wires)

estimation_wires = [1,2,3]

dev = qml.device("default.qubit", wires = 4, shots = 1)


@qml.qnode(dev)
def circuit_qpe():
    # we initialize the eigenvalue |10>
    qml.PauliX(wires=0)

    # We apply the QPE subroutine

    for wire in estimation_wires:
        qml.Hadamard(wires=wire)

    qml.ControlledSequence(U(wires = 0), control=estimation_wires)

    qml.adjoint(qml.QFT)(wires=estimation_wires)

    return qml.sample(wires=estimation_wires)


results = circuit_qpe()
print(f"The estimated phase is: 0.{''.join([str(r) for r in results])}")


##############################################################################
# Great! We have obtained just the expected phase. In this case, one shot is enough since the binary representation is exact.
# In situations where we do not have enough bits to represent the phase, a small deviation will appear in the solution.
#
# Reducing the number of qubits
# ------------------------------
#
# We just saw that the number of precision bits we want in the algorithm is related :math:`1` to :math:`1` to the number of estimation qubits.
# What Iterative Quantum Phase Estimation achieves is to reduce that number to a single auxiliary qubit,
# in which we will perform different measurements throughout the circuit.
#
# Although it may seem surprising, this is something that can be achieved with a very simple tool: Phase KickBack.
# In this :doc:`demo <tutorial_phase_kickback>` you can find the details of the algorithm but to give the general idea, suppose that we have
# an operator :math:`U` and an eigenvector :math:`|\psi\rangle` with eigenvalue :math:`\lambda` that can take the values :math:`1` or :math:`-1`.
# Phase KickBack provides us with a circuit capable of telling us which of the two cases we are in with the help of an auxiliary qubit:
#
# .. figure:: ../static/demonstration_assets/iterative_quantum_phase_estimation/phase_kick_back.jpeg
#   :align: center
#   :width: 50%
#   :target: javascript:void(0)
#
# Let's look at an example with :math:`|\psi\rangle = |1\rangle` and :math:`U = \text{Z}`:

dev = qml.device("default.qubit", wires = 2, shots = 1)

@qml.qnode(dev)
def circuit():

  # We generate psi
  qml.PauliX(wires = 1)

  # We apply Phase KickBack
  qml.Hadamard(wires = 0)
  qml.ctrl(qml.PauliZ(wires = 1), control = 0)
  qml.Hadamard(wires = 0)

  return qml.sample(wires = 0)

qml.draw_mpl(circuit, wire_order=[0,1], style="pennylane")()
plt.show()

print(f"The output in the ancilla wire is: |{circuit()}>")


##############################################################################
# In this case we have obtained :math:`1` because the eigenvalue is :math:`\lambda = -1`. In the case of starting in the state
# :math:`|0\rangle` you can verify that the output will be 0 since the associated eigenvalue is :math:`\lambda =1`. However,
# although talking about :math:`1` and :math:`-1` is intuitively simple, there is an equivalent notation that will be more
# comfortable for us:
#
# .. math::
#   1 = e^{0} = e^{2\pi i 0}
#
# .. math::
#   -1 = e^{i\pi} = e^{2\pi i \frac{1}{2}}.
#
# Note that :math:`0` can be written with a decimal value as :math:`\overline{0.0}` and :math:`\frac{1}{2}` can be represented as :math:`\overline{0.1}` in binary. Therefore if we represent:
#
# .. math::
#   \lambda = e^{2\pi i \overline{0.\theta_0}},
#
# we will be obtaining the value of :math:`\theta_0` in the auxiliary qubit.
# Although it seems that this already solves our problem, we still have to be careful, if we apply this technique
# directly to our original eigenvalue :math:`\theta = \overline{0.\theta_0\theta_1\theta_2}` we will not obtain the value
# :math:`\theta_0` since the values of :math:`\theta_1` and :math:`\theta_2` do not have to be :math:`0` and this means that the previous
# situation does not directly apply.
# This is not a limitation and there is a very nice trick: instead of first approximating :math:`\theta_0`, we can first
# approximate :math:`\theta_2` by applying KickBack phase to :math:`U^4` instead of :math:`U`.
# Let's see what would happen with the example in which :math:`\theta = 0.375_{10} = 0.011_2`.
# To do this we will take :math:`|\psi\rangle = |1\rangle` and :math:`U = \text{PhaseShift}`:

@qml.qnode(dev)
def circuit():

  # We generate psi
  qml.PauliX(wires = 1)

  # We apply Phase KickBack
  qml.Hadamard(wires = 0)
  qml.ctrl(qml.pow(U(wires = 1), z = 4), control = 0)
  qml.Hadamard(wires = 0)

  return qml.sample(wires = 0)

qml.draw_mpl(circuit, wire_order=[0,1], style="pennylane")()
plt.show()

theta_2 = circuit()
print(f"The value of θ2 is: {theta_2}")


##############################################################################
# Why this works may attract attention at first but as you will see it is not complicated:
#
# .. math::
#   \left( e^{2\pi i \overline{0.\theta_0\theta_1\theta_2}} \right)^4 = e^{2\pi i \overline{\theta_0\theta_1.\theta_2}} = e^{2\pi i \overline{\theta_0\theta_1} + 2\pi i \overline{0.\theta_2}} = e^{2\pi i \overline{0.\theta_2}}.
#
# Basically we are first multiplying by :math:`4` and it is equivalent to moving the decimal point two units to the right.
# After simplifying the expression we can see that we arrive at the particular case of Phase KickBack so we will obtain
# exactly :math:`\theta_2`.
#
# At this point, the next step will be to obtain :math:`\theta_1`, and by the same reasoning we would have to apply :math:`U^2`.
# In this way, you would know that the eigenvalue we are looking for is of the form :math:`e^{2 \pi i \overline{0.\theta_1\theta_2}}`.
# In this case, it is still not equivalent, but since we already know :math:`\theta_2`, we can subtract that value from the
# expression to ensure that our final eigenvalue is :math:`e^{2 \pi i \overline{0.\theta_10}} `. This is something we can do with a :math:`R_Z` gate:


@qml.qnode(dev)
def circuit():

  # We generate psi
  qml.PauliX(wires = 1)

  # We apply Phase KickBack
  qml.Hadamard(wires = 0)
  qml.ctrl(qml.pow(U(wires = 1), z = 2), control = 0)

  qml.PhaseShift(- 2 * np.pi * theta_2 * (1 / 4), wires = 0) # 1/4 = 0.01 en binario

  qml.Hadamard(wires = 0)

  return qml.sample(wires = 0)

qml.draw_mpl(circuit, wire_order=[0,1], style="pennylane")()
plt.show()

theta_1 = circuit()

print(f"The value of θ1 is: {theta_1}")


###############################################################################
# Brilliant! With this we already have one more value, and now we can do the same to finally calculate :math:`\theta_0`:

@qml.qnode(dev)
def circuit():

  # We generate psi
  qml.PauliX(wires = 1)

  # We apply Phase KickBack
  qml.Hadamard(wires = 0)
  qml.ctrl(U(wires = 1), control = 0)

  qml.PhaseShift(- 2 * np.pi * theta_1 * (1 / 4), wires = 0) # 1/4 = 0.01 en binario
  qml.PhaseShift(- 2 * np.pi * theta_2 * (1 / 8), wires = 0) # 1/8 = 0.001 en binario

  qml.Hadamard(wires = 0)

  return qml.sample(wires = 0)

qml.draw_mpl(circuit, wire_order=[0,1], style="pennylane")()
plt.show()

theta_0 = circuit()

print(f"The value of θ0 is: {theta_0}")

###############################################################################
# In this case, we have been conditioning the rotation gates manually, but this is something that can be done through
# :func:`~.pennylane.measure` and :func:`~.pennylane.cond` in the same circuit. Next we will encode it in a generic way where `iters` will refer
# to the number of precision bits we want:

@qml.qnode(dev)
def circuit_iterative_qpe(iters):

    # We generate psi
    qml.PauliX(wires = 1)


    measurements = []

    for i in range(iters):
        qml.Hadamard(wires=0)
        qml.ctrl(qml.pow(U(wires = 1), z=2 ** (iters - i - 1)), control=0)

        for ind, meas in enumerate(measurements):
            qml.cond(meas, qml.PhaseShift)(-2.0 * np.pi / 2 ** (ind + 2), wires=0)

        qml.Hadamard(wires=0)
        measurements.insert(0, qml.measure(wires=0, reset=True))

    return [qml.sample(op = meas) for meas in measurements]


results = circuit_iterative_qpe(iters = 3)
print(f"The estimated phase is: 0.{''.join([str(r) for r in results])}")

###############################################################################
# As you can see, what we are doing is storing the output in a measurement. That measurement will decide whether to apply the rotation to correct the bit or not.
# You have all this implemented in PennyLane natively with the :func:`~.pennylane.iterative_qpe` functionality, let's see how it works:

@qml.qnode(dev)
def circuit():

  # We generate psi
  qml.PauliX(wires = 1)

  # We apply Phase KickBack
  measurements = qml.iterative_qpe(U(wires = 1), ancilla = 0, iters = 3) # iters es el número de bits
  return [qml.sample(meas) for meas in measurements]

qml.draw_mpl(circuit, wire_order=[0,1], style="pennylane")()
plt.show()

##############################################################################
# Very nice, isn't it? The double line indicates that we are working with classical bits from this measurement.
# On that same line we can see black dots that have the same effect as the control we are used to.
# Try modify the number of iterations and see how the circuit changes.
#
# Conclusion
# --------------
#
# In this demo we have shown that we can reduce the number of resources of advanced algorithms such as Quantum Phase
# Estimation. However, we must be careful because in this case, the algorithms will be equivalent only if the input is
# an eigenvector. I invite you to research further and exploit the new iterative QPE functionality we have developed
# to make it easier for you to construct your own ideas.

##############################################################################
# About the author
# ----------------
