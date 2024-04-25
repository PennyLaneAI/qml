r"""
How to: Track Algorithmic error using PennyLane
======================================

.. meta::
    :property="og:description": Error propagation with PennyLane
    :property="og:image": https://pennylane.ai/qml/_static/brain_board.png

*Authors: Jay Soni, — Posted: 25 April 2024.*

Introduction
------------

In order to accurately determine the resources required to run a given quantum workflow, one must carefully track 
and propagate the sources of error within the many algorithms that make up the workflow.  

There are a variety of different errors to keep track of:
- Input / Embedding Error
- Algorithm Specific Error
- Approximation Error 
- Hardware Noise
- Measurement Uncertainty

In this demo, we show you how to use pennylane to tackle algorithm specific error and approximation error. 
Typically, these types of computations are performed by hand due to the variety of error metrics to track and 
the specific handling of such errors for each sub-routine. In this demo, we present the latest tools in pennylane 
to help track algorithmic error.

Quantifying the effects of errors / approximations in our gates and how they relate to the error in our
final measurement outcomes is very useful for mordern quantum computing workflows (especially in the 
NISQ era). Typically, these types of computations are performed by hand due to the varity of error metrics
to track and the specific handling of such errors for each sub-routine. To the best of our knowledge, 
there is currently no generally agreed upon systematic approach to tracking and "propagating" errors 
through a quantum workflow. 

Quantify Error using the Spectral Norm
--------------------------------------

One way to quantify the error of an operator is to compute the spectral norm of the difference between the approximate
operator and the true operator. We can use the new `SpectralNormError()` class to compute and represent this error. 
Consider for example, that instead of applying `qml.RX(1.234)` we incure some rounding error and apply `qml.RX(1.2)`; 
how much error would we have? 

We can compute this as follows:
"""

import pennylane as qml
from pennylane.resource import SpectralNormError

exact_op = qml.RX(1.234, wires=0)

thetas = [1.0, 1.2, 1.23]
ops = [qml.RX(theta, wires=0) for theta in thetas]

for approx_op, theta in zip(ops, thetas):
    error = SpectralNormError.get_error(exact_op, approx_op)
    print(f"Spectral Norm error (theta = {theta}): {error}")


###############################################################################
# Tracking Errors in Hamiltonian Simulation
# -----------------------------------------
# One way to evolve a state under a given hamiltonian is to use the method of product formulas. One of the most common
# product formulas is Suzuki-Trotter product formula, which ***approximates*** the exponential operator.
#
# Let's compute the error in this approximation for a simple hamiltonian:

time = 0.1
Hamiltonian = qml.X(0) + qml.Y(0)

exact_op = qml.exp(Hamiltonian, 1j * time)  #  U = e^iHt ~ TrotterProduct(..., order=2)
approx_op = qml.TrotterProduct(Hamiltonian, time)  #  eg: e^iHt ~ e^iXt/2 * e^iYt * e^iXt/2

error = SpectralNormError.get_error(exact_op, approx_op)  # Expensive to compute for large systems
print(error)


###############################################################################
# In generally, computing the Spectral norm error is computationally expensive for larger systems.
# For this reason, we tend to use upper bounds on the Spectral Norm to bound the error in the product formulas.

# We provide two different methods for bounding the error, according to (reference Theory of Trotter Error paper).
# They can be accessed by using the new operator method :code:`op.error()`:

op = qml.TrotterProduct(
    Hamiltonian, time, n=10, order=4
)  # n, order are parameters which tune the approximation

one_norm_error_bound = op.error(method="one-norm")  # one-norm based scaling
commutator_error_bound = op.error(method="commutator")  # commutator based scaling

print("one-norm bound:   ", one_norm_error_bound)
print("commutator bound: ", commutator_error_bound)


###############################################################################
# *(Optional)* With this, one can analyze how the error bounds scale as we tune the parameters of the
# product formula:

# Optional
steps = range(1, 11)
one_norm_error = []
commutator_error = []

for num_steps in steps:
    op = qml.TrotterProduct(Hamiltonian, time, n=num_steps, order=4)

    e_one_norm = op.error(method="one-norm").error
    e_commutator = op.error(method="commutator").error

    one_norm_error.append(e_one_norm)
    commutator_error.append(e_commutator)


# Optional Plot
import matplotlib.pyplot as plt

plt.title("Error vs. Num Trotter Steps")
plt.ylabel("Spectral Norm Error")
plt.xlabel("Number of Trotter steps (n)")

plt.plot(steps, one_norm_error, "-*", label="one-norm")
plt.plot(steps, commutator_error, "-*", label="commutator")

plt.hlines(y=1e-4, xmin=0.5, xmax=10.5, colors="black", linestyles="--", label="epsilon")
plt.yscale("log")
plt.legend()
plt.show()


###############################################################################
# Custom Error Operations
# -----------------------
# With the new abstract classes, it's easy for anyone to define and track custom operations with error.
# All that we need to do, is specify how the error is computed. Lets consider the following example for
# an approximate decomposition of the RX gate:
#
# Notice that the sequence H * T * H = RX(pi/4) up to a global phase (e^i*pi/8):

from pennylane import numpy as np

op1 = qml.RX(np.pi / 4, 0)
op2 = qml.GlobalPhase(np.pi / 8) @ qml.Hadamard(0) @ qml.T(0) @ qml.Hadamard(0)

np.allclose(qml.matrix(op1), qml.matrix(op2))


###############################################################################
# We can then approximate the RX gate by *rounding* the rotation angle to the lowest multiple pi/4 and
# using that many iterations of the decomposition above as our approximate gate.

from pennylane.resource.error import ErrorOperation


class Approximate_RX(ErrorOperation):

    def __init__(self, phi, wires):
        """Approximate decomposition for RX gate"""
        return super().__init__(phi, wires)

    @staticmethod
    def compute_decomposition(phi, wires):
        """Defining the gate decomposition"""
        num_iterations = int(phi // (np.pi / 4))  # how many rotations of pi/4 to apply
        global_phase = num_iterations * np.pi / 8

        decomposition = [qml.GlobalPhase(global_phase)]
        for _ in range(num_iterations):
            decomposition += [qml.Hadamard(0), qml.T(0), qml.Hadamard(0)]

        return decomposition

    def error(self):
        """The error in our approximation"""
        phi = self.parameters[0]  # The error depends on the true rotation angle
        theta = (np.pi / 2) - (phi % (np.pi / 4)) / 2
        maximum_error = np.sqrt(2 - 2 * np.sin(theta))
        return SpectralNormError(maximum_error)


###############################################################################
# We can verify that evaluating the expression for the approximation error gives us the same result as
# explicitly computing the error. Notice that we can access the error of our new operator in the same way
# we did for hamiltonian simulation, using :func:`op.error()`.

phi = 1.23
true_op = qml.RX(phi, wires=0)
approx_op = Approximate_RX(phi, wires=0)

error_from_theory = approx_op.error()
explicit_comp = SpectralNormError.get_error(true_op, approx_op)

print("Explicit computation: ", explicit_comp)
print("Error from function: ", error_from_theory)

###############################################################################
# Bring it All Together
# ---------------------
# Tracking the error for each of these components individually is great, but we ultimately want to put these
# pieces together in a quantum circuit. Pennylane now automatically tracks and propagates these errors through
# a circuit. This means we can write our circuits as usual, but get all the benefits of error tracking:

dev = qml.device("default.qubit")


@qml.qnode(dev)
def circ(H, t, phi1, phi2):

    qml.Hadamard(0)
    qml.Hadamard(1)

    # Approx decomposition
    Approximate_RX(phi1, 0)
    Approximate_RX(phi2, 1)

    qml.CNOT([0, 1])

    # Approx Time evolution:
    qml.TrotterProduct(H, t, order=2)

    # Measurement:
    return qml.state()


###############################################################################
# Along with executing the circuit, we can also compute the error in the circuit through :func:`qml.specs()`:

phi1, phi2 = (phi + 0.12, phi - 3.45)
print("State:")
print(circ(Hamiltonian, time, phi1, phi2), "\n")

errors_dict = qml.specs(circ)(Hamiltonian, time, phi1, phi2)["errors"]
error = errors_dict["SpectralNormError"]
print("Error:")
print(error)


###############################################################################
# Conclusion
# -------------------------------
# In this demo, we showcased the class:`~.pennylane.error_prop.ErrorOperation` classes in PennyLane.
# We explained how to construct a custom error operation. We used this in a simple circuit to track and
# propagate the error through the circuit. We hope that you can use this
# tools in cutting edge research workflows to estimate error.
#
##############################################################################
# About the author
# ----------------
# .. include:: ../_static/authors/jay_soni.txt
