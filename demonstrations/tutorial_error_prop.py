r"""
How to track algorithmic error using PennyLane
======================================

In order to accurately determine the resources required to run a given quantum workflow, one must carefully track 
and propagate the sources of error within the many algorithms that make up the workflow. Furthermore, there are a 
variety of different errors to keep track of:

- **Input / Encoding:** The error from embedding classical data into the quantum circuit (eg. initial state prep).
- **Algorithm Specific Error:** The error caused by the algorithm itself (eg. QPE with limited readout qubits).
- **Approximate gates and Decompositions:** Decomposing gates approximately (eg. Clifford + T decomposition).
- **Hardware Noise:** The error introduced by noisy quantum channels (eg. Bitflip, Phaseflip).
- **Measurement Uncertainty:** The nature of quantum measurement uncertainity. 
 
We refer to the first three of these as "Algorithmic Errors"; typically, these types of computations are performed by 
hand due to the variety of error metrics to track and the specific handling of such errors for each subroutine. In 
this demo, we present the latest tools in PennyLane which *automatically* track algorithmic error. 


Quantify Error using the Spectral Norm
--------------------------------------

Before we can track the error in our quantum workflow, we need to quantify it. A common method for quantifying the error 
between operators is to compute the "distance" between them; specifically, the spectral norm of the difference between
the operators. We can use the new :class:`~.pennylane.resource.SpectralNormError` class to compute and represent this error. 
Consider for example, that instead of applying :code:`qml.RX(1.234)` we incure some *rounding* error in the rotation angle;
how much error would the resulting operators have?

We can compute this easily with PennyLane:
"""

import pennylane as qml
from pennylane.resource import SpectralNormError

exact_op = qml.RX(1.234, wires=0)

thetas = [1.23, 1.2, 1.0]
ops = [qml.RX(theta, wires=0) for theta in thetas]

for approx_op, theta in zip(ops, thetas):
    error = SpectralNormError.get_error(exact_op, approx_op)
    print(f"Spectral Norm error (theta = {theta:.2f}): {error:.5f}")


###############################################################################
# The error in the operator increases as we round the rotation angle to fewer decimal places as expected.
# Now that we can quantify the error, let's track the error for one of the most common workflows in quantum
# computing. Tracking the error for time evolving a quantum state under a given Hamiltonian!
#
# Tracking Errors in Hamiltonian Simulation
# -----------------------------------------
# Time evolving a quantum state under a hamiltonian requires generating the unitary :math:`\hat{U} = exp(iHt)`.
# In general it is difficult to prepare this operator exactly, so it is instead prepared approximately.
# The most common method to accomplish this the Suzuki-Trotter product formula. This subroutine introduces
# *algorithm specific error* as it produces an approximation to the matrix exponential operator.
#
# Let's explicitly compute the error from this algorithm for a simple hamiltonian:

time = 0.1
Hamiltonian = qml.X(0) + qml.Y(0)

exact_op = qml.exp(Hamiltonian, 1j * time)  #  U = e^iHt ~ TrotterProduct(..., order=2)
approx_op = qml.TrotterProduct(  #  eg: e^iHt ~ e^iXt/2 * e^iYt * e^iXt/2
    Hamiltonian,
    time,
    order=2,
)

error = SpectralNormError.get_error(exact_op, approx_op)  # Expensive to compute
print(f"Error from Suzuki-Trotter algorithm: {error:.5f}")


###############################################################################
# In general, exactly computing the spectral norm is computationally expensive for larger systems as it requires
# diagonalizing the operators. For this reason, we typically use upper bounds on the spectral norm error
# in the product formulas.
#
# We provide two common methods for bounding the error from literature [#TrotterError]_.
# They can be accessed by using :code:`op.error()` and specifying the :code:`method` keyword argument:

op = qml.TrotterProduct(Hamiltonian, time, order=2)

one_norm_error_bound = op.error(method="one-norm")  # one-norm based scaling
commutator_error_bound = op.error(method="commutator")  # commutator based scaling

print("one-norm bound:   ", one_norm_error_bound)
print("commutator bound: ", commutator_error_bound)


###############################################################################
# Custom Error Operations
# -----------------------
# With the new abstract classes it's easy for anyone to define and track custom operations with error.
# All we need to do, is specify how the error is computed. Once the error function is defined, PennyLane
# tracks and propagates the error through the circuit. This makes it easy for us to add and combine multiple
# error operations together in a quantum circuit. In this example we define a custom operation with error to
# act as an approximate decomposition:

# Suppose, for example, that our quantum
# hardware does not natively support rotation gates (:class:`~.pennylane.RX`,
# :class:`~.pennylane.RY`, :class:`~.pennylane.RZ`). How could we decompose the RX gate?
#
# Notice that the sequence :math:`\hat{H} \cdot \hat{T} \cdot \hat{H}` is equivalent
# to :math:`\hat{RX}(\frac{\pi}{4}) \ ` (up to a global phase :math:`e^{i \frac{\pi}{8}}`):

from pennylane import numpy as np

op1 = qml.RX(np.pi / 4, 0)
op2 = qml.GlobalPhase(np.pi / 8) @ qml.Hadamard(0) @ qml.T(0) @ qml.Hadamard(0)

np.allclose(qml.matrix(op1), qml.matrix(op2))


###############################################################################
# We can approximate the RX gate by *rounding* the rotation angle to the lowest multiple
# of :math:`\frac{\pi}{4}`, then using multiple iterations of the sequence above.
#
# The *approximation* error we incure from this decomposition is given by the expression:
#
# .. math::
#
#     \epsilon = \sqrt{2 - 2 * sin(\theta)}
#
# Where :math:`\theta = \frac{\pi \ - \ \Delta_{\phi}}{2}` and :math:`\Delta_{\phi}` is the
# absolute difference between the true rotation angle and the next lowest mulitple of :math:`\frac{\pi}{4}`.
#
# We can take this approximate decomposition and turn it into a PennyLane operation simply by inheriting
# from the :class:`~.pennylane.resource.ErrorOperation` class, and defining the error method:

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
            decomposition += [qml.Hadamard(wires), qml.T(wires), qml.Hadamard(wires)]

        return decomposition

    def error(self):
        """The error in our approximation"""
        phi = self.parameters[0]  # The error depends on the true rotation angle
        delta_phi = phi % (np.pi / 4)

        theta = (np.pi - delta_phi) / 2
        error = np.sqrt(2 - 2 * np.sin(theta))
        return SpectralNormError(error)


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
# Bringing it All Together
# ------------------------
# Tracking the error for each of these components individually is great, but we ultimately want to put these
# pieces together in a quantum circuit. PennyLane now automatically tracks and propagates these errors through
# the circuit. This means we can write our circuits as usual and get all the benefits of error tracking for free.

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
# ----------
# In this demo, we showcased the new :class:`~.pennylane.resource.SpectralNormError` and
# :class:`~.pennylane.resource.ErrorOperation` classes in PennyLane. We highlighted the new functionality
# in :class:`~.pennylane.TrotterProduct` class to compute error bounds in product formulas.
# We explained how to construct a custom error operation. We used this in a simple circuit to track and
# propagate the error through the circuit. We hope that you can make use of these
# tools in cutting edge research workflows to track errors.
#
#
# References
# ----------
#
# .. [#TrotterError]
#
#     Andrew M. Childs, Yuan Su, Minh C. Tran, Nathan Wiebe, and Shuchen Zhu,
#     "Theory of Trotter Error with Commutator Scaling".
#     `Phys. Rev. X 11, 011020 (2021)
#     <https://journals.aps.org/prx/abstract/10.1103/PhysRevX.11.011020>`__
#
#
# About the author
# ----------------
# .. include:: ../_static/authors/jay_soni.txt
