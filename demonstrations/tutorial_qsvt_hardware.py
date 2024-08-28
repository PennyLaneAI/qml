r"""How to implement QSVT on hardware
=======================================

Performing polynomial transformations on matrices or operators is a task of great interest,
which we have previously addressed in our introductory demos on :doc:`QSVT </demos/tutorial_intro_qsvt>` and its :doc:`practical applications </demos/tutorial_apply_qsvt>`.
In this how-to guide, we will show how we can implement the QSVT subroutine in a hardware-compatible way,
taking your applications to the next level.

Angles calculation
------------------

Our goal is to apply a polynomial transformation to a given Hamiltonian, i.e., :math:`p(\mathcal{H})`. To achieve this, we must consider the two
fundamental components of the QSVT algorithm:

- **Projection angles**: A list of angles that will determine the coefficients of the polynomial to be applied.
- **Block Encoding**: The strategy used to encode the Hamiltonian. We will use :class:`~.qml.PrepSelPrep`.

Calculating angles is not a trivial task but there are tools such as ``pyqsp`` that do the job for us.
For instance, to find the angles to apply the polynomial :math:`p(x) = -x + \frac{x^3}{2}+ \frac{x^5}{2}`, we can run this code:

.. code-block:: python

   from pyqsp.angle_sequence import QuantumSignalProcessingPhases
   import numpy as np

   # Define the polynomial, the coefficients are in the order of the polynomial degree.
   poly = np.array([0,-1, 0, 0.5, 0 , 0.5])

   ang_seq = QuantumSignalProcessingPhases(poly, signal_operator="Wx")

The angles obtained after execution are as follows:

"""

ang_seq = [
    -1.5115007723754004,
    0.6300762184670975,
    0.8813995564082947,
    -2.2601930971815003,
    3.7716688720568885,
    0.059295554419495855,
]

######################################################################
# We use these angles to apply the polynomial transformation.
# However, we are not finished yet: these angles have been calculated following the "Wx"
# convention, while :class:`~.qml.PrepSelPrep` follows a different one. Moreover, the angles obtained in the
# context of QSP (the ones given by ``pyqsp``) are not the same as the ones we have to use in QSVT. That is why
# we must transform the angles:

import numpy as np


def convert_angles(angles):
    num_angles = len(angles)
    update_vals = np.zeros(num_angles)

    update_vals[0] = 3 * np.pi / 4 - (3 + len(angles) % 4) * np.pi / 2
    update_vals[1:-1] = np.pi / 2
    update_vals[-1] = -np.pi / 4

    return angles + update_vals


angles = convert_angles(ang_seq)
print(angles)

######################################################################
# Using these angles, we can now start working with the template.
#
# QSVT on Hardware
# -----------------
#
# The :class:`~.qml.QSVT` template expects two inputs. The first one is the block encoding operator, :class:`~.qml.PrepSelPrep`,
# and the second one is a set of projection operators, :class:`~.qml.PCPhase`, that encode the angles properly.
# We will see how to apply them later, but first let's define
# a Hamiltonian and manually apply the polynomial of interest:

import pennylane as qml
from numpy.linalg import matrix_power as mpow

coeffs = np.array([0.2, -0.7, -0.6])
coeffs /= np.linalg.norm(coeffs, ord=1)  # Normalize the coefficients

obs = [qml.X(3), qml.X(3) @ qml.Z(4), qml.Z(3) @ qml.Y(4)]

H = qml.dot(coeffs, obs)

H_mat = qml.matrix(H, wire_order=[3, 4])

# We calculate p(H) = -H + 0.5 * H^3 + 0.5 * H^5
H_poly = -H_mat + 0.5 * mpow(H_mat, 3) + 0.5 * mpow(H_mat, 5)

print(np.round(H_poly, 4))

######################################################################
# Great, we already know what the target result is. Let's now see how apply the polynomial with a quantum circuit.
# We start by defining the proper input operators for the :class:`~.qml.QSVT` template.

# We need |log2(len(coeffs))| = 2 control wires to encode the Hamiltonian
control_wires = [1, 2]
block_encode = qml.PrepSelPrep(H, control=control_wires)

projectors = [
    qml.PCPhase(angles[i], dim=2 ** len(H.wires), wires=control_wires + H.wires)
    for i in range(len(angles))
]


@qml.qnode(qml.device("default.qubit"))
def circuit():

    qml.Hadamard(0)
    qml.ctrl(qml.QSVT, control=0, control_values=[1])(block_encode, projectors)
    qml.ctrl(qml.adjoint(qml.QSVT), control=0, control_values=[0])(block_encode, projectors)
    qml.Hadamard(0)

    return qml.state()


matrix = qml.matrix(circuit, wire_order=[0] + control_wires + H.wires)()
print(np.round(matrix[: 2 ** len(H.wires), : 2 ** len(H.wires)], 4))

######################################################################
# The matrix obtained using QSVT is the same as the one obtained by applying the polynomial
# directly to the Hamiltonian! That means the circuit is encoding :math:`p(\mathcal{H})` correctly.
# The great advantage of this approach is that all the building blocks used in the circuit can be
# decomposed into basic gates easily.
#
# Please also note that QSVT encodes the desired polynomial :math:`p(\mathcal{H})` as well as
# a polynomial :math:`i q(\mathcal{H})`. To isolate :math:`p(\mathcal{H})`, we have used an auxiliary qubit and considered the that
# the sum of a complex number and its conjugate gives us twice its real part. We
# recommend :doc:`this demo </demos/tutorial_apply_qsvt>` to learn more about the structure
# of the circuit.
#
# Conclusion
# ----------
# In this brief how-to we demonstrated applying QSVT on a sample Hamiltonian. Note that the algorithm is sensitive to
# the block-encoding method, so please make sure that the projection angles are converted to the proper format.
# This how-to serves as a guide for running your own workflows and experimenting with more advanced Hamiltonians and polynomial functions.
#
# About the author
# ----------------
#
