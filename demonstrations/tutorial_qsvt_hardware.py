r"""How to implement QSVT on hardware
=======================================

Performing polynomial transformations on matrices or operators is a task of great interest,
which we have previously addressed in our introductory demos on :doc:`QSVT </demos/tutorial_intro_qsvt>` and its :doc:`practical applications </demos/tutorial_apply_qsvt>`.
However, until now, we have focused more on the algebraic aspect rather than the implementation in hardware.
In this how-to guide, we will show how we can implement the QSVT subroutine in a hardware-compatible way,
taking your applications to the next level.

Angles calculation
------------------

Our goal is to apply a polynomial transformation to a given Hamiltonian (i.e., :math:`p(\mathcal{H})`). To achieve this, we must consider the two
fundamental components of the QSVT algorithm:

- **Projection angles**: A list of angles that will determine the coefficients of the polynomial to be applied.
- **Block Encoding**: The strategy used to introduce the Hamiltonian into the quantum computer. We will use :class:`~.qml.PrepSelPrep`.

Calculating angles is not a trivial task but there are some frameworks like ``pyqsp`` that do that job for us.
Let's see how we can find the angles to apply the polynomial :math:`p(x) = -x + \frac{x^3}{2}+ \frac{x^5}{2}`:

"""

from pyqsp.angle_sequence import QuantumSignalProcessingPhases
import numpy as np

# Define the polynomial, the coefficients are in the order of the polynomial degree.
poly = np.array([0,-1, 0, 0.5, 0 , 0.5])

ang_seq = QuantumSignalProcessingPhases(poly, signal_operator="Wx")
print(ang_seq)

######################################################################
# The output is a list of angles that we can use to apply the polynomial transformation.
# However, we are not finished yet: these angles have been calculated following the "Wx"
# convention, while :class:`~.qml.PrepSelPrep` follows a different one. Moreover, the angles obtained in the
# context of QSP (the ones given by ``pyqsp``) are not the same as the ones we have to use in QSVT. That is why
# we must transform the angles:

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
# The :class:`~.qml.QSVT` template expects two inputs. The first one is the block encoding operator (:class:`~.qml.PrepSelPrep`)
# and the next one are the projection operators (:class:`~.qml.PCPhase`) that encodes the angles properly. Let's define
# a Hamiltonian and the operators:

import pennylane as qml

coeffs = np.array([0.2, -.7, -.6])
coeffs /= np.linalg.norm(coeffs, ord = 1) # Normalize the coefficients

obs = [qml.X(3), qml.X(3) @ qml.Z(4), qml.Z(3) @ qml.Y(4)]

H = qml.dot(coeffs, obs)

# We need |log2(len(coeffs))| = 2 control wires to encode the Hamiltonian
control_wires = [1,2]
block_encode = qml.PrepSelPrep(H, control = control_wires)

projectors = [qml.PCPhase(angles[i], dim=2**len(H.wires), wires= control_wires + H.wires)
              for i in range(len(angles))]

######################################################################
# We now have all the pieces to run the algorithm. Let's put it together and see the resulting matrix:

dev = qml.device("default.qubit")

@qml.qnode(dev)
def circuit():

    qml.Hadamard(0)
    qml.ctrl(qml.QSVT, control=0, control_values=[1])(block_encode, projectors)
    qml.ctrl(qml.adjoint(qml.QSVT), control=0, control_values=[0])(block_encode, projectors)
    qml.Hadamard(0)

    return qml.state()

matrix = qml.matrix(circuit, wire_order = [0] + control_wires + H.wires)()
print(np.round(matrix[: 2**len(H.wires), :2**len(H.wires)],4))

######################################################################
# The circuit is encoding :math:`p(\mathcal{H})` in the top left block of its matrix.
#
# The idea behind this circuit is that QSVT encodes the desired polynomial :math:`p(\mathcal{H})` but also
# a polynomial :math:`i q(\mathcal{H})`. To isolate :math:`p(\mathcal{H})` we have used an auxiliary qubit and the property that
# the sum of a complex number and its conjugate gives us twice its real part. We
# recommend :doc:`this demo </demos/tutorial_apply_qsvt>` to learn more about the structure
# of the circuit.
# Finally, we can verify that the results obtained are as expected:

from numpy.linalg import matrix_power as mpow

H_mat = qml.matrix(H, wire_order = [3,4])

# We calculate p(H) = -H + 0.5 * H^3 + 0.5 * H^5
H_poly = -H_mat + 0.5 * mpow(H_mat, 3) + 0.5* mpow(H_mat, 5)

print(np.round(H_poly,4))

######################################################################
# The matrix obtained from the QSVT subroutine is the same as the one obtained by applying the polynomial
# directly to the Hamiltonian! The great advantage of this approach is that all the templates used can be
# decomposed into basic gates easily.
#
#
# Conclusion
# ----------
# In this brief how-to we have seen how to apply QSVT on a Hamiltonian. Note that the algorithm is sensitive to
# the encoding used so make sure that the angles are converted to the proper format.
# We hope this how-to will serve as a guide to running your own workflows and experimenting with more advanced Hamiltonians and functions.
#
# About the author
# ----------------
#