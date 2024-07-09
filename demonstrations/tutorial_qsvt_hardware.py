r"""How to implement QSVT on hardware
=======================================

Performing polynomial transformations on matrices is a task of great interest,
which we have previously addressed in our introductory demos on :doc:`QSVT </demos/tutorial_intro_qsvt>` and its practical :doc:`applications </demos/tutorial_apply_qsvt>`.
However, until now, we had focused more on the algebraic aspect rather than on the implementation in real hardware.
In this how-to guide, you will see how we can implement the QSVT subroutine in a hardware-compatible way,
taking your applications to the next level.

Conventions and Angle Calculation 
-------------------------------------

Our goal is to apply a polynomial transformation to a given Hamiltonian. To achieve this, we must consider the two
fundamental components of the QSVT algorithm:

- **Subroutine angles**: A list of angles that will determine the coefficients of the polynomial to be applied.
- **Block Encoding**: The strategy used to introduce the Hamiltonian into our computer. We will use :class:`~.qml.PrepSelPrep`.

Calculating angles is not a trivial task but there are some frameworks like ``pyqsp`` that do that job for us.
Let's see how we can find the angles to apply the polynomial :math:`-x + \frac{x^3}{2}+ \frac{x^5}{2}`:

"""

from pyqsp.angle_sequence import QuantumSignalProcessingPhases
import numpy as np

# Define the polynomial. The coefficients are in the order of the polynomial degree.
poly = np.array([0,-1, 0, 0.5, 0 , 0.5])

ang_seq = QuantumSignalProcessingPhases(poly, signal_operator="Wx")
print(ang_seq)

######################################################################
# The output is a list of angles that we can use to apply the polynomial transformation.
# However, we are not finished yet. These angles have been calculated assuming one type of encoding following the "Wx"
# convention while our encoding of the Hamiltonian follows another convention. Moreover, the angles obtained in the
# context of QSP (the ones given by ``pyqsp``) are different from the ones we have to use in QSVT since these
# subroutines have different decompositions. That is why we must make a transformation of the angles:

def convert_angles(angles):
    num_angles = len(angles)
    update_vals = np.empty(num_angles)

    update_vals[0] = 3 * np.pi / 4 - np.pi / 2
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

obs = [qml.X(3), qml.Z(4), qml.Z(5)]

H = qml.dot(coeffs, obs)

# We need |log2(len(coeffs))| = 2 control wires to encode the Hamiltonian
control_wires = [1,2]
block_encode = qml.PrepSelPrep(H, control = control_wires)

projectors = [qml.PCPhase(angles[i], dim=2**len(H.wires), wires= control_wires + H.wires) for i in range(len(angles))]

######################################################################
# We now have all the pieces to run the algorithm. Let's put it together and see the resulting matrix:

dev = qml.device("default.qubit")

@qml.qnode(dev)
def circuit():
    qml.QSVT(block_encode, projectors)
    return qml.state()

matrix = qml.matrix(circuit, wire_order = control_wires + H.wires)()[: 2**len(H.wires), :2**len(H.wires)]

print(np.round(matrix,4).real)

######################################################################
# The real part of the circuit matrix is equivalent to the application of our polynomial to the Hamiltonian.
# Let's see that we obtained the correct result.

H_mat = qml.matrix(H, wire_order = [3,4,5])

H_poly = -H_mat + 0.5 * np.linalg.matrix_power(H_mat, 3) + 0.5* np.linalg.matrix_power(H_mat, 5)

print(np.round(H_poly,4).real)

######################################################################
# As we can see, the matrix obtained from the QSVT subroutine is the same as the one obtained by applying the polynomial
# directly to the Hamiltonian. The great advantage of this approach is that the :class:`~.qml.BlockEncode` operator has not been used,
# making it possible to decompose the algorithm into basic gates easily.
#
# Extracting the real part
# ------------------------
#
# In the previous example we have manually extracted the real part of the matrix to check that it actually worked.
# However, this is not something we can do in hardware. For this reason we present here a way to extract the real
# part with the help of an auxiliary qubit:

@qml.qnode(dev)
def circuit():

    qml.Hadamard(0)
    qml.ctrl(qml.QSVT, control=0, control_values=[1])(block_encode, projectors)
    qml.ctrl(qml.adjoint(qml.QSVT), control=0, control_values=[0])(block_encode, projectors)
    qml.Hadamard(0)

    return qml.state()

matrix = qml.matrix(circuit, wire_order = [0] + control_wires + H.wires)()[: 2**len(H.wires), :2**len(H.wires)]
print(np.round(matrix,4))

######################################################################
# As we can see, the real part has been correctly encoded into the circuit.
# The idea behind this circuit is based on the fact that the sum of a complex number and its conjugate gives us
# its real part. I recommend :doc:`this demo </demos/tutorial_lcu_blockencoding>` to learn more about the structure
# of the circuit.
#
# Conclusion
# ----------
# In this brief how-to we have seen how we can apply QSVT on a Hamiltonian. Note that the algorithm is sensitive to
# the encoding used so make sure that the angles are being converted to the proper format.
# I hope this how-to will serve as a guide to run your own workflows and experiment with more advanced Hamiltonians and functions.
#
# About the author
# ----------------
# .. include:: ../_static/authors/guillermo_alonso.txt
