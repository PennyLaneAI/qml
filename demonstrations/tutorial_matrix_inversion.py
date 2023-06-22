r"""How to: Use QSVT for Matrix Inversion
=============================================================

*Author: Jay Soni, Jarret — Posted: .*

The Quantum Singular Value Transformation[insert link to paper] (QSVT) is a powerful tool in the world of quantum algorithms.
We have explored this topic from a pedagogical / theoretical prespective a few times (see these demos [insert link to other demos]).

PennyLane has built in functionality to apply QSVT to a quantum circuit.
In this demo we provide a practical guide on how to use it, focusing on matrix inversion as a guiding example.
First lets recall how to apply QSVT in a circuit:
"""
import pennylane as qml
from pennylane import numpy as np

dev = qml.device("default.qubit", wires=[0, 1])

A = np.array([[0.1, 0.2],
              [0.3, 0.4]])


@qml.qnode(dev)
def my_circ(phase_angles):
    qml.qsvt(A, phase_angles, wires=[0, 1])
    return qml.state()


phase_angles = np.array([0.0, 1.0, 2.0, 3.0])
my_circ(phase_angles)
print(my_circ.tape.expand().draw())


##############################################################################
# Obtaining Phase Angles
# -----------------------
#
# The specific transformation performed by QSVT depends on the phase angles :math:`\vect{\phi}` used.
# While we may have a particular transformation in mind, its not trivial to compute the phase angles which will produce
# that transformation. We highlight two approaches to obtain the phase angles:
#
# - Use external packages (eg. PyQSP [insert link to pyqsp here])
# - Use optimization to learn the optimal phase angles
#
# First we highlight how to use external packages and a known transformation:

import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial as Poly

my_poly = [0, -3/2, 0, 5/2]        # specify some polynomial, this one is: 1/2 * (5x^3 - 3x)
x_vals = np.linspace(-1, 1, 50)
y_vals = Poly(my_poly)(x_vals)

plt.plot(x_vals, y_vals, label="target")
plt.axvline(0, color="k", linestyle="--")
plt.axhline(0, color="k", linestyle="--")
plt.legend()
plt.show()

from pyqsp.angle_sequence import QuantumSignalProcessingPhases  # import the external angle solver (PyQSP)

ang_seq = QuantumSignalProcessingPhases(my_poly, signal_operator="Wx")  # get phase angles for polynomial
print("pyqsp angles:", ang_seq)

qsvt_y_vals = []

for a in x_vals:
    poly_a = qml.matrix(qml.qsvt)(a, ang_seq, wires=[0], convention="Wx")  # note make sure the conventions match!
    qsvt_y_vals.append(np.real(poly_a[0,0]))  # the polynomial transformation is applied to the top left entry

plt.plot(x_vals, y_vals, label="target")
plt.plot(x_vals, qsvt_y_vals, "*", label="qsvt")  # plot the computed values

plt.axvline(0, color="k", linestyle="--")
plt.axhline(0, color="k", linestyle="--")

plt.legend()
plt.show()

##############################################################################
# Alternatively, we can leverage PennyLane's fully differentiable workflow and optimize the phase angles via gradient
# descent. We can use a simple mean squared error (MSE) loss function which compares our expected transformation with
# our target transformation.

def target_func(x):
    return (1 / 2) * (5 * x ** 3 - 3 * x)  # target polynomial


def mean_squared_error(phi):
    norm = 1 / len(x_vals)

    sum_square_error = 0
    for x in x_vals:
        qsvt_val = qml.matrix(qml.qsvt(x, phi, wires=[0]))[0, 0]
        sum_square_error += (np.real(qsvt_val) - target_func(x)) ** 2 + (np.imag(qsvt_val)) ** 2

    return norm * sum_square_error  # Add a note to explain that you can use more complicated loss functions


# Initialize parameters:
phi = np.random.rand(4)
x_vals = np.linspace(-1, 1, 50)

# Optimization:
opt = qml.AdagradOptimizer(0.3)
for epoch in range(30):
    phi, cost = opt.step_and_cost(mean_squared_error, phi)
    if (epoch + 1) % 5 == 0:
        print(f"iter: {epoch + 1}, cost: {cost}")

print("Completed Optimization!")

print(f"optimized phase angles: {phi}")

y_vals = []
qsvt_y_vals = []

for a in x_vals:
    y_vals.append(target_func(a))

    poly_a = qml.matrix(qml.qsvt)(a, phi, wires=[0])
    qsvt_y_vals.append(np.real(poly_a[0, 0]))

plt.plot(x_vals, y_vals, label="target")
plt.plot(x_vals, qsvt_y_vals, "*", label="qsvt")  # plot the computed values

plt.axvline(0, color="k", linestyle="--")
plt.axhline(0, color="k", linestyle="--")

plt.legend()
plt.show()

##############################################################################
# Matrix Inversion
# ----------------
# Lets use these techniques to solve the problem of matrix inversion! A few things to take care off:
#
# - 1/x is not bounded on the domain, so we need re-normalize the transform by the condition number kappa
# [1 / (kappa * x)]
# - This is fine since kappa is a fixed constant, thus we can multiply it back in after we have computed the inverse!
# - We can use this information to focus our optimization protocol to train in the domain of interest

dev = qml.device("default.qubit", wires=["ancilla", 0])


def my_circ(x, phi):
    phi1 = phi[:len(phi) // 2]  # Use the first half of the angles for U1
    phi2 = phi[len(phi) // 2:]  # Use the remaining angles for U2

    # LCU for even and odd QSVT:
    qml.Hadamard(wires="ancilla")  # Prep

    qml.ctrl(qml.qsvt, control=("ancilla",), control_values=(0,))(x, phi1, wires=[0])  # Sel U1
    qml.ctrl(qml.qsvt, control=("ancilla",), control_values=(1,))(x, phi2, wires=[0])  # Sel U2

    qml.Hadamard(wires="ancilla")  # Prep_dagger


np.random.seed(42)
import time

def target_func(x):
    return 1 / (kappa * x)


def mean_squared_error(phi):
    norm = 1 / len(samples_a)

    sum_square_error = 0
    for s in samples_a:
        qsvt_val = qml.matrix(my_circ)(s, phi)[0, 0]
        sum_square_error += (np.real(qsvt_val) - target_func(s)) ** 2 + (np.imag(qsvt_val)) ** 2

    return norm * sum_square_error


# Initialize parameters:
kappa = 10
phi = np.random.rand(101)  # degree 50 + degree 51 polynomial approximation
samples_a = np.linspace(1 / kappa, 1, 75)  # Taking 50 samples in (1/kappa, 1) to train

# Optimization:
opt = qml.AdagradOptimizer(0.3)
epoch = 0
cost = 1
store_best_phi = [cost, phi, epoch]
store_other_phis = [(phi, cost)]

t0 = time.time()
while cost > 0.001:
    epoch += 1
    phi, cost = opt.step_and_cost(mean_squared_error, phi)

    if cost <= store_best_phi[0]:
        store_best_phi = [cost, phi, epoch]

    if epoch % 5 == 0:
        print(f"iter: {epoch}, cost: {cost}")

    if epoch % 10 == 0:
        store_other_phis.append((phi, cost))

    if epoch > 150:
        print("Epoch limit reached!")
        break

import matplotlib.pyplot as plt

samples_inv = np.linspace(1/kappa, 1, 50)
inv_x = [target_func(a) for a in samples_inv]

print(store_best_phi[0])
samples_a = np.linspace(0, 1, 100)
y_vals = [qml.matrix(my_circ)(x, store_best_phi[1])[0,0] for x in samples_a]


plt.plot(samples_a, y_vals, label="qsvt")
plt.plot(samples_inv, inv_x, label="1/(kappa * x)")

plt.vlines(1/kappa, -1.0, 1.0, linestyle="--", color="black", label="1/kappa")
plt.vlines(0.0, -1.0, 1.0, color="black")
plt.hlines(0.0, -0.1, 1.0, color="black")

plt.legend()
plt.show()

t1 = time.time()
print(f"Completed Optimization! (Time = {t1 - t0})")


# Target problem to solve: Ax = b

A=np.array([[0.1, 0.2],
            [0.3, 0.4]])

target_x= np.array([1.23, -4.5])
b = np.array([-0.777, -1.431])   # here b satisfies: A @ x = b


# We need to make sure the operators and states we use are normalized:

ANorm=max(np.linalg.norm(A @ A.conj().T,ord=np.inf),np.linalg.norm(A.conj().T @ A,ord=np.inf))
Normed_A = qml.math.array(A) / ANorm

bNorm=np.sqrt(np.sum(np.abs(b)**2))
Normed_b=b/bNorm

xNorm=np.sqrt(np.sum(np.abs(target_x)**2))
Normed_x=target_x/xNorm


@qml.qnode(qml.device('default.qubit', wires=["ancilla", 0, 1]))
def circuit(phi):
    # Partition angles:
    phi1 = phi[:len(phi) // 2]
    phi2 = phi[len(phi) // 2:]

    # Prepare b:
    qml.QubitStateVector(Normed_b, wires=[1])

    # Apply A^(-1)
    qml.Hadamard(wires="ancilla")

    qml.ctrl(qml.qsvt, control=("ancilla",), control_values=(0,))(Normed_A, phi1, wires=[0, 1])
    qml.ctrl(qml.qsvt, control=("ancilla",), control_values=(1,))(Normed_A, phi2, wires=[0, 1])

    qml.Hadamard(wires="ancilla")

    return qml.state()


from pennylane.math import fidelity

lst_epoch = 10 * np.arange(len(store_other_phis))
lst_computed_x = [circuit(phi)[:2] * kappa * np.sqrt(bNorm) * (1/ANorm) for phi, _ in store_other_phis]
best_computed_x = circuit(store_best_phi[1])[:2] * kappa * np.sqrt(bNorm) * (1/ANorm)

lst_cost = [cost for _, cost in store_other_phis]
lst_fidelity = [fidelity(Normed_x, computed_x / np.linalg.norm(computed_x)) for computed_x in lst_computed_x]


# Plot the fidelity between the target state and input state as we trained:

plt.plot(lst_epoch[:], lst_fidelity[:], "--.b")
plt.plot([store_best_phi[2]],
         [fidelity(Normed_x, best_computed_x / np.linalg.norm(best_computed_x))],
         "*",
         label="best")

plt.xlabel("Training Iterations")
plt.ylabel("Fidelity")
plt.legend()
plt.show()


# Plot the loss value as we train the parameters

plt.plot(lst_epoch[:], lst_cost[:], "--.b")
plt.plot([store_best_phi[2]], [store_best_phi[0]], "*", label="best")

plt.xlabel("Training Iterations")
plt.ylabel("Training Cost")
plt.legend()
plt.show()

###############################################################################
# References
# ----------
#
##############################################################################
# About the author
# ----------------
# .. include:: ../_static/authors/jay_soni.txt