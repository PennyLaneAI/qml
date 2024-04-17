r"""Ground State and Excited State of H2 Molecule using VQE and VQD
===============================================================

Introduction
------------

Quantum computing holds the promise of revolutionizing many fields, and one of the most exciting
applications is in computational chemistry. Traditional methods for simulating molecular systems
become computationally intractable as the size of the system increases. Quantum computers offer a
potential solution to this problem by exploiting the quantum properties of matter to efficiently
simulate molecular behavior.

In this notebook, we will employ two quantum algorithms, the Variational Quantum Eigensolver (VQE)
and the Variational Quantum Deflation (VQD), to find the ground state and excited state of the
:math:`H_2` molecule.
"""

import functools 

import pennylane as qml
from pennylane import numpy as np
import optax
import jax

jax.config.update("jax_platform_name", "cpu")
jax.config.update('jax_enable_x64', True)

h2_dataset = qml.data.load("qchem", molname="H2", bondlength=0.742, basis="STO-3G")
h2 = h2_dataset[0]
H, qubits = h2.hamiltonian, len(h2.hamiltonian.wires)

h2.hf_state

######################################################################
# VQE
# ---
# 
# The VQE needs the following: - An Ansatz - Loss function
# 

print("Number of qubits = ", qubits)
print("The Hamiltonian is ", H)

######################################################################
# Groudtruth
# ----------
# 
# Let’s look at some of the eperical measured value - Ground state energy: - :math:`H` atom:
# :math:`E_1=-13.6eV` - :math:`H_2` molecule: :math:`-1.136*27.21 Ha=-30.91 eV` - 1st level excitation
# energy for :math:`H` atom: :math:`E_2=\frac{-13.6}{4}=-3.4eV` - The energy to transition from
# :math:`E_1` to :math:`E_2` for :math:`H` atom: :math:`10.2eV`
# 

def hatree_energy_to_ev(hatree: float):
    return hatree*27.2107

def ev_energy_to_hatree(ev: float):
    return ev/27.2107

######################################################################
# Begin training
# --------------
# 
# Let’s set some expectation for the optimization process. Thankfully, :math:`H_2` is well studied and
# we have all we need in the ``dataset`` library to know the ground truth
# 

######################################################################
# Ansatz
# ~~~~~~
# 
# Before any run, we can assume that the Jordan Wigner representation ``[1 1 0 0]`` has the lowest
# energy. Let’s calculate that
# 

dev = qml.device("default.qubit", wires=qubits)
@qml.qnode(dev)
def circuit_expected():
    qml.BasisState(h2.hf_state, wires = range(qubits))
    for op in h2.vqe_gates:
        qml.apply(op)
    return qml.probs(), qml.state(), qml.expval(H)

print(f"HF state: {h2.hf_state}")
prob, state, expval = circuit_expected()
print(f"Ground state energy H_2: {expval}")

hatree_energy_to_ev(expval)

######################################################################
# Taking the superposition with themselves and the higher/lower energy level (excite/de-excite). Note
# that in ``h2.vqe_gates`` we already have the value for :math:`\theta`
# 

print(qml.draw(circuit_expected)())

######################################################################
# We would define the same circuit but without the :math:`\theta`. Given 2 :math:`H` and 4 qubits,
# after a double excitation, the HF is the superposition of the states
# 
# .. math:: \alpha\ket{1100}+\beta\ket{0011}:=\cos(\theta)\ket{1100}-\sin(\theta)\ket{0011}
# 

@qml.qnode(dev, diff_method="backprop")
def circuit(param):
    qml.BasisState(h2.hf_state, wires=range(qubits))
    qml.DoubleExcitation(param, wires=[0, 1, 2, 3])
    return qml.state(), qml.expval(H)

######################################################################
# Define the lost function
# ~~~~~~~~~~~~~~~~~~~~~~~~
# 
# Remember that the lost function is the second ingredient. We use the first two equations in `this
# paper <https://www.nature.com/articles/s41524-023-00965-1>`__ :raw-latex:`\begin{align}
# C_0\left( {{{\mathbf{\theta }}}} \right) &= \left\langle {{\Psi}\left( {{{\mathbf{\theta }}}} \right)\left| {\hat H} \right|{\Psi}\left( {{{\mathbf{\theta }}}} \right)} \right\rangle \label{eq:loss_1} \tag{1} \\
# C_1\left( {{{\mathbf{\theta }}}} \right) &= \left\langle {{\Psi}\left( {{{\mathbf{\theta }}}} \right)\left| {\hat H} \right|{\Psi}\left( {{{\mathbf{\theta }}}} \right)} \right\rangle + \beta \left| {\left\langle {{\Psi}\left( {{{\mathbf{\theta }}}} \right)\left| {{\Psi}_0} \right.} \right\rangle } \right|^2 \label{eq:loss_2} \tag{2}
# \end{align}`
# 
# We can then define a lost function
# 
# At first sight, it might raises some eyebrow for someone who is from a ML background, because we
# define the loss function based on the predicted and the groundtruth. However we do not have any
# groundtruth value here. In this context, a loss function is just a function that we want to
# minimize.
# 
# Now we proceed to optimize the variational parameters. Note that :raw-latex:`\eqref{eq:loss_1}` has
# been implemented in ``circuit()``. For the term
# :math:`\beta \left| {\left\langle {{\Psi}\left( {{{\mathbf{\theta }}}} \right)\left| {{\Psi}_0} \right.} \right\rangle } \right|^2`
# in equation :raw-latex:`\eqref{eq:loss_2}`, there is no straight-forward method to compute it
# directly in a quantum machine. To make everything pure quantum, we rely on a swap test as below
# 

dev_swap = qml.device("default.qubit", wires=qubits*2+1)

@qml.qnode(dev_swap)
def circuit_loss_2(param, theta_0):
    """
    args:
    param: rotation angle for the Double Exciment gate, to be found
    theta_0: The rotantion angle corresponding to ground energy
    If psi and phi are orthogonal (|⟨psi|phi⟩|^2 = 1) then the probability that 0 is measured is 1/2 
    If the states are equal (|⟨psi|phi⟩|^2 = 1), then the probability that 0 is measured is 1.
    The measurement on the 0th wire, or 1st qubit is 0.5+0.5(|⟨psi|phi⟩|^2)    
    """
    # The Hamiltonian reserves wire 0 to 3, so they are reserved for the excitement state calculation
    # Wire 4 to 7 are to calcluate the ground state of H_2
    # Wire 8 is for the Hadamard gate
    qml.BasisState(h2.hf_state, wires=range(0, qubits))
    qml.BasisState(h2.hf_state, wires=range(qubits, qubits*2))
    qml.DoubleExcitation(param, wires=range(0, qubits))
    qml.DoubleExcitation(theta_0, wires=range(qubits, qubits*2))
    qml.Hadamard(8)
    for i in range(0, (qubits)):
        qml.CSWAP([8,i,i+qubits])
    qml.Hadamard(8)
    return qml.expval(H), qml.probs(8)

######################################################################
# Let’s preview the circuit
# 

print(qml.draw(circuit_loss_2)(param=0,theta_0=1))

def loss_fn_1(theta):
    """
    Pure expectation value
    """
    _, expval = circuit(theta)
    return expval

def loss_fn_2(theta, theta_0, beta):
    """
    Expectation value
    Depends on the molecule, beta must be large enough to jump over the gap
    """
    expval, measurement = circuit_loss_2(theta, theta_0)    
    return expval + beta*(measurement[0] - 0.5)/0.5

def optimize(loss_f, **kwargs):
    theta = np.array(0.)

    # store the values of the cost function
    energy = [loss_fn_1(theta)]
    conv_tol = 1e-6
    max_iterations = 100
    opt = optax.sgd(learning_rate=0.4)
    
    # store the values of the circuit parameter
    angle = [theta]
    
    opt_state = opt.init(theta)
    
    for n in range(max_iterations):
        gradient = jax.grad(loss_f)(theta, **kwargs)
        updates, opt_state = opt.update(gradient, opt_state)
        theta = optax.apply_updates(theta, updates)
        
        angle.append(theta)
        energy.append(loss_fn_1(theta))
    
        conv = np.abs(energy[-1] - energy[-2])
    
        if n % 5 == 0:
            print(f"Step = {n},  Energy = {energy[-1]:.8f} Ha")
    
        if conv <= conv_tol:
            break
    return angle[-1], energy[-1]

######################################################################
# Run the ground state optimization
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 

ground_state_theta, ground_state_energy = optimize(loss_fn_1)

######################################################################
# Run the 1st excited state optimization
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 

######################################################################
# Next we are going to choose the value for :math:`\beta`, such that :math:`\beta > E_1 - E_0`. In
# other word, :math:`\beta` needs to be larger than the gap between the ground state energy and the
# first excited state energy.
# 

beta = 5

first_excite_theta, first_excite_energy = optimize(loss_fn_2, theta_0=ground_state_theta, beta=beta)

hatree_energy_to_ev(ground_state_energy), hatree_energy_to_ev(first_excite_energy)

######################################################################
# The result should produce something close to the first ionization energy of :math:`H_2` is
# :math:`1312.0 kJ/mol` according to Wikipedia. We now see how close the result is to reality.
# 
# A Hatree is :math:`2625.5 kJ/mol`
# 

kj_per_mol_per_hatree = 2625.5
ground_truth_in_kj_per_mol = 1312
prediction_in_kj_per_mol = first_excite_energy*kj_per_mol_per_hatree

error = np.abs(prediction_in_kj_per_mol-ground_truth_in_kj_per_mol)

print(f"The result is {error} kJ/mol different from reality, or {100-(prediction_in_kj_per_mol/ground_truth_in_kj_per_mol*100)} percent")

######################################################################
# About the author
# ----------------
# # .. include:: ../_static/authors/minh_chau.txt
