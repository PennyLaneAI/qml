r"""Calculate the excited state energy with VQD
===============================================================

Understanding the ground state and excited state energies of quantum systems is paramount in various scientific fields. The **ground state energy** represents the lowest energy configuration of a system, crucial for predicting its stability, chemical reactivity, and electronic properties. **Excited state energies**, on the other hand, reveal the system's potential for transitions to higher energy levels. Both ground and excited state energies provide insights into fundamental properties of matter, guiding research in diverse areas such as drug discovery, semiconductor physics, and renewable energy technologies.

In this demo, we find the first excitation energy of Hydrogen using the ground state energy combined with Variational Quantum Deflation algorithm [#Vqd]_ . To benefit the most from this tutorial, we recommend a familiarization with the `VQE tutorial from Pennylane <https://pennylane.ai/qml/demos/tutorial_vqe/>`_.
"""

######################################################################
# Defining the Hydrogen molecule
# -------------------------------------------
# The `datasets` package from Pennylane makes it a breeze to find the Hamiltonian and the Hartree Fock state
# of some molecules, which fortunately contain our molecule of interest.
# Let's see how we can build the ground state in a simple way:
#

import copy
import jax
import optax
import pennylane as qml
from functools import reduce
from pennylane import numpy as np

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

h2_dataset = qml.data.load("qchem", molname="H2", bondlength=0.742, basis="STO-3G")
h2 = h2_dataset[0]
H, qubits = h2.hamiltonian, len(h2.hamiltonian.wires)
print("Number of qubits = ", qubits)
print("The Hamiltonian is ", H)

######################################################################
# The `hf_state` is our starting point to find the ground state energy. Let’s see what it is.
#
h2.hf_state

######################################################################
# In the Hartree Fock representation, a qubit with state :math:`1`/:math:`0` means that there is/isn’t an
# electron occupying the respective spinning molecular orbital. Here we are starting from the config where the two electrons occupy the lowest two energy levels.
#
# Let’s also see the gates used to evolve the hf state.
#
print(h2.vqe_gates)

######################################################################
# This is a single Double excitement gate with the rotation angle of ~ 0.2732 radians.
#
excitation_angle = 0.27324054462951564

######################################################################
# Ansatz and the ground state
# ------
# Although the usual way to find the ground state is through the first equation in [#Vqd]_
# .. math:: C_0(\theta) = \left\langle\Psi(\theta)|\hat H |\Psi (\theta) \right\rangle,
# this is not enough in this case since we are not looking for the ground state energy.
# Therefore, we must add a penalty term, which results in the second equation in [#Vqd]_.
#
# .. math:: C_1(\theta) = \left\langle\Psi(\theta)|\hat H |\Psi (\theta) \right\rangle + \beta | \left\langle \Psi (\theta)| \Psi_0 \right\rangle|^2
#
# Setting a sufficiently large hyperparameter :math:`\beta` can guarantee that the penalty big is enough to encourage the learning progress.
# From a physics perspective, :math:`\beta` should be larger than the energy gap between the excitement level.
# In addition, we could iteratively calculate other excited states by varying :math:`\beta` by choosing a large enough :math:`\beta_1, \beta_2, ... \beta_k`
# corresponding to the 1st, 2nd ... :math:`k`-th state of excitement energy
#
# After nailing the theory down, let's define an ansatz to simulate the promoting to higher orbitals of electrons when they receive external energy,
# or excited. The Givens rotation ansatz (See `related tutorial <https://pennylane.ai/qml/demos/tutorial_givens_rotations/>`_) describes such phenomenon.
#

dev = qml.device("default.qubit", wires=qubits)

def generate_ground_state(wires):
    qml.BasisState(h2.hf_state, wires=wires)

    for op in h2.vqe_gates:  # use the gates data the datasets package provided
        op = qml.map_wires(op, {op.wires[i]: wires[i] for i in range(len(wires))})
        qml.apply(op)


@qml.qnode(dev)
def circuit():
    generate_ground_state(range(qubits))
    return qml.expval(H)


print(qml.draw(circuit))

######################################################################
# Let's find the ground state energy.
#

gs_energy = circuit()
gs_energy

######################################################################
# VQD in Pennylane
# ----------------
#
# Let's define the circuit for finding the excited state.
#

dev_swap = qml.device("default.qubit", wires=qubits * 2 + 1)

# We have the Hamiltonian for the H2 defined, but it is fixed for wires 0 to 3. Let's adapt the Hamilton for our case
def map_wires(old_H, wires_map):
    """Map the wires of an Hamiltonian according to a wires map.

    Args:
        old_H (Hamiltonian or Tensor or Observable): Hamiltonian to remap the wires of.
        wires_map (dict): Wires map with `(origin, destination)` pairs as key-value pairs.

    Returns:
        Hamiltonian: A copy of the original Hamiltonian with remapped wires.
    """
    new_ops = []
    for op in old_H.ops:
        new_op = copy.copy(op)
        if hasattr(new_op, '__len__'):
            for sub_op in new_op:
                sub_op._wires = sub_op.wires.map(wires_map)
            new_op = reduce(lambda x,y: x@y, new_op.obs)
        else:
            new_op._wires = new_op.wires.map(wires_map)
        new_ops.append(new_op)
    new_H = qml.Hamiltonian(old_H.coeffs, new_ops)
    return new_H

shifted_H = map_wires(H, {0:4, 1:5, 2:6, 3:7})
print(shifted_H)

@qml.qnode(dev_swap)
def circuit_vqd(param):
    """
    Args:
    param (float): Rotation angle for the Double Excitation gate, to be optimized.
    theta_0 (float): The rotation angle corresponding to ground energy.

    Returns:
    Probability distribution of measurement outcomes on the 8th wire.
    """
    qml.BasisState(h2.hf_state, wires=range(0, qubits))
    qml.BasisState(h2.hf_state, wires=range(qubits, qubits * 2))
    for op in h2.vqe_gates:  # use the gates data the datasets package provided
        qml.apply(op)
    qml.DoubleExcitation(param, wires=range(qubits, qubits * 2))
    qml.Hadamard(8)
    for i in range(0, qubits):
        qml.CSWAP([8, i, i + qubits])
    qml.Hadamard(8)
    return qml.expval(shifted_H), qml.probs(8)


######################################################################
# Let’s preview the circuit initialized using a placeholder value of 1.
#


print(qml.draw(circuit_vqd)(param=1))


######################################################################
# The circuit consists of operations to prepare the initial states for the excited and ground states of :math:`H_2`
# and the swap test.
# Here we reserve wires 0 to 3 for the excited state calculation and wires 4 to 7 for the ground state of :math:`H_2`.
#
#
# VQD adds a penalization at the second term, which minimizes when the excited eigenstate is orthogonal to the ground state. It is due to the third postulate of quantum mechanics and the fact that the eigenbasis are orthogonal. For this purpose, we implement the function  `swap test <https://en.wikipedia.org/wiki/Swap_test>`_.
# Let's see it in action.
#


def loss_f(theta, beta):
    exp_h, measurement = circuit_vqd(theta)
    return exp_h + beta * (measurement[0] - 0.5) / 0.5


def optimize(beta):
    theta = 0.0

    # store the values of the cost function
    energy = [loss_f(theta, beta)]
    conv_tol = 1e-6
    max_iterations = 100
    opt = optax.sgd(learning_rate=0.4)

    # store the values of the circuit parameter
    angle = [theta]

    opt_state = opt.init(theta)

    for n in range(max_iterations):
        gradient = jax.grad(loss_f)(theta, beta)
        updates, opt_state = opt.update(gradient, opt_state)
        theta = optax.apply_updates(theta, updates)
        angle.append(theta)
        energy.append(loss_f(theta, beta))

        conv = np.abs(energy[-1] - energy[-2])

        if n % 1 == 0:
            print(f"Step = {n},  Energy = {energy[-1]:.8f} Ha, {theta}")

        if conv <= conv_tol:
            break
    return angle[-1], energy[-1]


######################################################################
# We now have all we need to run the ground state and 1st excited state optimization.
#
# For the excited state, we are going to choose the value for :math:`\beta`, such that :math:`\beta > E_1 - E_0`. In
# other word, :math:`\beta` needs to be larger than the gap between the ground state energy and the
# first excited state energy.
#

beta = 6

first_excite_theta, first_excite_energy = optimize(beta=beta)

print(f"First level excite energy: {first_excite_energy - gs_energy}")

######################################################################
# The result is close to the result we expected.
#
# Conclusion
# ----------
# We have used VQD to find the excited state of the :math:`H_2` molecule. One of the applications is
# in photovoltaic devices. For example, the design of solar cells relies on optimizing the energy levels of donor and acceptor
# materials to facilitate charge separation and collection, thereby enhancing solar energy conversion efficiency.
#
# To build up on this work, we recommend readers to run this script with more complex molecules and/or find the energy needed for
# higher excitation levels. Also do not forget check out other tutorials for Quantum chemistry here in Pennylane. Good luck on your Quantum chemistry journey!
#
# Acknowledgement
# ----------
# The author is grateful to Guillermo Alonso and Soran Jahangiri for their comments.
#
# References
# ----------
#
# .. [#Vqd]
#
#     Higgott, Oscar and Wang, Daochen and Brierley, Stephen
#     "Variational Quantum Computation of Excited States"
#     `Quantum 3, 156 (2019).: <https://dx.doi.org/10.22331/q-2019-07-01-156>`__.
#
# About the author
# ----------------
# .. include:: ../_static/authors/minh_chau.txt
#
