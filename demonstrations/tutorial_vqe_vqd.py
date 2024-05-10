r"""Ground State and Excited State of H2 Molecule using VQE and VQD
===============================================================

Understanding the ground state and excited state energies of quantum systems is paramount in various scientific fields. The **ground state energy** represents the lowest energy configuration of a system, crucial for predicting its stability, chemical reactivity, and electronic properties. **Excited state energies**, on the other hand, reveal the system's potential for transitions to higher energy levels, essential in fields like spectroscopy, materials science, and quantum computing. Both ground and excited state energies provide insights into fundamental properties of matter, guiding research in diverse areas such as drug discovery, semiconductor physics, and renewable energy technologies.

In this demo, we solve this problem by employ two quantum algorithms, the Variational Quantum Eigensolver [#Vqe]_ to find the energy of the ground state,
and the Variational Quantum Deflation [#Vqd]_ to find the excited state based on the above result.
We recommend readers to familiarize themselves with the `VQE tutorial from Pennylane <https://pennylane.ai/qml/demos/tutorial_vqe/>`_. 
"""

######################################################################
# Defining the Hydrogen molecule
# -------------------------------------------
# The `datasets` package from Pennylane makes it a breeze to find the Hamiltonian and the Hartree Fock state
# of some molecules, which fortunately contain :math:`H_2`.
#

import jax
import optax
import pennylane as qml
from pennylane import numpy as np

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

h2_dataset = qml.data.load("qchem", molname="H2", bondlength=0.742, basis="STO-3G")
h2 = h2_dataset[0]
H, qubits = h2.hamiltonian, len(h2.hamiltonian.wires)
print("Number of qubits = ", qubits)
print("The Hamiltonian is ", H)

######################################################################
# The `hf_state` will contain the orbital config with the lowest energy. Let's see what it is.
#
h2.hf_state

######################################################################
# In the Hartree Fock representation, a qubit with state :math:`1` means that there is an electron occupying the respective
# orbital. Chemistry teaches us that the first few orbitals config are :math:`1s^1, 1s^2, 1s^22s^1, 1s^22s^2, ...`. We can see
# that in :math:`H_2`, we start from the config where the two electrons occupy the lowest two energy levels.
#
# Let's also see the gates used to evolve the hf state to the ground state
#
print(h2.vqe_gates)
excitation_angle = 0.27324054462951564

######################################################################
# Setting expectation for VQE and VQD
# -------------------------------------------
#
# Before any training takes place, let’s first look at some of the empirical measured value.
# The energy of an atom at :math:`n` th excitement level is denoted as :math:`E_n`. Unlike computer scientists, in this case physicists starts the value
# of :math:`n` from :math:`1`. It is because :math:`E_n=\frac{E_I}{n^2}`, where :math:`E_I` is the ionization energy.
#
# - Ground state energy:
#     - :math:`H` atom: :math:`E_1=-13.6eV`
#     - :math:`H_2` molecule: :math:`4.52 eV` (source: `Florida State University <https://web1.eng.famu.fsu.edu/~dommelen/quantum/style_a/hmol.html>`_)
#
# - 1st level excitation energy
#     - :math:`H` atom: :math:`E_2=\frac{-13.6}{4}=-3.4eV`
#     - Therefore, to transition from :math:`E_1` to :math:`E_2` for :math:`H` atom: we need :math:`E_1-E_2=10.2eV`
#
# All the measures are in :math:`eV` (electron volt), but later when running the optimization circuit, we would meet another unit called :math:`Ha` (Hatree energy). They both measure energy, just like Joule or calorie
# but in the scale for basic particles. We would define a unit conversion function here
#

def hatree_energy_to_ev(hatree: float):
    return hatree * 27.2107


######################################################################
# Just like training a neural network, the VQE needs two ingredients to make it works. First we need to define
# an Ansatz (which plays the role of the neural network), then a loss function.
#

######################################################################
# Generating the ground state from a dataset
# ------
#
# Starting from the HF state ``[1 1 0 0]``, we will use the Given rotation ansatz below to generate the state
# with the lowest energy.
#

dev = qml.device("default.qubit", wires=qubits)


@qml.qnode(dev)
def circuit_expected(theta):
    qml.BasisState(h2.hf_state, wires=range(qubits))
    qml.DoubleExcitation(theta, wires=[0, 1, 2, 3])
    return qml.expval(H)

print(qml.draw(circuit_expected)(0))

######################################################################
# Let's find the ground energy
#

gs_energy = circuit_expected(excitation_angle)
gs_energy

######################################################################
# Define the lost function
# ------------------------
#
# Remember that the lost function is the second ingredient. We use the second equation in `this
# paper <https://www.nature.com/articles/s41524-023-00965-1>`_.
#
# .. math:: C_1(\theta) = \left\langle\Psi(\theta)|\hat H |\Psi (\theta) \right\rangle + \beta | \left\langle \Psi (\theta)| \Psi_0 \right\rangle|^2
#
# At first sight, it might raise some eyebrows for someone from an ML background, because we
# define the loss function based on the predicted and the ground truth. However, note that we do not have any
# ground truth value here. In this context, a loss function is just a function that we want to
# minimize.
#
# We can then define a lost function using the VQE and VQD methods. The power of VQD is due to the third postulate of quantum mechanics and the fact that
# the eigenbasis are orthogonal. Therefore, once we find the parameters through VQE, our loss function only penalized eigenvector in the second term.
# For this purpose, we implement the function with a quantum technique called `swap test <https://en.wikipedia.org/wiki/Swap_test>`_.
# Let's see it in action.
#

dev_swap = qml.device("default.qubit", wires=qubits * 2 + 1)


@qml.qnode(dev_swap)
def circuit_loss_2(param):
    """
    Constructs a quantum circuit for finding the excited state using swap test.

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
    return qml.probs(8)

######################################################################
# Let’s preview the circuit...
#

print(qml.draw(circuit_loss_2)(param=1))


######################################################################
# The circuit consists of operations to prepare the initial states for the excited and ground states of :math:`H_2`,
# apply the Double Excitation gate with the provided parameters, and the swap test.
# Here we reserve wires 0 to 3 for the excited state calculation and wires 4 to 7 for the ground state of :math:`H_2`.
#
# Now we will define the loss functions. The first (`loss_fn_1`) is using VQE to obtain the ground state
# energy and the second (`loss_fn_2`) use VQD to compute the excited energy using the results obtained by optimizing for
# `loss_fn_1`.
#


def loss_f(theta, beta):
    measurement = circuit_loss_2(theta)
    return beta * (measurement[0] - 0.5) / 0.5


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
        energy.append(circuit_expected(theta))

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

hatree_energy_to_ev(gs_energy), hatree_energy_to_ev(first_excite_energy)

######################################################################
# The result should produce something close to the first ionization energy of :math:`H_2` is
# :math:`1312.0 kJ/mol` according to `Wikipedia <https://en.wikipedia.org/wiki/Hydrogen>`_. Note that this is the ionization energy,
# at which the electron is completely removed from the molecule. Here we are calculating the excited state energy, where an electron
# moves to the outer shell only. Intuitively, we should a lower number than above. We now see how close the result is to reality.
#
# Conclusion
# ----------
# We have used VQE and VQD to find the ground state and the excited state of the :math:`H_2` molecule. One of the applications is
# in photovoltaic devices. For example, the design of solar cells relies on optimizing the energy levels of donor and acceptor
# materials to facilitate charge separation and collection, thereby enhancing solar energy conversion efficiency.
#
# To build up on this work, we recommend readers to run this script with more complex molecules and/or find the energy needed for
# higher excitation levels. Also do not forget check out other tutorials for Quantum chemistry here in Pennylane. Good luck on your Quantum chemistry journey!
#

######################################################################
# References
# ----------
#
# .. [#Vqe]
#
#    Peruzzo, A., McClean, J., Shadbolt, P. et al.
#    "A variational eigenvalue solver on a photonic quantum processor"
#    `Nat Commun 5, 4213 (2014). <https://doi.org/10.1038/ncomms5213>`__.
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
