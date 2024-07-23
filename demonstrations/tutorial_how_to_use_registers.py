r"""How to use wire registers
====================================

Quantum registers help us group wires that represent different data. In this tutorial, we will
explore how wire registers are constructed and used in PennyLane.
"""

######################################################################
# How to create a wire register
# -----------------------------------------------------
#
# The way to construct a wire register is simple, just use :func:`~pennylane.registers`. Here we
# pass a dictionary where the keys are the names for our registers and the values are the number of
# wires for said register:

wire_register = qml.registers({"alice": 1, "bob": 2, "cleo": 3})
print(wire_register)

# The wire_register created is a dictionary where the keys are the names of the registers and the
# values are :class:`~Wires` instances.
#
# You can also pass in a dictionary that has nested dictionaries as its values:

nested_register = qml.registers({"alice": 1, "bob": {"bob1": 1, "bob2": 1}, "cleo": 1})
print(nested_register)

# Note that :func:`~pennylane.registers` flattens any nested dictionaries, and the order of the
# elements is based on order of appearance and nestedness. For more details on the ordering, refer
# to the documentation for :func:`~pennylane.registers`.
#
######################################################################
# A simple example using wire registers
# -------------------------------------
#
# Wire registers help us group qubits and abstract away the finer details of running quantum algorithms.
# In this example, we demonstrate how to use registers to perform a SWAP test, which is a procedure
# for testing how alike two quantum states are. The SWAP test requires one ancilla qubit and takes
# two input states :math:`|\psi\rangle` and :math:`|\phi\rangle`. We can think of these components
# as three registers. Suppose states :math:`|\psi\rangle` and :math:`|\phi\rangle` are each represented
# with 3 wires. In PennyLane code, that would be:

import pennylane as qml

swap_register = qml.registers({"ancilla": 1, "psi": 3, "phi": 3})

# To perform the SWAP test, we first need to apply the Hadamard gate to our ancilla qubit.


def swap_test():
    qml.Hadamard(swap_register["ancilla"])


# We then need to apply a controlled SWAP operation to :math:`|\psi\rangle` and :math:`|\phi\rangle`
# with our ancilla qubit as the control qubit. To do so, we need to apply CSWAPs to all the wires
# in our registers.


def swap_test():
    qml.Hadamard(swap_register["ancilla"])
    for i in range(len(swap_register["psi"])):
        qml.CSWAP(
            swap_register["ancilla"] | swap_register["psi"][i] | swap_register["phi"][i]
        )


# Finally, we apply the Hadamard gate to our ancilla qubit once again, and measure in the Z basis.


def swap_test():
    qml.Hadamard(swap_register["ancilla"])
    for i in range(len(swap_register["psi"])):
        qml.CSWAP(
            swap_register["ancilla"] | swap_register["psi"][i] | swap_register["phi"][i]
        )
    qml.Hadamard(swap_register["ancilla"])
    return qml.expval(qml.Z(wires=swap_register["ancilla"]))


######################################################################
# A real world example
# --------------------
#
# In this example, we use Quantum Phase Estimation (QPE) and qubitization to calculate the eigenvalues of a Hamiltonian.
# We won't go over the details of how QPE works here, but you can find a great explanation in our `demo <https://pennylane.ai/qml/demos/tutorial_qpe/>`_.
# Generally, QPE is described as having two sets of registers. One register is known as the
# "estimation" (or measurement) register and the other is the state register where we apply our
# unitary operators :math:`U`. Using wire registers, we can define these two registers like so:

import pennylane as qml

wire_register = qml.registers({"state": 4, "estimation": 10})

# To make things more interesting, let's build our unitary operator :math:`U` using qubitization.
# In order to do so, we will need to define an additional "control" register like so:

wire_register = qml.registers({"state": 4, "estimation": 10, "control": 4})

# Finally, let's define our Hamiltonian. We'll choose the H2 molecule for simplicitiy, but feel
# free to try this with any other Hamiltonian you want to find the eigenvalues of.

import pennylane as qml
import numpy as np

symbols = ["H", "H"]
coordinates = np.array([[0.0, 0.0, -0.66140414], [0.0, 0.0, 0.66140414]])
molecule = qml.qchem.Molecule(symbols, coordinates)
H, qubits = qml.qchem.molecular_hamiltonian(molecule)

# For QPE to work, we need to initialize the "state" register with an initial state that has good
# overlap with the eigenstate we want the eigenvalue of. For simplicity's sake, we use the
# Hartree-Fock state. In PennyLane code:

electrons = 2
hf_state = qml.qchem.hf_state(electrons, qubits)

# With this, we can now define our QPE & qubitization circuit like so:

dev = qml.device("lightning.qubit", wires=18)


@qml.qnode(dev)
def circuit():
    # Initialize state register to Hartree-Fock State
    qml.BasisState(hf_state, wires=wire_register["state"])

    # Apply Hadamard gate to all wires in estimation register
    for wire in wire_register["estimation"]:
        qml.Hadamard(wires=wire)

    qml.ControlledSequence(
        qml.Qubitization(H, wire_register["control"]),
        control=wire_register["estimation"],
    )

    qml.adjoint(qml.QFT)(wires=wire_register["estimation"])

    return qml.probs(wires=wire_register["estimation"])


# Finally, we can run our circuit to verify that the results of our calculation is close to what
# we expect:

results = circuit()

lamb = sum([abs(coeff) for coeff in H.terms()[0]])

print(
    "E = ",
    lamb
    * np.cos(2 * np.pi * np.argmax(results) / 2 ** (len(wire_register["estimation"]))),
)

# This should give an eigenvalue of about -1.1359091600247835, which is close to the real value.
# Feel free to change the number of wires in your estimation register to see how the error changes
# accordingly.

######################################################################
# Conclusion
# ----------
#
# Wire registers provide a neat way of organizing and managing wires. Often, algorithms are
# described as acting upon registers and sub-registers, and using wire registers can greatly
# streamline the implementation from theory to code. In this demo, we showed how to construct
# wire registers and use them to implement quantum algorithms.
#
######################################################################
# About the author
# ----------------
#
#
