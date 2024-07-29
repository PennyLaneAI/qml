r"""How to use wire registers
====================================
Wire registers help us group qubits and abstract away the finer details of running quantum
algorithms. In this tutorial, we will explore how wire registers are constructed and used
in PennyLane.
"""

######################################################################
# How to create a wire register
# -----------------------------
#
# The way to construct a wire register is simple --- just use :func:`~pennylane.registers`. Here we
# pass a dictionary where the keys are the register names and the values are the number of
# wires for each register:

import pennylane as qml

wire_register = qml.registers({"alice": 1, "bob": 2, "cleo": 3})
print(wire_register)

# The wire_register created is a dictionary where the keys are the names of the registers and the
# values are :class:`~Wires` instances.
#
# You can also pass in a dictionary that has nested dictionaries as its values.

nested_register = qml.registers(
    {"all_registers": {"alice": 1, "bob": {"bob1": {"bob1a": 1}, "bob2": 1}, "cleo": 1}}
)
print(nested_register)

# Note that :func:`~pennylane.registers` flattens any nested dictionaries, and the order of the
# elements is based on order of appearance and nestedness. For more details on the ordering, refer
# to the documentation for :func:`~pennylane.registers`.
#
# Accessing elements in your registers is the same as accessing any element in a dictionary.

print(nested_register["alice"])
print(nested_register["bob1a"])

# You can access a specific wire index via its index in a register.

print(nested_register["all_registers"][2])
print(nested_register["bob1a"][0])

# You can also create registers using set operations. For more details on what set operations are
# supported, refer to the documentation of :func:`~pennylane.registers`.

new_register = nested_register["alice"] | nested_register["cleo"]
print(new_register)
######################################################################
# A simple example using wire registers
# -------------------------------------
#
# In this example, we demonstrate how one can implement the SWAP test with registers. 
# The `SWAP test <https://en.wikipedia.org/wiki/Swap_test>`_
# is an algorithm that calculates the squared inner
# product of two input states. It requires one ancilla qubit and takes two input states :math:`|\psi\rangle`
# and :math:`|\phi\rangle`. We can think of these components as three registers. Suppose states
# :math:`|\psi\rangle` and :math:`|\phi\rangle` are each represented with 3 wires. In PennyLane
# code, that would be:

import pennylane as qml

swap_register = qml.registers({"ancilla": 1, "psi": 3, "phi": 3})


def swap_test():
    # Prepare phi and psi in some arbitrary state
    for state in ["phi", "psi"]:
        qml.BasisState([1, 1, 0], swap_register[state])

    qml.Hadamard(swap_register["ancilla"])
    for i in range(len(swap_register["psi"])):
        qml.CSWAP(
            swap_register["ancilla"] | swap_register["psi"][i] | swap_register["phi"][i]
        )
    qml.Hadamard(swap_register["ancilla"])
    return qml.expval(qml.Z(wires=swap_register["ancilla"]))


print(swap_test())

######################################################################
# Advanced example
# --------------------
#
# Building quantum algorithms often requires working with certain constraints or trade-offs.
# For example, sometimes one needs to trade circuit depth for qubit count and vice versa. This
# often means experimenting with a variety of subroutines to benchmark and test the efficiency of
# a quantum algorithm. Using registers can greatly streamline the process of modifying a workflow
# by simplifying wire management.
#
# In this example, we use Quantum Phase Estimation (QPE) to calculate the eigenvalues of a Hamiltonian
# and compare how we would define such a workflow with and without using registers.
# We won't go over the details of how QPE works here, but you can find a great explanation in our `demo <https://pennylane.ai/qml/demos/tutorial_qpe/>`_.
# Generally, QPE is described as having two sets of registers. One register is known as the
# "estimation" (or measurement) register and the other is the state register where we apply our
# unitary operators :math:`U`. We can define these two registers using registers or with wires.
# For comparison in PennyLane code:

import pennylane as qml

wire_register = qml.registers({"state": 4, "estimation": 8})  # registers

state = range(4)  # using wires instead
estimation = range(4, 12)

# To build our unitary operator :math:`U`, there are a variety of options. We can opt to use a
# straight-forward block encoding, or choose to use a subroutine like qubitization.
# We'll use qubitization, which means we have to define another "control" register.
# Full code block now looks like this:

wire_register = qml.registers({"state": 4, "estimation": 8, "control": 4})  # registers

state = range(4)  # using wires instead
estimation = range(4, 12)
control = range(12, 16)

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

# With this, we can now define our QPE circuit like so:

dev = qml.device("lightning.qubit", wires=16)


@qml.qnode(dev)
def registers_circuit():  # Using registers
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


@qml.qnode(dev)
def wires_circuit():  # Using wires
    # Initialize state register to Hartree-Fock State
    qml.BasisState(hf_state, wires=state)

    # Apply Hadamard gate to all wires in estimation register
    for wire in estimation:
        qml.Hadamard(wires=wire)

    qml.ControlledSequence(
        qml.Qubitization(H, control),
        control=estimation,
    )

    qml.adjoint(qml.QFT)(wires=estimation)

    return qml.probs(wires=estimation)


# Finally, we can run our circuit to verify that the results of our calculation is close to what
# we expect. This should give an eigenvalue of about -1.1359091600247835, which is close to the
# real value.

results = registers_circuit()  # or wires_circuit()

lamb = sum([abs(coeff) for coeff in H.terms()[0]])

print(
    "E = ",
    lamb
    * np.cos(2 * np.pi * np.argmax(results) / 2 ** (len(wire_register["estimation"]))),
)

# Changing the number of wires in your estimation register is very easy
# with registers, but can be very error-prone when using wires:

wire_register = qml.registers({"state": 4, "estimation": 10, "control": 4})

state = range(4)  # no change
estimation = range(4, 14)  # change 12 to 14
control = range(14, 18)  # change 12 to 14, 16 to 18

# The complexity of wire management only gets more difficult as you start working with more
# and more registers. As you start building
# bigger and more complex algorithms, this can quickly become a serious issue!

######################################################################
# Conclusion
# ----------
#
# Wire registers provide a neat way of organizing and managing wires. Often, algorithms are
# described as acting upon registers and sub-registers; using wire registers can greatly
# streamline the implementation from theory to code. In this demo, we showed how to construct
# wire registers and use them to implement quantum algorithms.
#
######################################################################
# About the author
# ----------------
#
#
