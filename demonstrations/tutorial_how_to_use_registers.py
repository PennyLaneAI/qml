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
# The way to construct a wire register is simple --- just use :func:`~.pennylane.registers`. We need to
# pass a dictionary where the keys are the register names and the values are the number of
# wires in each register:
# 

import pennylane as qml

register = qml.registers({"alice": 1, "bob": 2, "charlie": 3})
print(register)

######################################################################
# The output is a dictionary where the keys are the names of the registers and the
# values are :class:`~Wires` instances.
#
# You can also pass in a dictionary that has nested dictionaries as its values.
# 

nested_register = qml.registers(
    {
        "all_registers": {
            "alice": 1,
            "bob": {"bob1": {"bob1a": 1}, "bob2": 1},
            "charlie": 1,
        }
    }
)
print(nested_register)

######################################################################
# Note that :func:`~.pennylane.registers` flattens any nested dictionaries, and the order of the
# elements is based on the order of appearance and nestedness. For more details on ordering, refer
# to the documentation for :func:`~pennylane.registers`.
#
# Accessing a particular register is the same as accessing any element in a dictionary.
#

print(nested_register["alice"])
print(nested_register["bob1a"])

######################################################################
# You can access a specific wire index via its index in a register.

print(nested_register["all_registers"][2])
print(nested_register["bob1a"][0])

######################################################################
# You can also combine registers using set operations. Here, we use the pipe operator ``|`` to
# perform the union operation on the ``alice`` register and the ``charlie`` register.

new_register = nested_register["alice"] | nested_register["charlie"]
print(new_register)

######################################################################
# For more details on what set operations are supported, refer to the documentation of
# :func:`~.pennylane.registers`.
######################################################################
# A simple example using wire registers
# -------------------------------------
#
# In this example, we demonstrate how one can implement the SWAP test with registers.
# The `SWAP test <https://en.wikipedia.org/wiki/Swap_test>`_
# is an algorithm that calculates the squared inner
# product of two input states. It requires one auxiliary qubit and takes two input states :math:`|\psi\rangle`
# and :math:`|\phi\rangle`. We can think of these components as three registers. Suppose states
# :math:`|\psi\rangle` and :math:`|\phi\rangle` are each represented with 3 wires. In PennyLane
# code, that would be:


swap_register = qml.registers({"auxiliary": 1, "psi": 3, "phi": 3})

dev = qml.device("default.qubit")

@qml.qnode(dev)
def swap_test():
    # Prepare phi and psi in some arbitrary state
    for state in ["phi", "psi"]:
        qml.BasisState([1, 1, 0], swap_register[state])

    qml.Hadamard(swap_register["auxiliary"])
    for i in range(len(swap_register["psi"])):
        # We can use the union operation to assemble our registers on the fly
        qml.CSWAP(
            swap_register["auxiliary"]
            | swap_register["psi"][i]
            | swap_register["phi"][i]
        )
    qml.Hadamard(swap_register["auxiliary"])
    return qml.expval(qml.Z(wires=swap_register["auxiliary"]))


print(swap_test())

######################################################################
# Advanced example
# --------------------
#
# Using registers can greatly streamline the process of modifying a workflow
# by simplifying wire management. In this example, we use :doc:`Quantum Phase Estimation (QPE) <tutorial_qpe>` to
# calculate the eigenvalues of a Hamiltonian.
# Generally, QPE is described as having two sets of registers. One register is known as the
# "estimation" (or measurement) register and the other is the state register where we apply our
# unitary operators :math:`U`. We can define these registers in PennyLane code:

register = qml.registers({"state": 4, "estimation": 8})

######################################################################
# To build our unitary operator :math:`U`, there are a variety of options. We can opt to use a
# straight-forward block encoding, or choose to use a subroutine like qubitization.
# Let's opt for :class:`~.pennylane.Qubitization`, which means we have to define another "prep" register.
# Our registers now look like this:

register = qml.registers({"state": 4, "estimation": 8, "prep": 4})

######################################################################
# Finally, let's define our Hamiltonian. We'll use the Transverse-Field Ising model from
# PennyLane's `quantum datasets <https://pennylane.ai/datasets/qspin/transverse-field-ising-model>`_,
# but feel free to try this with any other Hamiltonian you want to find the eigenvalues of.


[dataset] = qml.data.load(
    "qspin", sysname="Ising", periodicity="open", lattice="chain", layout="1x4"
)
H = dataset.hamiltonians[0]
print(H)

######################################################################
# For QPE to work, we need to initialize the "state" register with an initial state that has good
# overlap with the eigenstate we want the eigenvalue of.

initial_state = dataset.ground_states[0]

######################################################################
# With this, we can now define our QPE circuit like so:

dev = qml.device("lightning.qubit", wires=16)

@qml.qnode(dev)
def circuit():
    # Initialize state register to initial state
    qml.BasisState(initial_state, wires=register["state"])

    # Apply Hadamard gate to all wires in estimation register
    for wire in register["estimation"]:
        qml.Hadamard(wires=wire)

    qml.ControlledSequence(
        qml.Qubitization(H, register["prep"]),
        control=register["estimation"],
    )

    qml.adjoint(qml.QFT)(wires=register["estimation"])

    return qml.probs(wires=register["estimation"])

######################################################################
# Changing the number of wires in your estimation register is very easy with registers, but can
# be very error-prone when using wires. The complexity of wire management only gets more difficult
# as you start working with more and more registers. As you start building bigger and more complex
# algorithms, this can quickly become a serious issue!
#
# Conclusion
# -----------------------
# In this demo, we showed how to construct
# wire registers and use them to implement quantum algorithms.
# Wire registers provide a neat way of organizing and managing wires. Often, algorithms are
# described as acting upon registers and sub-registers; using wire registers can greatly
# streamline the implementation from theory to code. 

######################################################################
# About the author
# ----------------
#
# .. include:: ../_static/authors/austin_huang.txt
#
