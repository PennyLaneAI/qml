r"""

Building a quantum lock using phase kickback
============================================

.. meta::
    :property="og:description": Use phase kickback to create an unbreakable quantum lock
    :property="og:image": https://pennylane.ai/qml/_images/thumbnail_tutorial_phase_kickback.png

.. related::
   tutorial_qubit_rotation Basic tutorial: qubit rotation

Greetings, quantum adventurers! In this exciting tutorial, we'll explore the quantum phase kickback concept, used in many quantum algorithms such as the Deutsch–Jozsa algorithm and quantum phase estimation. Here, we'll utilize it to create a *quantum lock*. Are you ready to dive into the quantum world and learn how to create an unbreakable lock? Let's go!
""" 

######################################################################
# Introduction to phase kickback
# ------------------------------
#
# Phase kickback is a powerful quantum phenomenon that uses entanglement properties to transfer phase information from a target register to a control qubit. It plays a vital role in the design of many quantum algorithms.
# In a phase kickback circuit, an ancilla qubit is prepared in a superposition state using a Hadamard gate, and it acts as a control qubit for a controlled unitary gate applied to the target register. When the target register is in an eigenstate of the unitary gate, the corresponding eigenvalue's phase is *kicked back* to the control qubit. A subsequent Hadamard gate on the ancilla qubit enables phase information extraction through measurement.
#

##############################################################################
# .. figure:: ../demonstrations/phase_kickback/Phase_Kickback.png
#    :align: center
#    :width: 50%
#
# If you want to know more about the details, do not hesitate to consult the node `[P.1] <https://codebook.xanadu.ai/P.1>`_ of the Xanadu Quantum Codebook.

######################################################################
# Setting up PennyLane
# --------------------
#
# First, let's import the necessary PennyLane libraries and create a device to run our quantum circuits. Here we will work with five qubits. We will use qubit `[0]` as the control ancilla qubit, and qubits `[1,2,3,4]` will be our target qubits where we will encode :math:`|\psi\rangle`.
#

import pennylane as qml
from pennylane import numpy as np

num_wires = 5
dev = qml.device("default.qubit", wires=num_wires, shots=1)

######################################################################
# Building the quantum lock
# -------------------------
#
# Now let's create the most formidable lock in the universe: the *quantum lock*! Here our lock is represented by a unitary U, which has all but one eigenvalue equal to 1. Our one *key* eigenstate has eigenvalue -1:
#
# .. math:: U|\text{key}\rangle = -|\text{key}\rangle
#
# But how can we differentiate the *key* eigenstate from the other eigenstate when the information is contained in the phase? That's where phase kickback comes in! When the correct eigenstate is input, the -1 phase imparted by U is kicked back to the ancilla, effectively changing its state from :math:`|+\rangle` to :math:`|-\rangle`.
# Then the measurement outcome on the control qubit tells us whether the correct eigenstate was inputted. In this case, :math:`|1\rangle = H|-\rangle` represents unlocking the lock, and :math:`|0\rangle = H|+\rangle` represents failure.
# To simplify things, we'll work with a lock-in computational basis. In this setting, the key corresponds to a binary encoded integer :math:`m`, which will be our key eigenstate:
#
# .. math::
#
#
#      U|n\rangle =
#      \begin{cases}
#        -|n\rangle, & \text{if } n=m \\
#        |n\rangle, & \text{if } n\neq m
#      \end{cases}
#
# We’ll make use of :class:`~.pennylane.FlipSign` to build our lock:
#


def quantum_lock(secret_key):
    return qml.FlipSign(secret_key, wires=list(range(1, num_wires)))


######################################################################
# Next, we must prepare the corresponding eigenstate for a key we want to try. Remember, only the *key* eigenstate with eigenvalue -1 unlocks the lock. We’ll make use of
# :class:`~.pennylane.BasisState` to build the key:
#


def build_key(key):
    return qml.BasisState(key, wires=list(range(1, num_wires)))


######################################################################
# Now we'll put it all together to build our quantum locking mechanism:
#


@qml.qnode(dev)
def quantum_locking_mechanism(lock, key):
    build_key(key)
    qml.Hadamard(wires=0)  # Hadamard on ancilla qubit
    qml.ctrl(lock, control=0)  # Controlled unitary operation
    qml.Hadamard(wires=0)  # Hadamard again on ancilla qubit
    return qml.sample(wires=0)


def check_key(lock, key):
    if quantum_locking_mechanism(lock, key) == 1:
        print("Great job, you have uncovered the mysteries of the quantum universe!")
    else:
        print("Nice try, but that's not the right key!")


######################################################################
# Opening the Quantum Lock
# ------------------------
# We'll need the correct input state or *quantum key* to open the *quantum lock*. Let's see how the quantum system evolves when we input the right key.
# We first apply a Hadamard to our control qubit:
#
# .. math:: \frac{|0\rangle|\text{key}\rangle + |1\rangle|\text{key}\rangle}{\sqrt{2}}
#
# By applying the controlled unitary operation, we get the following:
#
# .. math:: \frac{|0\rangle|\text{key}\rangle - |1\rangle|\text{key}\rangle}{\sqrt{2}} = |-\rangle|\text{key}\rangle
#
# Finally, we apply a Hadamard to our control qubit again to get:
#
# .. math:: |1\rangle|\text{key}\rangle
#
# And just like that, we've uncovered the quantum secrets hidden by the lock. Let's now crack open our quantum lock-in code!
#

secret_key = np.array([0, 1, 1, 1])
lock = quantum_lock(secret_key)

check_key(lock, secret_key)

######################################################################
# What happens with an incorrect quantum key?
# -------------------------------------------
#
# Now, we'll try using the wrong key and see if we can still unlock the quantum lock. Will we be able to break through its quantum defenses? Let's see how the quantum system evolves when we input the wrong key.
# We first apply a Hadamard to our control qubit:
#
# .. math:: \frac{|0\rangle|\text{incorrect key}\rangle + |1\rangle|\text{incorrect key}\rangle}{\sqrt{2}}
#
# Applying the controlled unitary operation, in this case, acts as the identity gate. Hence we get the following:
#
# .. math:: \frac{|0\rangle|\text{incorrect key}\rangle + |1\rangle|\text{incorrect key}\rangle}{\sqrt{2}} = |+\rangle|\text{incorrect key}\rangle
#
# Finally, we apply a Hadamard to our control qubit again to get:
#
# .. math:: |0\rangle|\text{incorrect key}\rangle
#
# As you can see, we were unable to fool the almighty lock. Don't believe me? See for yourself!
#

incorrect_key = np.array([1, 1, 1, 1])

check_key(lock, incorrect_key)

######################################################################
# Conclusion
# ----------
#
# Congratulations! 🎉 You've successfully explored the remarkable phenomenon of phase kickback and created an unbreakable *quantum lock*. You can impress your friends with your newfound quantum knowledge and incredible quantum lock-picking skills!
#
#
# About the author
# ----------------
# .. include:: ../_static/authors/danial_motlagh.txt
