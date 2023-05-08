r"""Phase Kickback: Building a Quantum Lock! 🔒
==============================================

*Author: Danial Motlagh.*

Greetings, quantum adventurers! In this exciting tutorial we’ll be exploring the concept of quantum
phase kickback and utilizing it to create a “quantum lock”. Are you ready to dive into the quantum
world and learn how to create an unbreakable lock? Let’s go!
"""

######################################################################
# 🧪 Setting up PennyLane
# -----------------------
#
# First, let’s import the necessary PennyLane libraries and create a device to run our quantum
# circuits.
#

import pennylane as qml
from pennylane import numpy as np

num_wires = 5
dev = qml.device("default.qubit", wires=num_wires, shots=1)

######################################################################
# 🎓 Introduction to Phase Kickback
# ---------------------------------
#
# Phase kickback is a powerful quantum phenomenon that allows the transfer of phase information from a
# target register to a control qubit through entanglement. It plays a vital role in many quantum
# algorithms, including Deutsch’s algorithm, the Deutsch-Jozsa algorithm, and Quantum Phase
# Estimation.
#
# In a phase kickback circuit, an ancilla qubit is prepared in a superposition state using a Hadamard
# gate and acts as a control qubit for a controlled unitary gate applied to the target register. When
# the target register is in an eigenstate of the unitary gate, the corresponding eigenvalue’s phase is
# “kicked back” to the control qubit. A subsequent Hadamard gate on the ancilla qubit enables the
# extraction of the phase information through measurement.
#

##############################################################################
# .. figure:: ../demonstrations/phase_kickback/Phase_Kickback.png
#    :align: center
#    :width: 50%

######################################################################
# 🔨 Building the Quantum Lock
# ----------------------------
#
# Now let’s create the most formidable lock in the universe: the “quantum lock”! Here our lock is
# represented by a unitary :math:`U`, whose eigenvalues are all 1 except for our “key” eigenstate
# which has eigenvalue -1:
#
# .. math:: U|\text{key}\rangle = -|\text{key}\rangle
#
# The outcome of the measurement on the control qubit tells us whether we were able to successfully
# unlock the quantum lock with 1 representing unlocking the lock and 0 representing a failure. To make
# things simple, here we’ll work with a lock in the computational basis. In this setting, the key
# corresponds to a binary encoded integer :math:`m` which will be our key eigenstate:
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
# We’ll make use of ``qml.FlipSign`` to build our lock:
#


def quantum_lock(secret_key):
    return qml.FlipSign(secret_key, wires=list(range(1, num_wires)))


######################################################################
# Next, we need to prepare the corresponding input state given a trial key. We’ll make use of
# ``qml.BasisState`` to do this:
#


def build_key(key):
    return qml.BasisState(key, wires=list(range(1, num_wires)))


######################################################################
# Now we’ll put it all together to build our quantum locking mechanism:
#


@qml.qnode(dev)
def quantum_locking_mechanism(lock, key):
    build_key(key)
    qml.Hadamard(wires=0)  # Hadamard on ancilla qubit
    qml.ctrl(lock, 0)  # Controlled unitary operation
    qml.Hadamard(wires=0)  # Hadamard again on ancilla qubit
    return qml.sample(wires=0)


def check_key(lock, key):
    if quantum_locking_mechanism(lock, key):
        print("Great job, you have uncovered the mysteries of the quantum universe!")
    else:
        print("Nice try, but that's not the right key!")


######################################################################
# 🔑 Opening the Quantum Lock
# ---------------------------
#
# To open the quantum lock, we’ll need the correct input state or “quantum key”. Let’s see how the
# quantum system evolves when we input the right key.
#
# We first apply a Hadamard to our control qubit:
#
# .. math:: \rightarrow \frac{|0\rangle|\text{key}\rangle + |1\rangle|\text{key}\rangle}{\sqrt{2}}
#
# Applying the controlled unitay operation we get:
#
# .. math:: \rightarrow \frac{|0\rangle|\text{key}\rangle - |1\rangle|\text{key}\rangle}{\sqrt{2}} = |-\rangle|\text{key}\rangle
#
# Finally we apply a Hadamard to our control qubit again to get:
#
# .. math:: \rightarrow |1\rangle|\text{key}\rangle
#
# And just like that, we’ve uncovered the quantum secrets hidden by the lock. Let’s now crack open our
# quantum lock in code!
#

secret_key = np.array([0, 1, 1, 1])
lock = quantum_lock(secret_key)

check_key(lock, secret_key)

######################################################################
# 🕵️‍♂️ What Happens with an Incorrect Quantum Key?
# ----------------------------------------------
#
# Now, we’ll try using the wrong key and see if we can still unlock the quantum lock. Will we be able
# to break through its quantum defenses? Let’s see how the quantum system evolves when we input the
# wrong key.
#
# We first apply a Hadamard to our control qubit:
#
# .. math:: \rightarrow \frac{|0\rangle|\text{incorrect key}\rangle + |1\rangle|\text{incorrect key}\rangle}{\sqrt{2}}
#
# Applying the controlled unitay operation in this case acts as the identity gate, hence we get:
#
# .. math:: \rightarrow \frac{|0\rangle|\text{incorrect key}\rangle + |1\rangle|\text{incorrect key}\rangle}{\sqrt{2}} = |+\rangle|\text{incorrect key}\rangle
#
# Finally we apply a Hadamard to our control qubit again to get:
#
# .. math:: \rightarrow |0\rangle|\text{incorrect key}\rangle
#
# As you can see, we were unable to fool the almighty lock, don’t believe me? See for youself!
#

incorrect_key = np.array([1, 1, 1, 1])

check_key(lock, incorrect_key)

######################################################################
# 🎉 Congratulations!
# -------------------
#
# You’ve successfully explored the remarkable phenomenon of phase kickback and created an unbreakable
# “quantum lock”. Now you can impress your friends with your newfound quantum knowledge and your
# incredible quantum lock-picking skills!
#
