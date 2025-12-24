r"""What are magic states?
==========================

Quantum computers rely on very fragile physical systems. They are easily disturbed, leading to errors 
being produced and propagated. In order to scale and harness the capabilities of quantum computing, 
we need to develop fault-tolerant arquitectures (FTQC). 
For this purpose, besides being able to detect and correct errors, we must be able to implement gates 
that can correctly perform their intended operations without creating or spreading existing errors 
across the encoded information.

To achieve universal quantum computing (UQC), we must be able to implement a universal gate set, such as
the :math:`\textrm{Clifford + T}` set, ``{H, S, CNOT, T}``. As previously stated, all these gates must be 
implemented in a fault-tolerant manner. 
For most quantum error correction architectures, gates of the Clifford group are much 
simpler to implement, often using transversal operations, than their non-Clifford 
counterparts. 

This means that when you implement a logical gate, the physical gates that compose it only affect one 
physical qubit at most such that the errors stay localized and do not propagate within each code block. 
Still could write something about NC gates not being applied transversal (Eastin-Knill) to reduce error 
propagation. 

This is where magic states appear: they provide a way to implement FT non-Clifford gates. The idea is to
effectively implement such a gate by consuming a special, pre-prepared quantum state, called "magic state", 
and teleporting it to the circuit. 

In this demo, we will explore what is so magic about magic states, how they are prepared through 
distillation and cultivation, and outline the current research and open problems that concern them.

Where is the magic?
-------------------

Let's beging with a little but of history. Bravyi and Kitaev formalized the idea and coined the term 
"magic states" in [#Bravyi]_. 
They proved that the capability of preparing magic states in combination with Clifford operations 
access a set ideal Clifford gates, the creation of ancillas :math:`0` and measurements capabilities 
in the Z-basis on all qubits are enough to enable UQC. 

In short, magic states are a class of states whose injection into a circuit results in the same 
output as implementing a determined non-Clifford gate. 
There are several kinds of magic states, unfortunately their nomenclature is not consistent across papers

Let's choose one and see it in action. We will use the magic state 
:math:`|H\rangle=\frac{1}{\sqrt{2}}(|0\rangle+e^{i\pi/4}|1\rangle)` and perform gate teleportation
(see the figure below) to perform the T operation to an arbitrary 
one-qubit state defined in code. The process can be found explicitly in this PennyLane
`glossary page <https://pennylane.ai/qml/glossary/what-are-magic-states>`__. 
"""

import pennylane as qml
from pennylane import numpy as np
from functools import partial


def prepare_magic_state():
    """Prepares the |H> magic state on wire 1."""
    qml.Hadamard(wires=1)
    qml.T(wires=1)


dev = qml.device("default.qubit")

@partial(qml.set_shots, shots=10000)
@qml.qnode(dev)
def t_gate_teleportation_circuit(target_state_params):

    # Prepare the initial target state (e.g., Ry rotation)
    qml.RY(target_state_params, wires=0)

    # Prepare the Magic State on Qubit 1
    prepare_magic_state()

    # Apply the Clifford operations for injection
    qml.CNOT(wires=[0, 1])

    m_1 = qml.measure(1)
    # The outcome of M_a dictates the final correction
    qml.cond(m_1 == 1, qml.S)(wires=0) 


    return qml.probs()

print(t_gate_teleportation_circuit(np.pi/3))

######################################################################
# We see from the output that ...
# 
# **Why "magic"?** In their paper [#Bravyi]_, Bravyi and Kitaev not only presented a way to achieve 
# UQC using magic states, but also proposed a method to distill them starting with imperfect copies
# of magic states and then on using only Clifford operations, which we will detail below. 
# These two properties are the reason why they called them "magic states".
#
# Preparing magic states 
# ----------------------
#
# It is exciting that magic states offer a workaround for complicated non-Clifford gate implementations. 
# However, they are still remain expensive to prepare. Let's see two examples of methods to prepare them.
#
# Magic state distillation
# ~~~~~~~~~~~~~~~~~~~~~~~~
#
# As the name suggests, distillation protocols rely on using multiple copies of noisy 
# magic states to "purify" them. By consuming these noisy inputs, the protocol
# produces fewer but higher-fidelity magic states. This cycle can be repeated to achieve an arbitrarily 
# low error rate. However, as you might suspect, there is a strict distillation threshold:
# the initial noisy states :math:`\rho` must have a fidelity above a certain limit for the process 
# to converge toward a pure state [#Bravyi]_. If the initial states are too noisy, the protocol will 
# fail to improve them.
#
# A typical protocol follows these steps:
# 1. Preparation of the initial imperfect states using non-Clifford gates (starting the "magic"). 
# 2. Several copies of these states are processed using Clifford operations to map them onto an 
# error-correcting code.
# 3. Perform measurements (akin to syndrome measurements) on the auxiliary qubits. These consist on
# performing the controlled `stabilizer <https://pennylane.ai/qml/demos/tutorial_stabilizer_codes>`__ generators. 
# 4. The measurement results indicate whether the state is still in the "clean" codespace. 
# If the syndrome is trivial, the remaining qubit is kept as a higher-purity state; if an error is 
# detected, the state is discarded.
#
# It is worth noticing that the main operations used in distillation are logical. This means that 
# they are executed across several error correcting codes resulting in a big overhead of resources. 
#
# See this `demo <https://pennylane.ai/qml/demos/tutorial_magic_state_distillation>`__ 
# for a practical implementation of a distillation protocol using Catalyst. 
#
# Magic state cultivation
# ~~~~~~~~~~~~~~~~~~~~~~~
#
# It is currently considered the state-of-the-art for preparing high-fidelity magic states. 
# Formulated by Gidney et al [#Gidney2024]_, 
# it as a way to improve magic state preparation from a practical perspective. 
# This means that the objective is to find an efficient way to prepare magic states with acceptable 
# fidelities suitable for relevant quantum computations. 
#
# Unlike distillation, which consumes many noisy states to "filter" out a clean one, cultivation 
# starts with a single seed state and improves it "in-place." The entire process occurs 
# within a single code patch using physical-level operations.
#
# The protocol consists of four primary stages:
#
# 1. Injection: prepare a magic state encoded in a small code of distance 3 or better. 
# 2. Cultivation: gradually improve the magic state by performing Clifford checks and postselection. 
# However, as the fault distance of the state increases, the code needs to be improved as well 
# so it can hold a more fault-tolerant state. For this reason, this stage also includes a "grow" 
# phase that increases the size of the code. Several cycles of check, grow, and stabilize take place.
# 3. Escape: after the cultivation stage, the magic state reaches its target fidelity,
# and becomes too good for the code that holds it. The cultivated state needs to **escape** into a 
# much larger code as fast as possible to ensure that such high fidelity is preserved and it
# can be used in a computation.
#
# Perspective on magic states: current research
# ---------------------------------------------
#
# References
# ----------
#
# .. [#bravyi2005]
#
#    Sergey Bravyi and Alexei Kitaev. "Universal quantum computation with
#    ideal Clifford gates and noisy ancillas." `Physical Review A 71.2 (2005).
#    <https://journals.aps.org/pra/abstract/10.1103/PhysRevA.71.022316>`__
#
# .. [#Gidney2024]
#
#    Craig Gidney, Noah Shutty, and Cody Jones. "Magic state cultivation: growing T states 
#    as cheap as CNOT gates."
#    `arXiv preprint arXiv:2409.17595 (2024) <https://arxiv.org/abs/2409.17595>`__.
#
