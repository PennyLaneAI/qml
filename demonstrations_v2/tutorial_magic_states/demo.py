r"""What are magic states?
==========================

Quantum computers rely on very fragile physical systems. They are easily disturbed, leading to 
the rapid generation and propagation of errors. In order to scale and harness the full potential 
of quantum computing, we must develop fault-tolerant architectures (FTQC). For this purpose, besides 
simply detecting and correcting errors, we must be able to realize gates that correctly perform their 
intended operations without introducing or spreading noise across the encoded information.

To achieve universal quantum computing (UQC), we need to have access to a universal gate set, such as
the :math:`\textrm{Clifford + T}` set, ``{H, S, CNOT, T}``. As previously stated, all these gates must be 
executed in a fault-tolerant manner. 
For most quantum error correction architectures, gates of the Clifford group are much 
simpler to implement, often using `transversal operations <https://arthurpesah.me/blog/2023-12-25-transversal-gates/>`__, 
than their non-Clifford counterparts. 

This is where magic states become essential: they provide a mechanism to build non-Clifford gates in a 
fault-tolerant manner. The idea is to effectively apply such a gate by consuming a special, pre-prepared 
quantum state, called "magic state", and teleporting its logical action into the circuit.

In this demo, we will explore the unique properties of magic states, how they are prepared through 
distillation and cultivation, and outline the current research challenges and open problems in the field.

Where is the magic?
-------------------

Let's beging with a little bit of history. Bravyi and Kitaev formalized the concept and coined the
term "magic states" in [#Bravyi2005]_. 
In this work, they proved that the capability to prepare magic states, when combined with a set of ideal Clifford 
gates, the preparation of :math:`0` ancillas, and Z-basis measurements capabilities 
in the on all qubits, is sufficient to enable UQC. Essentially, magic states are a class of states 
that when injected into a circuit implement a specific non-Clifford gate. 

There are different types of magic states and their nomenclature often varies across the literature. 
Let’s examine a specific state, which we will denote as :math:`|H\rangle`, to see it in action:

.. math:: |H\rangle=\frac{1}{\sqrt{2}}(|0\rangle+e^{i\pi/4}|1\rangle)=T|+\rangle.

Notice that this state is obtained by applying a T gate to the :math:`|+\rangle` state 
(the +1 eigenstate of the Pauli X operator).  
Using magic state injection (see the circuit illustration),  we can apply a T operation to an 
arbitrary single-qubit state (wire 0 in the code). A detailed step-by-step breakdown of this process 
can be found in this PennyLane `glossary page <https://pennylane.ai/qml/glossary/what-are-magic-states>`__.

"""

import pennylane as qml
from pennylane import numpy as np
from functools import partial


def prepare_magic_state():
    """Prepares the |H> magic state on wire 1."""
    qml.Hadamard(wires=1)
    qml.T(wires=1)


dev = qml.device("default.qubit")


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

    return qml.density_matrix(wires=[0])


print(qml.draw(t_gate_teleportation_circuit)(np.pi / 3))
print(t_gate_teleportation_circuit(np.pi / 3))

######################################################################
# We see from the output that ...
# 
# **Why "magic"?** In their paper [#Bravyi2005]_, Bravyi and Kitaev not only presented a way to achieve 
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
# to converge toward a pure state [#Bravyi2005]_. If the initial states are too noisy, the protocol will 
# fail to improve them.
#
# A typical protocol follows these steps:
#
# 1. Preparation of the initial imperfect states using non-Clifford gates (starting the "magic").
# 2. Several copies of these states are processed using Clifford operations to map them onto an 
#    error-correcting code.
# 3. Perform measurements (akin to syndrome measurements) on the auxiliary qubits. These consist on
#    performing the controlled `stabilizer <https://pennylane.ai/qml/demos/tutorial_stabilizer_codes>`__ generators.
# 4. The measurement results indicate whether the state is still in the "clean" codespace.
#    If the syndrome is trivial, the remaining qubit is kept as a higher-purity state; if an error is 
#    detected, the state is discarded.
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
# 1. Injection: prepare an initial, noisy magic state encoded in a small code of distance 3 or better.
# 2. Cultivation: gradually improve the magic state through repeated Clifford checks and postselection.
#    To reach higher fidelities, the code distance must be increased; otherwise, the "noise floor" 
#    of the small code would limit the state's purity. This stage then includes cycles of growing
#    the code, stabilizing it, and checking the magic state. 
# 3. Escape: rapidly expand the code hosting the state. After the cultivation stage, the magic state 
#    reaches its target fidelity, and becomes "too good for the code". 
#    To preserve such high fidelity, the state needs to **escape** 
#    into a much larger code as fast as possible. This can be achieved via code-morphing or lattice surgery.
# 4. Decoding: decide whether to accept the final state using standard error correction. Post-selection is 
#    no longer used since the circuit is too large. The decoder computes a threshold called complementary gap
#    that acts as a confidence score of the final state and accept or discard the state accordingly. 
#     
#    
# Perspective on magic states: current research
# ---------------------------------------------
#
# References
# ----------
#
# .. [#Bravyi2005]
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
