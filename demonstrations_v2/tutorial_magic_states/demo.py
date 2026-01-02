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

This is where magic states enter the picture: they provide a mechanism to build non-Clifford gates in a 
fault-tolerant manner. The idea is to effectively apply such a gate by consuming a special, pre-prepared 
quantum state, called magic state, and teleporting its logical action into the circuit.

In this demo, we will explore the unique properties of magic states, how they are prepared through 
distillation and cultivation, and outline the current research challenges and open problems in the field.

Where is the magic?
-------------------

Let's beging with a little bit of history. Bravyi and Kitaev formalized the concept and coined the
term "magic states" in [#Bravyi2005]_. 
In this work, they proved that the ability to prepare magic states, when combined with a set of ideal Clifford 
gates, the preparation of :math:`|0\rangle` ancillas, and Z-basis measurement capabilities 
on all qubits, is sufficient to enable UQC. Essentially, magic states are a class of states 
that when injected into a circuit implement a specific non-Clifford gate. 

There are different types of magic states and their nomenclature often varies across the literature. 
Let’s examine a specific state, which we will denote as :math:`|H\rangle`, to see it in action:

.. math:: |H\rangle=\frac{1}{\sqrt{2}}(|0\rangle+e^{i\pi/4}|1\rangle)=T|+\rangle.

Notice that this state is obtained by applying a T gate to the :math:`|+\rangle` state 
(the +1 eigenstate of the Pauli X operator).  
Using **magic state injection** (see the circuit illustration below), we can apply a T operation to an 
arbitrary single-qubit state (wire 0 in the code). A step-by-step breakdown of this process 
can be found in this PennyLane `glossary page <https://pennylane.ai/qml/glossary/what-are-magic-states>`__.

"""

import pennylane as qml
from pennylane import numpy as np


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

    # The outcome of m_1 dictates the final correction
    m_1 = qml.measure(1)
    qml.cond(m_1 == 1, qml.S)(wires=0)

    return qml.density_matrix(wires=[0])


print(qml.draw(t_gate_teleportation_circuit)(np.pi / 3))
print(t_gate_teleportation_circuit(np.pi / 3))

######################################################################
# The output displays the density matrix of the target state after the T gate has been applied via
# teleportation:
# 
# .. math:: T|\psi\rangle=T (R_y(\pi/3)|0\rangle)=\frac{\sqrt{3}}{2}|0\rangle+\frac{1}{2}e^{i\pi/4}|1\rangle.
#
# This confirms the intended effect of applying a non-Clifford rotation by consuming the magic state :math:`|H\rangle`.
# 
# **Why "magic"?** In their influential paper [#Bravyi2005]_, Bravyi and Kitaev not only presented a path to  
# UQC via magic states but also proposed a method to prepare them starting from imperfect copies
# of magic states. This purification process relies solely on Clifford operations. 
# The synergy of these two properties---enabling universality and being distillable through restricted 
# (Clifford) operations---is precisely why they were named "magic states".
#
# Preparing magic states 
# ----------------------
#
# While magic states offer an elegant workaround for complex non-Clifford gate implementations, 
# they remain computationally expensive to prepare. Let's examine two methods for their preparation.
#
# Magic state distillation
# ~~~~~~~~~~~~~~~~~~~~~~~~
#
# As the name suggests, distillation protocols rely on using multiple copies of noisy 
# magic states to purify them. By consuming these noisy inputs, the protocol
# produces a smaller number of higher-fidelity magic states. This cycle can be repeated to achieve 
# an arbitrarily low error rate. 
#
# However, as one might suspect, there is a strict distillation threshold:
# the initial noisy states :math:`\rho` must have a fidelity above a certain limit for the process 
# to converge toward a pure state [#Bravyi2005]_. If the initial states are too noisy, the protocol will 
# fail to improve them.
#
# A typical protocol follows these steps:
#
# 1. Prepare initial, imperfect states using non-Clifford gates (starting the "magic").
#    These input states are encoded in error-correcting (inner) codes such as surface codes. 
# 2. Process several copies of these logical qubits using Clifford operations to map them onto an 
#    error-correcting (outer) code such as the Reed-Muller code.
# 3. Perform a syndrome measurement by measuring certain `stabilizers <https://pennylane.ai/qml/demos/tutorial_stabilizer_codes>`__
#    across this multi-block structure.
# 4. Decide whether to keep or discard the state by looking at the measurement results that indicate 
#    if the state remains in the "clean" codespace.
#    If the syndrome is trivial, the reduced state is kept as a higher-purity state; if an error is 
#    detected, the state is discarded.
#
# It is worth noting the underlying structure of code concatenation; there is an outer code and 
# several smaller inner codes. As such, the main operations in a distillation protocol are 
# logical, being executed across multiple error-correcting blocks, resulting in a significant resource overhead. 
#
# See this `demo <https://pennylane.ai/qml/demos/tutorial_magic_state_distillation>`__ 
# for an implementation of a distillation protocol using Catalyst. 
#
# Magic state cultivation
# ~~~~~~~~~~~~~~~~~~~~~~~
#
# Currently considered the state-of-the-art method for generating high-fidelity magic states, 
# formulated by Gidney et al [#Gidney2024]_ as a practical optimization. 
# The primary objective is to synthesize magic states with the specific fidelities required 
# for large-scale quantum computations as efficiently as possible.
#
# Unlike distillation, which consumes numerous noisy states to filter out a clean one, cultivation 
# starts with a single *seed* state and improves it "in-place." The entire process occurs 
# within a single code patch using physical-level operations.
#
# The protocol consists of four primary stages:
#
# 1. Injection: prepare an initial, noisy magic state encoded in a small code of distance 3 or better.
# 2. Cultivation: gradually improve the state through repeated Clifford checks and postselection.
#    To reach higher fidelities, the code distance must be increased; otherwise, the noise floor 
#    of the small code would limit the state's purity. This stage then involves cycles of growing
#    the code, stabilizing it, and checking the magic state. 
# 3. Escape: rapidly expand the code hosting the state. Once the cultivation stage is complete, the magic state 
#    reaches its target fidelity, and becomes "too good for the code". 
#    To preserve this high fidelity, the state needs to *escape*
#    into a much larger code as quickly as possible, typically via `grafting <https://arxiv.org/abs/2409.17595>`__
#    `code morphing <https://arxiv.org/abs/2112.01446>`__ or 
#    `lattice surgery <https://pennylane.ai/qml/demos/tutorial_lattice_surgery>`__.
# 4. Decoding: determine whether to accept the final state using standard error correction. Since the circuit 
#    is now too large for efficient post-selection, a decoder computes a complementary gap. This metric acts as 
#    a confidence score for the final state, allowing the system to accept or discard the state accordingly.
#     
#    
# Current research
# ----------------
#
# Despite the theoretical foundations of magic states being well-established, translating them into 
# scalable hardware remains a challenge. Current research in the field focuses on developing more 
# efficient protocols for generating high-fidelity magic states and achieving robust experimental 
# implementations across various qubit architectures. 
#
# A distillation protocol was recently implemented at the logical level on a neutral-atom quantum 
# computer [#QuEra2025]_, leveraging the parallel control and high connectivity offered by atom platforms. 
# Additionally, a pathway for optimizing these methods involves performing distillation, or at least 
# portions of it, at the physical level [#Ruiz2025]_, [#Itogawa2025]_. In [#Ruiz2025]_, for example, 
# the authors exploit the biased-noise properties of cat qubits to "unfold" a 3D outer code 
# into a 2D layout, improving both the time and qubit overheads. 
#
# Parallel to these efforts, the first experimental verification and characterization of a cultivation 
# protocol was recently achieved on a superconducting-qubit architecture [#Google2025]_. 
# However, this is only the beginning; multiple proposals are currently being developed to further 
# improve magic state cultivation (MSC). For instance, addressing the complexity
# of the Escape stage by introducing easier-to-decode expansion methods [#Chen2025]_, and 
# providing alternatives to the "grafting" method, used in the original MSC proposal, such as 
# lattice surgery [#Hirano2025]_.  
#
# Conclusion
# ----------
#
# In this tutorial, we reviewed the concept of magic states, an essential and creative tool for achieving FTQC and UQC.
# While traditional distillation provided the theoretical foundation for universality, the shift 
# toward cultivation reflects a new era of practical fault-tolerance focused on resource efficiency.
# As we move toward large-scale implementations, the challenge lies in optimizing these "magic state factories."
# 
# For those interested in the deeper math, there is a nice geometric perspective of magic states 
# involving the stabilizer polytope on the Bloch sphere [#Heinrich2019]_. In this framework, magic states 
# emerge as the maximal non-stabilizer states, lying as far as possible from the
# classically-simulatable region defined by the Clifford group.
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
# .. [#Heinrich2019]
#
#    Markus Heinrich and David Gross. "Robustness of Magic and Symmetries of the Stabiliser 
#    Polytope."
#    `Quantum 3, 132 (2019) <https://quantum-journal.org/papers/q-2019-04-08-132>`__.
#
# .. [#QuEra2025]
#
#    Pedro Sales Rodriguez, et al. "Experimental demonstration of logical magic state distillation."
#    `Nature 645, 620–625 (2025) <https://www.nature.com/articles/s41586-025-09367-3>`__.
#
# .. [#Ruiz2025]
#
#    Diego Ruiz, Jérémie Guillaud, Christophe Vuillot, and Mazyar Mirrahimi. 
#    "Unfolded distillation: very low-cost magic state preparation for biased-noise qubits."
#    `arXiv preprint arXiv:2507.12511 <https://arxiv.org/abs/2507.12511>`__.
#
# .. [#Itogawa2025]
#
#    Tomohiro Itogawa, Yugo Takada, Yutaka Hirano, and Keisuke Fujii. 
#    "Efficient magic state distillation by zero-Level distillation."
#    `PRX Quantum 6, 020356 (2025) <https://journals.aps.org/prxquantum/abstract/10.1103/thxx-njr6>`__.
#
# .. [#Chen2025]
#
#    Zi-Han Chen, Ming-Cheng Chen, Chao-Yang Lu, and Jian-Wei Pan. 
#    "Efficient Magic State Cultivation on ℝℙ2."
#    `arXiv preprint arXiv:2503.18657 <https://arxiv.org/abs/2503.18657>`__.
#
# .. [#Hirano2025]
#
#    Yutaka Hirano, Riki Toshio, Tomohiro Itogawa, and Keisuke Fujii.
#    "Efficient magic state cultivation with lattice surgery."
#    `arXiv preprint arXiv:2510.24615 <https://arxiv.org/abs/2510.24615>`__.
#
# .. [#Google2025]
#
#    Emma Rosenfeld, et al.
#    "Magic state cultivation on a superconducting quantum processor."
#    `arXiv preprint arXiv:2512.13908 <https://arxiv.org/abs/2512.13908>`__.
#
