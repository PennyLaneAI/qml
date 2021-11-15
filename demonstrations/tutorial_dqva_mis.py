r""".. _trapped_ions:

Solving MIS with the DQVA Ansatz
================================

.. meta::
    :property="og:description": Description and assessment of trapped ion quantum computers
    :property="og:image": https://pennylane.ai/qml/_images/trapped_ions_tn.png

.. related::
   tutorial_pasqal Quantum computation with neutral atoms

*Author: PennyLane dev team. Posted: 10 November 2021. Last updated: 10 November 2021.*

DQVA ansatz intro

.. container:: alert alert-block alert-info
    
    Alert block

"""

##############################################################################
#
# QAOA
# ~~~~~~~~~~~~~~~~~~
#
# Something about QAOA
#
# .. figure:: ../demonstrations/trapped_ions/confining.png
#    :align: center
#    :width: 70%
#
#    ..
#
#    Confining potential
#
# What are the limitations for the general QAOA? 
#
# .. figure:: ../demonstrations/trapped_ions/saddle_potential.png
#    :align: center
#    :width: 70%
#
#    ..
#
#    Image caption
#



import pennylane as qml
from pennylane import qaoa
import numpy as np
from matplotlib import pyplot as plt
import networkx as nx

G = nx.Graph()
G.add_edges_from([(0, 1), (1, 2), (2, 0), (2, 3)])

nx.draw(graph, with_labels=True)
plt.show()

def hamming_weight(bitstr):
    return sum([1 for bit in bitstr if bit == '1'])
    
def mixer_layer(alpha, init_state, G, mixer_order):
    """
    Apply the mixer unitary U_M(alpha) to circ

    Input
    -----
    alpha : list[float]
        The angle values of the parametrized gates
    init_state : str
        The current initial state for the ansatz, bits which are "1" are hit
        with an X-gate at the beginning of the circuit and their partial mixers
        are turned off. Bitstring is little-endian ordered.
    G : NetworkX Graph
        The graph we want to solve MIS on
    mixer_order : list[int]
        The order that the partial mixers should be applied in. For a list
        such as [1,2,0,3] qubit 1's mixer is applied, then qubit 2's, and so on
    """

    # Apply partial mixers V_i(alpha_i)
    if mixer_order is None:
        mixer_order = list(G.nodes)
    print('Mixer order:', mixer_order)

    # Pad the given alpha parameters to account for the zeroed angles
    pad_alpha = [None]*len(init_state)
    next_alpha = 0
    for qubit in mixer_order:
        bit = list(reversed(init_state))[qubit]
        if bit == '1' or next_alpha >= len(alpha):
            continue
        else:
            pad_alpha[qubit] = alpha[next_alpha]
            next_alpha += 1

    print('init_state: {}, alpha: {}, pad_alpha: {}'.format(init_state,
                                                              alpha, pad_alpha))

    for qubit in mixer_order:
        if pad_alpha[qubit] == None or not G.has_node(qubit):
            # Turn off mixers for qubits which are already 1
            continue

        neighbors = list(G.neighbors(qubit))
        anc_idx = 4

        qml.MultiControlledX(control_wires= range(len(neighbors)),wires = len(neighbors), control_values ='0'*len(neighbors))

        # apply an X rotation controlled by the state of the ancilla qubit
        qml.CRX(2*pad_alpha[qubit], wires = [anc_idx, qubit])
        
        qml.MultiControlledX(control_wires= range(len(neighbors)),wires = len(neighbors), control_values ='0'*len(neighbors))

def cost_layer(gamma, G):
    """
    Apply a parameterized Z-rotation to every qubit
    """
    for qb in G.nodes:
        qml.RZ(2*gamma, wires=qb)

def dqva_layer(G, P=1, params=[], init_state=None, mixer_order=None):
    nq = len(G.nodes)
    print(nq)

    # Step 1: Jump Start
    # Run an efficient classical approximation algorithm to warm-start the optimization
    if init_state is None:
        init_state = '0'*nq

    # Step 2: Mixer Initialization
    for qb, bit in enumerate(reversed(init_state)):
        if bit == '1':
            qml.PauliX(wires = qb)
            
    # parse the variational parameters
    # The dqva ansatz dynamically turns off partial mixers for qubits in |1>
    # and adds extra mixers to the end of the circuit
    num_nonzero = nq - hamming_weight(init_state)
    assert (len(params) == (nq + 1) * P), "Incorrect number of parameters!"
    alpha_list = []
    gamma_list = []
    last_idx = 0
    for p in range(P):
        chunk = num_nonzero + 1
        cur_section = params[p*chunk:(p+1)*chunk]
        alpha_list.append(cur_section[:-1])
        gamma_list.append(cur_section[-1])
        last_idx = (p+1)*chunk

    # Add the leftover parameters as extra mixers
    alpha_list.append(params[last_idx:])

    for i in range(len(alpha_list)):
        print('alpha_{}: {}'.format(i, alpha_list[i]))
        if i < len(gamma_list):
            print('gamma_{}: {}'.format(i, gamma_list[i]))

    # Construct the dqva ansatz
    #for alphas, gamma in zip(alpha_list, gamma_list):
    for i in range(len(alpha_list)):
        alphas = alpha_list[i]
        mixer_layer(alphas, init_state, G,  mixer_order)


        if i < len(gamma_list):
            gamma = gamma_list[i]
            cost_layer(gamma, G)
##############################################################################
# Text in between code goes here

dev = qml.device("default.qubit", wires=1)


@qml.qnode(dev)
def ion_hadamard(state):

    if state == 1:
        qml.PauliX(wires=0)
    
    """We use a series of seemingly arbitrary pulses that will give the Hadamard gate.
    Why this is the case will become clear later"""

    qml.QubitUnitary(evolution(0, -np.pi / 2 / Omega), wires=0)
    qml.QubitUnitary(evolution(np.pi / 2, np.pi / 2 / Omega), wires=0)
    qml.QubitUnitary(evolution(0, np.pi / 2 / Omega), wires=0)
    qml.QubitUnitary(evolution(np.pi / 2, np.pi / 2 / Omega), wires=0)
    qml.QubitUnitary(evolution(0, np.pi / 2 / Omega), wires=0)

    return qml.state()

#For comparison, we use the Hadamard built into PennyLane
@qml.qnode(dev)
def hadamard(state):

    if state == 1:
        qml.PauliX(wires=0)

    qml.Hadamard(wires=0)

    return qml.state()

#We confirm that the values given by both functions are the same up to numerical error
print(np.isclose(1j * ion_hadamard(0), hadamard(0)))
print(np.isclose(1j * ion_hadamard(1), hadamard(1)))

##############################################################################
# Note that the desired gate was obtained up to a global phase factor.
# A similar exercise can be done for the :math:`T` gate:


@qml.qnode(dev)
def ion_Tgate(state):

    if state == 1:
        qml.PauliX(wires=0)

    qml.QubitUnitary(evolution(0, -np.pi / 2 / Omega), wires=0)
    qml.QubitUnitary(evolution(np.pi / 2, np.pi / 4 / Omega), wires=0)
    qml.QubitUnitary(evolution(0, np.pi / 2 / Omega), wires=0)

    return qml.state()


@qml.qnode(dev)
def tgate(state):

    if state == 1:
        qml.PauliX(wires=0)

    qml.T(wires=0)

    return qml.state()


print(np.isclose(np.exp(1j * np.pi / 8) * ion_Tgate(0), tgate(0)))
print(np.isclose(np.exp(1j * np.pi / 8) * ion_Tgate(1), tgate(1)))

##############################################################################
# Text

import matplotlib.pyplot as plt


@qml.qnode(dev)
def evolution_prob(t):

    qml.QubitUnitary(evolution(0, t / Omega), wires=0)

    return qml.probs(wires=0)


t = np.linspace(0, 4 * np.pi, 101)
s = [evolution_prob(i)[1].numpy() for i in t]

fig1, ax1 = plt.subplots(figsize=(9, 6))

ax1.plot(t, s, color="#9D2EC5")

ax1.set(
    xlabel="time (in units of 1/Ω)", 
    ylabel="Probability", 
    title="Probability of measuring the excited state"
)
ax1.grid()

plt.show()


#
# References
# ----------
#
# .. [#DiVincenzo2000]
#
#     D. DiVincenzo. (2000) "The Physical Implementation of Quantum Computation",
#     `Fortschritte der Physik 48 (9–11): 771–783
#     <https://onlinelibrary.wiley.com/doi/10.1002/1521-3978(200009)48:9/11%3C771::AID-PROP771%3E3.0.CO;2-E>`__.
#     (`arXiv <https://arxiv.org/abs/quant-ph/0002077>`__)
#
# .. [#Paul1953]
#
#     W. Paul, H. Steinwedel. (1953) "Ein neues Massenspektrometer ohne Magnetfeld",
#     RZeitschrift für Naturforschung A 8 (7): 448-450.
#
# .. [#CiracZoller]
#
#     J. Cirac, P. Zoller. (1995) "Quantum Computations with Cold Trapped Ions".
#     Physical Review Letters 74 (20): 4091–4094.
#
# .. [#Malinowski]
#
#     M. Malinowski. (2021) "Unitary and Dissipative Trapped-​Ion Entanglement Using
#     Integrated Optics". PhD Thesis retrieved from `ETH thesis repository
#     <https://ethz.ch/content/dam/ethz/special-interest/phys/quantum-electronics/tiqi-dam/documents/phd_theses/Thesis-Maciej-Malinowski>`__.
#
# .. [#NandC2000]
#
#     M. A. Nielsen, and I. L. Chuang (2000) "Quantum Computation and Quantum Information",
#     Cambridge University Press.
#
# .. [#Hughes2020]
#
#     A. Hughes, V. Schafer, K. Thirumalai, et al. (2020)
#     "Benchmarking a High-Fidelity Mixed-Species Entangling Gate"
#     `Phys. Rev. Lett. 125, 080504
#     <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.125.080504>`__.
#     (`arXiv <https://arxiv.org/abs/2004.08162>`__)
#
# .. [#Bergou2021]
#
#     J. Bergou, M. Hillery, and M. Saffman. (2021) "Quantum Information Processing",
#     Springer.
#
# .. [#Molmer1999]
#
#     A. Sørensen, K. Mølmer.  (1999) "Multi-particle entanglement of hot trapped ions",
#     `Physical Review Letters. 82 (9): 1835–1838
#     <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.82.1835>`__.
#     (`arXiv <https://arxiv.org/abs/quant-ph/9810040>`__)
#
# .. [#Brown2019]
#
#     M. Brown, M. Newman, and K. Brown. (2019)
#     "Handling leakage with subsystem codes",
#     `New J. Phys. 21 073055
#     <https://iopscience.iop.org/article/10.1088/1367-2630/ab3372>`__.
#     (`arXiv <https://arxiv.org/abs/1903.03937>`__)
#
# .. [#Monroe2014]
#
#     C. Monroe, R. Ruassendorf, A Ruthven, et al. (2019)
#     "Large scale modular quantum computer architecture with atomic memory and photonic interconnects",
#     `Phys. Rev. A 89 022317
#     <https://journals.aps.org/pra/abstract/10.1103/PhysRevA.89.022317>`__.
#     (`arXiv <https://arxiv.org/abs/1208.0391>`__)
#
# .. [#QCCD2002]
#
#     D. Kielpinski, C. Monroe, and D. Wineland. (2002)
#     "Architecture for a large-scale ion-trap quantum computer",
#     `Nature 417, 709–711 (2002).
#     <https://www.nature.com/articles/nature00784>`__.
#
# .. [#Amini2010]
#
#     J. Amini, H. Uys, J. Wesenberg, et al. (2010)
#     "Toward scalable ion traps for quantum information processing",
#     `New J. Phys 12 033031
#     <https://iopscience.iop.org/article/10.1088/1367-2630/12/3/033031/meta>`__.
#     (`arXiv <https://arxiv.org/abs/0909.2464>`__)
#
#
# .. [#Pino2021]
#
#     J. Pino, J. Dreiling, J, C, Figgatt, et al. (2021)
#     "Demonstration of the trapped-ion quantum CCD computer architecture".
#     `Nature 592, 209–213
#     <https://www.nature.com/articles/s41586-021-03318-4>`__.
#     (`arXiv <https://arxiv.org/abs/2003.01293>`__)
#
# .. [#Blumel2021]
#
#     R. Blumel, N. Grzesiak, N. Nguyen, et al. (2021)
#     "Efficient Stabilized Two-Qubit Gates on a Trapped-Ion Quantum Computer"
#     `Phys. Rev. Lett. 126, 220503
#     <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.126.220503>`__.
#     (`arXiv <https://arxiv.org/abs/2101.07887>`__)
#
# .. [#Niffenegger2020]
#
#     R. Niffenegger, J. Stuart, C.Sorace-Agaskar, et al. (2020)
#     "Integrated multi-wavelength control of an ion qubit"
#     `Nature volume 586, pages538–542
#     <https://www.nature.com/articles/s41586-020-2811-x>`__.
#     (`arXiv <https://arxiv.org/abs/2001.05052>`__)
