r""".. _trapped_ions:

Solving MIS with the DQVA Ansatz
================================

.. meta::
    :property="og:description": Description and assessment of trapped ion quantum computers
    :property="og:image": https://pennylane.ai/qml/_images/trapped_ions_tn.png

.. related::
   tutorial_pasqal Quantum computation with neutral atoms

*Author: *

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
#    Image caption
#
# What are the limitations for the general QAOA? 
#
#



import pennylane as qml
from pennylane import qaoa
import numpy as np
from matplotlib import pyplot as plt
import networkx as nx

G = nx.Graph()
G.add_edges_from([(0, 1), (1, 2), (2, 0), (2, 3)])

nx.draw(G, with_labels=True)
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



#
# References
# ----------
#
# .. [#Saleem2020]
#
#     Z. H. Saleem, T. Tomesh, B. Tariq, M. Suchara. (2000) "Approaches to Constrained Quantum Approximate Optimization",
#     `arXiv preprint arXiv:2010.06660 <https://arxiv.org/abs/2010.06660>`__.
#
