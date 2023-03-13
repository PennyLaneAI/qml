"""
An equivariant graph embedding
==============================

.. meta::
    :property="og:description": Find out more about how to embedd graphs into quantum states.
    :property="og:image": https://pennylane.ai/qml/_images/thumbnail_tutorial_quivariant_graph_embedding.jpg

.. related::
   tutorial_geometric_qml Geometric quantum machine learning


*Author: Maria Schuld — Posted: DAY MONTH 2023.*
"""
######################################################################
# A notorious problem when data comes in the form of graphs -- think of molecules or social media
# networks -- is that the numerical representation of a graph in a computer is not unique. 
# For example, if we describe a graph via an [adjacency matrix](https://en.wikipedia.org/wiki/Adjacency_matrix), 
# any permutation of this matrix represents the same graph. 
# 
# .. figure:: ../demonstrations/equivariant_graph_embedding/adjacency-matrices.png
#    :width: 70%
#    :align: center
#    :alt: adjacency-matrices
#    
# If we want to do machine learning with graph data, we usually want our models to "know" that all
# permuted adjacency matrices refer to the same object, so we do not waste resources on learning 
# this property. In mathematical terms, this means that the model should be in- or 
# equivariant (more about this distinction below) with respect to permutations. 
# This is the basic idea of _Geometric Deep Learning_, which has found its way into 
# quantum machine learning. 
# 
# This tutorial shows how to implement an example of a permutation equivariant graph embedding 
# into quantum states as proposed in [Skolik et al. (2022)](https://arxiv.org/pdf/2205.06109.pdf). 
# It will not go into much of the mathematical formalism, which is covered in the PennyLane 
# demo on [geometric QML](https://pennylane.ai/qml/demos/tutorial_geometric_qml.html).
# 
# Permuted adjacency matrices describe the same graph
# ---------------------------------------------------
# 
# Let us first verify that permuted adjacency matrices really describe one and the same graph. 
# We also gain some useful data generation functions for later.
#
# Let's create a random adjacency matrix for an undirected graph. 
# The entry $a_{ij}$ of this matrix corresponds to the weight of the edge between nodes 
# $i$ and $j$ in the graph. We assume that graphs have no self-loops; instead, 
# the diagonal elements of the adjacency matrix are interpreted as node attributes. 
# 
# Taking the example of a twitter user retweet network, the nodes would be users, 
# edge weights indicate how often two users retweet each other and node attributes 
# could indicate the follower count of a user.
#

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def create_data(n):
    """
    Returns a random undirected adjacency matrix of dimension (n,n). 
    The diagonal elements are interpreted as node attributes.
    """
    mat = np.random.rand(n, n)
    A = (mat + np.transpose(mat))/2    
    return np.round(A, decimals=2)

A = create_data(3)
print(A)

######################################################################
# Let's also write a function to generate permuted versions of this adjacency matrix.
#

def permute(A, permutation):
    """
    Returns a copy of A with rows and columns swapped according to permutation. 
    For example, the permutation [1, 2, 0] swaps 0->1, 1->2, 2->0.
    """
    
    # construct permutation matrix
    P = np.zeros((len(A), len(A)))
    for i,j in zip(range(len(permutation)),permutation):
        P[i,j] = 1

    return P @ A @ np.transpose(P)

A_perm = permute(A, [1, 2, 0])
print(A_perm)

######################################################################
# If we create `networkx` graphs from both adjacency matrices and plot them, 
# we see that they are identical as claimed.
#

# TODO: NODE WEIGHTS

fig, (ax1, ax2) = plt.subplots(1, 2)

G1 = nx.Graph(A)
pos1=nx.spring_layout(G1)
nx.draw(G1, pos1, ax=ax1)
labels = nx.get_edge_attributes(G1,'weight')
nx.draw_networkx_edge_labels(G1,pos1,edge_labels=labels, ax=ax1)

G2 = nx.Graph(A_perm)
pos2=nx.spring_layout(G2)
nx.draw(G2, pos2, ax=ax2)
labels = nx.get_edge_attributes(G2,'weight')
nx.draw_networkx_edge_labels(G2,pos2,edge_labels=labels, ax=ax2)

plt.show()

######################################################################
# .. note:: 
# 
#     Permutation invariance of graphs ultimately stems from the fact that the nodes in a graph 
#     do not have an order, and by labelling them to write them into a data structure like a matrix
#     we impose an arbitrary order.
#
# Permutation equivariant embeddings
# ----------------------------------
# 
# When we design a machine learning model that takes graph data, the first step is to encode 
# the adjacency matrix into a quantum state (i.e., a "quantum feature map").
# 
# .. math:: 
# 
#     A \xrightarrow{\phi} |\phi(A)\rangle
# 
# We may want the resulting quantum state to be the same for all adjacency matrices describing 
# the same graph, or in mathematical terms, an **invariant** embedding with respect to 
# permutations $\pi(A)$ of the adjacency matrix:
# 
# .. math:: 
# 
#     \pi(A) \xrightarrow{\phi} |\phi(A)\rangle
# 
# However, this is often too strong a constraint. Think for example of an encoding that 
# associates each node in the graph with a qubit. We might want permutations of the adjacency 
# matrix to lead to the same state _up to an equivalent permutation of the qubits_. 
# (Such a permutation can be described by a permutation operator $P_{\pi}$). This results in 
# an **equivariant** embedding with respect to permutations of the adjacency matrix:
# 
# .. math:: 
# 
#     \pi(A) \xrightarrow{\phi} P|\phi(A)\rangle
#     
#  
# This is exactly what the following quantum embedding is aiming to do! The mathematical details
# behind these concepts use  group theory are beautiful, but can be a bit daunting. Have a look
# [here](https://pennylane.ai/qml/demos/tutorial_geometric_qml.html) if you are interested.
#
# Implementation in PennyLane
# ---------------------------
# 
# Let's get our hands dirty. The permutation-equivariant embedding suggested in 
# [Skolik et al. (2022)](https://arxiv.org/pdf/2205.06109.pdf) has this structure:
# 
# .. figure:: ../demonstrations/equivariant_graph_embedding/circuit.png
#    :width: 70%
#    :align: center
#    :alt: Equivariant embedding
# 
# In PennyLane this looks as follows:
#


import pennylane as qml

def perm_equivariant_embedding(A, betas, gammas):
    """
    Ansatz to embedd a weighted graph with node attributes into a quantum state.
    
    The adjacency matrix A contains the edge weights, as well as the node attributes 
    on its diagonal.
    
    The embedding contains trainable weights 'betas' and 'gammas'.
    """
    n_nodes = len(A)
    n_layers = len(betas)
    
    alphas = np.diag(A)
    
    # initialise state
    for i in range(n_nodes):
        qml.Hadamard(i)
    
    # apply layers
    for l in range(n_layers):

        # apply UG
        for i in range(n_nodes):
            for j in range(i):
                qml.IsingZZ(2*gammas[l]*A[i,j], wires=[i,j]) # Factor of 2 due to definition of gate

        # apply UN
        for i in range(n_nodes):
            qml.RX(alphas[i]*betas[l], wires=i)

######################################################################
# We can use this ansatz in a circuit. 

n_qubits = 5
n_layers = 2

dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def eqc(A, observable, betas, gammas):
    """Circuit that uses the permutation equivariant embedding"""
    
    perm_equivariant_embedding(A, betas, gammas)
    return qml.expval(observable)


A = create_data(n_qubits)
betas = np.random.rand(n_layers)
gammas = np.random.rand(n_layers)
observable = qml.PauliX(0) @ qml.PauliX(1) @ qml.PauliX(3)

print(qml.draw_mpl(eqc, decimals=2)(A, observable, beta, gamma))


######################################################################
# Validating the equivariance
# ---------------------------
# 
# Let's now check if the circuit is really equivariant!
# 
# This is the expectation value we get using the original adjacency matrix as an input:
#

resA = eqc(A, observable, betas, gammas)
print("Model output for A:", resA)


######################################################################
# If we permute the adjacency matrix, this is what we get:
#

perm = [2, 3, 0, 1, 4]
A_perm = permute(A, perm)
resAperm = eqc(A_perm, observable, betas, gammas)
print("Model output for permutation of A: ", resAperm)


######################################################################
# Why are the two values different? Well, we constructed an equivariant ansatz, 
# not an invariant one! The final state before measurement is only the same if we 
# permute the qubits whenever we permute the input adjacency matrix. We could insert a 
# permutation operator (`qml.Permute`) to achieve this, or we simply permute the wires 
# of the observables!
#

observable_perm = qml.PauliX(perm[0]) @ qml.PauliX(perm[1]) @ qml.PauliX(perm[3])

######################################################################
# Now everything should work out!
#

resAperm = eqc(A_perm, observable_perm, betas, gammas)
print("Model output for permutation of A, and with permuted observable: ", resAperm)

######################################################################
# Et voila!
# 
# 
# Conclusion
# ----------
# 
# This example of a permutation-equivariant embedding is just one of many ways to design 
# equivariant quantum machine learning models. [Skolik et al. (2022)](https://arxiv.org/pdf/2205.06109.pdf) use their ansatz to compute functions of quantum expectations that are overall
# permutation equivariant, and use these 
# functions as parts of a reinforcement learning scheme that serves as a heuristic for the 
# traveling salesman problem. Their simulations benchmark this circuit against others that break 
# permutation equivariance and show that it performs better, confirming that if we know 
# about structure in our data, we should try to use this knowledge in machine learning.
#




