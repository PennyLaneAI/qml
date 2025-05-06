r"""Intro to the Quantum Chebyshev
=============================================================

This demo is inspired by the paper `"Quantum Chebyshev transform: mapping, embedding, learning and sampling distributions" <https://arxiv.org/abs/2306.17026>`_ wherein the authors describe a workflow for quantum Chebyshev-based model building. 
They demonstrate the use of the Chebyshev basis space in generative modeling for probability distributions. Crucial to their implementation of learning models in Chebyshev space is the quantum Chebyshev transform (QChT), which is used to swap between the computational basis and the Chebyshev basis. 

We will start by discussing Chebyshev polynomials and why you may want to work in Chebyshev space. Then we will show how the QChT can be implemented in PennyLane. 


.. figure:: ../_static/demonstration_assets/qft/socialthumbnail_large_QFT_2024-04-04.png
    :align: center
    :width: 60%
    :target: javascript:void(0)


What are Chebyshev polynomials?
---------------------------------------

`Chebyshev polynomials <https://en.wikipedia.org/wiki/Chebyshev_polynomials#As_a_basis_set>`_ of the first kind :math:`T_n(x)` are a set of orthogonal polynomials that are complete on the interval :math:`[-1,1]`. They can be defined as 

.. math::
  T_n(x) \equiv \cos(n \arccos(x))

where :math:`n` is the order of the polynomial. We can write out the first few orders explicitly.

.. math::
  T_0(x) &= 1 \\
  T_1(x) &= x \\
  T_2(x) &= 2x^2-1 \\
  T_3(x) &= 4x^3 - 3x \\
  T_4(x) &= 8x^4 - 8x^2 + 1 \\
  &\ \,\vdots \\
  T_{n+1}(x) &= 2xT_n(x) - T_{n-1}(x)

The recursion relation in the last line can be used to compute the next orders. Observe that odd and even order :math:`T_n` are odd and even functions, respectively. 
The roots of the :math:`T_n(x)` occur at the values 

.. math::
  x_n^\mathrm{Ch} = \cos\left(\frac{2k+1}{2n}\pi\right)\,, \quad k=0, ..., n-1\,. 

These are known as the `Chebyshev nodes <https://en.wikipedia.org/wiki/Chebyshev_nodes>`_.

.. figure:: ../_static/demonstration_assets/quantum_chebyshev_transform/chebyshev_polynomials.png
    :align: center
    :width: 60%
    :target: javascript:void(0)

The nodes are plotted above along with the corresponding polynomials. Note that the polynomials are normalized such that $T_n(1)=1$, and they satisfy a discrete orthogonality condition on the nodes of $T_N(x)$ in the following way for $k$, $\ell<N$

.. math::
  \sum^{N-1}_{j=0}T_k(x_j^\mathrm{Ch})T_\ell(x_j^\mathrm{Ch}) =  
    \begin{cases}
      0 & k \neq \ell\,,\\
      N & k = \ell = 0\,,\\
      N/2 & k = \ell \neq 0
    \end{cases}

The Chebyshev polynomials have a lot of *nice* properties. Because they are complete, any function :math:`f(x)` on the interval :math:`x\in [-1,1]` can be expanded in :math:`T_n(x)` up to order :math`N` as :math:`f(x) = \sum_{j=0}^N a_j T_j(x)`. \
To do this process numerically for a discrete set of sampling points would take :math:`\mathcal{O}(N^2)` operations for a general set of complete functions \emoji{snail}. 
However, because of the way the Chebyshev polynomials are defined in terms of cosine, the `discrete Chebyshev transformation (DChT) <https://en.wikipedia.org/wiki/Discrete_Chebyshev_transform>`_ can be related to the `discrete cosine transform (DCT) <https://en.wikipedia.org/wiki/Discrete_cosine_transform>`_ to leverage the efficiency of the `fast-Fourier-transform <https://en.wikipedia.org/wiki/Fast_Fourier_transform>`_-style algorithms, which take :math:`\mathcal{O}(N \log N)` operations \emoji{rocket}.

The DChT is sampled on the nodes of :math:`T_N(x)`, which are non-equidistant on the :math:`[-1,1]` interval. 
This non-uniform sampling has more resolution at the boundary, but less in the middle. This can be a benefit if you are, for example, solving a differential equation and expect more interesting features at the boundary, so the extra resolution there is useful. 
In general, working in the Chebyshev basis can have advantages over the Fourier basis for polynomial decomposition.


Quantum Chebyshev basis
---------------------------------------
The quantum Chebyshev transform (QChT) circuit described in Ref. [#williams2023]_ maps :math:`2^N` computational states :math:`\{\ket{x_j}\}_{j=0}^{2^N-1}` to Chebyshev states :math:`\{\ket{\tau(x_j^\mathrm{Ch})}\}_{j=0}^{2^N-1}` which have amplitudes given by the Chebyshev polynomials of the first kind. 

.. math::
  \ket{\tau(x)} = \frac1{2^{N/2}}T_0(x)\ket{0} + \frac1{2^{(N-1)/2}}\sum_{k=1}^{2^N-1}T_k(x)\ket{k}

where :math:`\ket{k}` are the computational basis states and :math:`N` is the number of qubits. These states are orthonormal at the Chebyshev nodes due to the orthogonality condition, that is

.. math::
  \braket{\tau(x_j^\mathrm{Ch}}{\tau(x_{j'}^\mathrm{Ch})} = \delta_{j, j'}

The squared overlap if one of the variables is not at a node can be derived analytically as

.. math:: 
  \abs{\braket{\tau(x_j^\mathrm{Ch})}{\tau(x)}}^2 = \frac{\left(T_{2^N+1}(x_j^\mathrm{Ch})T_{2^N}(x)-T_{2^N}(x_j^\mathrm{Ch})T_{2^N+1}(x)\right)^2}{2^{2N}(x_j^\mathrm{Ch}-x)^2}

The goal is to design a circuit that applies the operation :math:`\mathcal{U}_\mathrm{QChT} = \sum_{j=0}^{2^N-1} \ket{\tau(x_j^\mathrm{Ch})}\bra{x_j}`.


Designing the transform circuit
---------------------------------------
Let's start from the end and look at the circuit diagram generated from the code we want to write. 
An ancilla qubit is required, which will be the :math:`0` indexed qubit, and the rest compose the state :math:`\ket{x}` which starts in the computational basis, shown below as :math:`\ket{\psi}`. We demonstrate for :math:`N=4` non-ancilla qubits.

.. figure:: ../_static/demonstration_assets/quantum_chebyshev_transform/QChT_diagram_4qubits.png
    :align: center
    :width: 60%
    :target: javascript:void(0)

The intuition for the structure of the above circuit comes from the link between the DChT and the DCT. 
Notice the use of the `quantum Fourier transform (QFT) <https://pennylane.ai/qml/demos/tutorial_qft/>`_ applied on all qubits. 
The QChT is an extended QFT circuit with some added interference and mixing of the elements.

Let's break down the circuit above into pieces that we will use inside our circuit function. 
First, a Hadamard gate is applied to the ancilla, and then a CNOT ladder is applied, controlled on the ancilla. 
To start, we will define a function for the CNOT ladder.

"""

import pennylane as qml

# number of qubits (non-ancilla)
N = 4

def CNOT_ladder():
    for wire in range(1, N+1):
        qml.CNOT([0, wire])

#############################################
# After the initial CNOT ladder comes an :math:`N+1` QFT circuit, which can be implemented using ``qml.QFT``. 

# Next are phase rotations and shifts. 
# In particular, there is a phase shift on the ancilla by :math:`-\pi/2^{(N+1)}` followed by a :math:`Z` rotation of :math:`-\pi(2^N - 1)/2^{N+1}`. 
# The other other qubits are rotated in :math:`Z` by :math:`\pi/2^{(j+1)}`, where :math:`j` is the index of the qubit as labelled in the circuit diagram.
# 




#############################################
# References
# ----------
#
# .. [#williams2023]
#
#     Chelsea A. Williams, Annie E. Paine, Hsin-Yu Wu, Vincent E. Elfving and Oleksandr Kyriienk. "Quantum Chebyshev 
#     transform: mapping, embedding, learning and sampling distributions." `arxiv:2306.17026
#     <https://arxiv.org/abs/2306.17026>`__ (2023).
#
# About the author
# ----------------
#
