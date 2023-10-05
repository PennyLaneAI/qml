r"""

Initial state preparation for quantum chemistry
===============================================

.. meta::
    :property="og:description": Understand the concept of the initial state, and learn how to prepare it with PennyLane
    :property="og:image": https://pennylane.ai/qml/_images/thumbnail_tutorial_initial_state_preparation.png

.. related::
    tutorial_quantum_chemistry Building molecular Hamiltonians
    tutorial_vqe A brief overview of VQE

*Author: Stepan Fomichev — Posted: 20 October 2023. Last updated: 20 October 2023.*

It was 1968, the height of the Cold War, and on a routine surveillance mission the 
US submarine Scorpion goes missing. Stakes are high -- a nuclear sub is lost at sea! -- 
but despite the round-the-clock efforts of dozens of ships and aircraft, after a week 
the search is called off. The search area -- "somewhere off the Eastern Seaboard" -- 
is simply too big to comb through by brute force. 

But then a group of statisticians gets involved. Combining information from underwater 
sonar listening stations and averaging insights from a variety of experts, they are 
able to zero in on the a few most promising search quadrants -- and soon after, the sub 
is found in one of them.  

Much like searching the oceanic floor for a sunken ship, searching for the ground state 
in the gigantic Hilbert space of a typical molecule requires expert guidance -- an 
initial guess for what the state could be. In this demo, you will learn about different 
strategies for preparing such initial states, and specifically how to do that in 
PennyLane.

How do initial states affect quantum algorithms?
------------------------------------------------
From the variational quantum eigensolver (VQE) to quantum phase estimation (QPE), to even 
the recent ISQ-era algorithms like the Lin-Tong approach, many quantum algorithms for 
obtaining the ground state of a chemical system require a good initial state to be 
useful. (add three images here)

    1. In the case of VQE, as we will see later in this demo, a good initial state 
    directly translates into fewer optimization steps. 
    2. In QPE, the probability of measuring the ground-state energy is directly 
    proportional to the overlap squared $|c_0|^2$ of the initial and ground states

.. math::

    \ket{\psi_{\text{in}}} = c_0 \ket{\psi_0} + c_1 \ket{\psi_1} + ...

    3. Finally, in Lin-Tong the overlap with the ground-state affects the size of the 
    step in the cumulative distribution function, the bigger step making it easier to 
    detect the jump and thus resolve the position of the ground-state energy.

We see that in all these cases, having a high-quality initial state can seriously 
reduce the runtime of the algorithm. By high-quality, we just mean that the prepared 
state in some sense minimizes the effort of the quantum algorithm.

Where do I get good initial states?
-----------------------------------
Much like when searching for a sunken submarine, there are a lot of things you might try 
to prepare a good guess for the ground-state. 

    1. The adiabatic principle tells us that the eigenstates of Hamiltonians smoothly 
    evolving with some parameter :math:`\lambda` are  This is the reasoning behind the adiabatic approach: 


"""

from pennylane import FermiC, FermiA

a0_dag = FermiC(0)
a1 = FermiA(1)

##############################################################################
# We used the compact notations ``a0_dag`` to denote a creation operator applied to
# the :math:`0\text{th}` orbital and ``a1`` to denote an annihilation operator applied to the
# :math:`1\text{st}` orbital. Once created, these operators can be multiplied or added to each other
# to create new operators. A product of fermionic operators will be called a *Fermi word* and a
# linear combination of Fermi words will be called a *Fermi sentence*.

fermi_word = a0_dag * a1
fermi_sentence = 1.3 * a0_dag * a1 + 2.4 * a1
fermi_sentence

##############################################################################
# In this simple example, we first created the operator :math:`a^{\dagger}_0 a_1` and then created
# the linear combination :math:`1.3 a^{\dagger}_0 a_1 + 2.4 a_1`. We can also perform
# arithmetic operations between Fermi words and Fermi sentences.

fermi_sentence = fermi_sentence * fermi_word + 2.3 * fermi_word
fermi_sentence

##############################################################################
# Summary
# -------
# This demo explains how to create and manipulate fermionic operators in PennyLane, which is as
# easy as writing the operators on paper. PennyLane supports several arithmetic operations between
# fermionic operators and provides tools for mapping them to the qubit basis. This makes it easy and
# intuitive to construct complicated fermionic Hamiltonians such as
# `molecular Hamiltonians <https://pennylane.ai/qml/demos/tutorial_quantum_chemistry>`_.
#
# References
# ----------
#
# .. [#surjan]
#
#     Peter R. Surjan, "Second Quantized Approach to Quantum Chemistry". Springer-Verlag, 1989.
#
#
# About the author
# ----------------
# .. include:: ../_static/authors/soran_jahangiri.txt
