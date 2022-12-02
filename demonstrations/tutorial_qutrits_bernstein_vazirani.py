r"""

Qutrits and quantum algorithms
==============================

.. meta::
    :property="og:description": Learn how to interpret the Bernstein-Vazirani algorithm with qutrits
    :property="og:image": https://pennylane.ai/qml/_images/oracle_qutrit.jpg

.. related::


*Author: Guillermo Alonso-Linaje — Posted: XXX. Last updated: XXX*

One of the best known quantum gates is the Toffoli gate. It is an operator that we use all the time but surprisingly it is not implemented natively in hardware. To build it, we will have to decompose it into a series of simpler gates and in particular we will need :math:`6` CNOTs! It was recently discovered that if we unlock the third level of energy and work with qutrits, we can reduce the number of control gates to just :math:`3`.
For this reason, it is important to start developing the intuition behind this new basic unit of information to see in what other situations we can find advantages. The objective of this demo will be to start working with qutrits from an algorithmic point of view. To do so, we will start with the Berstein-Vacerani algorithm, which we will see initially from the point of view of qubits and later on from the point of view of qutrits.


Algoritmo de Berstein-Vazerani
------------------------------

Let us imagine that we are given a function of the form :math:`f(\vec{x}) = \vec{a}\cdot\vec{x} \pmod 2` where :math:`\vec{a}:=[a_0,a_1,...,a_n]` and :math:`\vec{x}:=[x_0,x_1,...,x_n]` are bit strings of length :math:`n` with :math:`a_i, x_i \in \{0,1\}`. Our challenge will be to discover the hidden value of :math:`\vec{a}` by playing with the function :math:`f`. We don't know anything about :math:`\vec{a}` so the only thing we can do is to evaluate :math:`f` at different points :math:`\vec{x}` with the idea of gaining hidden information. I invite you to take your time to think of a possible strategy (at the classical level) in order to determine :math:`\vec{a}` with the minimum number of evaluations of the function :math:`f`. The optimal solution requires only :math:`n` calls to the function! Let's see how we can do this.

Knowing the form of :math:`\vec{a}` and :math:`\vec{x}`, we can rewrite :math:`f` as:

.. math::
  f(\vec{x})=\sum_{i=0}^{n-1}a_ix_i \pmod 2.

The strategy will be to deduce an element of :math:`\vec{a}` with each call to the function. To do this, imagine that we want to determine the value :math:`a_i`. If we notice, we will simply have to choose :math:`\vec{x}` as a vector of all zeros except a one in the i-th position, since in this case:


.. math::
    f(\vec{x})= 0\cdot a_0 + 0\cdot a_1 + ... + 1\cdot a_i + ... + 0\cdot a_n \pmod 2 \quad= a_i \pmod 2.

It is trivial to see therefore that :math:`n` questions are needed. The question therefore is, can we do it more efficiently with a quantum computer? The answer is yes, and in fact, it is simply one of the calls we have to make to our function! The first step is to see how we can represent this statement in a circuit. In this case, we will assume an oracle :math:`U_f` that encodes the function as we can see in the picture

.. figure:: ../demonstrations/qutrits_bernstein_vazirani/oracle_qutrit.jpg
   :scale: 65%
   :alt: Oracle representation
   :align: center

 In general, :math:`U_f` sends the basic state :math:`|\vec{x} \rangle |y\rangle` into the state :math:`| \vec{x} \rangle |y + \vec{a} \cdot \vec{x} \rangle \pmod{2} \rangle`.
 Suppose, for example, that :math:`\vec{a}=[0,1,0]`. Then :math:`U_f|1110\rangle = |1111\rangle`, since we are evaluating :math:`f` at the point :math:`\vec{x} = [1,1,1]`. Since the scalar product between the two values is :math:`1`, the last qubit of the output will take the value :math:`1`. That said, the Bernstein-Vazirani algorithm states the following

.. figure:: ../demonstrations/qutrits_bernstein_vazirani/bernstein_vazirani_algorithm.jpg
   :scale: 65%
   :alt: Bernstein-Vazirani's algorithm
   :align: center

 What we can see is that by simply using Hadamard gates before and after the oracle, what we are going to get is that with a single run, the output of the circuit is exactly the hidden value of :math:`\vec{a}`. Let's do a little math to verify that this is so.

-   First, the input to our circuit is :math:`|0001\rangle`.

-   The second step is to apply Hadamard gates on this state, and for this we must remember the following property:
.. math::
        H^n|\vec{x}\rangle = \frac{1}{\sqrt{2^n}}\sum_{\vec{z} \in \{0,1\}^n}(-1)^{\vec{x}\cdot\vec{z}}|\vec{z}\rangle.

Taking as input the value :math:`|0001\rangle`, we obtain the state:
.. math::
        |\phi_1\rangle:=H^4|0001\rangle = H^3|000\rangle\otimes H|1\rangle = \frac{1}{\sqrt{2^3}}\left(\sum_{z \in \{0,1\}^3}|\vec{z}\rangle\right)\left(\frac{|0\rangle-|1\rangle}{\sqrt{2}}\right).
As you can see, we have separated the first three qubits from the fourth for simplicity.

-   If we now apply our operator :math:`U_f`,
.. math::
      |\phi_2\rangle:= U_f |\phi_1\rangle = \frac{1}{\sqrt{2^3}}\left(\sum_{\vec{z} \in \{0,1\}^3}|\vec{z}\rangle\right)\left(\frac{|\vec{a}\cdot\vec{z} \pmod 2\rangle-|1 + \vec{a}\cdot\vec{z} \pmod 2\rangle}{\sqrt{2}}\right).
Depending on the value of :math:`f(\vec{x})` the final part of the expression can take two values and it can be checked that it is satisfied that

.. math::
      |\phi_2\rangle = \frac{1}{\sqrt{2^3}}\left(\sum_{\vec{z} \in \{0,1\}^3}|\vec{z}\rangle\right)(-1)^{\vec{a}\cdot\vec{z}}\left(\frac{|0\rangle-|1\rangle}{\sqrt{2}}\right).

This is because if :math:`\vec{a}\cdot\vec{z}` takes the value :math:`0`, we will have the :math:`\frac{|0\rangle - |1\rangle}{\sqrt{2}}` and if takes the value $1$, the result will be :math:`\frac{|1\rangle - |0\rangle}{\sqrt{2}} = - \frac{|0\rangle - |1\rangle}{\sqrt{2}}`. Therefore, by calculating :math:`(-1)^{\vec{a}\cdot\vec{z}}` we cover both cases.

-   After this, we can enter the minus sign in the left-hand term and disregard the last qubit since we are not going to use it again:
.. math::
        |\phi_2\rangle =\frac{1}{\sqrt{2^3}}\sum_{\vec{z} \in \{0,1\}^3}(-1)^{\vec{a}\cdot\vec{z}}|\vec{z}\rangle.
-   Finally, we will reapply the rule of the first step to calculate the result after using the Hadamard:
.. math::
        |\phi_3\rangle := H^3|\phi_2\rangle = \frac{1}{2^3}\sum_{\vec{z} \in \{0,1\}^3}(-1)^{\vec{a}\cdot\vec{z}}\left(\sum_{\vec{y} \in \{0,1\}^3}(-1)^{\vec{z}\cdot\vec{y}}|\vec{y}\rangle\right).

Rearranging this expression we obtain that

.. math::
      |\phi_3\rangle  = \frac{1}{2^3}\sum_{\vec{y} \in \{0,1\}^3}\left(\sum_{\vec{z} \in \{0,1\}^3}(-1)^{\vec{a}\cdot\vec{z}+\vec{y}\cdot\vec{z}}\right)|\vec{y}\rangle.

Perfect! The only thing left to check is that in fact, the previous state is exactly :math:`|\vec{a}\rangle`. It may seem complicated but I invite you to demonstrate it by seeing that :math:`\langle \vec{a}|\phi_3\rangle = 1`. Let's go to the code and check that it works. First, let's do the classical approximation:

"""

import pennylane as qml

dev = qml.device("default.qubit", wires = 4, shots = 1)

def Uf():
    qml.CNOT(wires = [1,3])
    qml.CNOT(wires = [2,3])

@qml.qnode(dev)
def circuit1():

    # Initialize x = [1,0,0]
    qml.PauliX(wires = 0)

    # We apply our function
    Uf()

    # We measure the last qubit
    return qml.sample(wires = 3)

@qml.qnode(dev)
def circuit2():

    # Initialize x = [0,1,0]
    qml.PauliX(wires = 1)

    # We apply our function
    Uf()

    # We measure the last qubit
    return qml.sample(wires = 3)

@qml.qnode(dev)
def circuit3():

    # Initialize x = [0,0,1]
    qml.PauliX(wires = 2)

    # We apply our function
    Uf()

    # We measure the last qubit
    return qml.sample(wires = 3)

# We run for x = [1,0,0]
a0 = circuit1()

# We run for x = [0,1,0]
a1 = circuit2()

# We run for x = [0,0,1]
a2 = circuit3()

print(f"The value of 'a' is [{a0},{a1},{a2}]")

##############################################################################
#
# In this case, with 3 queries (:math:`n=3`), we have discovered the value of :math:`\vec{a}`. Let's run the Berstein-Vazerani subroutine to check that one call is enough:


@qml.qnode(dev)
def circuit():

    # We initialize to |0001>
    qml.PauliX(wires = 3)

    # We run the Hadamard, the operator and the Hadamard again.

    for i in range(4):
        qml.Hadamard(wires = i)

    Uf()

    for i in range(3):
        qml.Hadamard(wires = i)

    # We measure in the first 3 qubits
    return qml.sample(wires = range(3))

a = circuit()

print(f"El valor de a es {a}")

##############################################################################
# Great! Everything works as expected, we have just successfully executed the Berstein-vazerani algorithm.
# Now things get more interesting, let's imagine that our basic unit of information is now not the qubit but the qutrit.
# We can generalize the statement as follows:
#
#
# Now we are given a function of the form :math:`f(\vec{x}) = \vec{a}\cdot\vec{x} \pmod 3` where :math:`\vec{a}:=\{a_0,a_1,...,a_n\}` and :math:`\vec{x}:=\{x_0,x_1,...,x_n\}` are strings of length :math:`n` with :math:`a_i, x_i \in \{0,1,2\}`. How can we minimize the number of calls to the function to discover :math:`\vec{a}`? In this case, the classical procedure to detect the value of :math:`\vec{a}` is the same as in the case of qubits: we will evaluate the output of the inputs :math:`[1,0,0]`, :math:`[0,1,0]` and :math:`[0,0,1]`.
# To do this, we must use qubit operators, and in particular we will use that `qml.TShift` is the equivalent gate to `qml.PauliX` for qubits. It has the property of
#
# .. math::
#   \text{TShift}|0\rangle = |1\rangle
#
# .. math::
#   \text{TShift}|1\rangle = |2\rangle
#
# .. math::
#   \text{TShift}|2\rangle = |0\rangle
#
# Therefore we can use this gate to initialize each of the states.


import pennylane as qml
from pennylane import numpy as np

dev = qml.device("default.qutrit", wires = 4, shots = 1)

def Uf():
    qml.TAdd(wires = [1,3])
    qml.TAdd(wires = [1,3])
    qml.TAdd(wires = [2,3])

@qml.qnode(dev)
def circuit1():

    # Initialize x = [1,0,0]
    qml.TShift(wires = 0)

    # We apply our function
    Uf()

    # We measure the last qubit
    return qml.sample(wires = 3)

@qml.qnode(dev)
def circuit2():

    # Initialize x = [0,1,0]
    qml.TShift(wires = 1)

    # We apply our function
    Uf()

    # We measure the last qubit
    return qml.sample(wires = 3)

@qml.qnode(dev)
def circuit3():

    # Initialize x = [0,0,1]
    qml.TShift(wires = 2)

    # We apply our function
    Uf()

    # We measure the last qubit
    return qml.sample(wires = 3)

# We run for x = [1,0,0]
a0 = circuit1()

# We run for x = [0,1,0]
a1 = circuit2()

# We run for x = [0,0,1]
a2 = circuit3()

print(f"El valor de a es [{a0},{a1},{a2}]")

##############################################################################
#
# The question is, can we perform the same procedure as we have done before to find :math:`\vec{a}` in one shot? I mean, there exists for qutrits the `qml.THadamard` gate, we could try to simply substitute and see what happens!
#
#
# The definition of the `THadamard` gate is:
#
# .. math::
#   \text{THadamard}=\begin{pmatrix}
#   1 & 1 & 1\\
#   1 & w & w^2\\
#   1 & w^2 & w
#   \end{pmatrix},
#
# where :math:`w = e^{\frac{2 \pi i}{3}}`.
# Let's go to the code and see how to run this in PennyLane.



w = np.e ** (2 * np.pi * 1j/ 3)

matrix = 1/(w-w**2)*np.array([[1.,1.,1.],
         [1.,w,w**2],
         [1.,w**2,w]])

Hadamard = lambda wires: qml.QutritUnitary(matrix, wires = wires)

@qml.qnode(dev)
def circuit():

    # We initialize to |0001>
    qml.TShift(wires = 3)

    # We run the Hadamard, the operator and the Hadamard again.

    for i in range(4):
        Hadamard(wires = i)

    Uf()

    for i in range(3):
        Hadamard(wires = i)

    # We measure in the first 3 qubits
    return qml.sample(wires = range(3))

a = circuit()

print(f"The value of 'a' is {a}")

##############################################################################
#
# Perfect! The Berstein-Vazerani algorithm generalizes perfectly to qutrits! Let's do the mathematical development again to see that it does indeed make sense!
#
#
# - As before, the input of our circuit has been :math:`|0001\rangle:math:.
# - We will then use the Hadamard definition in qutrits:
# .. math::
#     H^n|\vec{x}\rangle = \frac{1}{\sqrt{3^n}}\sum_{\vec{z} \in \{0,1,2\}^n}w^{\vec{x}\cdot\vec{z}}|\vec{z}\rangle.
# Therefore, applying it to the state :math:`|0001\rangle`, we obtain the state
#
# .. math::
#    |\phi_1\rangle:=H^4|0001\rangle = H^3|000\rangle\otimes H|1\rangle = \frac{1}{\sqrt{3^3}}\left(\sum_{z \in \{0,1,2\}^3}|\vec{z}\rangle\right)\left(\frac{|0\rangle+w|1\rangle+w^2|2\rangle}{\sqrt{3}}\right).
#
# - Then we apply the operator :math:`U_f` to obtain:
#
# .. math::
#   |\phi_2\rangle:= U_f |\phi_1\rangle = \frac{1}{\sqrt{3^3}}\left(\sum_{\vec{z} \in \{0,1,2\}^3}|\vec{z}\rangle\right)\left(\frac{|0 + \vec{a}\cdot\vec{z} \pmod 3 \rangle+w|1+ \vec{a}\cdot\vec{z} \pmod 3 \rangle+w^2|2+ \vec{a}\cdot\vec{z} \pmod 3 \rangle}{\sqrt{3}}\right).
#
# Depending on the value of :math:`f(\vec{x})`, as before, we obtain three possible states:
#   - If :math:`\vec{a}\cdot\vec{z} = 0`, we have :math:`\frac{1}{\sqrt{3}}\left(|0\rangle+w|1\rangle+w^2|2\rangle\right)`.
#   - If :math:`\vec{a}\cdot\vec{z} = 1`, we have :math:`\frac{w^2}{\sqrt{3}}\left(|0\rangle+|1\rangle+w|2\rangle\right)`.
#   - If :math:`\vec{a}\cdot\vec{z} = 2`, we have :math:`\frac{w}{\sqrt{3}}\left(|0\rangle+w^2|1\rangle+|2\rangle\right)`.
#
# Based on this, we can group the three states as :math:`\frac{|w^{-\vec{a}\cdot\vec{z}}}{\sqrt{3}}\left(|0\rangle+w|1\rangle+w^2|2\rangle\right)`.
#
# - After this, we can enter the coefficient in the left-hand term and, as before, disregard the last qubit since we are not going to use it again:
# .. math::
#   |\phi_2\rangle =\frac{1}{\sqrt{3^3}}\sum_{\vec{z} \in \{0,1,2\}^3}w^{-\vec{a}\cdot\vec{z}}|\vec{z}\rangle.
# - Finally, we re-apply the THadamard:
# .. math::
#     |\phi_3\rangle := H^3|\phi_2\rangle = \frac{1}{3^3}\sum_{\vec{z} \in \{0,1,2\}^3}w^{-\vec{a}\cdot\vec{z}}\left(\sum_{\vec{y} \in \{0,1,2\}^3}w^{\vec{z}\cdot\vec{y}}|\vec{y}\rangle\right).
# Rearranging this expression we obtain that
#
# .. math::
#     |\phi_3\rangle  = \frac{1}{3^3}\sum_{\vec{y} \in \{0,1,2\}^3}\left(\sum_{\vec{z} \in \{0,1,2\}^3}w^{-\vec{a}\cdot\vec{z}+\vec{y}\cdot\vec{z}}\right)|\vec{y}\rangle.
#
# In the same way as before, it can be easily checked that :math:`\langle \vec{a}|\phi_3\rangle = 1` and therefore, when measuring, one shot will be enough to obtain the value of :math:`\vec{a}`!
#
# Conclusion
# ----------
#
# In this demo we have practiced the use of basic qutrit gates such as `TShift` or `Thadamard` throughout the Berstein-Vazerani algorithm. In this case the generalization has been straightforward and we have found that it makes mathematical sense, but we cannot always substitute qubit gates for qubit gates as we have seen in the demo. To give an easy example of this, we know the property that :math:`X = HZH`, but it turns out that this property does not generalize! The general property is :math:`X = H^{\dagger}ZH`. In the case of qubits it holds that :math:`H = H^{\dagger}` but in other dimensions it does not. I invite you to continue practicing with other types of algorithms, will the Deustch-Jozsa algorithm generalize well? Take a pen and paper and check it out.
#
# About the author
# ----------------
# .. include:: ../_static/authors/guillermo_alonso.txt