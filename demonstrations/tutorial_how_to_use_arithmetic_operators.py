r"""
How-to use Quantum Arithmetic Operators
=======================================

Classical computers handle arithmetic operations like addition, subtraction, multiplication, and exponentiation with ease, thanks to decades of development. For instance, you can multiply two numbers on your phone in milliseconds!

In contrast, performing the same tasks on quantum computers isn't as straightforward. While they can excel at solving certain problems faster than classical computers, basic arithmetic isn't their strongest point. So why do we want to do quantum arithmetic anyway?

Well, quantum arithmetic isn't necessarily “better” at basic operations, but it becomes essential when it serves as a part of a more complex quantum algorithm. For example:

1. In Shor's algorithm for factoring large numbers, quantum arithmetic is crucial for performing modular exponentiation in order to execute the algorithm efficiently.

2. Grover's algorithm might need to use quantum arithmetic to construct oracles that help in speeding up search problems, as shown in [#qft_arith_demo]_.

3. Loading functions into quantum computers, which might require several quantum arithmetic operations.

These arithmetic operations are like building blocks. Alone, they might not offer a speedup, but when incorporated into larger algorithms, they enable the kind of powerful computations that quantum computers are designed for.

With PennyLane, you'll see how easy it is to build these quantum arithmetic operations and use them as subroutines in your quantum algorithms!


Loading a function :math:`f(x, y)`
----------------------------------

In this how-to guide, we will show how we can load a general function into a quantum computer. 
In particular we will load a function of this shape:

.. math:: f(x,y)=a+b\cdot x+c\cdot y + d \cdot x \cdot y.


InPlace and OutPlace Operations
-------------------------------

We can load the target function into the quantum computer using different quantum arithmetic operations. 
In particular we will break down into pieces. And we'll do a step by step load of the function :math:`f(x, y)`.

First let's do the necesary imports, give some value to the constants :math:`a,b,c,d` and define the registers:
"""

import pennylane as qml

a, b, c, d = 4, 5, 3, 3
wires = qml.registers({"x": 4, "y":4, "output":5,"work_wires": 4})

######################################################################
# Then, what we want to build is a quantum circuit that performs the following operation:
# .. figure:: ../_static/demonstration_assets/how_to_use_arithmetic_operators/circuit_fxy.jpg
#   :align: center
#   :width: 90%
# Now, let's break down this problem into pieces.
# 
# We will start off building the initial quantum circuit

dev = qml.device("default.qubit", shots=1)
@qml.qnode(dev)
def initial_circuit(x,y):
    qml.BasisState(x, wires=wires["x"])
    qml.BasisState(y, wires=wires["y"])
    qml.BasisState(0, wires=wires["output"])
    return qml.sample(wires=wires["output"])

######################################################################
# We can now check that this circuit performs the correct initialization by setting example values 
# for :math:`x`, and :math:`y` such as :math:`x=1` and :math:`y=4`. In general, the variables :math:`x` and :math:`y` can represent any 
# quantum state, but in this how-to they will be the quantum states [0 0 1 1] and [0 1 0 0] which represent the 
# numbers 3 and 4 respectively.

print(initial_circuit(x=3,y=4))

######################################################################
# Now, we can introduce the first quantum arithmetic operation to load :math:`f(x, y)`. The first step will be
#  to load the constant :math:`a` by using the Inplace addition operator :class:`~.pennylane.Adder`:

@qml.qnode(dev)
def first_circuit(x,y):
    # Initial circuit
    qml.BasisState(x, wires=wires["x"])
    qml.BasisState(y, wires=wires["y"])
    qml.BasisState(0, wires=wires["output"])
    # Loading `a`
    qml.Adder(a, wires["output"])
    return qml.sample(wires=wires["output"])
print(first_circuit(x=1,y=4))

######################################################################
# We obtained the state [0 1 0 0], i.e. :math:`a=4`, as expected!
#
# The next step will be to load the mixed term :math:`d \cdot x \cdot y` by using the
# Inplace and Outplace multiplication operators, :class:`~.pennylane.Multiplier`  and :class:`~.pennylane.OutMultiplier` respectively.

@qml.qnode(dev)
def load_mixed_term(x,y):
    # Initial circuit
    qml.BasisState(x, wires=wires["x"])
    qml.BasisState(y, wires=wires["y"])
    qml.BasisState(0, wires=wires["output"])
    # Loading `a`
    qml.Adder(a, wires["output"])
    # Loading `d*x*y`
    qml.Multiplier(d, wires["x"], work_wires=wires["work_wires"])
    qml.OutMultiplier(wires["x"], wires["y"], wires["output"])
    # we clean up the wires["x"] by multiplying the state |d*x \rangle by d^-1
    qml.Multiplier(pow(d, -1, 2**len(wires["x"])), wires["x"], work_wires=wires["work_wires"])
    return qml.sample(wires=wires["output"])
print(load_mixed_term(x=1,y=4))

######################################################################
#Nice! The state [1 0 0 0 0] represents :math:`a+d\cdot x \cdot y = 4+ 3 \cdot 1 \cdot 4 =16`.
#
#The last step will involve to load the monomial terms :math:`b \cdot x$ and $ c \cdot y` by using 
# the OutPlace addition operator :class:`~.pennylane.OutAdder` and the :class:`~.pennylane.Multiplier` previously employed.

@qml.qnode(dev)
def load_f_x_y(x,y):
    # Initial circuit
    qml.BasisState(x, wires=wires["x"])
    qml.BasisState(y, wires=wires["y"])
    qml.BasisState(0, wires=wires["output"])
    # Loading `a`
    qml.Adder(a, wires["output"])
    # Loading `d*x*y`
    qml.Multiplier(d, wires["x"], work_wires=wires["work_wires"])
    qml.OutMultiplier(wires["x"], wires["y"], wires["output"])
    # we clean up the wires["x"] by multiplying the state |d*x \rangle by d^-1
    qml.Multiplier(pow(d, -1, 2**len(wires["x"])), wires["x"], work_wires=wires["work_wires"])
    # Loading `b*x*` and `c*y`
    qml.Multiplier(b, wires["x"], work_wires=wires["work_wires"])
    qml.Multiplier(c, wires["y"], work_wires=wires["work_wires"])
    qml.OutAdder(wires["x"], wires["y"], wires["output"])
    # we clean up the wires["y"] and wires["y"] by multiplying the state |b*x \rangle by b^-1 and the state |c*y \rangle by c^-1
    qml.Multiplier(pow(b, -1, 2**len(wires["x"])), wires["x"], work_wires=wires["work_wires"])
    qml.Multiplier(pow(c, -1, 2**len(wires["y"])), wires["y"], work_wires=wires["work_wires"])
    return qml.sample(wires=wires["output"])
print(load_f_x_y(x=1,y=4))

######################################################################
# The result obtained doesn't look quite right, since one would expect to obtain :math:`f(x=1,y=4)=4+ 5\cdot1+3\cdot4+ 3 \cdot 1 \cdot 4=33`... 
# Could you guess what is going on?
#
# What's happening here is that we're running into overflow. The number 33 is too large for the number of wires we have defined in
#  `wires[output]`. With 5 wires, we can represent numbers up to :math:`2^5=32`. Any number larger than that gets reduced to its modulo with 
#  respect to :math:`2^5`. We have to keep in mind that all the quantum arithmetic is modular. So, every operation we perform is with respect
#  to a given modulo that can be set by the user, but by default will be :math:`mod=2^{\text{len(wires)}}`.
#
# To fix this  and get the correct result :math:`f(x=1,y=4)=33`, the simplest solution is to rerun the previous cell, adding one more wire 
# to the output.

wires = qml.registers({"x": 4, "y": 4, "output": 6,"work_wires": 4})
@qml.qnode(dev)
def load_f_x_y(x,y):
    # Initial circuit
    qml.BasisState(x, wires=wires["x"])
    qml.BasisState(y, wires=wires["y"])
    qml.BasisState(0, wires=wires["output"])
    # Loading `a`
    qml.Adder(a, wires["output"])
    # Loading `d*x*y`
    qml.Multiplier(d, wires["x"], work_wires=wires["work_wires"])
    qml.OutMultiplier(wires["x"], wires["y"], wires["output"])
    # we clean up the wires["x"] by multiplying the state |d*x \rangle by d^-1
    qml.Multiplier(pow(d, -1, 2**len(wires["x"])), wires["x"], work_wires=wires["work_wires"])
    # Loading `b*x` and `c*y`
    qml.Multiplier(b, wires["x"], work_wires=wires["work_wires"])
    qml.Multiplier(c, wires["y"], work_wires=wires["work_wires"])
    qml.OutAdder(wires["x"], wires["y"], wires["output"])
    # we clean up the wires["y"] and wires["y"] by multiplying the state |b*x \rangle by b^-1 and the state |c*y \rangle by c^-1
    qml.Multiplier(pow(b, -1, 2**len(wires["x"])), wires["x"], work_wires=wires["work_wires"])
    qml.Multiplier(pow(c, -1, 2**len(wires["y"])), wires["y"], work_wires=wires["work_wires"])
    return qml.sample(wires=wires["output"])
print(load_f_x_y(x=1,y=4))

######################################################################
# Now we get the correct result :math:`f(x=1,y=4)=33`!

######################################################################
# Using OutPoly
# ------------------------------------------
# In the last section, we showed how to use different arithmetic operations to load 
# a function onto a quantum computer. But what if I told you there’s an easier way to do all this using just one
# PennyLane function that handles the arithmetic for you? Pretty cool, right? I’m talking about :class:`~.pennylane.OutPoly`. 
# This handy operator lets you load polynomials directly into quantum states, taking care of all the arithmetic in one go. 
# Let’s check out how to load a function like :math:`f(x, y)` using :class:`~.pennylane.OutPoly`!
#
# Let's first start by explicitly defining our function:

def f(x, y):
   return a+b*x+c*y+d*x*y

######################################################################
# Now, let's load it into a quantum circuit!

######################################################################

@qml.qnode(dev)
def load_f_x_y_with_OutPoly(x,y):

   # Initial circuit
   qml.BasisState(x, wires=wires["x"])
   qml.BasisState(y, wires=wires["y"])
   qml.BasisState(0, wires=wires["output"])

   # applying the polynomial
   qml.OutPoly(f, registers_wires=[wires["x"], wires["y"], wires["output"]])
   
   return qml.sample(wires = wires["output"])
print(load_f_x_y_with_OutPoly(x=1,y=4))

######################################################################
# Eureka! We’ve just seen how much easier it can be to implement arithmetic operations in one step. 
# Now, it's up to you to decide, depending on the problem you're tackling, whether to go for the versatility 
# of defining your own arithmetic operations or the convenience of using the :class:`~.pennylane.OutPoly` function.

######################################################################
# Conclusions
# ------------------------------------------
# In conclusion, understanding and implementing quantum arithmetic is a key step toward unlocking the full potential
#  of quantum computing. While it may not replace classical efficiency for simple tasks, its role in complex algorithms 
# is undeniable. By leveraging tools like `qml.OutPoly`, you can streamline the coding of your quantum algorithms. So, 
# whether you choose to customize your arithmetic operations or take advantage of the built-in convenience offered by PennyLane 
# operators, you're now equipped to tackle exciting quantum challenges ahead!
#
# References
# ----------
#
# .. [#demo_qft_arith]
#
#    Guillermo Alonso
#    "Basic arithmetic with the quantum Fourier transform (QFT)",
#    `Pennylane: Basic arithmetic with the quantum Fourier transform (QFT)  <https://pennylane.ai/qml/demos/tutorial_qft_arithmetics/>`__, 2024
#
# About the authors
# -----------------

