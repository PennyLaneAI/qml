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

In this how-to guide, we will show how we can apply a polynomial function in a quantum computer using basic arithmetic.
We will use as an example the function :math:`f(x,y)=a+b\cdot x+c\cdot y + d \cdot xy ` where the variables and the coefficients
are integer values. We will take the values :math:`a = 4`, :math:`b = 5`,:math:`c = 3` and :math:`d = 3`, defining the desired operator as:

.. math::

    U|x\rangle |y\rangle |0\rangle = |x\rangle |y\rangle |4 + 5x + 3y + 3xy\rangle,

where :math:`x` and :math:`y` are the binary representations of the integers on which we want to apply the function.


InPlace and OutPlace Operations
-------------------------------

We can load the target function into the quantum computer using different quantum arithmetic operations. 
We will break down into pieces. We'll do a step by step load of the function :math:`f(x, y)`.

First let's do the necessary imports, give the values to the constants :math:`a,b,c,d` and define the registers:
"""

import pennylane as qml

a, b, c, d = 4, 5, 3, 3

# we indicate the name of the registers and their number of qubits
wires = qml.registers({"x": 4, "y":4, "output":5,"work_wires": 4})

######################################################################
# We will start off building the initial quantum circuit

def prepare_initial_state(x,y):
    qml.BasisState(x, wires=wires["x"])
    qml.BasisState(y, wires=wires["y"])


dev = qml.device("default.qubit", shots=1)
@qml.qnode(dev)
def circuit(x,y):
    prepare_initial_state(x, y)
    return [qml.sample(wires=wires[name]) for name in ["x", "y", "output"]]

######################################################################
# We can now check that this circuit performs the correct initialization by setting example values 
# for :math:`x`, and :math:`y` such as :math:`x=1` and :math:`y=4`. In general, the variables :math:`x` and :math:`y` can represent any
# quantum state, but in this how-to they will be the quantum states [0 0 0 1] and [0 1 0 0] which represent the
# numbers 1 and 4 respectively.

output = circuit(x=1,y=4)

print("x register: ", output[0])
print("y register: ", output[1])
print("output register: ", output[2])


######################################################################
# Now, we can introduce the first quantum arithmetic operation to load :math:`f(x, y)`. The first step will be
#  to load the constant :math:`a = 4` by using the Inplace addition operator :class:`~.pennylane.Adder`:

@qml.qnode(dev)
def circuit(x,y):

    prepare_initial_state(x, y)     #    |x> |y> |0>
    qml.Adder(a, wires["output"])   #    |x> |y> |4>

    return qml.sample(wires=wires["output"])

print(circuit(x=1,y=4))

######################################################################
# We obtained the state [0 1 0 0], i.e. :math:`a=4`, as expected!
#
# The next step will be to add the term :math:`3xy` by using the
# Inplace and Outplace multiplication operators, :class:`~.pennylane.Multiplier`  and :class:`~.pennylane.OutMultiplier` respectively.
# To do this, we first turn :math:`|x\rangle` into :math:`|3x\rangle` and then multiply it by :math:`|y\rangle`.

def adding_3xy():
    # |x> --->  |3x>
    qml.Multiplier(d, wires["x"], work_wires=wires["work_wires"])

    # |3x>|y>|0> ---> |3x>|y>|3xy>
    qml.OutMultiplier(wires["x"], wires["y"], wires["output"])

    # We return the x-register to its original value
    # |3x>|y>|3xy>  ---> |x>|y>|3xy>
    qml.adjoint(qml.Multiplier)(d, wires["x"], work_wires=wires["work_wires"])

@qml.qnode(dev)
def circuit(x,y):

    prepare_initial_state(x, y)    #    |x> |y> |0>
    qml.Adder(a, wires["output"])  #    |x> |y> |4>
    adding_3xy()                   #    |x> |y> |3 + 3xy>

    return qml.sample(wires=wires["output"])

print(circuit(x=1,y=4))

######################################################################
#Nice! The state [1 0 0 0 0] represents :math:`4+3\cdot xy =16`.
#
#The last step will involve to load the monomial terms :math:`5x` and math:`3y` by using
# the OutPlace addition operator :class:`~.pennylane.OutAdder` and the :class:`~.pennylane.Multiplier` previously employed.

def adding_5x_3y():

    # |x>|y> --->  |5x>|3y>
    qml.Multiplier(b, wires["x"], work_wires=wires["work_wires"])
    qml.Multiplier(c, wires["y"], work_wires=wires["work_wires"])

    # |5x>|3y>|0> --->  |5x>|3y>|5x + 3y>
    qml.OutAdder(wires["x"], wires["y"], wires["output"])

    # We return the x and y registers to its original value
    # |5x>|3y>|5x + 3y> --->  |x>|y>|5x + 3y>
    qml.adjoint(qml.Multiplier)(b, wires["x"], work_wires=wires["work_wires"])
    qml.adjoint(qml.Multiplier)(c, wires["y"], work_wires=wires["work_wires"])


@qml.qnode(dev)
def circuit(x,y):

    prepare_initial_state(x, y)    #    |x> |y> |0>
    qml.Adder(a, wires["output"])  #    |x> |y> |4>
    adding_3xy()                   #    |x> |y> |4 + 3xy>
    adding_5x_3y()                 #    |x> |y> |4 + 5x + 3y + 3xy>


    return qml.sample(wires=wires["output"])

print(circuit(x=1,y=4))

######################################################################
# The result obtained doesn't look quite right, since one would expect to obtain :math:`f(x=1,y=4)=4+ 5\cdot1+3\cdot4+ 3 \cdot 1 \cdot 4=33`... 
# Could you guess what is going on?
#
# What's happening here is that we're running into overflow. The number 33 is too large for the number of wires we have defined in
#  `wires[output]`. With 5 wires, we can represent numbers up to :math:`2^5=32`. Any number larger than that gets reduced to its modulo with 
#  respect to :math:`2^5`. We have to keep in mind that all the quantum arithmetic is modular. So, every operation we perform is with respect
#  to a given modulo that can be set by the user, but by default will be :math:`mod=2^{\text{len(wires)}}`.
#
# To fix this  and get the correct result :math:`f(x=1,y=4)=33`, the simplest solution is to redefine the registers
# adding one more wire to the output.

wires = qml.registers({"x": 4, "y": 4, "output": 6,"work_wires": 4})
print(circuit(x=1, y=4))

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
def circuit_with_Poly(x,y):

   prepare_initial_state(x, y)
   #qml.OutPoly(f, registers_wires=[wires["x"], wires["y"], wires["output"]])
   
   return qml.sample(wires = wires["output"])

print(circuit_with_Poly(x=1,y=4))

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

