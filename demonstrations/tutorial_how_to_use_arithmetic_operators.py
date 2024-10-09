r"""
How-to use Quantum Arithmetic Operators
=======================================

Classical computers handle arithmetic operations like addition, subtraction, multiplication, and exponentiation with ease. 
For instance, you can multiply two numbers on your phone in milliseconds!

Quantum computers, however, aren't as efficient at basic arithmetic. So why do we need quantum arithmetic?
While it's not "better" for basic operations, it's essential in more complex quantum algorithms. For example:

1. In Shor's algorithm quantum arithmetic is crucial for performing modular exponentiation. 

2. Grover's algorithm might need to use quantum arithmetic to construct oracles, as shown in [#demo_qft_arith]_.

3. Loading functions into quantum computers, which often requires several quantum arithmetic operations.

These arithmetic operations act as building blocks that enable powerful quantum computations when integrated into larger algorithms.
With PennyLane, you'll see how easy it is to build these operations as subroutines for your quantum algorithms!


Loading a function :math:`f(x, y)`
----------------------------------

In this how-to guide, we will show how we can apply a polynomial function in a quantum computer using basic arithmetic.
We will use as an example the function :math:`f(x,y)= 4 + 3xy + 5 x+ 3 y` where the variables :math:`x` and :math:`y`
are integer values. Therefore, the operator we want to build is:

.. math::

    U|x\rangle |y\rangle |0\rangle = |x\rangle |y\rangle |4 + 3xy + 5x + 3y\rangle,

where :math:`x` and :math:`y` are the binary representations of the integers on which we want to apply the function.

We will show how to load this function in two different ways: first, by concatenating simple arithmetic operators,
and finally, using the :class:`~.pennylane.OutPoly` operator.

InPlace and OutPlace Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We can load the target function into the quantum computer using different quantum arithmetic operations. 
We will break down into pieces. We'll do a step by step load of the function :math:`f(x, y)`.

The first thing to do is to define the `registers of wires <https://pennylane.ai/qml/demos/tutorial_how_to_use_registers/>`_
we will work with:"""

import pennylane as qml

# we indicate the name of the registers and their number of qubits
wires = qml.registers({"x": 4, "y":4, "output":5,"work_wires": 4})

######################################################################
# Before we start applying our arithmetic operators, we can initialize specific values to :math:`x` and :math:`y`
# which will help us to check that the results obtained are correct.

def prepare_initial_state(x,y):
    qml.BasisState(x, wires=wires["x"])
    qml.BasisState(y, wires=wires["y"])


dev = qml.device("default.qubit", shots=1)
@qml.qnode(dev)
def circuit(x,y):
    prepare_initial_state(x, y)
    return [qml.sample(wires=wires[name]) for name in ["x", "y", "output"]]

output = circuit(x=1,y=4)

print("x register: ", output[0])
print("y register: ", output[1])
print("output register: ", output[2])

######################################################################
# In this example we are setting :math:`x=1` and :math:`y=4` whose binary representation are :math:`[0 0 0 1]` and :math:`[0 1 0 0]` respectively.
#
# Now, the first step to load :math:`f(x, y) =4 + 3xy + 5 x+ 3 y` will be
# to add the constant :math:`4` by using the Inplace addition operator :class:`~.pennylane.Adder` directly on the output wires:
#
# .. math::
#
#   \text{Adder}(k) |w \rangle = | w+k \rangle.
#
# In our example, we are considering the input :math:`|w\rangle = |0\rangle` and :math:`k = 4`.
# Let's see how it looks like in code:

@qml.qnode(dev)
def circuit(x,y):

    prepare_initial_state(x, y)        #    |x> |y> |0>
    qml.Adder(4, wires["output"])      #    |x> |y> |4>

    return qml.sample(wires=wires["output"])

print(circuit(x=1,y=4))

######################################################################
# We obtained the state [0 0 1 0 0], i.e. :math:`4`, as expected.
# Note that we have not used the :math:`x` and :math:`y` wires for the moment since the constant term does not depend on these values.
#
# The next step will be to add the term :math:`3xy` by using the 
# `Inplace multiplication <https://docs.pennylane.ai/en/stable/code/api/pennylane.Multiplier.html>`_
#
# .. math::
#
#   \text{Multiplier}(k) |w \rangle = | kw \rangle,
#
# and the `Outplace multiplication <https://docs.pennylane.ai/en/stable/code/api/pennylane.OutMultiplier.html>`_
#
# .. math::
#
#   \text{OutMultiplier} |w \rangle |z \rangle |0 \rangle = |w \rangle |z \rangle |wz \rangle.
#
# To do this, we first turn :math:`|x\rangle` into :math:`|3x\rangle` with the Inplace multiplication. After that
# we will multiply the result by :math:`|y\rangle`
# using the Outplace operator:

def adding_3xy():
    # |x> --->  |3x>
    qml.Multiplier(3, wires["x"], work_wires=wires["work_wires"])

    # |3x>|y>|0> ---> |3x>|y>|3xy>
    qml.OutMultiplier(wires["x"], wires["y"], wires["output"])

    # We return the x-register to its original value
    # |3x>|y>|3xy>  ---> |x>|y>|3xy>
    qml.adjoint(qml.Multiplier)(3, wires["x"], work_wires=wires["work_wires"])

@qml.qnode(dev)
def circuit(x,y):

    prepare_initial_state(x, y)    #    |x> |y> |0>
    qml.Adder(4, wires["output"])  #    |x> |y> |4>
    adding_3xy()                   #    |x> |y> |4 + 3xy>

    return qml.sample(wires=wires["output"])

print(circuit(x=1,y=4))

######################################################################
# Nice! The state [1 0 0 0 0] represents :math:`4+3xy = 4 + 12 =16`.
#
# The final step to compute :math:`f(x, y)` is to generate the monomial terms :math:`5x` and :math:`3y` using the previously employed
# :class:`~.pennylane.Multiplier`. These terms are then added to the output register using the :class:`~.pennylane.OutAdder` operator:
#
# .. math::
#
#   \text{OutAdder} |w \rangle |z \rangle |0 \rangle = |w \rangle |z \rangle |w + z \rangle,
#
# where in our case, :math:`|w\rangle` and :math:`|z\rangle` are :math:`|5x\rangle` and :math:`|3y\rangle` respectively.
#

def adding_5x_3y():

    # |x>|y> --->  |5x>|3y>
    qml.Multiplier(5, wires["x"], work_wires=wires["work_wires"])
    qml.Multiplier(3, wires["y"], work_wires=wires["work_wires"])

    # |5x>|3y>|0> --->  |5x>|3y>|5x + 3y>
    qml.OutAdder(wires["x"], wires["y"], wires["output"])

    # We return the x and y registers to its original value
    # |5x>|3y>|5x + 3y> --->  |x>|y>|5x + 3y>
    qml.adjoint(qml.Multiplier)(5, wires["x"], work_wires=wires["work_wires"])
    qml.adjoint(qml.Multiplier)(3, wires["y"], work_wires=wires["work_wires"])


@qml.qnode(dev)
def circuit(x,y):

    prepare_initial_state(x, y)    #    |x> |y> |0>
    qml.Adder(4, wires["output"])  #    |x> |y> |4>
    adding_3xy()                   #    |x> |y> |4 + 3xy>
    adding_5x_3y()                 #    |x> |y> |4 + 3xy + 5x + 3y>


    return qml.sample(wires=wires["output"])

print(circuit(x=1,y=4))

######################################################################
# The result doesn't seem right, as we expect to get :math:`f(x=1, y=4) = 33`. Can you guess why?
#
# The issue is overflow. The number 33 exceeds what can be represented with the 5 wires defined in `wires["output"]`.
# With 5 wires, we can only represent numbers up to :math:`2^5 = 32`. Anything larger is reduced modulo :math:`2^5`.
# Quantum arithmetic is modular, so every operation is mod-based.
#
# To fix this and get :math:`f(x=1, y=4) = 33`, we could just add one more wire to the output register.
#

wires = qml.registers({"x": 4, "y": 4, "output": 6,"work_wires": 4})
print(circuit(x=1, y=4))

######################################################################
# Now we get the correct result where :math:`[1 0 0 0 0 1]` is the binary representation of :math:`33`.

######################################################################
# Using OutPoly
# ~~~~~~~~~~~~~
# In the last section, we showed how to use different arithmetic operations to load 
# a function onto a quantum computer. But what if I told you there's an easier way to do all this using just one
# PennyLane function that handles the arithmetic for you? Pretty cool, right? I'm talking about :class:`~.pennylane.OutPoly`. 
# This handy operator lets you load polynomials directly into quantum states, taking care of all the arithmetic in one go. 
# Let's check out how to load a function like :math:`f(x, y)` using :class:`~.pennylane.OutPoly`.
#
# Let's first start by explicitly defining our function:

def f(x, y):
   return 4 + 3*x*y + 5*x + 3*y

######################################################################
# Now, we load it into a quantum circuit.

######################################################################

@qml.qnode(dev)
def circuit_with_Poly(x,y):

   prepare_initial_state(x, y)
   qml.OutPoly(f, registers_wires=[wires["x"], wires["y"], wires["output"]])
   
   return qml.sample(wires = wires["output"])

print(circuit_with_Poly(x=1,y=4))

######################################################################
# Eureka! We've just seen how much easier it can be to implement arithmetic operations in one step. 
# Now, it's up to you to decide, depending on the problem you're tackling, whether to go for the versatility 
# of defining your own arithmetic operations or the convenience of using the :class:`~.pennylane.OutPoly` function.

######################################################################
# Conclusion
# ------------------------------------------
# Understanding and implementing quantum arithmetic is a key step toward unlocking the full potential
# of quantum computing. While it may not replace classical efficiency for simple tasks, its role in complex algorithms 
# is undeniable. By leveraging tools like :class:`~.pennylane.OutPoly`, you can streamline the coding of your quantum algorithms. So, 
# whether you choose to customize your arithmetic operations or take advantage of the built-in convenience offered by PennyLane 
# operators, you're now equipped to tackle the exciting quantum challenges ahead.
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

