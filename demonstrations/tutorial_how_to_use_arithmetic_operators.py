r"""
How-to use Quantum Arithmetic Operators
=======================================

Classical computers handle arithmetic operations like addition, subtraction, multiplication, and exponentiation with ease. 
For instance, you can multiply two numbers on your phone in milliseconds!

Quantum computers can handle these operations too, but their true value lies beyond basic calculations. While we don't plan to use 
quantum computers as calculators for everyday arithmetic, these operations play a crucial role in more advanced quantum algorithms, 
serving as fundamental building blocks in their design and execution. For example:

1. In Shor's algorithm quantum arithmetic is crucial for performing modular exponentiation [#shor_exp]_. 

2. Grover's algorithm might need to use quantum arithmetic to construct oracles, as shown in [#demo_qft_arith]_.

3. Loading functions into quantum computers often requires several quantum arithmetic operations [#sanders]_.

With PennyLane, you will see how easy it is to build these operations as subroutines for your quantum algorithms!


InPlace and OutPlace arithmetic operations
------------------------------------------
Let's begin by defining the terms "Inplace" and "Outplace" in the context of arithmetic operators. 
Inplace operators, like the Adder and Multiplier, directly modify the original quantum state by updating a 
specific register's state. In contrast, Outplace operators, such as the OutAdder and OutMultiplier, 
combine multiple states and store the result in a new register, leaving the original states unchanged.

In quantum computing, all arithmetic operations are inherently modular. This means that the result of any 
operation is reduced modulo :math:mod, where the default is :math:mod=2^n, with :math:n being the number of 
qubits in the register.Quantum states can represent numbers up to this limit, which is why PennyLane uses it 
as the default modulo for arithmetic operations. However, users can specify a custom value smaller than this 
default. It's important to keep this modular behavior in mind when working with quantum arithmetic, as using 
too few qubits in a quantum register could lead to overflow errors. We will come back to this point later.

Next, we will explore how to define and implement addition and multiplication operators in PennyLane.

Let's start by defining the Inplace and Outplace arithmetic addition operations. 

Addition operators
~~~~~~~~~~~~~~~~~~

There are two addition operators in PennyLane: the :class:`~.pennylane.Adder` and the :class:`~.pennylane.OutAdder`.

The :class:`~.pennylane.Adder` performs an **Inplace** operation, adding an integer value :math:`k` to the state of the wires :math:`|w \rangle`. It is defined as:

.. math::

   \text{Adder}(k) |w \rangle = | w+k \rangle.

On the other hand, the :class:`~.pennylane.OutAdder` performs an **Outplace** operation, where the states of two 
wires, :math:`|x \rangle` and :math:`|y \rangle` are 
added together and the result is stored in a third wire:

.. math::

   \text{OutAdder} |x \rangle |y \rangle |0 \rangle = |x \rangle |y \rangle |x + y \rangle.

Let's see how to implement these operators in Pennylane.

The first thing to do is to define the `registers of wires <https://pennylane.ai/qml/demos/tutorial_how_to_use_registers/>`_
we will work with:
"""

import pennylane as qml

# we indicate the name of the registers and their number of qubits
wires = qml.registers({"x": 4, "y":4, "output":6,"work_wires": 4})

######################################################################
# Now, we write a circuit to prepare the state :math:`|x \rangle|y \rangle|0 \rangle`, since it will be needed for the Outplace 
# operation, where we initialize specific values to :math:`x` and :math:`y`. Note that in this example we use basic states, but
# you could introduce any quantum state as input.

def prepare_initial_state(x,y):
    qml.BasisState(x, wires=wires["x"])
    qml.BasisState(y, wires=wires["y"])

dev = qml.device("default.qubit", shots=1)
@qml.qnode(dev)
def circuit(x,y):
    prepare_initial_state(x, y)
    return [qml.sample(wires=wires[name]) for name in ["x", "y", "output"]]

output = circuit(x=1,y=4)

def state_to_decimal(binary_array):
    # Convert a binary array to a decimal number
    return sum(bit * (2 ** idx) for idx, bit in enumerate(reversed(binary_array)))

print("x register: ", output[0]," which represents the number ", state_to_decimal(output[0]))
print("y register: ", output[1]," which represents the number ", state_to_decimal(output[1]))
print("output register: ", output[2]," which represents the number ", state_to_decimal(output[2]))

######################################################################
# In this example we are setting :math:`x=1` and :math:`y=4` and checking the results are as expected.
# Note that we are sampling from the circuit instead of returning the quantum state to demonstrate 
# its immediate applicability to hardware. With a single shot, the circuit produces the expected state.
#
# Now we can implement an example for the :class:`~.pennylane.Adder`. We will add the integer :math:`4` to the ``wires["x"]`` register:

@qml.qnode(dev)
def circuit():

    # |0> --->  |0+4>
    qml.Adder(4, wires["x"])    

    return qml.sample(wires=wires["x"])

print(circuit(), " which represents the number ", state_to_decimal(circuit()))

######################################################################
# We obtained the result :math:`4`, as expected. At this point, it's worth taking a moment to look
# at the decomposition of the circuit into quantum gates and operators. 

fig, _ = qml.draw_mpl(circuit, decimals = 2, style = "pennylane", level='device')()
fig.show()

######################################################################
# Taking a look at the decomposition of :class:`~.pennylane.Adder` we can see that the addition is performed 
# in the Fourier basis. This includes a QFT transformation, followed by rotations to perform the addition, and 
# concludes with an inverse QFT transformation. A more detailed explanation on the decomposition of arithmetic operators can be found in
# [#demo_qft_arith]_. 
#
# Now, let's see an example for the :class:`~.pennylane.OutAdder` operator to add the states 
# :math:`|x \rangle` and :math:`|y \rangle` to the output register.

@qml.qnode(dev)
def circuit(x,y):

    prepare_initial_state(x, y)                                #    |x> |y> |0>
    qml.OutAdder(wires["x"], wires["y"], wires["output"])      #    |x> |y> |x+y>

    return qml.sample(wires=wires["output"])

print(circuit(x=1,y=4), " which represents the number ", state_to_decimal(circuit(x=1,y=4)))

######################################################################
# We obtained the result :math:`5`, just as we expected.
# 
# Multiplication  operators
# ~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# There are two multiplication operators in PennyLane: the :class:`~.pennylane.Multiplier` and the :class:`~.pennylane.OutMultiplier`.
# The class :class:`~.pennylane.Multiplier` performs an **Inplace** operation, multiplying the state of the wires :math:`|w \rangle` by an integer :math:`k`. It is defined as:
#
# .. math::
#
#   \text{Multiplier}(k) |w \rangle = | kw \rangle,
#
# while the :class:`~.pennylane.OutMultiplier` performs an **Outplace** operation, where the states of two wires,
# :math:`|x \rangle` and :math:`|y \rangle`, 
# are multiplied together and the result is stored in a third register:
#
# .. math::
#
#   \text{OutMultiplier} |x \rangle |y \rangle |0 \rangle = |x \rangle |y \rangle |xy \rangle.
#  
# We proceed to implement these operators in PennyLane.
#
# First, let's see an example for the :class:`~.pennylane.Multiplier` operator. We will multiply the state of the wire :math:`|x \rangle=|1 \rangle` by the integer term,
# for instance :math:`k=3`:

@qml.qnode(dev)
def circuit():

    qml.BasisState(1, wires=wires["x"])                                #    |1>                                     
    qml.Multiplier(3, wires["x"], work_wires=wires["work_wires"])      #    |1·3> 

    return qml.sample(wires=wires["x"])

print(circuit(), " which represents the number ", state_to_decimal(circuit()))

######################################################################
# We got the expected result of :math:`3`.
#
# Now, let's look at an example using the :class:`~.pennylane.OutMultiplier` operator to multiply the states :math:`|x \rangle` and
# :math:`|y \rangle`, storing the result in the output register.

@qml.qnode(dev)
def circuit(x,y):

    prepare_initial_state(x, y)                                     #    |x> |y> |0>
    qml.OutMultiplier(wires["x"], wires["y"], wires["output"])      #    |x> |y> |xy>

    return qml.sample(wires=wires["output"])

print(circuit(x=1,y=4), " which represents the number ", state_to_decimal(circuit(x=1,y=4)))

######################################################################
# Nice! 
# 
# After going through these educational examples, you might be curious if it's possible to 
# use these operations for subtraction or division in a quantum computer. The answer is yes!
# This is possible through the :func:`~.pennylane.adjoint` operation, which allows us to reverse a multiplication or addition operation, hence enabling
# modular division and subtraction. We will see this in action in the following section.
#
# Loading a polynomial into a quantum computer
# --------------------------------------------
#
# Now that you are familiar with these operations, let's take it a step further and see how we can use them for something practical. 
# We will explore how to implement a polynomial function on a quantum computer using basic arithmetic.
# In particular, we will take as an example the function :math:`f(x,y)= 4 + 3xy + 5 x+ 3 y` where the variables :math:`x` and :math:`y`
# are integer values. Therefore, the operator we want to build is:
#
# .. math::
# 
#    U|x\rangle |y\rangle |0\rangle = |x\rangle |y\rangle |4 + 3xy + 5x + 3y\rangle,
# 
# where :math:`x` and :math:`y` are the binary representations of the integers on which we want to apply the function.
#
# We will show how to load this function in two different ways: first, by concatenating simple modular arithmetic operators,
# and finally, using the :class:`~.pennylane.OutPoly` operator.
#
# Concatenating arithmetic operations
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Let's start by defining the arithmetic operations to load the function :math:`f(x,y) = 4 + 3xy + 5x + 3y` into a quantum state.
#
# First, we need to define a function that will add the term :math:`3xy` to the output register. We will utilize
# the :class:`~.pennylane.Multiplier` and :class:`~.pennylane.OutMultiplier` operators for this. Note that we will need to use the
# adjoint function to undo certain multiplications and clean up the input registers after performing the operations.

def adding_3xy():
    # |x> --->  |3x>
    qml.Multiplier(3, wires["x"], work_wires=wires["work_wires"])

    # |3x>|y>|0> ---> |3x>|y>|3xy>
    qml.OutMultiplier(wires["x"], wires["y"], wires["output"])

    # We return the x-register to its original value using the adjoint operation
    # |3x>|y>|3xy>  ---> |x>|y>|3xy>
    qml.adjoint(qml.Multiplier)(3, wires["x"], work_wires=wires["work_wires"])

######################################################################
# Then we need to add the term :math:`5x + 3y` to the output register, which can be done by using the
# :class:`~.pennylane.Multiplier` and :class:`~.pennylane.OutAdder` operators.
def adding_5x_3y():

    # |x>|y> --->  |5x>|3y>
    qml.Multiplier(5, wires["x"], work_wires=wires["work_wires"])
    qml.Multiplier(3, wires["y"], work_wires=wires["work_wires"])

    # |5x>|3y>|0> --->  |5x>|3y>|5x + 3y>
    qml.OutAdder(wires["x"], wires["y"], wires["output"])

    # We return the x and y registers to its original value using the adjoint operation
    # |5x>|3y>|5x + 3y> --->  |x>|y>|5x + 3y>
    qml.adjoint(qml.Multiplier)(5, wires["x"], work_wires=wires["work_wires"])
    qml.adjoint(qml.Multiplier)(3, wires["y"], work_wires=wires["work_wires"])

######################################################################
# In this functions, we showed how to undo the multiplication by using the adjoint operation, which in this case is useful to clean
# the input registers ``wires["x"]`` and ``wires["y"]`` after the operation.
#
# Now we can combine all these functions to load the function :math:`f(x,y)= 4 + 3xy + 5 x+ 3 y` into a quantum state.
@qml.qnode(dev)
def circuit(x,y):

    prepare_initial_state(x, y)    #    |x> |y> |0>
    qml.Adder(4, wires["output"])  #    |x> |y> |4>
    adding_3xy()                   #    |x> |y> |4 + 3xy>
    adding_5x_3y()                 #    |x> |y> |4 + 3xy + 5x + 3y>

    return qml.sample(wires=wires["output"])

print(circuit(x=1,y=4), " which represents the number ", state_to_decimal(circuit(x=1,y=4)))

######################################################################
# Cool, we get the correct result 33!
#
# At this point, it's interesting to consider what would happen if we had chosen a smaller number of wires for the output.
# For instance, if we had selected one less wire for the output, we would have obtained the result :math:`33 \mod 2^5 = 1`.

wires = qml.registers({"x": 4, "y": 4, "output": 5})
print(circuit(x=1,y=4), " which represents the number ", state_to_decimal(circuit(x=1,y=4)))

######################################################################
# With one less wire, we get :math:`1`, just like we predicted. Remember, we are working with modular arithmetic!
#
# Using OutPoly
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# In the last section, we showed how to use different arithmetic operations to load 
# a function onto a quantum computer. But what if I told you there's an easier way to do all this using just one
# PennyLane function that handles the arithmetic for you? I'm talking about :class:`~.pennylane.OutPoly`. 
# This handy operator lets you load polynomials directly into quantum states, taking care of all the arithmetic in one go. 
# Let's check out how to load a function like :math:`f(x, y)` using :class:`~.pennylane.OutPoly`.
#
# We will start by explicitly defining our function:

def f(x, y):
   return 4 + 3*x*y + 5*x + 3*y

######################################################################
# Now, we load it into a quantum circuit.

######################################################################

wires = qml.registers({"x": 4, "y":4, "output":6})
@qml.qnode(dev)
def circuit_with_Poly(x,y):

   prepare_initial_state(x, y)
   qml.OutPoly(
       f, 
       x_wires = wires["x"],
       y_wires = wires["y"],
       output_wires = wires["output"])
   
   return qml.sample(wires = wires["output"])

print(circuit_with_Poly(x=1,y=4), " which represents the number ", state_to_decimal(circuit_with_Poly(x=1,y=4)))

######################################################################
# Eureka! We've just seen how much easier it can be to implement arithmetic operations in one step. 
# Now, it's up to you to decide, depending on the problem you are tackling, whether to go for the versatility 
# of defining your own arithmetic operations or the convenience of using the :class:`~.pennylane.OutPoly` function.

######################################################################
# Conclusion
# ------------------------------------------
# Understanding and implementing quantum arithmetic is a key step toward unlocking the full potential
# of quantum computing. By leveraging tools like :class:`~.pennylane.OutPoly`, you can streamline the coding of your quantum algorithms. So, 
# whether you choose to customize your arithmetic operations or take advantage of the built-in convenience offered by PennyLane 
# operators, you are now equipped to tackle the exciting quantum challenges ahead.
#
# References
# ----------
# 
# .. [#shor_exp]
#
#     Robert L Singleton Jr
#     "Shor's Factoring Algorithm and Modular Exponentiation Operators.",
#     `arXiv:2306.09122 <https://arxiv.org/abs/2306.09122/>`__, 2023.
#
# .. [#demo_qft_arith]
#
#     Guillermo Alonso
#     "Basic arithmetic with the quantum Fourier transform (QFT).",
#     `Pennylane: Basic arithmetic with the quantum Fourier transform (QFT) <https://pennylane.ai/qml/demos/tutorial_qft_arithmetics/>`__, 2024.
#
#  .. [#sanders]
#
#     Yuval R. Sanders, Guang Hao Low, Artur Scherer, Dominic W. Berry
#     "Black-box quantum state preparation without arithmetic.",
#     `arXiv:1807.03206 <https://arxiv.org/abs/1807.03206/>`__, 2018.
#
# About the authors
# -----------------

