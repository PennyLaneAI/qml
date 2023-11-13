r"""
Error Propagation
======================================

.. meta::
    :property="og:description": Error propagation with PennyLane
    :property="og:image": https://pennylane.ai/qml/_static/brain_board.png

*Authors: Jay Soni, — Posted: 10 November 2023.*

Quantifying the effects of errors / approximations in our gates and how they relate to the error in our
final measurement outcomes is very useful for mordern quantum computing workflows (especially in the 
NISQ era). Typically, these types of computations are performed by hand due to the varity of error metrics
to track and the specific handling of such errors for each sub-routine. To the best of our knowledge, 
there is currently no generally agreed upon systematic approach to tracking and "propagating" errors 
through a quantum workflow. 

Introduction
------------
In this demo, we explore (with little modification to PennyLane) a simple workflow for tracking and 
propagation errors. We take a Hamiltonian simulation problem as an initial case study to work through the
process. Let's begin by creating an object to represent our error object.

We inherit from the abstract :class:`~.error_prop.OperatorError` class which requires implementing two 
abstract methods. The first is the :code:`__add__` method which is responsible for combining 
two erroneous operations in a quantum circuit.
"""

import pennylane as qml
from pennylane import numpy as qnp

from pennylane.error_prop import OperatorError

class SpectralNorm_Error(OperatorError):
    
    def __add__(self, other):
        """Abstract combination function to combine two instances of Spectral Norm error 
        in a circuit. In this case, it is simply additive

        Args:
            other (SpectralNorm_Error): The error in the other operation to be combined.

        Returns:
            SpectralNorm_Error: The final error after combination. 
        """
        return self.__class__(self.error + other.error)  # an instance of the class with combined error 
    
    def get_error(op1, op2):
        """A function to compute the spectral norm error between two operations.

        .. math:: \epsilon = ||U_{true} - U_{approx}||

        Args:
            op1 (.Operator): The target (true) operator we aim to apply.
            op2 (.Operator): The approximate operator we will actually apply.

        Returns:
            float: The spectral norm error.
        """
        return qnp.linalg.norm(qml.matrix(op1) - qml.matrix(op2), ord="fro")  # frobenius norm bounds spectral norm
    
    
###############################################################################
# Next we use PennyLane's built in :class:`.ResourcesOperation` as a mixin to build our 
# approximate operation. By inheriting from the base :code:`TrotterProduct` class, we can 
# allow our approximate operation to be executable in a quantum circuit!
#
# The only additional logic we need to provide, is that of tracking the resources. This is 
# where we would introduce the subroutine specific formulas to compute the error. In this case,
# we use the simple time order bound for the Trotter error.
#

from pennylane.resource import Resources, ResourcesOperation

class My_Approx_Trotter(ResourcesOperation, qml.TrotterProduct):
    
    def resources(self):
        """A method to compute the resources of the operation using 
        its parameters and hyperparameters.

        Returns:
            .Resources: A resources container object with all of the resources to track. 
        """
        n = self.hyperparameters["n"]
        time = self.parameters[0]
        order = self.hyperparameters["order"]
        
        # Time-Order scaling 
        op_error = (time**(order + 1)) / n  
        
        return Resources(                          # Pennylane resources container object for storing resources
            num_wires=self.num_wires,
            num_gates=len(self.decomposition()),
            gate_types={"My_Approx_Trotter": 1},
            gate_sizes={1:1},
            depth=len(self.decomposition()),
            error=SpectralNorm_Error(op_error),
        )


###############################################################################
# Problem:
# --------
# Suppose that we are interested in evolving an initial state under a target hamiltonian and 
# then measuring some quantity of interest from the result. There are many different approaches 
# to approximating the time evolution unitary, having a method to benchmark this would be useful 
# in determining which approach to use under which circumstances. 
#
# Below we define our initial state, target hamiltonian and final measurement:

wires = [0]
dev = qml.device("default.qubit")

# Hamiltonian to time evolve wrt: H = 123 * XY - 45 * ZZ + 0.6 * YX + IZ
H = qml.sum(
    qml.s_prod(123, qml.prod(qml.PauliX(0), qml.PauliY(1))),
    qml.s_prod(-45, qml.prod(qml.PauliZ(0), qml.PauliZ(1))),
    qml.s_prod(0.6, qml.prod(qml.PauliY(0), qml.PauliX(1))),
    qml.s_prod(1, qml.PauliZ(1))
)

def prep_initial_state(wires):
    """Prepare |+> in all qubits."""
    for w in wires:
        qml.Hadamard(w)


@qml.qnode(dev, interface=None)
def circuit(time_evo_op):
    
    prep_initial_state(wires)
    
    qml.apply(time_evo_op)
    
    return qml.expval(qml.prod(qml.PauliZ(0), qml.Hadamard(1)))  # measure < Z(0) @ Hadamard(1) >

###############################################################################
# Now we use the circuit above and execute with two different time evolution sub-routines, 
# one exact, the other approximate: 

# Exact time evolution:
time_evo_op1 = qml.exp(H, coeff=1j)

print(circuit(time_evo_op1))
print(qml.draw(circuit, expansion_strategy="device")(time_evo_op1), "\n\n")


# Approximate time evolution: 
time_evo_op2 = My_Approx_Trotter(H, time=1, order=1, n=1)

print(circuit(time_evo_op2))
print(qml.draw(circuit, expansion_strategy="device")(time_evo_op2))

###############################################################################
# We see that there is a difference in the computed expectation value. We can use the error 
# tracked to bound the error in expectation.
#
# Resource Analysis: 
# ------------------
# The error can be extracted from the circuit in the same way we track resources, using the 
# :func:`~.pennylane.specs` function. Simply query the :code:`"resources"` key of the specs 
# dictionary and extract the error attribute. This will be an instance of the error 
# class we defined above :code:`SpectralNorm_Error`.

circ_resources = qml.specs(circuit)(time_evo_op2)["resources"]  # extract resources from circuit specs
error = circ_resources.error[0]                   # Spectral Norm error propagated through the circuit.


print("Error in expval is: ", abs(circuit(time_evo_op1) - circuit(time_evo_op2)))
print("Which is less than: ", error.error)  # The expected value is correct within 2 * norm(H) * error 

###############################################################################
# Let's use everything we have built and apply it to explore the error scaling of Trotter product 
# formulas of higher order for increasing number of trotter-steps:
#

import matplotlib.pyplot as plt 

time_evo_op1 = qml.exp(H, coeff=1j)

first_order_trotter = []
second_order_trotter = []
error_bound = []

order_lst = [2**i for i in range(1, 9)]

for n in order_lst:
    time_evo_op2 = My_Approx_Trotter(H, time=1, order=1, n=n)
    first_order_trotter.append(abs(circuit(time_evo_op1) - circuit(time_evo_op2)))
    
    time_evo_op3 = My_Approx_Trotter(H, time=1, order=2, n=n)
    second_order_trotter.append(abs(circuit(time_evo_op1) - circuit(time_evo_op3)))

    circ_resources = qml.specs(circuit)(time_evo_op2)["resources"]
    error = circ_resources.error
    
    error_bound.append(error.error)

plt.plot(order_lst, first_order_trotter, "--*", label="1st order trotter")
plt.plot(order_lst, second_order_trotter, "--*", label="2nd order trotter")

plt.plot(order_lst, error_bound, "--*", label="simple order bound")

plt.yscale("log")
plt.ylabel("Error")
plt.xlabel("Trotter-Step")

plt.legend()
plt.show()

###############################################################################
# We can see that the simple error bound is higher than the computed error for both 1st order and 
# 2nd order trotter approximations.
#
# Conclusion
# -------------------------------
# In this demo, we showcased the :class:`~.pennylane.ResourcesOperation`, and the 
# :class:`~.pennylane.error_prop.OperatorError` classes in PennyLane. We explained how to construct 
# a custom resource operation and custom error type. We used this in a simple circuit to track and 
# propagate the error through the circuit to the final measurement. We hope that you can use this 
# tools in cutting edge research workflows to estimate error. 
#
##############################################################################
# About the author
# ----------------
# .. include:: ../_static/authors/jay_soni.txt
