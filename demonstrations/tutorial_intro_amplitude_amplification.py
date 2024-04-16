r"""Intro to Amplitude Amplification
=============================================================


Search problems have been with us since the dawn of time and finding efficient ways to perform this task is extremely useful.
It is surprising that it was not until the 17th century that we realized that sorting words alphabetically in a
dictionary could make searching easier.

However, the problem becomes complicated again when, given a word definition, we want to find that word in the dictionary.
We have lost the order and there is no possible classical strategy to help us do this quickly. Could we achieve it quantumly?


Preparing the problem
-----------------------

Taking a look to this problem from the point of view of quantum computing, our goal is to generate (and therefore, find)
an unknown state :math:`|\phi\rangle`, which, referring to the previous example, would be the word we just know the definition.

Having no information at all, we have few options, so a good first approach is to generate a state :math:`U|0\rangle := |\Psi\rangle`
with the intention that it contains some amount of the searched state :math:`|\phi\rangle`. Generically we can represent
:math:`|\Psi\rangle` in the computational basis as:

.. math::
    |\Psi\rangle = \sum_i c_i |i\rangle,

but we could represent it with another basis if necessary. In fact, it is not difficult to show that there exists one
where :math:`|\phi\rangle` is an element of such a basis and :math:`|\phi^{\perp}\rangle` is another element orthogonal
to the previous one such that:

.. math::
    |\Psi\rangle = \alpha |\phi\rangle + \beta |\phi^{\perp}\rangle,

where :math:`\alpha, \beta \in \mathbb{R}`.

.. note::

    An example of :math:`|\Psi\rangle` could be the uniform superposition of all the possible words since we know that,
    at least, the one we are looking for is included there. Also in that example :math:`|\phi^{\perp}\rangle` is the
    uniform superposition of the words we are not interested in and :math:`\alpha` can be calculated as :math:`\sqrt{\frac{1}{n}}` where :math:`n` is the size of the dictionary.

A great advantage of representing the state :math:`|\Psi\rangle` in this way is that we can now visualize it
in a two-dimensional space:

.. figure:: ../_static/demonstration_assets/intro_amplitude_amplification/ampamp1.jpeg
    :align: center
    :width: 60%
    :target: javascript:void(0)
Our goal, therefore, is to find an operator that moves the initial vector as close to :math:`|\phi\rangle` as we can.

Finding the operator
-----------------------

Ideally, if we could create a rotational gate in this subspace, we would be done. We would simply have to apply a certain
angle to our initial state and we would arrive at :math:`|\phi\rangle`. However, without information about that state,
it is really hard to think that this is possible. This is where a simple but great idea is born: What if instead of
rotations we think of reflections?

Reflection is a well studied operator and can help us to move our state in a controlled way through this subspace.
Looking at our two-dimensional representation we could think of three possible reflections: with respect to
:math:`|\phi\rangle`, :math:`|\phi^{\perp}\rangle` or :math:`|\Psi\rangle`.

With a little intuition we can see that there is a sequence of two reflections that really meets our objective.
The first is the reflection with respect to :math:`|\phi^{\perp}\rangle` and the second with respect to :math:`|\Psi\rangle`.
Let's go step by step:

.. figure:: ../_static/demonstration_assets/intro_amplitude_amplification/ampamp2.jpeg
    :align: center
    :width: 60%
    :target: javascript:void(0)
After applying this reflection it seems that we are moving away from our objective, why do that?
There is a nice phrase that says that sometimes it is necessary to take a step backwards in order to take two steps
forward, and that is exactly what we will do.
The :math:`|\phi^{\perp}\rangle` reflection  may seem somewhat complex to create since we do not have access to such a state.
However, the operator is well defined:

.. math::
     \begin{cases}
     R_{\phi^{\perp}}|\phi\rangle = -|\phi\rangle, \\
     R_{\phi^{\perp}}|\phi^{\perp}\rangle = |\phi^{\perp}\rangle.
     \end{cases}

This operator must be able to change the sign of the solution state.
The way to build it depends on the problem, but in general it is just a classic check:
if the given state meets the properties of being a solution, we change its sign.

.. note::

    In the example we have been working on, this operator would take a word as input, look up its dictionary definition
    and if it matches ours, it applies a phase to that state.

The second reflection is the one with respect to :math:`|\Psi\rangle`. This operator is much easier to build since
we know the gate :math:`U` that generate it. This can be built directly in PennyLane with :class:`~.pennylane.Reflection`.

.. figure:: ../_static/demonstration_assets/intro_amplitude_amplification/ampamp3.jpeg
    :align: center
    :width: 60%
    :target: javascript:void(0)


These two reflections are equivalent to rotate the state :math:`2\theta` degrees, where :math:`\theta` is the initial
angle that forms our state. To approach the target state, we must perform this rotation :math:`\text{iters}` times where:

.. math::
    \text{iters} \sim \frac{\pi}{4 \arcsin \alpha}-\frac{1}{2}.

Time to code
-----------------------

We are going to work with a simplified version of the dictionary problem. We will have :math:`8` different words that
we represent with :math:`3` qubits and to each one we assign a different definition. For simplicity, the definition
will be determined by a binary vector of size :math:`5`. Our dictionary will be defined as follows:

"""

dictionary = {
    # (word): (definition),
    (0,0,0): (1,1,0,0,0),
    (0,0,1): (0,0,1,0,0),
    (0,1,0): (0,0,1,1,0),
    (0,1,1): (0,0,0,1,1),
    (1,0,0): (0,0,1,0,1),
    (1,0,1): (1,1,1,0,1),
    (1,1,0): (0,0,0,0,0),
    (1,1,1): (0,0,1,1,1)
}

definition = (0,0,1,0,1)

##############################################################################
# Our goal is to find the word whose definition is :math:`(0,0,1,0,1)`.
# We are going to choose the superposition of all possible words as the initial state :math:`|\Psi\rangle`.
# This something we can do by applying three Hadamard gates.

import pennylane as qml
import matplotlib.pyplot as plt
plt.style.use('pennylane.drawer.plot')

@qml.prod
def U():
    for wire in range(3):
        qml.Hadamard(wires=wire)

dev = qml.device("default.qubit")

@qml.qnode(dev)
def circuit():
    U()
    return qml.state()

output = circuit()[:8].real

basis = ["|000⟩","|001⟩", "|010⟩", "|011⟩", "|100⟩","|101⟩", "|110⟩", "|111⟩"]
plt.bar(basis, output)
plt.ylim(-0.4, 0.9)
plt.show()





##############################################################################
# The next step is to reflect on the :math:`|\phi^{\perp}\rangle` state, i.e., to change sign only to the solution state.
# To do this we must first define a dictionary operator such that:
#
# .. math::
#     \text{Dic}|\text{word}\rangle|0\rangle = |\text{word}\rangle|\text{definition}\rangle
#

def Dic():
    for word, definition in dictionary.items():
        qml.ctrl(qml.BasisEmbedding(definition, wires = range(3,8)), control=range(3), control_values=word)

##############################################################################
# .. note::
#
#     This is a didactic example, this operator has not been built in an efficient way since it has a complexity :math:`\mathcal{O}(n)`.
#     Therefore, we are not taking advantage of the quadratic advantage of the algorithm.

# With this operator we can now define the searched reflection: we simply access the definition of each word and change
# the sign of the searched word.

@qml.prod
def R_perp():

    # Apply the dictionary operator
    Dic()

    # Flip the sign of the searched definition (therefore, the sign of the searched word)
    qml.FlipSign(definition, wires=range(3, 8))

    # Set auxiliar qubits to |0>
    qml.adjoint(Dic)()


@qml.qnode(dev)
def circuit():

    # Generate initial state
    U()

    # Apply reflection
    R_perp()
    return qml.state()


output = circuit()[0::2 ** 5].real
plt.bar(basis, output)
plt.ylim(-0.4, 0.9)
plt.show()

##############################################################################
# Great, we have flipped the sign of the searched word without knowing what it is, simply by making use of its
# definition. In the literature this operator is known as the Oracle.
#
# The next step is to reflect on the :math:`|\Psi\rangle` state.


def R_psi():
    qml.Reflection(U())

@qml.qnode(dev)
def circuit():

    # Generate initial state
    U()

    # Apply the two reflections
    R_perp()
    R_psi()

    return qml.state()


output = circuit()[0::2 ** 5].real
plt.bar(basis, output)
plt.ylim(-0.4, 0.9)
plt.show()

##############################################################################
# After applying the reflections, we have amplified the desired state. The combination of this two reflections is
# implemented in PennyLane as :class:`~.AmplitudeAmplification`, template that we will use to see the evolution of the
# state as a function of the number of iterations.

@qml.qnode(dev)
def circuit(iters):

    # Apply the initial state
    U()

    # Apply the two reflections iters times
    qml.AmplitudeAmplification(U = U(), O = R_perp(), iters = iters)

    return qml.probs(wires = range(3))

fig, axs = plt.subplots(2, 2)
for i in range(4):
    output = circuit(iters=i)
    ax = axs[i // 2, i % 2]
    ax.bar(basis, output)
    ax.set_ylim(0, 1)
    ax.set_title(f"Iteration {i}")

plt.tight_layout()
plt.subplots_adjust(bottom=0.3)
plt.show()

##############################################################################
# As the number of iterations increases, the probability of success increases as well, but be careful not to overdo
# it or the results will start to get worse. This phenomenon is known as the "overcooking" of the state and is a
# consequence rotate the state too much.
#
#
# Oblivious amplitude amplification
# ---------------------------------
# Amplitude Amplification, as we have shown, is a technique that allows us to generate desired states. Thus, at this
# point, we may ask ourselves a very intriguing question: can we use this method to generate desired operators?
#
# In this case, our task will be to generate a particular unitary operator :math:`V`. To do this, we know an operator
# :math:`U` that generates part of this operator:
#
# .. math::
#   U(\ket{0} \otimes \mathbb{I}) = \alpha \ket{0} \otimes V + \beta \ket{0^\perp} \otimes W,
#
# where :math:`|0^\perp\rangle` is a state orthogonal to :math:`\0\rangle`. To do that we will follow the idea of the
# main algorithm. The first step is to create a two-dimensional subspace. This task is not obvious since we are working
# with operators instead of vectors. For this reason, for convenience, we will multiply by any state :math:`\ket{\phi}`.
# In this way, we have that:
#
# .. figure::
#   ../../_static/quantum_learning/oblivious_amplitude_amplification_1.jpeg
#   :width: 50%
#   :align: center
#   :target: javascript:void(0);
#
# The first reflection is carried out with respect to the X-axis, and even without knowing anything about
# :math:`|\phi\rangle` or :math:`V`, this is something that can be accomplished by changing the sign of those states
# whose first register is :math:`|0\rangle`, that is :math:`R_0 \otimes \mathbb{I}`.
#
# .. figure::
#   ../../_static/quantum_learning/oblivious_amplitude_amplification_2.jpeg
#   :width: 50%
#   :align: center
#   :target: javascript:void(0);
#
# Complications arise in the second reflection for one reason: we do not know the state :math:`|\phi\rangle`. Somehow,
# the reflection must be performed _oblivious_ to it. To achieve this, we will use an auxiliary subspace, chosen in
# such a way that it facilitates this reflection. In this case, the space will be the result of multiplying all
# vectors by :math:`U^\dagger`. Since multiplications by unitary operators preserve angles, we can represent the two
# spaces in the following way:
#
# .. figure::
#   ../../_static/quantum_learning/oblivious_amplitude_amplification_3.jpeg
#   :width: 80%
#   :align: center
#   :target: javascript:void(0);
#
# The advantage of choosing this new space is that the reflection with respect to the vector of interest is equivalent
# to performing it with respect to :math:`|0\rangle` in the first register. Therefore, the strategy will be to first
# move to the new space by applying :math:`U^\dagger`, then perform the reflection :math:`R_0 \otimes \mathbb{I}`,
# and finally return to the original space by applying :math:`U`. It means:
#
# .. math::
#   R_{\Psi} = U R_0 \otimes \mathbb{I} U^\dagger.
#
# Coding OAA
# ~~~~~~~~~~
#

















##############################################################################
# About the author
# ----------------
