r"""Intro to Amplitude Amplification (and its variants)
=======================================================================

Search problems have been with us since the dawn of time and finding efficient ways to perform this task is extremely useful.
It is surprising that it was not until the 17th century that we realized that sorting words alphabetically in a
dictionary could make searching easier.

However, the problem becomes complicated again when, given only the word definition, we want to find the particular
word with that meaning in the dictionary. We have lost the order and there is no possible classical strategy to help us do this quickly. Could we achieve it quantumly?


Amplitude Amplification
-------------------------

Our goal is to prepare an unknown state :math:`|\phi\rangle`. A good first approach is to use a circuit :math:`U` to generate a state :math:`U|0\rangle := |\Psi\rangle`
that "contains" some amount of the target state :math:`|\phi\rangle`. Generically we can represent
:math:`|\Psi\rangle` in the computational basis as:

.. math::
    |\Psi\rangle = \sum_i c_i |i\rangle,

but we can do better. One choice is to make :math:`|\phi\rangle` an element of the basis. We can then write

.. math::
    |\Psi\rangle = \alpha |\phi\rangle + \beta |\phi^{\perp}\rangle,

where  :math:`|\phi^{\perp}\rangle` is some state orthogonal
to :math:`|\phi\rangle` and :math:`\alpha, \beta \in \mathbb{R}`. 


A great advantage of representing the state :math:`|\Psi\rangle` in this way is that we can now visualize it
in a two-dimensional space. This representation is even simpler than a Bloch sphere since the amplitudes are real numbers --- we can visualize all operations inside a "Bloch circle":

.. figure:: ../_static/demonstration_assets/intro_amplitude_amplification/ampamp1.jpeg
    :align: center
    :width: 60%
    :target: javascript:void(0)

We are going to _amplify_ the amplitude :math:`\alpha` to get closer
to :math:`|\phi\rangle`, hence the name Amplitude Amplification [#ampamp]_. We will try to find an operator that moves the initial vector :math:`|\Psi\rangle` as close to :math:`|\phi\rangle` as
possible.

Finding the operators
~~~~~~~~~~~~~~~~~~~~~

It would be enough if we could create a rotation gate in this subspace, then rotate the initial state counterclockwise by :math:`\pi/2 -\theta` to obtain :math:`|\phi\rangle`.  However, we don't explicitly know :math:`|\phi\rangle`, so it's unclear how this could be done.  This is where a great idea is born: What if instead of rotations we think of reflections?

We could think of reflections with respect to three states: :math:`|\phi\rangle`,  :math:`|\phi^{\perp}\rangle`, or :math:`|\Psi\rangle`.

The main insight of the Amplitude Amplification algorithm is that there is a sequence of **two reflections** that meets our objective.
The first is the reflection with respect to :math:`|\phi^{\perp}\rangle` and the second with respect to :math:`|\Psi\rangle`.
Let's go step by step:

.. figure:: ../_static/demonstration_assets/intro_amplitude_amplification/ampamp2.jpeg
    :align: center
    :width: 60%
    :target: javascript:void(0)

    Reflections with respect to :math:`|\phi^{\perp}\rangle`.

After applying this reflection it seems that we are moving away from :math:`|\phi\rangle`, why do that?
A nice phrase says that sometimes it is necessary to take a step backwards in order to take two steps
forward, and that is exactly what we will do.
The :math:`|\phi^{\perp}\rangle` reflection  may seem somewhat complex to create since we do not have access to such a state.
However, the operator is well-defined:

.. math::
     \begin{cases}
     R_{\phi^{\perp}}|\phi\rangle = -|\phi\rangle, \\
     R_{\phi^{\perp}}|\phi^{\perp}\rangle = |\phi^{\perp}\rangle.
     \end{cases}

This operator must be able to change the sign of the solution state.
The way to build it depends on the problem, but in general it is just a classic check:
if the given state meets the properties of being a solution, we change its sign.



The second reflection is the one with respect to :math:`|\Psi\rangle`. This operator is easier to build since
we know the gate :math:`U` that generate it. This can be built directly in PennyLane with :class:`~.pennylane.Reflection`.

.. figure:: ../_static/demonstration_assets/intro_amplitude_amplification/ampamp3.jpeg
    :align: center
    :width: 60%
    :target: javascript:void(0)

    Reflections with respect to :math:`|\Psi\rangle`.


These two reflections are equivalent to rotating the state by :math:`2\theta` degrees from its original position, where :math:`\theta=\arcsin(\alpha/\beta)` is the
angle that defines the initial state. To amplify the amplitude and approach the target state, we perform this sequence of rotations multiple times. More precisely, we repeat them:

.. math::
    \text{iters} \sim \frac{\pi}{4 \arcsin \alpha}-\frac{1}{2}.

Amplitude Amplification in PennyLane
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
plt.ylim(-0.4, 1)
plt.ylabel("Amplitude")
plt.axhline(0, color='black', linewidth=1)
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
#
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
plt.ylim(-0.4, 1)
plt.ylabel("Amplitude")
plt.axhline(0, color='black', linewidth=1)
plt.show()

##############################################################################
# Great, we have flipped the sign of the searched word without knowing what it is, simply by making use of its
# definition. In the literature this operator is known as the Oracle.
#
# The next step is to reflect on the :math:`|\Psi\rangle` state.


def R_psi():
    # We pass the operator that generates the state on which to reflect.
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
plt.ylim(-0.4, 1)
plt.ylabel("Amplitude")
plt.axhline(0, color='black', linewidth=1)
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

fig, axs = plt.subplots(2, 2, figsize=(14, 10))
for i in range(4):
    output = circuit(iters=i)
    ax = axs[i // 2, i % 2]
    ax.bar(basis, output)
    ax.set_ylim(0, 1)
    ax.set_title(f"Iteration {i}")

plt.tight_layout()
plt.subplots_adjust(bottom=0.1)
plt.axhline(0, color='black', linewidth=1)
plt.show()

##############################################################################
# As the number of iterations increases, the probability of success increases as well, but be careful not to overdo
# it or the results will start to get worse. This phenomenon is known as the "overcooking" of the state and is a
# consequence of rotating the state too much.
#
#
# Oblivious Amplitude Amplification
# ---------------------------------
# Amplitude Amplification, as we have shown, is a technique that allows us to generate desired states. Thus, at this
# point, we may ask ourselves a very intriguing question: can we use this method to generate desired operators?
# This is the idea behind the Oblivious Amplitude Amplification [#oblivious]_ variant.
#
# In this case, our task is to generate a particular unitary operator :math:`V`. To do this, we know an operator
# :math:`U` that generates part of this operator:
#
# .. math::
#   U(|0\rangle \otimes \mathbb{I}) = \alpha |0\rangle \otimes V + \beta |0^\perp\rangle \otimes W,
#
# where :math:`|0^\perp\rangle` is a state orthogonal to :math:`|0\rangle`. To do that we will follow the idea of the
# main algorithm. The first step is to create a two-dimensional subspace but this task is not obvious since we are working
# with operators instead of vectors. For this reason, for convenience, we will multiply by any state :math:`|\phi\rangle`.
# In this way, we have that:
#
# .. figure::
#   ../_static/demonstration_assets/intro_amplitude_amplification/oblivious_amplitude_amplification_1.jpeg
#   :width: 60%
#   :align: center
#   :target: javascript:void(0);
#
# The first reflection is carried out with respect to the X-axis, and even without knowing anything about
# :math:`|\phi\rangle` or :math:`V`, this is something that can be accomplished by changing the sign of those states
# whose first register is :math:`|0\rangle`, that is :math:`R_0 \otimes \mathbb{I}`.
#
# .. figure::
#   ../_static/demonstration_assets/intro_amplitude_amplification/oblivious_amplitude_amplification_2.jpeg
#   :width: 60%
#   :align: center
#   :target: javascript:void(0);
#
# Complications arise in the second reflection for one reason: we can't use :math:`|\phi\rangle` since this is an
# arbitrary state that we have chosen in order to draw the subspace. Somehow,
# the reflection must be performed *oblivious* to it. To achieve this, we will use an auxiliary subspace, chosen in
# such a way that it facilitates this reflection. In this case, the space will be the result of multiplying all
# vectors by :math:`U^\dagger`. Since multiplications by unitary operators preserve angles, we can represent the two
# spaces in the following way:
#
# .. figure::
#   ../_static/demonstration_assets/intro_amplitude_amplification/oblivious_amplitude_amplification_3.jpeg
#   :width: 90%
#   :align: center
#   :target: javascript:void(0);
#
# The advantage of choosing this new space is that the reflection with respect to the vector of interest is equivalent
# to performing it with respect to :math:`|0\rangle` in the first register. Therefore, the strategy will be to first
# move to the new space by applying :math:`U^\dagger`, then perform the reflection :math:`R_0 \otimes \mathbb{I}`,
# and finally return to the original space by applying :math:`U`. It means:
#
# .. math::
#   R_{\Psi} = U \cdot (R_0 \otimes \mathbb{I}) \cdot U^\dagger.
#
# Let's see an example where we have a :math:`U` of the form:
#
# .. math::
#   U(|0\rangle \otimes \mathbb{I}) = \cos\left(\frac{2\pi}{5}\right) |0\rangle \otimes X + \sin\left(\frac{2\pi}{5}\right) |0^\perp\rangle \otimes W.
#
# We will make use of the :class:`~.pennylane.AmplitudeAmplification` template. This time we will have to specify the
# wires of the first register, which we will call ``reflection_wires``.

import numpy as np

@qml.prod
def U():
    qml.RY(4 * np.pi / 5, wires=0)
    qml.ctrl(qml.PauliX(wires=1), control=0, control_values=0)


@qml.qnode(dev)
def circuit(iters):

    # Apply the initial state
    U()

    # Apply the two reflections iters times
    qml.AmplitudeAmplification(
        U=U(),
        O=qml.FlipSign(0, wires=0), # R_0
        reflection_wires=0, # First register
        iters=iters)

    return qml.state()


for iter in range(0, 5):
    print("iters: ", iter)
    print("alpha: ", abs(qml.matrix(circuit, wire_order=[0, 1])(iter)[0, 1]), "\n")

##############################################################################
# The results show that the state is converging to the desired state. I invite you to check that effectively with two
# iterations you get the X gate for any input. Note also that as before, we must be careful not to exceed the number
# of iterations, as you will be overcooking the operator.
#
# Fixed-point Quantum Search
# --------------------------
# Before finishing I would like to comment that there is another variant that you can also use with the same template.
# The Fixed-point quantum search variant [#fixedpoint]_. This technique will allow you to avoid the overcooking problem by using an
# extra qubit. To do this you only need to set ``fixed_point = True`` and select the auxiliary qubit.
# Let's see what happens with the same example as before:

@qml.qnode(dev)
def circuit(iters):

    # Apply the initial state
    U()

    # Apply the two reflections iters times
    qml.AmplitudeAmplification(
        U=U(),
        O=qml.FlipSign(0, wires=0), # R_0
        reflection_wires=0, # First register
        fixed_point=True,
        work_wire = 2,
        iters=iters)

    return qml.state()


alphas = []
range_ = range(1, 30,2)
for iter in range_:
    alphas.append(abs(qml.matrix(circuit, wire_order=[2, 0, 1])(iter)[0, 1]))

plt.plot(range_, alphas)
plt.xlabel("Iterations")
plt.ylabel("Alpha")
plt.show()

##############################################################################
# As we can see, once :math:`\alpha` has taken a high value it will not suffer from overcooking and will remain with a
# controlled value greater than :math:`0.9`. This boundary is something you can control with the ``p_min`` parameter of
# the template.
#
# Conclusion
# ----------
# In this demo we have shown from scratch, the process by which we can find an unknown state.
# Although the most famous algorithm for this is Amplitude Amplification, we have seen that there are very interesting
# variants that are worth having in our toolkit!
#
#
#
# References
# ----------
#
# .. [#ampamp]
#
#     Gilles Brassard, Peter Hoyer, Michele Mosca and Alain Tapp
#     "Quantum Amplitude Amplification and Estimation",
#     `arXiv:quant-ph/0005055 <https://arxiv.org/abs/quant-ph/0005055>`__ (2000)
#
# .. [#fixedpoint]
#
#     Theodore J. Yoder, Guang Hao Low and Isaac L. Chuang
#     "Fixed-point quantum search with an optimal number of queries",
#     `arXiv:1409.3305 <https://arxiv.org/abs/1409.3305>`__ (2014)
#
#
# .. [#oblivious]
#
#    Dominic W. Berry, et al.
#    "Simulating Hamiltonian dynamics with a truncated Taylor series",
#    `arXiv:1412.4687 <https://arxiv.org/pdf/1412.4687.pdf>`__, 2014















##############################################################################
# About the author
# ----------------
