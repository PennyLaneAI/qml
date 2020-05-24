r"""
The Stochastic Parameter-shift Rule
===================================

.. meta::
    :property="og:description": Differentiate any qubit gate with the stochastic parameter-shift rule.
    :property="og:image": https://pennylane.ai/qml/_images/some_image.png # TODO: add image

We demonstrate how the stochastic parameter-shift rule (Banchi and Crooks [#banchi2020]_)
can be used to differentiate arbitrary qubit gates, generalizing the original 
:doc:`parameter-shift rule </glossary/parameter_shift>`_, which applies only for gates of a particular 
(but widely encountered) form.

Background
----------

One of the main ideas encountered in near-term quantum machine learning is the 
:doc:`variational circuit </glossary/variational_circuit>`_. 
Evolving from earlier concepts pioneered by domain-specific algorithms like the 
:doc:`variational quantum eigensolver </demonstrations/tutorial_vqe>`_ and the 
:doc:`quantum approximate optimization algorithm </demonstrations/tutorial_qaoa_maxcut>`_,
this class of quantum algorithms makes heavy use of two distinguishing ingredients: 

i) Gates have free parameters
ii) Expectation values of measurements are taken 

These two ingredients allow one circuit to actually represent an entire _family of circuits_. 
An objective function---encapsulating some problem-specific goal---is built from the expectation values, 
and the circuit's free parameters are progressively tuned to optimize this function. 
At each step, the circuit has the same gate layout, but slightly different parameters, making 
this approach promising to run on constrained near-term devices.

But how should we actually update the circuit's parameters to move us closer to a good output? 
Borrowing a page from classical optimization and deep learning, we can use 
`gradient descent <https://en.wikipedia.org/wiki/Gradient_descent>`_. 
In this general-purpose method, we compute the derivative of a (smooth) function :math:`f` with 
respect to its parameters :math:`\theta`, i.e., its gradient :math:`\nabla_\theta f`. 
Since the gradient always points in the direction of steepest ascent/descent, if we make small updates 
to the parameters according to

.. math::

    \theta \rightarrow \theta - \eta \nabla_\theta f,
    
we can iteratively progress to lower and lower values of the function.

The Parameter-shift Rule
------------------------

In the quantum case, the expectation value of a circuit with respect to an measurement operator 
:math:`\hat{C}` depends smoothly on the the circuit's gate parameters :math:`\theta`. We can write this
expectation value as :math:`\langle \hat{C}(\theta)\rangle`. This means that the derivatives 
:math:`\frac{\partial \langle \hat{C} \rangle}{\partial \theta}` exist and gradient descent can be used. 

Before digging deeper, we will first set establish some basic notation. For simplicity, though a circuit 
may contain many gates, we can concentrate on just a single gate :math:`U` that we want to differentiate
(other gates will follow the same pattern).

.. figure:: ../demonstrations/stochastic_parameter_shift/quantum_circuit.png
    :align: center
    :width: 90%

All gates appearing before :math:`U` can be absorbed into an initial state preparation 
:math:`\vert \psi_0 \rangle`, and all gates appearing after :math:`U` can be absorbed with the measurement
operator :math:`\hat{A}` to make a new effective measurement operator :math:`\hat{A}`.
The expectation value :math:`\hat{A}` in the simpler one-gate circuit is identical to 
the expectation value :math:`\hat{C}` in the larger circuit.

We can also write any unitary gate in the form

.. math::

    U(\theta) = e^{i\theta\hat{V}},
    
where :math:`\hat{V}` is the Hermitian _generator_ of the gate :math:`U`.

Now, how do we actually obtain the numerical values for the derivatives necessary for gradient descent? 

This is where the parameter-shift rule [#li2016], [#mitarai2018], [#schuld2018] enters the story. In short, the parameter-shift rule says that for 
many gates of interest---including all single-qubit gates---we can obtain the value of the derivative 
:math:`\frac{\partial \langle \hat{A}(\theta) \rangle}{\partial \theta}` by subtracting two related 
circuit evaluations:

..math::

   \frac{\partial \langle \hat{A} \rangle}{\partial \theta} = 
   \langle \hat{A}(\theta + \tfrac{\pi}{4}) \rangle -
   \langle \hat{A}(\theta - \tfrac{\pi}{4}) \rangle 
   
.. figure:: ../demonstrations/stochastic_parameter_shift/parameter_shift_circuits.png
    :align: center
    :width: 90%
    
.. note::

    Depending on the convention used, there may be additional constant multipliers appearing in this formula. 
    For example, PennyLane actually uses a convention where most qubit gates are parametrized as 
    :math:`exp(-i\tfrac{\theta}{2}\hat{V})`. This results in a slightly different, but equivalent, form 
    for the parameter-shift rule. Here we use the simplest form, which also tracks the notation from
    [#banchi2020]_.

The parameter-shift rule is _exact_, i.e., the formula for the gradient doesn't involve any approximations. 
For quantum hardware, we can only take a finite number of samples, so we can never determine a circuit's expectation values _exactly_. However, the parameter-shift rule provides the guarantee that it is an _unbiased estimator_, meaning that if we could take a infinite number of samples, it converges to the correct gradient value. 

Let's jump into some code and take a look at the parameter-shift rule in action. 

"""
import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

np.random.seed(143)
angles = np.linspace(0, 2 * np.pi, 50)
dev = qml.device('default.qubit', wires=2)


##############################################################################
# We will consider a very simple circuit, containing just a single-qubit 
# rotation about the x-axis, followed by a measurement along the z-axis.

@qml.qnode(dev)
def rotation_circuit(theta):
    qml.RX(theta, wires=0)
    return qml.expval(qml.PauliZ(0))


##############################################################################
# We will examine the gradient with respect to the parameter :math:`\theta`.
# The parameter-shift prescribes taking the difference of two  circuit 
# evaluations, with a forward/backward shift in angles.
# PennyLane also provides a convenience function ``grad`` to automatically
# compute the gradient. We can use it here for comparison.

def param_shift(theta):
    r_plus = 0.5 * rotation_circuit(theta + np.pi / 2)
    r_minus = 0.5 * rotation_circuit(theta - np.pi / 2)
    return r_plus - r_minus

gradient = qml.grad(rotation_circuit, argnum=0)

expvals = [rotation_circuit(theta) for theta in angles]
grad_vals = [gradient(theta) for theta in angles]
param_shift_vals = [param_shift(theta) for theta in angles]
plt.plot(angles, expvals, 'b', label="Expecation value")
plt.plot(angles, grad_vals, 'r', label="Gradient")
plt.plot(angles, param_shift_vals, 'mx', label="Parameter-shift rule")
plt.legend();


##############################################################################
# We have evaluated the expectation value at all possible values for the angle
# :math:`theta`. By inspection, we can see that the functional dependence is
# :math:`\cos(\theta)`. The parameter-shift evaluations are plotted with x's.
# Again, by inspection, we can see that these have the functional form 
# :math:`-\sin(\theta)`, and they match the values provided by the ``grad``
# function.
#
# The parameter-shift works really nicely for many gates---like the rotaiton
# gate we used in our example above. But it does have constraints. There are 
# some technical conditions that, if a gate satisfies them, we can guarantee
# it has a parameter-shift rule [#schuld2018]. Furthermore, we can derive
# similar parameter-shift recipes for some other gates that _don't_ meet 
# those technical conditions. 
#
# But, in general, the parameter-shift rule is not universally applicable.
# In cases where it doesn't hold (or is not yet known to hold). you would
# either have to decompose the gate into compatible gates, or use an
# alternate estimator for the gradient, e.g., the finite-difference
# approximation. But both of these alternatives can have drawbacks due
# to increased circuit complexity or potential errors in the gradient
# value. If only there was a method that could be used for _any_
# qubit gate.
#
# The Stochastic Parameter-shift Rule
# -----------------------------------
#
# Here's where the stochastic parameter-shift rule makes its appearance
# on the stage. 

# The stochastic parameter-shift rule introduces two new ingredients to
# the parameter-shift recipe:  
#
# i) A random parameter :math:`s`, sampled uniformly from :math:`[0,1]`
#    (this is the origin of the "stochastic" in the name);
# ii) Sandwiching the "shifted" gate applicaion with one additional 
#     gate on each side.
#
# These additions allow the stochastic parameter-shift rule to work 
# for arbitrary qubit gates. Every gate is unitary, which means they 
# have the form :math:`e^{i\theta \hat{G}` for some generator :math:`G`. 
# Additionally, every multi-qubit operator can be expressed as a 
# sum of tensor products of Pauli operators, so let's assume, 
# without loss of generality, the following form for :math:`G`:
#
#  .. math::
#
#      G = \hat{H} + \theta \hat{V}, 
#
# where :math:`\hat{V}` is a "Pauli word", i.e., a tensor 
# product of Pauli operators (e.g., 
# :math:`\hat{Z}_0\otimes\hat{Y}_1) and :math:`\hat{H}` can 
# be an arbitrary linear combination of Pauli-operator
# tensor products. For simplicity, we assume that the parameter
# :math:`\theta` appears only in front of :math:`\hat{V}` (other
# cases can be handled using the chain rule). 
#
# The stochastic parameter-shift rule gives the following recipe for
# computing the gradient of the expectation value 
# :math:`\langle \hat{A} (\theta) \rangle`:
#
# i) Sample a value for the variable :math:`s` uniformly form 
#    :math:`[0,1]`.
# ii) In place of gate :math:`U(\theta)`, apply the following
#     three gates:
#
#     a) :math:`e^{i(1-s)\hat{G}}`
#     b) :math:`e^{+i\tfrac{\pi}{4}\hat{V}}`
#     c) :math:`e^{is\hat{G}}`
#
#     Call the resulting expectation value of :math:`r_+`
#
# iii) Repeat step ii), but flip the sign of the generator
#      in part b). Call the resulting expectation value
#      :math:`r_-`.
#
# The gradient can be obtained from the average value of
# :math:`r_+ - r_-`, i.e.,
#
# .. math::
#
#     \mathbb{E}_{s\in\mathcal{U}[0,1]}[r_+ - r_-]
#
# .. figure:: ../demonstrations/stochastic_parameter_shift/stochastic_parameter_shift_circuit.png
#    :align: center
#    :width: 90%
#
# Let's see this method in action.
#
# Following [#banchi2020], we will use the cross-resonance gate as a
# working example. This gate is defined as
#
# .. math::
#     U_{CR}(t, b, c) = exp\left[ it(\hat{X}\otimes\hat{\mathbb{1} - 
#                                   b\hat{Z}\otimes\hat{X} + 
#                                   c\hat{\mathbb{1}}\otimes\hat{X}
#                                   ) \right]

# First we define some basic Pauli matrices
I = np.eye(2)
X = np.array([[0, 1], [1, 0]])
Z = np.array([[1, 0], [0, -1]])

def Generator(t, b, c):
    # the inputs will show up as Pennylane variables;
    # we have to extract their numerical values
    G = t.val * (np.kron(X, I) - 
        b.val * np.kron(Z, X) + 
        c.val * np.kron(I, X))
    return G
    
# A simple circuit that contains the cross-resonance gate
@qml.qnode(dev)
def crossres_circuit(gate_pars):
    G = Generator(*gate_pars)
    qml.QubitUnitary(expm(-1j * G), wires=[0, 1])
    return qml.expval(qml.PauliZ(0))
    
# Subcircuit implementing the gates necessary for the
# stochastic parameter-shift rule. 
# In this example, we will differentiate the first term of
# the circuit (i.e., our variable is :math:`\theta = t`).
def SPSRgates(gate_pars, s, sign):
    G = Generator(*gate_pars)
    # step a)
    qml.QubitUnitary(expm(1j * (1 - s) * G), wires=[0, 1])
    # step b)
    qml.QubitUnitary(expm(1j * sign * np.pi / 4 * X), wires=0)
    # step c)
    qml.QubitUnitary(expm(1j * s * G), wires=[0,1])
    
# Function which can obtain all expectation vals needed 
# for the stochastic parameter-shift rule
@qml.qnode(dev)
def spsr_circuit(gate_pars, s=None, sign=+1):
    SPSRgates(gate_pars, s, sign)
    return qml.expval(qml.PauliZ(0))

# Fix the other parameters of the gate
b, c = -0.15, 1.6

# Obtain r+ and r-
# Even 10 samples gives a good result for this example
pos_vals = np.array([[spsr_circuit([t, b, c], s=s, sign=+1) 
                      for s in np.random.uniform(size=10)]
                      for t in angles])
neg_vals = np.array([[spsr_circuit([t, b, c], s=s, sign=-1) 
                      for s in np.random.uniform(size=10)]
                      for t in angles])

# Plot the results
evals = [crossres_circuit([t, -0.15, 1.6]) for t in angles]
spsr_vals = (pos_vals - neg_vals).mean(axis=1)

plt.plot(angles, evals, 'b', label="Expectation Value") # looks like cos(2*theta)
plt.plot(angles, spsr_vals, 'r', label="Stochastic PSR") # looks like -2 * sin(2 * theta)!
plt.legend();


##############################################################################
# References
# ----------
#
# .. [#banchi2020]
#
#     Leonardo Banchi and Gavin E. Crooks. "Measuring Analytic Gradients of 
#     General Quantum Evolution with the Stochastic Parameter Shift Rule." 
#     `arXiv:2005.10299 <https://arxiv.org/abs/2005.10299>`__ (2020).
#
# .. [#li2016]
#
#     Jun Li, Xiaodong Yang, Xinhua Peng, and Chang-Pu Sun.
#     "Hybrid Quantum-Classical Approach to Quantum Optimal Control."
#     `arXiv:1608.00677 <https://arxiv.org/abs/1608.00677>`__ (2016).
#
# .. [#mitarai2018]
#
#     Kosuke Mitarai, Makoto Negoro, Masahiro Kitagawa, and Keisuke Fujii.
#     "Quantum Circuit Learning." 
#     `arXiv:1803.00745 <https://arxiv.org/abs/1803.00745>`__ (2020).
#
# .. [#schuld2018]
#
#     Maria Schuld, Ville Bergholm, Christian Gogolin, Josh Izaac, and 
#     Nathan Killoran. "Evaluating analytic gradients on quantum hardware."
#     `arXiv:1811.11184 <https://arxiv.org/abs/1811.11184>`__ (2019).

