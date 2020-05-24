r"""
The Stochastic Parameter-shift Rule
===================================

.. meta::
    :property="og:description": You can differentiate any qubit gate with the stochastic parameter-shift rule.
    :property="og:image": https://pennylane.ai/qml/_images/some_image.png # TODO: add image

In this tutorial we demonstrate how the stochastic parameter-shift rule (Banchi and Crooks [#banchi2020]_)
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

i) gates have free parameters
ii) expectation values of measurements are taken 

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
may contain many gates, we can concentrate on just a single gate :math:`G` that we want to differentiate
(other gates will follow the same pattern).

.. figure:: ../demonstrations/stochastic_parameter_shift/quantum_circuit.png
    :align: center
    :width: 90%

All gates appearing before :math:`G` can be absorbed into an initial state preparation 
:math:`\vert \psi_0 \rangle`, and all gates appearing after :math:`G` can be absorbed with the measurement
operator :math:`\hat{A}` to make a new effective measurement operator :math:`\hat{A}`.
The expectation value :math:`\hat{A}` in the simpler one-gate circuit is identical to 
the expectation value :math:`\hat{C}` in the larger circuit.

We can also write any unitary gate in the form

.. math::

    G(\theta) = e^{i\theta\hat{V}},
    
where :math:`\hat{V}` is the Hermitian _generator_ of the gate :math:`G`.

Now, how do we actually obtain the numerical values for the derivatives necessary for gradient descent? 

This is where the parameter-shift rule [#li2016], [#mitarai2018], [#schuld2018] enters the story. In short, the parameter-shift rule says that for 
many gates of interest---including all single-qubit gates---we can obtain the value of the derivative 
:math:`\frac{\partial \langle \hat{A}(\theta) \rangle}{\partial \theta}` by subtracting two related 
circuit evaluations:

..math::

   \frac{\partial \langle \hat{A} \rangle}{\partial \theta} = 
   \langle \hat{A}(\theta + \tfrac{\pi}{4}) \rangle -
   \langle \hat{A}(\theta - \tfrac{\pi}{4}) \rangle 
   
.. figure:: ../demonstrations/stochastic_parameter_shift/quantum_circuit.png
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
def circuit1(x):
    qml.RX(x, wires=0)
    return qml.expval(qml.PauliZ(0))
    
param = 0.5
circuit1(param)

gradient = qml.grad(circuit1, argnum=0)
gradient(param)

def param_shift(x):
    return 0.5 * (circuit1(x + np.pi / 2) - circuit1(x - np.pi / 2))
                  
expvals = [circuit1(theta) for theta in angles]
grad_vals = [gradient(theta) for theta in angles]
param_shift_vals = [param_shift(theta) for theta in angles]
plt.plot(angles, expvals, 'b', label="Expecation value")
plt.plot(angles, grad_vals, 'r', label="Gradient")
plt.plot(angles, param_shift_vals, 'mx', label="Parameter-shift rule")
plt.legend();

# Now the Stochasic PSR

I = np.eye(2)
X = np.array([[0, 1], [1, 0]])
Z = np.array([[1, 0], [0, -1]])

def Generator(t, b, c):
    # the inputs will show up as Pennylane variables;
    # we have to extract their numerical values
    h = t.val * (np.kron(X, I) - 
                 b.val * np.kron(Z, X) + 
                 c.val * np.kron(I, X))
    return h
    
# This is the circuit that we will try to apply SPSR to
@qml.qnode(dev)
def circuit3(gate_pars):
    H = Generator(*gate_pars)
    qml.QubitUnitary(expm(-1j * H), wires=[0, 1])
    return qml.expval(qml.PauliZ(0))
    
# subcircuit implementing the quantum gates necessary for SPSR
# would be nicer if this was implemented in a way that was
# more easily modifiable
def SPSRgates(gate_pars, s, sign):
    H = Generator(*gate_pars)
    qml.QubitUnitary(expm(1j * (1 - s) * H), wires=[0, 1])
    # Note: we're differentiating first term of H (i.e., variable `t`)
    # Also: extra (-2) sign needed to match PL convention
    #qml.RX(sign * np.pi / 4 * (-2), wires=0) 
    # line below is equivalent to line above, 
    # but more clear (doesn't require extra sign)
    qml.QubitUnitary(expm(1j * sign * np.pi / 4 * X), wires=0)
    qml.QubitUnitary(expm(1j * s * H), wires=[0,1])
    
# obtain all expvals needed for for SPSR
@qml.qnode(dev)
def spsr_circuit3(gate_pars, s=None, sign=+1):
    SPSRgates(gate_pars, s, sign)
    return qml.expval(qml.PauliZ(0))

# QNodeCollections would be perfect here, but QNodes
# but QNodes don't seem to be compatible with `functools.partial`

b, c = -0.15, 1.6
pos_vals = np.array([[spsr_circuit3([t, b, c], s=s, sign=+1) 
                      for s in np.random.uniform(size=10)]
                      for t in angles])
neg_vals = np.array([[spsr_circuit3([t, b, c], s=s, sign=-1) 
                      for s in np.random.uniform(size=10)]
                      for t in angles])
# even 10 samples gives a good result for this example

evals = [circuit3([t, -0.15, 1.6]) for t in angles]
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

