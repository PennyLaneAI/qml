r"""Tour of Quantum Phase Estimation
=============================================================

Within scientific computing, there are specific routines that are repeated over and over again in all kinds of situations. A clear example of this is solving systems of linear equations with an infinite number of applications. However, there are tasks that are perhaps less well known but that any improvement in their execution would have consequences in a lot of different fields!

The task we are going to talk about today is the computation of the eigenvalue given an eigenvector of a matrix. Such a task can be done efficiently on a quantum computer with the well-known Quantum Phase Estimation algorithm. This opens up a range of applications such as quantum chemistry or optimization problems.


Presentation and motivation of the problem
-----------------------------------------
It is common to find in the literature the explanation of this algorithm with a very mathematical presentation and it is often difficult to understand why it is done what it is done. The goal of this demo will be to explain in an intuitive and visual way what is behind this famous subroutine!

The first thing is to understand a little better the problem we are trying to solve. Given a matrix :math:`A`, we will say that :math:`|v \rangle` is an eigenvector if there exists a value :math:`\lambda` such that:

.. math::
    A |v \rangle = \lambda |v \rangle.

In this case, we will say that :math:`\lambda` is the eigenvalue of :math:`|v \rangle`. If we go to the field of quantum computation, the matrices we work with are unitary so it is satisfied that the eigenvalue is in fact of the form:

.. math::
    \lambda = e^{i \phi},

For this reason, it will be equivalent to find :math:`\lambda` or simply the :math:`\theta` value. This :math:`\theta` is called phase, and it is the value that our algorithm will be able to approximate: hence the name *Quantum Phase Estimation*.

Okay, the problem is clear: we want to calculate the eigenvalue associated to an eigenvector, but why is a quantum computer supposed to solve this task better?

There are really few applications in which it has been demonstrated that a quantum computer actually outperforms the best classical algorithm. The most famous example of this is Shor's algorithm. What Peter Shor did was to transform a problem of interest - the factorization of prime numbers - into a problem that we know that a quantum computer is more efficient: the calculation of the period of functions.

As it turns out, that strategy makes a lot of sense and we are going to imitate it! We will translate the problem of finding the eigenvalue of an eigenvector into the problem of finding the period of a function. Since we already know that this task can be done efficiently with a quantum computer, we will take advantage of its potential.

Building the periodic function
----------------------------------

Having clear the motivation of the problem we can start working. The first thing we will do is to define a periodic function that encodes the element we are looking for. There are many different ways to do this but a very simple way is to use the geometry of complex numbers. If we take:

.. math::
    f(x) := e^{i \lambda x},

we will be defining a periodic function that is equivalent to going around the unit circle of the complex plane. The period is equivalent to a complete lap, that is, when the exponent takes the value :math:`2 \pi`. Doing a quick calculation we can see that the first revolution is completed when:

.. math::
    x = \frac{2 \pi}{\lambda},

so, effectively, in the period of this function the eigenvalue is encoded. Great, so now what we can do is to see how we classically calculate this period. We will start by generating a sample of :math:`100` elements and drawing the function. In this case we will separate real and imaginary part to be able to see it in two dimensions:

"""

import numpy as np
import matplotlib.pyplot as plt

plt.style.use('pennylane.drawer.plot')

lanbda = 1.4
x_range = 10

xs = np.arange(0, x_range, 0.1)
f_xs = np.exp(1j * lanbda * xs)

plt.plot(xs, f_xs.real, label="real part")
plt.plot(xs, f_xs.imag, label="imaginary part")
plt.legend()

plt.show()


##############################################################################
#
# As expected, the function shown is periodic. Probably one of the most important algorithms in history is the Fourier Transform (FT), which is capable of turning a function in the time domain into the frequency domain. This means that if we have a periodic function like the one above and we apply the FT to it, we will obtain its frequency (which is the inverse of the period).
#

# We apply the fourier transform provided by numpy
ft_result = np.abs(np.fft.fft(f_xs))

# Let's plot the first 30 elements of the result
plt.bar(xs[:30], ft_result[:30], width=0.1)
plt.xlabel("frequency")
plt.show()

freq = np.argmax(ft_result) / x_range
period = 1 / freq

print("lambda:", (2 * np.pi) / period)


##############################################################################
# The reason why the peak we get coincides with the frequency is a very nice argument for which I recommend you to watch this `video <https://www.youtube.com/watch?v=spUNpyF58BY>`__.
#
# However, here we have cheated a bit. We actually knew :math:`\lambda` beforehand and for that reason we have been able to encode it in the period of the function. So there are two tasks that remain to be done:
#
# - to understand how we can translate this example to the quantum world.
# - discover how we can encode the function without having access to :math:`\lambda`.
#
# The reasoning in quantum programming will be the same, but we will replace the Fourier Transform with the Quantum Fourier Transform. From the classical point of view, we were sending a discrete vector generated by a function :math:`f`, while now we will send a quantum state generated by the same function:
#
# .. math::
#     |\phi\rangle = \sum_x f(x)|x\rangle, \quad x \in \{0, ... , N-1\}.
#
# In particular, using the function we established at the beginning, we must generate the state:
#
# .. math::
#     |\phi\rangle = \frac{1}{\sqrt{N}}\sum_x e^{i \lambda x}|x\rangle.
#
# Note that we have added a normalization factor at the beginning so that the quantum state has a :math:`1` norm. If we apply QFT to such a state, and measure, we can obtain the frequency and with it, the desired eigenvalue!
# Let us now turn our attention to the circuit generated by Quantum Phase Estimation:
#
# .. figure::
#   ../demonstrations/quantum_phase_estimation/phase_estimation.jpeg
#   :align: center
#   :width: 80%
#   :target: javascript:void(0)
#
# We can distinguish three important blocks: a column of initial Hadamards (which we will call the window), a sequence of controls and the QFT adjoint.
#
# Well, what the first two blocks are in charge of is precisely to generate the :math:`|\phi \rangle`. Now we understand perfectly what the QFT of the algorithm does, to obtain the frequency of the generated function!
#
# .. note::
#     In the QPE algorithm, instead of the QFT, the adjoint of the QFT is being applied. We will not go into the details but assume that they serve the same purpose.
#
#
# Well but how do these blocks generate the desired state? The idea is quite nice and focuses on the central block. First, if :math:`|\psi\rangle` is our eigenvector of :math:`A` and :math:`\lambda` is the eigenvalue, we know that:
#
# .. math::
#     U|\psi\rangle = \lambda|\psi\rangle.
#
# However, this does not guarantee that :math:`A` is unitary, so we can use a very common trick in quantum computing, working with the complex exponential. Therefore if we define :math:`U = e^{iA}`, we have that:
#
# .. math::
#     U|\psi\rangle = e^{i \lambda}|\psi\rangle.
#
# In addition, this applies to any :math:`U` power, ie:
#
# .. math::
#     U^k|\psi\rangle = e^{k i \lambda}|\psi\rangle.
#
# With this in mind, given any integer input :math:`|x \rangle`, our central operator will do the following:
#
# .. math::
#     |x\rangle |\psi\rangle \rightarrow e^{x i \lambda}|x\rangle |\psi\rangle.
#
# The key idea of this is to play with the binary representation of the input and the powers of two as shown in the following example:
#
# .. figure::
#   ../demonstrations/quantum_phase_estimation/controlled_sequence.jpeg
#   :align: center
#   :width: 80%
#   :target: javascript:void(0)
#
# This block manages to apply the function that we were looking for in given any :math:`x` and without making use of previous information of the eigenvalue, this information is extracted of the application of :math:`|\phi\rangle` to the operator! But let's not lose sight of the goal, we have to obtain this state:
#
# .. math::
#     |\phi\rangle = \frac{1}{\sqrt{N}}\sum_x e^{i \lambda x}|x\rangle.
#
# I am convinced that at this point many readers have just figured out how to accomplish this. One would simply have to generate the superposition of all possible :math:`x` by making use of Hadamard gates! From this we have managed to make sense of why to start from this particular state and why each of the following blocks.
# The good thing about this approach is that we have hardly had to use any mathematical account to understand the whole procedure and be sure that the results we will get are the desired ones. As I like to say: the more math you know, the less math you need.
#
# Time to code!
# -----------------
#
# All very well but let's not stay in the theory, let's code all this to see that it makes sense. For it we are going to begin with a simple example of a :math:`4\times 4` matrix, which, we will generate using Pauli gates for convenience:

import pennylane as qml

A = -0.6 * qml.PauliZ(0) - 0.8 * qml.PauliZ(0) @ qml.PauliZ(1)
print(qml.matrix(A))

##############################################################################
# In this case, we will use a diagonal matrix since it is easier to visualize the eigenvalues and eigenvectors. All the reasoning will not be affected by this assumption and if there is any change, we will indicate it.
#
# The eigenvectors are just the computational basis:
#
# .. math::
#     v_0 = \begin{bmatrix}
#     1 \\
#     0 \\
#     0 \\
#     0
#     \end{bmatrix}, \quad v_1 = \begin{bmatrix}
#     0 \\
#     1 \\
#     0 \\
#     0
#     \end{bmatrix}, \quad v_2 = \begin{bmatrix}
#     0 \\
#     0 \\
#     1 \\
#     0
#     \end{bmatrix}, \quad v_3 = \begin{bmatrix}
#     0 \\
#     0 \\
#     0 \\
#     1
#     \end{bmatrix}
#
# and the associated eigenvalues coincide exactly with the elements of the diagonal, :math:`\lambda_0 = -1.4`, :math:`\lambda_1 = 0.2`, :math:`\lambda_2 = 1.4` and :math:`\lambda_3 = -0.2`. I invite you to check that the definition of eigenvalue and eigenvector is indeed satisfied with the :math:`A` matrix seen above.
#
# .. note::
#     These eigenvectors could be represented on a quantum computer as :math:`|00 \rangle`, :math:`|01 \rangle`, :math:`|10 \rangle` and :math:`|11 \rangle`.
#
# To follow the same example we have seen from the classical point of view, let us suppose that our eigenvector is :math:`|10\rangle` and try to predict that its eigenvector is :math:`1.4`.

estimation_wires = [2, 3, 4, 5]

dev = qml.device("default.qubit")


@qml.qnode(dev)
def circuit_qpe():
    # we initialize the eigenvalue |10>
    qml.PauliX(wires=0)

    # We create the superposition of all x

    for wire in estimation_wires:
        qml.Hadamard(wires=wire)

    # We apply the function f to all values

    qml.ControlledSequence(qml.TrotterProduct(A, time=1), control=estimation_wires)

    # We apply the inverse QFT to obtain the frequency

    qml.adjoint(qml.QFT)(wires=estimation_wires)

    return qml.probs(wires=estimation_wires)


circuit_qpe()
results = circuit_qpe()
plt.bar(range(len(results)), results)
plt.xlabel("frequency")
plt.ylabel("prob")
plt.show()

##############################################################################
# With this we would already have programmed QPE. The :class:`~pennylane.ControlledSequence` operator is in charge of performing the central block of the algorithm. Note that we have passed as operator :class:`~pennylane.TrotterProduct` of :math:`A`. This is because we want the complex exponential of :math:`A`. In this case, the complex exponential of the operator can be obtained exactly with a Trotter iteration. If another type of Hamiltonian were to be used, you would probably need to approximate the exponential more accurately.
#
# As before, we get a peak in the frequency domain so by doing the calculation done above we can recover the eigenvalue:

freq = np.argmax(results) / len(results)
period = 1 / freq

print("lambda:", (2 * np.pi) / period)

##############################################################################
# Congratulations, you have managed to approximate the :math:`1.4` value we were looking for!
# I invite you to increase the number of estimation qubits to see how we're doing closer to that value. With this you have become an expert in Quantum Phase Estimation, but let's not stop there.
# In the following sections we will see more advanced techniques regarding this interesting subroutine.
#
# Mitigating leakage
# -------------------
#
# Although we have so far presented the QPE algorithm in a more traditional way, there are small modifications that can help us solve certain problems. But what are the problems with the procedure presented above?
# In principle everything seems to work fine but we have been manually accessing the vector of our quantum state to obtain the frequency with higher probability. When it comes down to it, this is not something we will be able to do and we will have to obtain this information by sampling. In the previous example, we have seen some kind of tails in our state:
#
# .. figure::
#   ../demonstrations/quantum_phase_estimation/leakage.jpeg
#   :align: center
#   :width: 80%
#   :target: javascript:void(0)
#
# These tails are called leaks and cause us to obtain unwanted values. In an ideal world in which we could work with all the points of our function these errors would not appear. However, having to work with a finite number of points we will have to deal with this situation. To give you an intuitive idea, although in all the previous section we have talked about calculating the period of the function :math:`f` in general, we are really limiting ourselves to an interval as seen in this image:
#
# .. figure::
#   ../demonstrations/quantum_phase_estimation/leakage2.jpeg
#   :align: center
#   :width: 80%
#   :target: javascript:void(0)
#
# The function that we generate with the help of the quantum computer or in the classical method is truncated to an interval marked by :math:`h(x)`. Such a function will take the value :math:`1` if :math:`x` belongs to the interval we want to work on and :math:`0` if we are outside. In the quantum case, the size of our window will be :math:`[0,...,2^m-1]` where :math:`m` is the number of estimation wires. For this reason, increasing the number of qubits will improve the accuracy, because we are working on more points of the function.
#
# Now that we know that our input is really the function :math:`f(x) \cdot h(x)`, we can deduce where the error comes from when trying to calculate the frequency. It is that we are not applying the QFT to :math:`f` but to the product of those two functions. But here we can make use of a very interesting property of the fourier transform and it is that:
#
# .. math::
#   \mathcal{F}(f \cdot h) = \mathcal{F}(f) \ast \mathcal{F}(h)
#
# where :math:`\ast` refers to convolution. This gives us a big clue to find suitable functions to bound with. The best ones will be those whose Fourier transform approximates a delta function (since it is the identity for the convolution) and hence:
#
# .. math::
#   \mathcal{F}(f(x)\cdot h(x)) = \mathcal{F}(f(x)) \ast \mathcal{F}(h(x)) \approx  \mathcal{F}(f(x)).
#
# This is a field widely studied in signal theory from a classical point of view. Applying all Hadamard gates generates what is known as the `BoxCar window <https://en.wikipedia.org/wiki/Boxcar_function>`__. However now with Pennylane you have access to more advanced windows such as the :class:`~pennylane.CosineWindow`.
# Let's see an example to see if it works:

estimation_wires = [2, 3, 4, 5]

dev = qml.device("default.qubit")


@qml.qnode(dev)
def circuit_qpe():
    # we initialize the eigenvalue |10>
    qml.PauliX(wires=0)

    # We apply the cosine window.
    qml.CosineWindow(wires=estimation_wires)

    # We apply the function f to all values

    qml.ControlledSequence(qml.TrotterProduct(A, time=1), control=estimation_wires)

    # We apply the inverse QFT to obtain the frequency

    qml.adjoint(qml.QFT)(wires=estimation_wires)

    return qml.probs(wires=estimation_wires)


circuit_qpe()
results = circuit_qpe()
plt.bar(range(len(results)), results)
plt.xlabel("frequency")
plt.ylabel("prob")
plt.show()

##############################################################################
# Great! A small modification in the previous code has mitigated the errors that were generated. Also the Cosine Window can be implemented efficiently on a quantum computer!
#
# Conclusion
# ----------
# In this demo we have seen how Quantum Phase Estimation works probably from a different perspective than you are used to. The great advantage of this approach we have shown you is that it creates a perfect bridge between classical signal processing and QPE. Therefore, future very important lines of research can be oriented to translate all the discoveries made in this work from the classical point of view into the quantum field.
# I invite you to experiment with other examples and demonstrate what you have learned!
#
# About the author
# ----------------
# .. include:: ../_static/authors/guillermo_alonso.txt
