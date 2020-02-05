.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_app_tutorial_QGAN.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_app_tutorial_QGAN.py:


.. _quantum_GAN:

Quantum Generative Adversarial Networks with Cirq + TensorFlow
==============================================================

This demo constructs a Quantum Generative Adversarial Network (QGAN)
(`Lloyd and Weedbrook
(2018) <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.121.040502>`__,
`Dallaire-Demers and Killoran
(2018) <https://journals.aps.org/pra/abstract/10.1103/PhysRevA.98.012324>`__)
using two subcircuits, a *generator* and a *discriminator*. The
generator attempts to generate synthetic quantum data to match a pattern
of "real" data, while the discriminator tries to discern real data from
fake data (see image below). The gradient of the discriminator’s output provides a
training signal for the generator to improve its fake generated data.

|

.. figure:: ../implementations/QGAN/qgan.png
    :align: center
    :width: 75%
    :target: javascript:void(0)

|



Using Cirq + TensorFlow
~~~~~~~~~~~~~~~~~~~~~~~
PennyLane allows us to mix and match quantum devices and classical machine
learning software. For this demo, we will link together
Google's `Cirq <https://cirq.readthedocs.io/en/stable/>`_ and `TensorFlow <https://www.tensorflow.org/>`_ libraries.

We begin by importing PennyLane, NumPy, and TensorFlow.


.. code-block:: default


    import pennylane as qml
    import numpy as np
    import tensorflow as tf








We also declare a 3-qubit simulator device running in Cirq.


.. code-block:: default


    dev  = qml.device('cirq.simulator', wires=3)








Generator and Discriminator
~~~~~~~~~~~~~~~~~~~~~~~~~~~

In classical GANs, the starting point is to draw samples either from
some “real data” distribution, or from the generator, and feed them to
the discriminator. In this QGAN example, we will use a quantum circuit
to generate the real data.

For this simple example, our real data will be a qubit that has been
rotated (from the starting state :math:`\left|0\right\rangle`) to some
arbitrary, but fixed, state.


.. code-block:: default


    def real(phi, theta, omega):
        qml.Rot(phi, theta, omega, wires=0)








For the generator and discriminator, we will choose the same basic
circuit structure, but acting on different wires.

Both the real data circuit and the generator will output on wire 0,
which will be connected as an input to the discriminator. Wire 1 is
provided as a workspace for the generator, while the discriminator’s
output will be on wire 2.


.. code-block:: default


    def generator(w):
        qml.RX(w[0], wires=0)
        qml.RX(w[1], wires=1)
        qml.RY(w[2], wires=0)
        qml.RY(w[3], wires=1)
        qml.RZ(w[4], wires=0)
        qml.RZ(w[5], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RX(w[6], wires=0)
        qml.RY(w[7], wires=0)
        qml.RZ(w[8], wires=0)


    def discriminator(w):
        qml.RX(w[0], wires=0)
        qml.RX(w[1], wires=2)
        qml.RY(w[2], wires=0)
        qml.RY(w[3], wires=2)
        qml.RZ(w[4], wires=0)
        qml.RZ(w[5], wires=2)
        qml.CNOT(wires=[1, 2])
        qml.RX(w[6], wires=2)
        qml.RY(w[7], wires=2)
        qml.RZ(w[8], wires=2)








We create two QNodes. One where the real data source is wired up to the
discriminator, and one where the generator is connected to the
discriminator. In order to pass TensorFlow Variables into the quantum
circuits, we specify the ``"tf"`` interface.


.. code-block:: default


    @qml.qnode(dev, interface="tf")
    def real_disc_circuit(phi, theta, omega, disc_weights):
        real(phi, theta, omega)
        discriminator(disc_weights)
        return qml.expval(qml.PauliZ(2))


    @qml.qnode(dev, interface="tf")
    def gen_disc_circuit(gen_weights, disc_weights):
        generator(gen_weights)
        discriminator(disc_weights)
        return qml.expval(qml.PauliZ(2))








QGAN cost functions
~~~~~~~~~~~~~~~~~~~

There are two cost functions of interest, corresponding to the two
stages of QGAN training. These cost functions are built from two pieces:
the first piece is the probability that the discriminator correctly
classifies real data as real. The second piece is the probability that the
discriminator classifies fake data (i.e., a state prepared by the
generator) as real.

The discriminator is trained to maximize the probability of
correctly classifying real data, while minimizing the probability of
mistakenly classifying fake data.

The generator is trained to maximize the probability that the
discriminator accepts fake data as real.


.. code-block:: default


    def prob_real_true(disc_weights):
        true_disc_output = real_disc_circuit(phi, theta, omega, disc_weights)
        # convert to probability
        prob_real_true = (true_disc_output + 1) / 2
        return prob_real_true


    def prob_fake_true(gen_weights, disc_weights):
        fake_disc_output = gen_disc_circuit(gen_weights, disc_weights)
        # convert to probability
        prob_fake_true = (fake_disc_output + 1) / 2
        return prob_fake_true


    def disc_cost(disc_weights):
        cost = prob_fake_true(gen_weights, disc_weights) - prob_real_true(disc_weights)
        return cost


    def gen_cost(gen_weights):
        return -prob_fake_true(gen_weights, disc_weights)








Training the QGAN
~~~~~~~~~~~~~~~~~

We initialize the fixed angles of the “real data” circuit, as well as
the initial parameters for both generator and discriminator. These are
chosen so that the generator initially prepares a state on wire 0 that
is very close to the :math:`\left| 1 \right\rangle` state.


.. code-block:: default


    phi = np.pi / 6
    theta = np.pi / 2
    omega = np.pi / 7
    np.random.seed(0)
    eps = 1e-2
    init_gen_weights = np.array([np.pi] + [0] * 8) + \
                       np.random.normal(scale=eps, size=(9,))
    init_disc_weights = np.random.normal(size=(9,))

    gen_weights = tf.Variable(init_gen_weights)
    disc_weights = tf.Variable(init_disc_weights)








We begin by creating the optimizer:


.. code-block:: default


    opt = tf.keras.optimizers.SGD(0.1)








In the first stage of training, we optimize the discriminator while
keeping the generator parameters fixed.


.. code-block:: default


    cost = lambda: disc_cost(disc_weights)

    for step in range(50):
        opt.minimize(cost, disc_weights)
        if step % 5 == 0:
            cost_val = cost().numpy()
            print("Step {}: cost = {}".format(step, cost_val))






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Step 0: cost = -0.1094201769647043
    Step 5: cost = -0.38998838139377767
    Step 10: cost = -0.6660191301125451
    Step 15: cost = -0.8550836384511058
    Step 20: cost = -0.9454460225917956
    Step 25: cost = -0.9805878255020275
    Step 30: cost = -0.9931367838787679
    Step 35: cost = -0.9974893059307561
    Step 40: cost = -0.9989861294952895
    Step 45: cost = -0.999499912682503


At the discriminator’s optimum, the probability for the discriminator to
correctly classify the real data should be close to one.


.. code-block:: default


    print("Prob(real classified as real): ", prob_real_true(disc_weights).numpy())






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Prob(real classified as real):  0.9998971697286834


For comparison, we check how the discriminator classifies the
generator’s (still unoptimized) fake data:


.. code-block:: default


    print("Prob(fake classified as real): ", prob_fake_true(gen_weights, disc_weights).numpy())






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Prob(fake classified as real):  0.0002428841737298626


In the adversarial game we now have to train the generator to better
fool the discriminator. For this demo, we only perform one stage of the
game. For more complex models, we would continue training the models in an
alternating fashion until we reach the optimum point of the two-player
adversarial game.


.. code-block:: default


    cost = lambda: gen_cost(gen_weights)

    for step in range(200):
        opt.minimize(cost, gen_weights)
        if step % 5 == 0:
            cost_val = cost().numpy()
            print("Step {}: cost = {}".format(step, cost_val))






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Step 0: cost = -0.0002663484193758947
    Step 5: cost = -0.0004265993908529886
    Step 10: cost = -0.0006873900747690342
    Step 15: cost = -0.0011113692917046336
    Step 20: cost = -0.0018002498164300107
    Step 25: cost = -0.002917927839000356
    Step 30: cost = -0.004727620673591559
    Step 35: cost = -0.007646681336993311
    Step 40: cost = -0.012325897896118931
    Step 45: cost = -0.019754579341508816
    Step 50: cost = -0.03136851320601863
    Step 55: cost = -0.04909775442706632
    Step 60: cost = -0.07520414905809503
    Step 65: cost = -0.11169057964980311
    Step 70: cost = -0.15917328641126005
    Step 75: cost = -0.21566084044070521
    Step 80: cost = -0.2763741277517511
    Step 85: cost = -0.335417384615198
    Step 90: cost = -0.38835027808318046
    Step 95: cost = -0.4337177035855575
    Step 100: cost = -0.4728487013510403
    Step 105: cost = -0.5087772953153831
    Step 110: cost = -0.5451969361121201
    Step 115: cost = -0.585662230533103
    Step 120: cost = -0.6327884996043167
    Step 125: cost = -0.6872451918443119
    Step 130: cost = -0.7468435674174998
    Step 135: cost = -0.8066396918874688
    Step 140: cost = -0.8607338632397017
    Step 145: cost = -0.904840228981648
    Step 150: cost = -0.9376678931274327
    Step 155: cost = -0.9604098274034136
    Step 160: cost = -0.9753705514718387
    Step 165: cost = -0.9848743661772765
    Step 170: cost = -0.9907762745791073
    Step 175: cost = -0.994389756948916
    Step 180: cost = -0.9965827705721335
    Step 185: cost = -0.9979066118186722
    Step 190: cost = -0.9987028814087249
    Step 195: cost = -0.9991812122988303


At the optimum of the generator, the probability for the discriminator
to be fooled should be close to 1.


.. code-block:: default


    print("Prob(fake classified as real): ", prob_fake_true(gen_weights, disc_weights).numpy())






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Prob(fake classified as real):  0.9994220374031011


At the joint optimum the discriminator cost will be close to zero,
indicating that the discriminator assigns equal probability to both real and
generated data.


.. code-block:: default


    print("Discriminator cost: ", disc_cost(disc_weights).numpy())

    # The generator has successfully learned how to simulate the real data
    # enough to fool the discriminator.




.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Discriminator cost:  -0.0004751323255822726



.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  35.530 seconds)


.. _sphx_glr_download_app_tutorial_QGAN.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: tutorial_QGAN.py <tutorial_QGAN.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: tutorial_QGAN.ipynb <tutorial_QGAN.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.readthedocs.io>`_
