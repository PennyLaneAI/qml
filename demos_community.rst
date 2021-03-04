 .. role:: html(raw)
   :format: html

Community
=========

.. meta::
   :property="og:description": PennyLane demonstrations created by the community showcasing quantum machine learning and other topics of interest.
   :property="og:image": https://pennylane.ai/qml/_static/demos_card.png

.. raw:: html

    <style>
    #right-column {
        max-width: 1200px;
    }
    .up-button {
        left: calc(50% - 650px);
    }
    .jumbotron {
        box-shadow: none!important;
    }
    </style>


    <div class="jumbotron p-0 other">
        <div class="view">
            <img src="_static/demo-quilt-wide.png" class="img-fluid" alt="Sample image">
            <a href="#">
            <div class="mask rgba-stylish-slight"></div>
            </a>
        </div>

        <div class="card-body text-center mb-3">
            <p class="card-text py-2 lead">
                Have an existing GitHub repository or Jupyter notebook showing
                off quantum machine learning with PennyLane? Read the guidelines
                and submission instructions <a href="demos_submission.html">here</a>,
                and have your demonstration and research featured on our community page.
            </p>
        </div>
    </div>

    <link href="https://cdnjs.cloudflare.com/ajax/libs/mdbootstrap/4.8.10/css/mdb.min.css" rel="stylesheet">

    <hr>
    <br>


.. For two cards per row, create a new card deck every two cards.

:html:`<div class="card-deck">`

.. Community card required options:
     - title
     - author
     - date
   All remaining options (code, paper, blog, color) are *optional*.
   All fields, including the description can contain arbitrary HTML.

.. community-card::
    :title: QCNN for Speech Commands Recognition
    :author: C.-H. Huck Yang
    :date: 02/03/2021
    :code: https://github.com/huckiyang/QuantumSpeech-QCNN
    :paper: https://arxiv.org/abs/2010.13309
    :color: heavy-rain-gradient

    We provide a hybrid model training with larger-scale acoustic features 3000 to 10000 with
    quantum convolution neural networks (QCNN) with Random layer, which still provide insightful
    convolution features without the encoding time cost from the CPU simulation and on the request
    queueing time from QPU. We further provide classical activation mapping, a neural saliency
    analysis, on the well-trained neural models (QCNN Self-Attention vs. CNN Self-Attention) to
    show that the QCNN Self-attention model did learn meaningful representations. An additional
    Connectionist Temporal Classification (CTC) loss on character recognition is also provided for
    continuous speech recognition.

.. community-card::
    :title: Layerwise learning for quantum neural networks
    :author: Felipe Oyarce Andrade
    :date: 26/01/2021
    :code: https://github.com/felipeoyarce/layerwise-learning
    :color: heavy-rain-gradient

    In this project we’ve implemented a strategy presented by <a
    href="https://arxiv.org/abs/2006.14904" target="_blank">Skolik et al., 2020</a> for
    effectively training quantum neural networks. In layerwise learning the
    strategy is to gradually increase the number of parameters by adding a few
    layers and training them while freezing the parameters of previous layers
    already trained.  An easy way for understanding this technique is to think
    that we’re dividing the problem into smaller circuits to successfully avoid
    falling into Barren Plateaus. We provide a proof-of-concept
    implementation of this technique in Pennylane’s Pytorch interface for binary
    classification in the MNIST dataset.    

:html:`<div class="card hidden-card"></div></div>`

:html:`</div></div><br><div style='clear:both'>`
:html:`<div class="card-deck">`

.. community-card::
    :title: A Quantum-Enhanced Transformer
    :author: Riccardo Di Sipio
    :date: 20/01/2021
    :code: https://github.com/rdisipio/qtransformer
    :blog: https://towardsdatascience.com/toward-a-quantum-transformer-a51566ed42c2
    :color: heavy-rain-gradient

    The Transformer neural network architecture revolutionized the analysis of
    text. Here we show an example of a Transformer with quantum-enhanced
    multi-headed attention. In the quantum-enhanced version, dense layers are
    replaced by simple Variational Quantum Circuits. An implementation based on
    PennyLane and TensorFlow-2.x illustrates the basic concept.

:html:`</div></div><br><div style='clear:both'>`
:html:`<div class="card-deck">`

.. community-card::
    :title: A Quantum-Enhanced LSTM Layer
    :author: Riccardo Di Sipio
    :date: 18/12/2020
    :code: https://github.com/rdisipio/qlstm/blob/main/POS_tagging.ipynb
    :blog: https://towardsdatascience.com/a-quantum-enhanced-lstm-layer-38a8c135dbfa
    :color: heavy-rain-gradient

    In Natural Language Processing, documents are usually presented as sequences
    of words.  One of the most successful techniques to manipulate this kind of
    data is the Recurrent Neural Network architecture, and in particular a
    variant called Long Short-Term Memory (LSTM). Using the PennyLane library
    and its PyTorch interface, one can easily define a LSTM network where
    Variational Quantum Circuits (VQCs) replace linear operations. An
    application to Part-of-Speech tagging is presented in this tutorial.

.. community-card::
    :title: Quantum Machine Learning Model Predictor for Continuous Variables
    :author: Roberth Saénz Pérez Alvarado
    :date: 16/12/2020
    :code: https://github.com/roberth2018/Quantum-Machine-Learning/blob/main/Quantum_Machine_Learning_Model_Predictor_for_Continuous_Variable_.ipynb
    :color: heavy-rain-gradient

    According to the paper "Predicting toxicity by quantum machine learning"
    (Teppei Suzuki, Michio Katouda 2020), it is possible to predict continuous
    variables—like those in the continuous-variable quantum neural network model
    described in Killoran et al.  (2018)—using 2 qubits per feature. This is
    done by applying encodings, variational circuits, and some linear
    transformations on expectation values in order to predict values close to
    the real target.  Based on an <a
    href="https://pennylane.ai/qml/demos/quantum_neural_net.html">example</a>
    from PennyLane, and using a small dataset which consists of a
    one-dimensional feature and one output (so that the processing does not take
    too much time), the algorithm showed reliable results.

:html:`</div></div><br><div style='clear:both'>`
:html:`<div class="card-deck">`

.. community-card::
    :title: Trainable Quanvolutional Neural Networks
    :author: Denny Mattern, Darya Martyniuk, Fabian Bergmann, and Henri Willems
    :date: 26/11/2020
    :code: https://github.com/PlanQK/TrainableQuantumConvolution
    :color: heavy-rain-gradient

    We implement a trainable version of Quanvolutional Neural Networks using
    parametrized <code>RandomCircuits</code>. Parameters are optimized using
    standard gradient descent. Our code is based on the <a
    href="https://pennylane.ai/qml/demos/tutorial_quanvolution.html">Quanvolutional
    Neural Networks</a> demo by Andrea Mari. This demo results from our research
    as part of the <a href="https://www.planqk.de">PlanQK consortium</a>.

.. community-card::
    :title: Using a Keras optimizer for Iris classification with a QNode and loss function
    :author: Hemant Gahankari
    :date: 09/11/2020
    :code: https://colab.research.google.com/drive/17Qri3jUBpjjkhmO6ZZZNXwm511svSVPw?usp=sharing
    :color: heavy-rain-gradient

    Using PennyLane, we explain how to create a quantum function and train a
    quantum function using a Keras optimizer directly, i.e., not using a Keras
    layer. The objective is to train a quantum function to predict classes of
    the Iris dataset.

:html:`</div></div><br><div style='clear:both'>`
:html:`<div class="card-deck">`

.. community-card::
    :title: Linear regression using angle embedding and a single qubit
    :author: Hemant Gahankari
    :date: 09/11/2020
    :code: https://colab.research.google.com/drive/1ABVtBjwcGNNIfmiwEXRdFdZ47K1vZ978?usp=sharing
    :color: heavy-rain-gradient

    In this example, we create a hybrid neural network (mix of classical and
    quantum layers), train it and get predictions from it. The data set
    consists of temperature readings in degrees Centigrade and corresponding
    Fahrenheit. The objective is to train a neural network that predicts
    Fahrenheit values given Centigrade values.


.. community-card::
    :title: Amplitude embedding in Iris classification with PennyLane's KerasLayer
    :author: Hemant Gahankari
    :date: 09/11/2020
    :code: https://colab.research.google.com/drive/12ls_GkSD2t0hr3Mx9-qzVvSWxR3-N0WI#scrollTo=4PQTkXpv52vZ
    :color: heavy-rain-gradient

    Using amplitude embedding from PennyLane, this demonstration aims to explain
    how to pass classical data into the quantum function and convert it to quantum
    data. It also shows how to create a PennyLane KerasLayer from a QNode, train it
    and check the performance of the model.

:html:`</div></div><br><div style='clear:both'>`
:html:`<div class="card-deck">`

.. community-card::
    :title: Angle embedding in Iris classification with PennyLane's KerasLayer
    :author: Hemant Gahankari
    :date: 09/11/2020
    :code: https://colab.research.google.com/drive/13PvS2D8mxBvlNw6_5EapUU2ePKdf_K53#scrollTo=1fJWDX5LxfvB
    :color: heavy-rain-gradient

    Using angle embedding from PennyLane, this demonstration aims to explain
    how to pass classical data into the quantum function and convert it to
    quantum data. It also shows how to create a PennyLane KerasLayer from a
    QNode, train it and check the performance of the model.


.. community-card::
    :title: Characterizing the loss landscape of variational quantum circuits
    :author: Patrick Huembeli and Alexandre Dauphin
    :date: 30/09/2020
    :code: https://github.com/PatrickHuembeli/vqc_loss_landscapes
    :paper: https://arxiv.org/abs/2008.02785
    :color: heavy-rain-gradient

    Using PennyLane and complex PyTorch, we compute the Hessian of the loss
    function of VQCs and show how to characterize the loss landscape with it. We
    show how the Hessian can be used to escape flat regions of the loss
    landscape.


.. If the final card deck only has a single card, we insert a 'hidden card'
   so that the card does not become full-width.
   :html:`<div class="card hidden-card"></div></div>`
   
.. toctree::
    :maxdepth: 2
    :hidden:
