.. raw:: html

    <div id="accordion">

      <div class="card">
        <div class="card-header" id="headingPennylane">
          <h5 class="mb-0">
            <button class="btn" data-toggle="collapse" data-target="#pennylane" aria-expanded="true" aria-controls="pennylane">
              What is PennyLane?
            </button>
          </h5>
        </div>
        <div id="pennylane" class="collapse show" aria-labelledby="headingPennylane" data-parent="#accordion">
          <div class="card-body">
            PennyLane is a <b> software framework for differentiable quantum programming</b>, similar to TensorFlow and
            PyTorch for classical computation. It facilitates the training of variational quantum circuits.
          </div>
        </div>
      </div>

      <div class="card">
        <div class="card-header" id="headingOther">
          <h5 class="mb-0">
            <button class="btn" data-toggle="collapse" data-target="#other" aria-expanded="false" aria-controls="other">
              What distinguishes PennyLane from other quantum programming languages?
            </button>
          </h5>
        </div>
        <div id="other" class="collapse" aria-labelledby="headingOther" data-parent="#accordion">
          <div class="card-body">
            While offering a lot of the functionality of standard quantum programming languages,
            PennyLane is built around the idea of <b> training quantum circuits using
            automatic differentiation</b>. This is especially important in applications such as quantum machine learning,
            quantum chemistry and quantum optimization.
          </div>
        </div>
      </div>

      <div class="card">
        <div class="card-header" id="headingVariational">
          <h5 class="mb-0">
            <button class="btn" data-toggle="collapse" data-target="#variational" aria-expanded="false" aria-controls="variational">
              What are variational circuits?
            </button>
          </h5>
        </div>
        <div id="variational" class="collapse" aria-labelledby="headingVariational" data-parent="#accordion">
          <div class="card-body">
            Variational quantum circuits, also called "parametrized quantum circuits" are <b> quantum algorithms that
            depend on tunable parameters</b>. For example, a variational circuit could contain
            so-called "Pauli rotation gates" which depend on a free parameter that determines the angle
            of a rotation applied to a qubit. Such free parameters can be trained by classical co-processors,
            to optimize the circuit for a given task.
          </div>
        </div>
      </div>

      <div class="card">
        <div class="card-header" id="headingGradients">
          <h5 class="mb-0">
            <button class="btn" data-toggle="collapse" data-target="#gradients" aria-expanded="false" aria-controls="gradients">
                How does PennyLane evaluate gradients of quantum circuits?
            </button>
          </h5>
        </div>
        <div id="gradients" class="collapse" aria-labelledby="headingGradients" data-parent="#accordion">
          <div class="card-body">
           Whereever possible, <b> PennyLane uses so-called "parameter shift rules"</b> to extract gradients of
           quantum circuits. These rules prescribe how to estimate a gradient by running a circuit twice
           or more times with slightly different parameters. In situations where no parameter shift rule
           can be applied, PennyLane uses the finite difference rule to approximate a gradient.
          </div>
        </div>
      </div>

      <div class="card">
        <div class="card-header" id="headingOpen">
          <h5 class="mb-0">
            <button class="btn" data-toggle="collapse" data-target="#open" aria-expanded="false" aria-controls="open">
                Is PennyLane open source?
            </button>
          </h5>
        </div>
        <div id="open" class="collapse" aria-labelledby="headingOpen" data-parent="#accordion">
          <div class="card-body">
           Yes, PennyLane is open source software developed under the Apache License.
          </div>
        </div>
      </div>

    </div>














