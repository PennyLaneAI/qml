r"""Generative quantum eigensolver (GQE) training using data generated with PennyLane
===================================================

In this demo, we will be (pre-)training a generative quantum eigensolver (GQE) and applying the technique described in `this
paper <https://arxiv.org/abs/2401.09253v1>`__, using the molecular data available in `PennyLane Datasets <https://pennylane.ai/datasets/>`__.
We will show that the model gradually better approximates the correct energies and, in turn, 
can sample energies close to the ground state energy calculated by Pennylane. 

.. figure:: ../_static/demo_thumbnails/opengraph_demo_thumbnails/OGthumbnail_generative_quantum_eigensolver.png
    :align: center
    :width: 70%
    :target: javascript:void(0)

The GQE algorithm is an alternative approach in estimating the ground state of a particular molecule.
Usually, this ground state estimation is done via the variational quantum eigensolver (VQE) approach, 
where the quantum state is represented as a quantum circuit with tunable parameters. The goal is then 
to find the optimal parameters that minimizes the corresponding energy :math:`E`. For more details on
VQEs, check out the `PennyLane Demo <https://pennylane.ai/qml/demos/tutorial_vqe/>`__ and 
`Documentation <https://docs.pennylane.ai/projects/catalyst/en/stable/demos/adaptive_circuits_demo.html>`__.

.. figure:: ../_static/demonstration_assets/gqe_training/paper_vqe_diagram.png
    :align: center
    :width: 90%
    :target: javascript:void(0)

There are some issues with VQE scalability, however. This shortcoming makes it less competitive against
the performance of classical ML algorithms for large problems. To bypass this, the GQE algorithm was 
proposed. A GQE is then a generative model where quantum states represented by quantum circuits are 
sampled. The generative model is then trained so that the states being sampled more closely approximates
the ground state. 

.. figure:: ../_static/demonstration_assets/gqe_training/paper_gqe_diagram.png
    :align: center
    :width: 90%
    :target: javascript:void(0)

The main difference between the two approaches is where the tunable parameters are embedded.
That is, it is the classical GQE model that is being optimized as opposed to the variable
quantum circuit of VQE. Potentially then, the loss landscape for GQE will be different and
will be amenable for larger problems.
"""

######################################################################
# Outline
# -------
# 
# 1. | **GPT-QE Background**
# 2. | **Dataset construction via Pennylane**
#    | 2a. Loading molecular information
#    | 2b. Defining energy function
#    | 2c. Token sequence generation with corresponding energies
# 
# 3. | **GPT (pre-)training**
#    | 3a. GPT implementation details
#    | 3b. GPT (pre-)training loop implementation
# 
# 4. | **Results**
#    | 4a. Loss curve
#    | 4b. GPT evaluation progress
#    | 4c. GPT sequence generation comparison
# 

######################################################################
# GPT-QE Background
# -----------------
# 
# In particular, the chosen model design in the paper was the generative pre-trained
# transformer (GPT) architecture. As a language model, GPTs are successful in generating
# sequences of words that closely resemble human natural language. This performance is
# harnessed by constructing quantum states :math:`\rho` as a sequence of unitary operators 
# which are in turn, represented by quantum circuits. That is, we let :math:`\rho = U\rho_0 U^{\dagger}`
# for some fixed initial state :math:`\rho_0` and the sequence is :math:`U = U_{j_N}U_{j_{N-1}}\cdots U_{j_1}`.
# The GPT model generates the sequence of integers :math:`j_1, j_2, ..., j_N` indexing a vocabulary
# of operators :math:`U_j`'s. The goal of training is then to minimize the corresponding energy
# :math:`E = \mbox{Tr}(\hat{H}\rho)` where :math:`\hat{H}` is the hamiltonian of the molecule in 
# question.
# 
# Each integer :math:`j_i` is sampled from the distribution :math:`\exp(-\beta w_{j_i})` where
# :math:`\beta` is an inverse temperature representing a trade-off parameter between exploration and 
# exploitation and :math:`w_{j_i}` is the logit returned by GPT for the index :math:`j_i`. We then
# observe that the probability of sampling a state through the method described above is 
# proportional to :math:`\exp(-\beta w_{\mbox{sum}})` where :math:`w_{\mbox{sum}} = \sum_{i=1}^N w_{j_i}` 
# and the probability for the corresponding energy is :math:`\exp(-\beta E)`. We thus have a constraint 
# for the total logit to be equal to the energy of the corresponding state :math:`w_{\mbox{sum}} = E` which
# can be imposed by minimizing the loss function :math:`C = (w_{\mbox{sum}} - E)^2`.
#   

######################################################################
# Generating molecules from Google Colab by Utkarsh
# -------------------------------------------------
# 
# For simplicity, let us consider the hydrogen gas molecule and load the correspoding data
# (especially, a pool of unitary operators) from Pennylane.
# 

import numpy as np
import pennylane as qml

def generate_molecule_data(molecules=["H2", "LiH", "BeH2", "H2O", "N2"]):
    # Get same molecules as the paper with addition of Water
    datasets = qml.data.load("qchem", molname=molecules)

    # Get the time set \mathcal{T}
    operator_times = np.sort(np.array([-2**k for k in range(1, 5)] + [2**k for k in range(1, 5)]) / 160)

    # Build operator set \mathcal{P} for each molecule
    molecule_data = dict()
    for dataset in datasets:
        molecule = dataset.molecule
        num_electrons, num_qubits = molecule.n_electrons, 2 * molecule.n_orbitals
        singles, doubles = qml.qchem.excitations(num_electrons, num_qubits)
        double_excs = [qml.DoubleExcitation(time, wires=double) for double in doubles for time in operator_times]
        single_excs = [qml.SingleExcitation(time, wires=single) for single in singles for time in operator_times]
        identity_ops = [qml.exp(qml.I(range(num_qubits)), 1j*time) for time in operator_times] # For Identity
        operator_pool = double_excs + single_excs + identity_ops
        molecule_data[dataset.molname] = {
            "op_pool": np.array(operator_pool), 
            "num_qubits": num_qubits,
            "hf_state": dataset.hf_state,
            "hamiltonian": dataset.hamiltonian,
            "expected_ground_state_E": dataset.fci_energy
        }
        print(f"Molecule: {dataset.molname}, n_ops: {len(operator_pool)}, num_qubits: {num_qubits}")
    return molecule_data

molecule_data = generate_molecule_data(molecules="H2")
h2_molecule = molecule_data["H2"]
op_pool = h2_molecule["op_pool"]
num_qubits = h2_molecule["num_qubits"]
init_state = h2_molecule["hf_state"]
hamiltonian = h2_molecule["hamiltonian"]
grd_E = h2_molecule["expected_ground_state_E"]
op_pool_size = len(op_pool)

######################################################################
# .. rst-class:: sphx-glr-script-out
# 
#  .. code-block:: none
# 
#     Molecule: H2, n_ops: 32, num_qubits: 4

######################################################################
# Define the energy function
# --------------------------
# 
# To generate a dataset, we would be creating an operator sequence of fixed length and calculating the
# energy corresponding to that sequence by using Eq. 1 of the paper. This function ``energy_circuit``
# was implemented by Utkarsh using Pennylane.
# 
# As a slight extension from the paper, we also calculate the energies for each subsequence of
# operators to help with the training of the model. That is, for a sequence of operators:
# ``[U_1, U_2, U_3]``, we also compute the energies for ``[U_1]`` and ``[U_1, U_2]`` instead of just
# the full sequence ``[U_1, U_2, U_3]`` described in the paper.
# 
# This is simply done in Pennylane using ``Snapshot``
# 

# This computes the energy for a chosen molecule with the selected operator pool
dev = qml.device("default.qubit", wires=num_qubits)

@qml.qnode(dev)
def energy_circuit(gqe_ops):
    """Computes Eq. 1 based on selected time evolution operators"""
    qml.BasisState(init_state, wires=range(num_qubits)) # Initial state <-- Hartree Fock state
    for op in gqe_ops:
        qml.Snapshot(measurement=qml.expval(hamiltonian))
        qml.apply(op) # Applies each of the time evolution operator
    return qml.expval(hamiltonian) # Computes the energy via (1)

energy_circuit = qml.snapshots(energy_circuit)

def get_subsequence_energies(op_seq):
    "Collates the energies of each subsequence for a batch of sequences"
    energies = []
    for ops in op_seq:
        es = energy_circuit(ops)
        energies.append(
            [es[k].item() for k in list(range(1, len(ops))) + ["execution_results"]]
        )
    return np.array(energies)

# Note: Energy offsets are included for other molecules in the paper

######################################################################
# Generate dataset for GPT (pre-)training
# ---------------------------------------
# 
# With these ingredients, we can now construct a dataset containing sequences of operators and their
# energies. Since we cannot feed the operators directly to the GPT model, we would need to tokenize
# them. The indices of ``op_pool`` seems to be a good candidate but we instead choose the tokens to be
# the ``op_pool`` indices shifted by 1. This is so that we can define a special token ``0`` that tells
# the GPT model the start of a sequence.
# 
# We generate a ``train_size`` number of random operator sequences of length ``seq_len`` for our
# purposes and calculate their energies (and their subsequences).
# 

# Generate sequence of indices of operators in vocab
train_size = 1024
seq_len = 4
train_op_pool_inds = np.random.randint(op_pool_size, size=(train_size, seq_len))

# Corresponding sequence of operators
train_op_seq = op_pool[train_op_pool_inds]

# Corresponding tokens with special starting token
train_token_seq = np.concatenate([
    np.zeros(shape=(train_size, 1), dtype=int), # starting token is 0
    train_op_pool_inds + 1 # shift operator inds by one
], axis=1)

# %%time 
train_sub_seq_en = get_subsequence_energies(train_op_seq)

######################################################################
# .. rst-class:: sphx-glr-script-out
# 
#  .. code-block:: none
# 
#     CPU times: user 47 s, sys: 37.5 ms, total: 47 s
#     Wall time: 47 s

######################################################################
# GPT implementation details
# --------------------------
# 
# The architecture of the GPT model we instantiate here is similar to that described in the paper.
# That is, we used 12 attention layers, 12 attention heads, and 768 embedding dimensions as specified
# in the default values of ``GPTConfig``. For our demo, the model then has a default of around 85
# million parameters and is ``324.25 MB`` in size. Its implementation is in `our
# repo <https://github.com/XanaduAI/gpt-qe/tree/dev>`__ and was originally from this `nanoGPT
# repo <https://github.com/karpathy/nanoGPT/blob/master/model.py>`__.
# 
# We needed to make some mandatory changes to the nanoGPT implementation to accommodate the details of
# the problem. - For pre-/training: - We returned all the logits generated by the network instead of
# just taking the last one. Since we are using a causal mask for the attention blocks (logits of the
# earlier tokens in the sequence are not affected by tokens in the later part of the sequence), we can
# directly use the logits corresponding to each token in the entire sequence. This would increase the
# training efficiency as well. - This was not clear in the paper. Instead, what they described is
# actually just the sampling procedure where the sequence is constructed per token and so, only the
# last logit was used to correspond to the next token for each forward pass of the network. - We
# defined the ``calculate_loss()`` as a function of the error between the total logits of the sequence
# and the corresponding energy. This is more direct and in line with their goal of logit-matching. To
# help with the training as mentioned earlier, we also considered the cummulative sum of the logits
# for each subsequence and matched them with the corresponding subsequence energies. This is
# calculated in ``calculate_loss_sub()`` and is used in this demo. - In contrast, the loss defined in
# the paper uses the error between the negative exponential of both the total logits and the energies.
# Since the exponential is 1-to-1, it wouldn’t really make a difference to the loss function. It may
# also introduce numerical instabilities because the numbers would be unnecessarily magnified
# exponentially.
# 
# -  For sequence generation:
# 
#    -  We sample next tokens proportionally to ``exp(-logits/temp)`` instead of the usual
#       ``exp(logits/temp)``
#    -  Since the token ``0`` only corresponds to the start of the sequence, we mask out the logits
#       corresponding to ``0`` so that it will never be chosen in the later parts of the sequence.
# 

import torch

from gpt_qe.nano_gpt.model import GPT, GPTConfig

tokens = torch.from_numpy(train_token_seq).to("cuda")
energies = torch.from_numpy(train_sub_seq_en).to("cuda")

gpt = GPT(GPTConfig(
    vocab_size=op_pool_size + 1,
    block_size=seq_len,
    # n_layer=2,
    # n_head=8,
    # n_embd=128,
    dropout=0.2,
    bias=False
)).to("cuda")
opt = gpt.configure_optimizers(device_type="cuda", learning_rate=5e-5)

######################################################################
# .. rst-class:: sphx-glr-script-out
# 
#  .. code-block:: none
# 
#     number of parameters: 84.98M
#     num decayed parameter tensors: 50, with 84,963,072 parameters
#     num non-decayed parameter tensors: 25, with 19,200 parameters

######################################################################
# .. rst-class:: sphx-glr-script-out
# 
#  .. code-block:: none
# 
#       from .autonotebook import tqdm as notebook_tqdm

######################################################################
# .. rst-class:: sphx-glr-script-out
# 
#  .. code-block:: none
# 
#     using fused AdamW: True

######################################################################
# GPT (pre-)training loop
# -----------------------
# 
# We now implement a training loop for our GPT model. This can be framed as a straightforward
# supervised learning problem.
# 
# For each training iteration/epoch: 1. Shuffle the training set and split it into ``n_batches``
# minibatches 2. For each minibatch, calculate the loss, the gradients, and take an optimizer step 3.
# For each nth iteration (500 here), evaluate the GPT model by: - Generating a batch of sequences and
# the predicted energies (total logits) -> not necessarily same sequences in training set - Calculate
# the true energies using Pennylane - Calculate some metrics to track the learning progress (and save
# everytime the metric gets better)
# 

# %%time 

# batch_size = 128
n_batches = 8
train_inds = np.arange(train_size)

losses = []
pred_Es_t = []
true_Es_t = []
current_mae = 10000
gpt.train()
for i in range(10000):
    # Shuffle batches of the training set
    np.random.shuffle(train_inds)
    token_batches = torch.tensor_split(tokens[train_inds], n_batches)
    energy_batches = torch.tensor_split(energies[train_inds], n_batches)
    
    # SGD on random minibatches
    loss_record = 0
    for token_batch, energy_batch in zip(token_batches, energy_batches):
        opt.zero_grad()
        loss = gpt.calculate_loss_sub(token_batch, energy_batch)
        loss.backward()
        opt.step()
        loss_record += loss.item() / n_batches
    losses.append(loss_record)

    if (i+1) % 500 == 0:
        # For evaluation of gpt
        gpt.eval()
        gen_token_seq, pred_Es = gpt.generate(
            n_sequences=100, 
            max_new_tokens=seq_len, 
            temperature=0.001, # use low temp to emphasize difference in logits
            device="cuda"
        )
        pred_Es = pred_Es.cpu().numpy()

        gen_inds = (gen_token_seq[:, 1:] - 1).cpu().numpy()
        gen_op_seq = op_pool[gen_inds]
        true_Es = get_subsequence_energies(gen_op_seq)[:, -1].reshape(-1, 1)

        mae = np.mean(np.abs(pred_Es - true_Es))
        ave_E = np.mean(true_Es)
        
        pred_Es_t.append(pred_Es)
        true_Es_t.append(true_Es)
        
        print(f"Iteration: {i+1}, Loss: {losses[-1]}, MAE: {mae}, Ave E: {ave_E}")
        
        if mae < current_mae:
            current_mae = mae
            torch.save(gpt, f"./seq_len={seq_len}/gqe.pt")
            print("Saved model!")
            
        gpt.train()
        
pred_Es_t = np.concatenate(pred_Es_t, axis=1)
true_Es_t = np.concatenate(true_Es_t, axis=1)

######################################################################
# .. rst-class:: sphx-glr-script-out
# 
#  .. code-block:: none
# 
#     Iteration: 500, Loss: 0.004496691238049528, MAE: 0.13945468622863236, Ave E: -1.1161227981406456
#     Saved model!
#     Iteration: 1000, Loss: 0.001162520404255374, MAE: 0.11792013497926974, Ave E: -1.116178063434579
#     Saved model!
#     Iteration: 1500, Loss: 0.0006311882560414964, MAE: 0.08421050347067748, Ave E: -1.1304435666682537
#     Saved model!
#     Iteration: 2000, Loss: 0.0002220232025956396, MAE: 0.03313205549288038, Ave E: -1.13411711385679
#     Saved model!
#     Iteration: 2500, Loss: 9.021296506465553e-05, MAE: 0.03720317687198404, Ave E: -1.1360217383940532
#     Iteration: 3000, Loss: 0.00011929328764308375, MAE: 0.010246824522607662, Ave E: -1.1355033629645301
#     Saved model!
#     Iteration: 3500, Loss: 4.015137835017087e-05, MAE: 0.008332604993116905, Ave E: -1.1362737218253494
#     Saved model!
#     Iteration: 4000, Loss: 0.00025425587370956726, MAE: 0.03346923599957368, Ave E: -1.13442109812976
#     Iteration: 4500, Loss: 4.590269966149363e-05, MAE: 0.0086580669691949, Ave E: -1.1344678899103924
#     Iteration: 5000, Loss: 2.7407370499136962e-05, MAE: 0.006680762382889203, Ave E: -1.136412143925528
#     Saved model!
#     Iteration: 5500, Loss: 3.778071550021417e-05, MAE: 0.014272903220676704, Ave E: -1.1362969016861684
#     Iteration: 6000, Loss: 2.2792776141250974e-05, MAE: 0.007428675818214263, Ave E: -1.1367647064449693
#     Iteration: 6500, Loss: 1.9002385742602413e-05, MAE: 0.004431537870071902, Ave E: -1.135880723613281
#     Saved model!
#     Iteration: 7000, Loss: 1.5268728079291623e-05, MAE: 0.002464256235883442, Ave E: -1.1356989137037925
#     Saved model!
#     Iteration: 7500, Loss: 1.1030378864566936e-05, MAE: 0.007000517223791054, Ave E: -1.1360445255294285
#     Iteration: 8000, Loss: 7.638036884241474e-06, MAE: 0.0044611951680048586, Ave E: -1.1352658877947734
#     Iteration: 8500, Loss: 1.616690860258467e-05, MAE: 0.004094392133172753, Ave E: -1.1356437076129735
#     Iteration: 9000, Loss: 7.37882245331426e-06, MAE: 0.004240113290004896, Ave E: -1.1358971131175264
#     Iteration: 9500, Loss: 1.004411104422562e-05, MAE: 0.010631562300185794, Ave E: -1.1368761600775912
#     Iteration: 10000, Loss: 1.809862392776087e-05, MAE: 0.01987725166307399, Ave E: -1.1345492765523346
#     CPU times: user 2h 12min 24s, sys: 8.18 s, total: 2h 12min 32s
#     Wall time: 2h 12min 32s

######################################################################
# GPT (pre-)training results
# --------------------------
# 

######################################################################
# Loss curve
# ~~~~~~~~~~
# 

import holoviews as hv
import hvplot.pandas
import pandas as pd
import numpy as np

hvplot.extension('matplotlib')

losses = pd.read_csv("./seq_len=4/trial7/losses.csv")["0"]
np.log(losses).hvplot(title="Training loss progress", ylabel="log(loss)", xlabel="Training epochs").opts(fig_size=500)

##############################################################################
#.. figure:: ../_static/demonstration_assets/gqe_training/gqe_training_loss.png
#    :align: center
#    :width: 90%

df_true = pd.read_csv("./seq_len=4/trial7/true_Es_t.csv").iloc[:, 1:]
df_pred = pd.read_csv("./seq_len=4/trial7/pred_Es_t.csv").iloc[:, 1:]

df_true.columns = df_true.columns.astype(int)
df_pred.columns = df_pred.columns.astype(int)

df_trues_stats = pd.concat([df_true.mean(axis=0), df_true.min(axis=0), df_true.max(axis=0)], axis=1)
df_trues_stats.columns = ["Ave True E", "Min True E", "Max True E"]

df_preds_stats = pd.concat([df_pred.mean(axis=0), df_pred.min(axis=0), df_pred.max(axis=0)], axis=1)
df_preds_stats.columns = ["Ave Pred E", "Min Pred E", "Max Pred E"]

######################################################################
# GPT model learning progress
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# In this plot, we track the performance of the GPT model throughout its training. As mentioned
# before, every after 500th iteration, we let the model generate a batch of sequences. Alongside, we
# also return the total logits (predicted energies) used in the sequence generation. In the figure,
# the average predicted energies corresponds to the red markers and the distribution of predicted
# energies is represented by the red area.
# 
# Once we have the generated sequences, we can also let Pennylane calculate the true sequence
# energies. Similarly in the figure then, the blue markers are the average true energies and the blue
# area represents the true energy distribution.
# 
# We now see that the energies predicted by the model gets more accurate at approximating the true
# energies during training. This in turn, samples lower energies as we see that the true energies
# sampled gets closer to the ground state energy (the dashed line).
# 

fig = (
    df_trues_stats.hvplot.scatter(y="Ave True E", label="Mean True Energies") * 
    df_trues_stats.hvplot.line(y="Ave True E", alpha=0.5, linewidth=1) * 
    df_trues_stats.hvplot.area(y="Min True E", y2="Max True E", alpha=0.1)
) * (
    df_preds_stats.hvplot.scatter(y="Ave Pred E", label="Mean Predicted Energies") * 
    df_preds_stats.hvplot.line(y="Ave Pred E", alpha=0.5, linewidth=1) * 
    df_preds_stats.hvplot.area(y="Min Pred E", y2="Max Pred E", alpha=0.1)
)
fig = fig * hv.Curve([[0, grd_E], [10000, grd_E]], label="Ground State Energy").opts(color="k", alpha=0.4, linestyle="dashed")
fig = fig.opts(ylabel="Sequence Energies", xlabel="Training Iterations", title="GQE Evaluations", fig_size=500)
fig

##############################################################################
#.. figure:: ../_static/demonstration_assets/gqe_training/gqe_performance.png
#    :align: center
#    :width: 90%

######################################################################
# Model sequence generation evaluation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 

######################################################################
# Random sequences
# ~~~~~~~~~~~~~~~~
# 
# -  Training set is generated randomly
# 
#    -  Already gives a minimum value close to the ground state energy
# 

grd_E

######################################################################
# .. rst-class:: sphx-glr-script-out
# 
#  .. code-block:: none
# 
#     -1.1372633205048763

train_ave_E = train_sub_seq_en[:, -1].mean()
train_ave_E

######################################################################
# .. rst-class:: sphx-glr-script-out
# 
#  .. code-block:: none
# 
#     -1.114531299423375

train_min_E = train_sub_seq_en[:, -1].min()
train_min_E

######################################################################
# .. rst-class:: sphx-glr-script-out
# 
#  .. code-block:: none
# 
#     -1.1369822088041555

train_pct_error_ave = abs(train_ave_E - grd_E) / abs(grd_E) *100
train_pct_error_ave

######################################################################
# .. rst-class:: sphx-glr-script-out
# 
#  .. code-block:: none
# 
#     1.9988353331759405

train_pct_error_min = abs(train_min_E - grd_E) / abs(grd_E) *100
train_pct_error_min

######################################################################
# .. rst-class:: sphx-glr-script-out
# 
#  .. code-block:: none
# 
#     0.02471825967235751

######################################################################
# Latest model
# ~~~~~~~~~~~~
# 
# -  Not best performing model but on average, very close to the ground state energy
# -  **Interesting that the model is able to generate a sequence with an energy that’s lower than the
#    energies in the training set**
# 

gen_token_seq_, _ = gpt.generate(
    n_sequences=1024, 
    max_new_tokens=seq_len, 
    temperature=0.001, 
    device="cuda"
)

gen_inds_ = (gen_token_seq_[:, 1:] - 1).cpu().numpy()
gen_op_seq_ = op_pool[gen_inds_]
true_Es_ = get_subsequence_energies(gen_op_seq_)[:, -1].reshape(-1, 1)
ave_trues_ = np.mean(true_Es_)
ave_trues_

######################################################################
# .. rst-class:: sphx-glr-script-out
# 
#  .. code-block:: none
# 
#     -1.1321997471813732

min_ave_trues_ = np.min(true_Es_)
min_ave_trues_

######################################################################
# .. rst-class:: sphx-glr-script-out
# 
#  .. code-block:: none
# 
#     -1.1370380157299025

pct_error_mean = abs(ave_trues_ - grd_E) / abs(grd_E) *100
pct_error_mean

######################################################################
# .. rst-class:: sphx-glr-script-out
# 
#  .. code-block:: none
# 
#     0.4452419446056904

pct_error_min = abs(min_ave_trues_ - grd_E) / abs(grd_E) *100
pct_error_min

######################################################################
# .. rst-class:: sphx-glr-script-out
# 
#  .. code-block:: none
# 
#     0.01981113528517062

######################################################################
# Best model saved
# ~~~~~~~~~~~~~~~~
# 
# -  Even better performance
# 

loaded = torch.load("./seq_len=4/trial7/gqe.pt")

loaded_token_seq_, _ = loaded.generate(
    n_sequences=1024, 
    max_new_tokens=seq_len, 
    temperature=0.001, 
    device="cuda"
)
# pred_Es = pred_Es.cpu().numpy()

loaded_inds_ = (loaded_token_seq_[:, 1:] - 1).cpu().numpy()
loaded_op_seq_ = op_pool[loaded_inds_]
loaded_true_Es_ = get_subsequence_energies(loaded_op_seq_)[:, -1].reshape(-1, 1)
loaded_ave_trues_ = np.mean(loaded_true_Es_)
loaded_ave_trues_

######################################################################
# .. rst-class:: sphx-glr-script-out
# 
#  .. code-block:: none
# 
#     -1.135559902483167

loaded_min_trues_ = np.min(loaded_true_Es_)
loaded_min_trues_

######################################################################
# .. rst-class:: sphx-glr-script-out
# 
#  .. code-block:: none
# 
#     -1.1372625492981558

pct_error_mean = abs(loaded_ave_trues_ - grd_E) / abs(grd_E) *100
pct_error_mean

######################################################################
# .. rst-class:: sphx-glr-script-out
# 
#  .. code-block:: none
# 
#     0.14978220004079995

pct_error_min = abs(loaded_min_trues_ - grd_E) / abs(grd_E) *100
pct_error_min

######################################################################
# .. rst-class:: sphx-glr-script-out
# 
#  .. code-block:: none
# 
#     6.781250274933829e-05

df_compare_Es = pd.DataFrame({
    "Source": ["Random", "Latest Model", "Best Model"], 
    "Aves": [train_sub_seq_en[:, -1].mean(), true_Es_.mean(), loaded_true_Es_.mean()],
    "Mins": [train_sub_seq_en[:, -1].min(), true_Es_.min(), loaded_true_Es_.min()],
    "Maxs": [train_sub_seq_en[:, -1].max(), true_Es_.max(), loaded_true_Es_.max()],
})
df_compare_Es

######################################################################
# .. rst-class:: sphx-glr-script-out
# 
#  .. code-block:: none
# 
#              Source      Aves      Mins      Maxs
#     0        Random -1.114531 -1.136982 -1.027878
#     1  Latest Model -1.132200 -1.137038 -1.125118
#     2    Best Model -1.135560 -1.137263 -1.125118

grd_E

######################################################################
# .. rst-class:: sphx-glr-script-out
# 
#  .. code-block:: none
# 
#     -1.1372633205048763

######################################################################
# ^ Model is able to generate energies outside of training set
# ------------------------------------------------------------
# 

# (
#     pd.Series(train_sub_seq_en[:, -1]).hvplot.kde(alpha=0.2, label="Random", xlabel="Energies") * 
#     pd.Series(true_Es_.ravel()).hvplot.kde(alpha=0.2, label="Latest Model") * 
#     pd.Series(loaded_true_Es_.ravel()).hvplot.kde(alpha=0.2, label="Best Model")
# ) * (
#     hv.VLine(train_sub_seq_en[:, -1].min()) * hv.VLine(true_Es_.min()) * hv.VLine(loaded_true_Es_.min())
# )

######################################################################
# Next steps
# ----------
# 

######################################################################
# -  learning rate schedule
# -  early stopping mechanism
# -  regularization
# -  optimize code, use all of gpu (evaluate energies on gpu?)
# -  try other generative algorithms?
# 

######################################################################
# Online learning -> training in the paper
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# For each training iteration/epoch: 1. Sample sequences from current GPT model - decrease temp of
# sampling 2. For each minibatch, calculate the loss, the gradients, and take an optimizer step 3. For
# each nth iteration, evaluate the GPT model
# 