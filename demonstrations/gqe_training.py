r"""Generative quantum eigensolver (GQE) training using data generated with PennyLane
===================================================

In this demo, we will be training a generative quantum eigensolver (GQE) and applying the technique described in `this
paper <https://arxiv.org/abs/2401.09253>`__, using the molecular data available in `PennyLane Datasets <https://pennylane.ai/datasets/>`__.
We will show that the model gradually better approximates the correct energies and, in turn, 
can sample energies close to the ground state energy calculated by PennyLane. 

.. figure:: ../_static/demo_thumbnails/opengraph_demo_thumbnails/OGthumbnail_generative_quantum_eigensolver.png
    :align: center
    :width: 70%
    :target: javascript:void(0)

The GQE algorithm is an alternative approach in estimating the ground state of a particular molecule.
Usually, this ground state estimation is done via the variational quantum eigensolver (VQE) approach, 
where the quantum state is represented as a quantum circuit with tunable parameters. The goal is then 
to find the optimal parameters that minimizes the corresponding energy :math:`E`. For more details on
VQEs, check out this `PennyLane Demo <https://pennylane.ai/qml/demos/tutorial_vqe/>`__ and 
`Documentation <https://docs.pennylane.ai/projects/catalyst/en/stable/demos/adaptive_circuits_demo.html>`__.

.. figure:: ../_static/demonstration_assets/gqe_training/paper_vqe_diagram.png
    :align: center
    :width: 90%
    :alt: Figure 1 from [#nakaji2024]_

There are some issues with VQE scalability, however. This shortcoming makes it less competitive against
the performance of classical ML algorithms for large problems. To bypass this, the GQE algorithm was 
proposed. A GQE is then a generative model where quantum states represented by quantum circuits are 
sampled. The generative model is then trained so that the states being sampled more closely approximates
the ground state. 

.. figure:: ../_static/demonstration_assets/gqe_training/paper_gqe_diagram.png
    :align: center
    :width: 90%
    :alt: Figure 1 from [#nakaji2024]_

The main difference between the two approaches is where the tunable parameters are embedded.
That is, it is the classical GQE model that is being optimized as opposed to the variable
quantum circuit of VQE. Potentially then, the loss landscape for GQE will be different and
will be amenable for larger problems.

This demo is organized as follows. In Section 1, we describe GPT-QE, (a particular
design of the GQE algorithm which uses a GPT model) what it generates, and how we train it. 
In Section 2, we generate the training dataset we will use by using PennyLane. In Section 3,
we give details on our GPT model architecture and training implementation. In Section 4, we 
evaluate the model throughout its training and discuss its performance in estimating the ground
state. And lastly in Section 5, we conclude.
"""

######################################################################
# 1. GPT-QE Background
# --------------------
# 
# In particular, the chosen model design in the paper was the generative pre-trained
# transformer (GPT) architecture. So, a GQE using a transformer is called GPT-QE. As a language model, GPTs are successful in generating
# sequences of words that closely resemble human natural language. This performance is
# harnessed for quantum chemistry by constructing quantum states :math:`\rho` as a sequence of unitary operators 
# which are in turn, represented by quantum circuits. That is, we let :math:`\rho = U\rho_0 U^{\dagger}`
# for some fixed initial state :math:`\rho_0` and the aforementioned sequence is :math:`U = U_{j_N}U_{j_{N-1}}\cdots U_{j_1}`.
# The GPT model generates the sequence of integers :math:`j_1, j_2, ..., j_N` indexing a pool 
# of operators :math:`U_j`'s. We interpret these integers as tokens and the pool as the vocabulary in the parlance for language models. 
# The goal of training is then to minimize the corresponding energy
# :math:`E = \mbox{Tr}(\hat{H}\rho)` where :math:`\hat{H}` is the hamiltonian of the molecule in 
# question.
# 
# Each token :math:`j_i` is sampled from the distribution :math:`\exp(-\beta w_{j_i})` where
# :math:`w_{j_i}` is the unnormalized log probability (or logit) returned by the GPT model for the token :math:`j_i`
# and :math:`\beta` is an inverse temperature representing a trade-off parameter between exploration and exploitation. We then
# observe that the probability of sampling a state through the method described above is 
# proportional to :math:`\exp(-\beta w_{\mbox{sum}})` where :math:`w_{\mbox{sum}} = \sum_{i=1}^N w_{j_i}` 
# and the probability for the corresponding energy is :math:`\exp(-\beta E)`. We thus have a constraint 
# for the total logit to be equal to the energy of the corresponding state: :math:`w_{\mbox{sum}} = E` which
# can be imposed by training GPT-QE to minimize the loss function :math:`C = (w_{\mbox{sum}} - E)^2`.
# With this constraint satisfied, GPT-QE would then be sampling states of smaller energies with increasing
# likelihood.
#  
# More concretely, we summarize the "pre-"training loop in the following diagram. This is called
# pre-training because the learning is done using a fixed dataset first before the "real" training 
# is done based on the data it generated on its own. In this demo, we will call the pre-training as offline
# training since the GPT model does not receive feedback from the sequences it samples, and online
# training if otherwise.

##############################################################################
#.. figure:: ../_static/demonstration_assets/gqe_training/gqe_training_diagram.png
#    :align: center
#    :width: 90%

######################################################################
# 2. Dataset construction via PennyLane
# -------------------------------------
# 
# Firstly, let us construct the static dataset we will use for offline training. We choose
# to generate our own dataset in order to illustrate the sequences and energies more concretely.
# Our dataset will be made from random sequences of tokens, which we recall corresponds to indices
# of a vocabulary of unitary operators. We then define an energy function in PennyLane to calculate 
# the energy of a state corresponding to a token sequence. Applying the aforementioned function,
# we would then have a dataset of token sequences and energies for the GPT model offline training.

######################################################################
# 2a. Loading molecular information
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# For simplicity, let us consider the  `hydrogen gas molecule <https://pennylane.ai/datasets/qchem/h2-molecule>`__ 
# and load the correspoding dataset from PennyLane.
# Recall that we would need a vocabulary of operators :math:`U_j`'s, an initial state :math:`\rho_0`, and 
# the hamiltonian :math:`\hat{H}` for hydrogen gas. We also get the ground state energy for later comparison
# with the results.
# 

import numpy as np
import pennylane as qml

def generate_molecule_data(molecules=["H2", "LiH", "BeH2", "H2O", "N2"]):
    # Get the same molecules as in the paper, with the addition of water
    datasets = qml.data.load("qchem", molname=molecules)

    # Get the time set T
    operator_times = np.sort(np.array([-2**k for k in range(1, 5)] + [2**k for k in range(1, 5)]) / 160)

    # Build operator set P for each molecule
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
# 2b. Defining the energy function
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# In PennyLane, we define the energy function :math:`E = \mbox{Tr}(\hat{H}U_{j_N}\cdots U_{j_1}\rho_0 U_{j_1}^{\dagger}\cdots U_{j_N}^{\dagger})`
# corresponding to Eq. 1 of [#nakaji2024]_. Here, ``energy_circuit`` takes in the operator sequence :math:`U_{j_1}, U_{j_2}, ..., U_{j_N}`
# and returns the energy of the corresponding quantum state.
#
# As a slight extension of the paper, we can also calculate the energies for each subsequence of
# operators to help with the training of the model. That is, for a sequence of three operators
# :math:`U_{j_1}, U_{j_2}, U_{j_3}` we compute the energies for :math:`U_{j_1}` and :math:`U_{j_1}, U_{j_2}` instead of just
# the full sequence of three operators, which was described in the paper. This can be done simply in PennyLane, using :class:`~.pennylane.Snapshot`.
# 

# This computes the energy for a chosen molecule with the selected operator pool
dev = qml.device("default.qubit", wires=num_qubits)

@qml.qnode(dev)
def energy_circuit(gqe_ops):
    # Computes Eq. 1 of [1] based on the selected unitary operators
    qml.BasisState(init_state, wires=range(num_qubits)) # Initial state <-- Hartree Fock state
    for op in gqe_ops:
        qml.Snapshot(measurement=qml.expval(hamiltonian))
        qml.apply(op) # Applies each of the unitary operators
    return qml.expval(hamiltonian) # Computes the energy expectation value as in Eq. 1 of [1]

energy_circuit = qml.snapshots(energy_circuit)

def get_subsequence_energies(op_seq):
    # Collates the energies of each subsequence for a batch of sequences
    energies = []
    for ops in op_seq:
        es = energy_circuit(ops)
        energies.append(
            [es[k].item() for k in list(range(1, len(ops))) + ["execution_results"]]
        )
    return np.array(energies)

######################################################################
# 2c. Token sequence generation with corresponding energies
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# With these ingredients, we can now construct a dataset containing sequences of tokens and their energies. 
# Since we cannot feed the operators directly to the GPT model, we would need to tokenize
# them. The indices of ``op_pool`` seems to be a good candidate, but we instead choose the tokens to be
# the ``op_pool`` indices shifted by 1. This is so that we can define a special token ``0`` that tells
# the GPT model where the sequence starts.
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

# Corresponding tokens with special starting tokens
train_token_seq = np.concatenate([
    np.zeros(shape=(train_size, 1), dtype=int), # starting token is 0
    train_op_pool_inds + 1 # shift operator inds by one
], axis=1)

# %%time 
# Calculate the energies for each subsequence in the training set
train_sub_seq_en = get_subsequence_energies(train_op_seq)

######################################################################
# .. rst-class:: sphx-glr-script-out
# 
#  .. code-block:: none
# 
#     CPU times: user 47 s, sys: 37.5 ms, total: 47 s
#     Wall time: 47 s

##############################################################################
# We also measure the time to calculate the subsequence energies. Currently, 47s is acceptable 
# but this can be a bottleneck when generating sequences of longer lengths and more complicated 
# molecules. Code optimization here can be done in the future.
#  

######################################################################
# 3. GPT-QE offline training
# --------------------------
# Having setup our training dataset, we can start implementing our offline training loop as
# illustrated by our diagram. We outline our implementation below.

######################################################################
# 3a. GPT model implementation details
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# The GPT model we will use in this demo is mostly implemented in the `nanoGPT repo
# <https://github.com/karpathy/nanoGPT>`__ as the 
# `class <https://github.com/karpathy/nanoGPT/blob/9755682b981a45507f6eb9b11eadef8cb83cebd5/model.py#L118>`__ 
# ``GPT`` with the model hyperparameters stored in the 
# `dataclass <https://github.com/karpathy/nanoGPT/blob/9755682b981a45507f6eb9b11eadef8cb83cebd5/model.py#L109>`__ 
# ``GPTConfig``. Namely, we will use 12 attention layers, 12 attention heads, and 768 embedding dimensions
# which are equal to those described in [#nakaji2024]_. 
# We can import from the nanoGPT repo directly by running the curl command below in the terminal
# or the jupyter notebook. Since nanoGPT is trained as a language model, its loss function and sampling method is 
# defined differently. We then define the subclass ``GPTQE`` below to override some nanoGPT methods in order to make it more
# suitable for our case. 
# 

# !curl -O https://raw.githubusercontent.com/karpathy/nanoGPT/master/model.py
from model import GPT, GPTConfig
import torch
from torch.nn import functional as F

class GPTQE(GPT):
    def forward(self, idx):
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits
    
    def calculate_loss(self, tokens, energies):
        current_tokens, next_tokens = tokens[:, :-1], tokens[:, 1:]
        # calculate the logits for the next possible tokens in the sequence
        logits = self(current_tokens)
        # get the logit for the actual next token in the sequence
        next_token_mask = torch.nn.functional.one_hot(
            next_tokens, num_classes=self.config.vocab_size
        )
        next_token_logits = (logits * next_token_mask).sum(axis=2)
        # calculate the cumulative logits for each subsequence
        cumsum_logits = torch.cumsum(next_token_logits, dim=1)
        # match cumulative logits to subsequence energies
        loss = torch.mean(torch.square(cumsum_logits - energies))
        return loss
    
    @torch.no_grad()
    def generate(self, n_sequences, max_new_tokens, temperature=1.0, device="cpu"):
        idx = torch.zeros(size=(n_sequences, 1), dtype=int, device=device)
        total_logits = torch.zeros(size=(n_sequences, 1), device=device)
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits = self(idx_cond)
            # pluck the logits at the final step 
            logits = logits[:, -1, :] 
            # set the logit of the first token so that its probability will be zero
            logits[:, 0] = float("inf")
            # apply softmax to convert logits to (normalized) probabilities and scale by desired temperature
            probs = F.softmax(-logits / temperature, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # # Accumulate logits
            total_logits += torch.gather(logits, index=idx_next, dim=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
        return idx, total_logits
 
# Strictly speaking however, the loss function ``calculate_loss`` we defined is different from that
# described in [#nakaji2024]_ which is :math:`(\exp(-w_{\mbox{sum}}) - \exp(-E))^2`. As described
# in Section 1, we directly compute the mean squared error between :math:`\mbox{sum}` and :math:`E`.
# Using the error between exponentials should be unnecessary since the exponential function is 1-to-1 and this may even 
# introduce numerical instabilities in the training since the loss would be taking differences of potentially large numbers.
# In addition to this change from [#nakaji2024]_, we also use the error between the cumulative sum of logits and the corresponding energy
# for each subsequence instead of just the error between total logits and the energy of an entire sequence. 
# This addition will give more training data to the model and should help with logit matching the intermediate tokens.
#
# Since it was not explicitly shown in [#nakaji2024]_, another possible deviation we made is the logit calculation during offline training. 
# It seems that the logits were accumulated by looping through the sequential generation of tokens. This is inefficient and unnecessary   
# for offline training since we can directly pass the fixed training sequences to the model and retrieve the relevant logits from it.
# This can be done because we are using a causal mask for the attention blocks so that the logits of the earlier tokens in the sequence are 
# not affected by tokens in the later part of the sequence. Thus, having the same effect of the sequential token generation.
# 
# We initialize our GPT model below and we see that it has around 85 million parameters and is ``324.25 MB`` in size.

tokens = torch.from_numpy(train_token_seq).to("cuda")
energies = torch.from_numpy(train_sub_seq_en).to("cuda")

gpt = GPTQE(GPTConfig(
    vocab_size=op_pool_size + 1,
    block_size=seq_len,
    dropout=0.2,
    bias=False
)).to("cuda")
opt = gpt.configure_optimizers(
    weight_decay=0.01, learning_rate=5e-5, betas=(0.9, 0.999), device_type="cuda"
)

######################################################################
# .. rst-class:: sphx-glr-script-out
# 
#  .. code-block:: none
# 
#     number of parameters: 84.98M
#     num decayed parameter tensors: 50, with 84,963,072 parameters
#     num non-decayed parameter tensors: 25, with 19,200 parameters

######################################################################
# 3b. GPT offline training loop implementation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# We now implement a training loop for our GPT model. This can be framed as a straightforward
# supervised learning problem. We sketch the steps for each training iteration/epoch below:
# 
# 1. Shuffle the training set and split it into ``n_batches`` minibatches
# 2. For each minibatch, calculate the average loss, the gradients, and take an optimizer step 
# 3. For each n-th iteration (500 here), evaluate the GPT model: 
# 
#    3a. Generate a batch of sequences and the predicted energies (total logits). Note that these are not necessarily same sequences in training set.
# 
#    3b. Calculate the true energies using PennyLane
# 
#    3c. Calculate the mean absolute error as a metric to track the learning progress and save the GPT model everytime the metric gets better
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
        loss = gpt.calculate_loss(token_batch, energy_batch)
        loss.backward()
        opt.step()
        loss_record += loss.item() / n_batches
    losses.append(loss_record)

    if (i+1) % 500 == 0:
        # For GPT evaluation
        gpt.eval()
        gen_token_seq, pred_Es = gpt.generate(
            n_sequences=100, 
            max_new_tokens=seq_len, 
            temperature=0.001, # Use a low temperature to emphasize the difference in logits
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
# 4. GPT-QE results
# -----------------
# Having finished the offline training, let's take a look at some of our results.

######################################################################
# 4a. Loss curve
# ~~~~~~~~~~~~~~
# One of the first things we can look at is the training loss curve. We see in Figure 1 that the average loss
# continues to decrease until around the 4000th iteration. There, the model was erroraneous but is quick
# to recover as training continues. This may signal that the GPT model started focusing on learning something
# erroraneous too quickly. So, more regularization noise (like ``dropout``) may be needed to help avoid this. 
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
#    :alt: Figure 1: The average subsequence loss for each training iteration

######################################################################
# 4b. Evaluation progress
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# We now track the performance of the GPT model throughout its training in Figure 2. As mentioned
# before, after every 500th iteration, we let the model generate a batch of sequences. Alongside, we
# also return the total logits (predicted energies) used in the sequence generation. In Figure 2,
# the average predicted energies corresponds to the red markers and the distribution of predicted
# energies is represented by the red area. Once we have the generated sequences, we can also let PennyLane calculate the true sequence
# energies. Similarly then, the blue markers are the average true energies and the blue
# area represents the true energy distribution.
# 

df_true = pd.read_csv("./seq_len=4/trial7/true_Es_t.csv").iloc[:, 1:]
df_pred = pd.read_csv("./seq_len=4/trial7/pred_Es_t.csv").iloc[:, 1:]

df_true.columns = df_true.columns.astype(int)
df_pred.columns = df_pred.columns.astype(int)

df_trues_stats = pd.concat([df_true.mean(axis=0), df_true.min(axis=0), df_true.max(axis=0)], axis=1)
df_trues_stats.columns = ["Ave True E", "Min True E", "Max True E"]

df_preds_stats = pd.concat([df_pred.mean(axis=0), df_pred.min(axis=0), df_pred.max(axis=0)], axis=1)
df_preds_stats.columns = ["Ave Pred E", "Min Pred E", "Max Pred E"]

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
#    :alt: Figure 2: True and predicted energies for sequences generated by the GPT model for each 500th training iteration

# We now see that the energies predicted by the model get more accurate at approximating the true
# energies during training. This in turn, samples lower energies as we see that the true energies
# sampled gets closer to the ground state energy (the dashed line).
# 
# Note that at around the 4000th iteration, the predicted energies are very far from the true energies.
# This makes sense considering our observation in Figure 1. Also note that at the 7000th
# iteration, the averages of the predicted and true energies are the closest and even their respective
# spreads seem to have good overlap. For later iterations however, the predicted energies no longer improved.
# This may indicate that the GPT model has started overfitting on the training dataset in the later iterations. 
# That is, the model became great at predicting the correct energies for the training set (as observed in the 
# loss curve) but not great at generalizing on those outside the training set (like the sequences that the model
# generated on its own). One solution to avoid overfitting could then be online training. This is so that 
# the GPT model is not restricted on a fixed dataset to overfit on.
# 

######################################################################
# 4c. Sequence generation comparison
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# Here, we compare some statistics of the true energies corresponding to sequences generated
# by a "random" model, the latest model after all the training iterations, and the best model
# saved based on the mean absolute error between the true and predicted energies during training 
# (for our case, this is the model saved at the 7000th iteration).
# Note that we consider the training set to be generated by a random model since the token
# sequences are just sampled uniformly. 
# 

# Latest model
gen_token_seq_, _ = gpt.generate(
    n_sequences=1024, 
    max_new_tokens=seq_len, 
    temperature=0.001, 
    device="cuda"
)
gen_inds_ = (gen_token_seq_[:, 1:] - 1).cpu().numpy()
gen_op_seq_ = op_pool[gen_inds_]
true_Es_ = get_subsequence_energies(gen_op_seq_)[:, -1].reshape(-1, 1)

# Best model
loaded = torch.load("./seq_len=4/trial7/gqe.pt")
loaded_token_seq_, _ = loaded.generate(
    n_sequences=1024, 
    max_new_tokens=seq_len, 
    temperature=0.001, 
    device="cuda"
)
loaded_inds_ = (loaded_token_seq_[:, 1:] - 1).cpu().numpy()
loaded_op_seq_ = op_pool[loaded_inds_]
loaded_true_Es_ = get_subsequence_energies(loaded_op_seq_)[:, -1].reshape(-1, 1)

# Summary table
df_compare_Es = pd.DataFrame({
    "Source": ["Random", "Latest Model", "Best Model"], 
    "Aves": [train_sub_seq_en[:, -1].mean(), true_Es_.mean(), loaded_true_Es_.mean()],
    "Mins": [train_sub_seq_en[:, -1].min(), true_Es_.min(), loaded_true_Es_.min()],
    "Maxs": [train_sub_seq_en[:, -1].max(), true_Es_.max(), loaded_true_Es_.max()],
    "Mins_error": [
        abs(train_sub_seq_en[:, -1].min() - grd_E),
        abs(true_Es_.min() - grd_E),
        abs(loaded_true_Es_.min() - grd_E),
    ],
})
df_compare_Es

######################################################################
# .. rst-class:: sphx-glr-script-out
# 
#  .. code-block:: none
# 
#              Source      Aves      Mins      Maxs   Mins_error
#     0        Random -1.114531 -1.136982 -1.027878 2.811117e-04
#     1  Latest Model -1.132200 -1.137038 -1.125118 2.253048e-04
#     2    Best Model -1.135560 -1.137263 -1.125118 7.712067e-07

# We observe that the minimum energy corresponding to the random training set is already close
# to the ground state energy ``grd_E=-1.1372633205048763`` with an error of around ``2.81e-04``. But we notice that the maximum energy
# is relatively far so the random sequences give a wider spread of energies.
# 
# The energies of the generated sequences from the GPT models however have a narrower spread and so,
# the average and the minimum energies are very close to ``grd_E``, closer than those in the random 
# training set. Namely, around a ``2.25e-04`` error for the minimum energy generated by the latest model
# and a ``7.71e-07`` error for the best model. It is then very interesting that
# the models can generate sequences that are better than those in the training set, even though every
# sequence the models have seen just came from that set. The models were able to generalize. 
# 
# Between the two GPT models, we see that the latest model is worse than the best model. The minimum energy
# error for the latest model has the same order of magnitude as the corresponding one for the random training set.
# Contrast this with the minimum energy error for the best model which is 3 orders of magnitude smaller.
# This behavior is supported by our observation in Figure 2 where the performance of the models saved after 
# the 7000th iteration worsened. That is, the predicted energies started to deviate further from the true energies
# which in turn caused the states being sampled from these predicted energies to be different from the intended 
# lower energy states.
# 

######################################################################
# 5. Conclusion
# -------------
# 
# In this demo, we see that GPT-QE is a viable alternative in estimating the ground state
# of a hydrogen gas molecule. The best underlying GPT model can generate a state whose energy is 
# only around ``7.71e-07`` away from ground state energy. The offline training algorithm (without the sequence 
# evaluations) is completely detached from the quantum simulations. Thus, the machinery of classical ML 
# can be harnessed and circumventing the issues of VQE.
# 
# The reader can also experiment with other molecules from PennyLane and tweak several hyperparameters
# of the GPT model (like ``dropout``, and ``n_layer``) and include standard ML callbacks to its training
# (like an early stopping mechanism, and a learning rate schedule). The code itself as well is open to
# optimization like using `PennyLane Lightning GPU <https://docs.pennylane.ai/projects/lightning/en/latest/lightning_gpu/device.html>`__ 
# to evaluate the energies faster. An online training loop can also be implemented similar to our offline
# version. The reader would just need to sample sequences from the current GPT model instead of a fixed
# dataset for each training iteration. To facilitate exploration and exploitation, one would also define
# a schedule for the inverse temperature. Initially letting the GPT model sample more randomly through
# a high temperature then gradually decreasing so that the GPT model focuses more on the higher probability
# states (low energies). 
# 

######################################################################
# Reference
# ---------
# 
# .. [#nakaji2024]
#
#     Kouhei Nakaji *et al.*, "The generative quantum eigensolver (GQE) and its application for ground state search". `Nature Communications 5, 4213 (2014).
#     `arXiv:2401.09253 <https://arxiv.org/abs/2401.09253>`__
# 