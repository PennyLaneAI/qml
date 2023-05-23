r"""
---

**Learning outcomes**

 - *Prove why we cannot design a quantum circuit that can clone arbitrary states.*
 - *Describe the quantum teleportation protocol.*
 
---

Suppose there are two labs, one led by Alice and the other by Bob. Alice and
Bob, having grown weary of their fame and the steady stream of requests to test
novel communications protocols, have taken an extended holiday. They've each
left a student, Ayanda and Brian, to manage their respective labs in their
absence. Over the course of her work, Ayanda has designed a complicated
apparatus that produces some very curious quantum states, and she would like to
send them over to Brian for further study. How can she relay her states to him?
One way of course is to physically move her equipment into Brian's lab and
produce the states there; but it would be very cumbersome to have to tear down
and rebuild her whole setup (furthermore, what if Brian's lab was halfway across
the world?).

One way for Ayanda to transmit a quantum state to Brian is by using **quantum
teleportation**. Teleportation is a protocol that transmits the *state* of a
qubit (not the qubit itself), from one location to another. The quantum circuit
for teleportation, which we will soon explore in great detail, is shown below:

<img src="images/teleportation.svg" width="400px">

Teleportation is a great way to end this introductory module, because it
incorporates all the key concepts we've seen so far:

<img src="images/teleportation-annotated.svg" width="500px">

## Why teleport?

You might be wondering why we need to teleport a state at all. Can't Ayanda
just make a copy of it and send the copy to Brian? It turns out that copying
arbitrary states is *prohibited*, which you can prove using something called the
**no-cloning theorem**. The proof is surprisingly straightforward. Suppose we
would like to design a circuit (unitary) $U$ that can perform the following
action:

$$
\begin{align*}
U(\vert \psi\rangle \otimes \vert s\rangle ) &= \vert \psi\rangle \otimes \vert \psi\rangle, \\
U(\vert \varphi\rangle \otimes \vert s\rangle ) &= \vert \varphi \rangle \otimes \vert \varphi \rangle,
\end{align*}\tag{1}
$$

where $\vert \psi\rangle$ and $\vert \varphi\rangle$ are arbitrary single-qubit
states, and $\vert s \rangle$ is some arbitrary starting state. Through the next
exercise, you'll prove that no such $U$ exists!

---

***Exercise I.15.1.*** Work through the following steps to prove the no-cloning
   theorem.

 1. Take the inner product of the left-hand-sides of the two equations.
 2. Take the inner product of the right-hand-sides of the two equations above.
 3. These inner products must be equal; if they are equal, what does this say about the properties of the states $\vert \psi\rangle$ and $\vert \varphi\rangle$? Can we use $U$ to clone arbitrary states?


<details>
  <summary><i>Hint.</i></summary>


The inner product works multiplicatively "across" the tensor
 product. Namely, $(\langle c \vert \otimes \langle d \vert ) (\vert a\rangle
 \otimes \vert b\rangle )$ = $(\langle c \vert a\rangle) \cdot (\langle d \vert
 b \rangle)$.

</details>


<details>
  <summary><i>Solution.</i></summary>

 1. The first inner product is $\langle \psi \vert  \varphi \rangle$.
 2. The second inner product is $(\langle \psi \vert  \varphi \rangle)^2$.
 3. If they are equal, the inner product is a number that squares to itself. The only valid values for the inner product then are 1 and 0. But if the inner product is 1, the states are the same; on the other hand, if the inner product is 0, the states are orthogonal. Therefore, we can't clone arbitrary states! ▢

</details>

---

## So, what is quantum teleportation?

Now that we know we can't arbitrarily copy states, we return to the task of
teleporting them. Teleportation relies on Ayanda and Brian having access to
shared entanglement. The protocol can be divided into roughly four parts. We'll
go through each of them in turn.

<img src="images/teleportation-4part.svg" width="500px">

### 1. State preparation

Teleportation involves three qubits. Two of them are held by Ayanda, and the
third by Brian. We'll denote their states using a subscript $\vert
\cdot\rangle_A$ and $\vert \cdot\rangle_B$ for clarity. Together, their starting
state is

$$
\begin{equation}
\vert 0\rangle_A \vert 0\rangle_A \vert 0\rangle_B. 
\end{equation}\tag{2}
$$

The first thing Ayanda does is prepare her first qubit in whichever state $\vert
\psi\rangle$ that she'd like to send to Brian, so that their combined state
becomes

$$
\begin{equation}
\vert \psi\rangle_A \vert 0\rangle_A \vert 0\rangle_B.
\end{equation} \tag{3}
$$

### 2. Shared entanglement

The reason why teleportation works as it does is the use of an *entangled state*
as a shared resource between Ayanda and Brian. You can imagine it being
constructed in the circuit, like so, but you can also imagine some sort of
centralized entangled state generator that produces Bell states, and sends one
qubit to each party.

Entangling the second and third qubits leads to the combined state

$$
\begin{equation}
\frac{1}{\sqrt{2}}\left( \vert \psi\rangle_A \vert 0\rangle_A \vert 0\rangle_B + \vert \psi\rangle_A \vert 1\rangle_A \vert 1\rangle_B \right)
\end{equation}.\tag{4}
$$

### 3. Change of basis

This is where things get tricky, but also very interesting. The third step of
the protocol is to apply a $CNOT$ and a Hadamard to the first two qubits. This is
done prior to the measurements, and labelled "change of basis". But what basis
is this? Notice how these two gates are the *opposite* of what we do to create a
Bell state. If we run them in the opposite direction, we transform the basis
back to the computational one, and simulate a measurement in the *Bell
basis*. The Bell basis is a set of four entangled states

$$
\begin{align*}
\vert \psi_+\rangle &= \frac{1}{\sqrt{2}} \left( \vert 00\rangle + \vert 11\rangle \right), \\
\vert \psi_-\rangle &= \frac{1}{\sqrt{2}} \left( \vert 00\rangle - \vert 11\rangle \right), \\
\vert \phi_+\rangle &= \frac{1}{\sqrt{2}} \left( \vert 01\rangle + \vert 10\rangle \right), \\
\vert \phi_-\rangle &= \frac{1}{\sqrt{2}} \left( \vert 01\rangle - \vert 10\rangle \right).
\end{align*} \tag{5}
$$

After the basis transform, if we observe the first two qubits to be in the state
$\vert 00\rangle$, this would correspond to the outcome $\vert \psi_+\rangle$ in
the bell basis, $\vert 11\rangle$ would correspond to $\vert \phi_-\rangle$,
etc.

---

***Exercise I.15.2.*** Write $\vert \psi\rangle = \alpha\vert 0\rangle +
   \beta\vert 1\rangle$, and evaluate the action of the $CNOT$ and Hadamard on
   the state prior to the change of basis. Then, rearrange the results so that
   you have a linear combination of terms where the first two qubits are in
   computational basis states.

<details>
  <summary><i>Solution.</i></summary>

Expanding out the terms (and removing the subscripts for brevity), we obtain
$$
\begin{equation}
\frac{1}{\sqrt{2}} ( \alpha\vert 000\rangle + 
\beta\vert 100\rangle + \alpha \vert 011\rangle + 
\beta\vert 111\rangle )
\end{equation}.
$$

Now let's apply a $CNOT$ between Ayanda's two qubits:

$$
\begin{equation}
\frac{1}{\sqrt{2}} ( \alpha\vert 000\rangle + 
\beta\vert 110\rangle + \alpha \vert 011\rangle + 
\beta\vert 101\rangle )
\end{equation}.
$$

And then a Hadamard on her first qubit:

$$
\frac{1}{2} ( \alpha \vert 000\rangle + \alpha\vert 100\rangle + \beta\vert 010\rangle - \beta\vert 110\rangle + \alpha \vert 011\rangle + \alpha \vert 111 \rangle + \beta\vert 001\rangle - \beta\vert 101 \rangle ).
$$

Now we need to do some rearranging. We group together the terms based on the
first two qubits:

$$
\frac{1}{2} \vert 00\rangle(\alpha\vert 0\rangle + \beta\vert 1\rangle) + \frac{1}{2}\vert 01\rangle (\beta\vert 0\rangle + \alpha\vert 1\rangle) + \frac{1}{2}\vert 10\rangle (\alpha\vert 0\rangle - \beta\vert 1\rangle) + \frac{1}{2}\vert 11\rangle (-\beta\vert 0\rangle + \alpha\vert 1\rangle).
$$

<div align="right"> ▢ </div>

</details>

### 4. Measurement

The last step of the protocol involves applying two controlled operations from
Ayanda's qubits to Brian, a controlled-$Z$, and a $CNOT$, followed by a
measurement. But why exactly are we doing this before the measurement? In the
previous step, we already performed a basis rotation back to the computational
basis, so shouldn't we be good to go?

Let's take a closer look at the state you should have obtained from the previous
exercise:

$$
\frac{1}{2} \vert 00\rangle(\alpha\vert 0\rangle + \beta\vert 1\rangle) + \frac{1}{2}\vert 01\rangle (\beta\vert 0\rangle + \alpha\vert 1\rangle) + \frac{1}{2}\vert 10\rangle (\alpha\vert 0\rangle - \beta\vert 1\rangle) + \frac{1}{2}\vert 11\rangle (-\beta\vert 0\rangle + \alpha\vert 1\rangle). \tag{6}
$$

If Ayanda measures her two qubits in the computational basis, she is equally
likely to obtain any of the four possible outcomes. If she observes the first two
qubits in the state $\vert 00 \rangle$, she would immediately know that Brian's
qubit was in the state $\alpha \vert 0 \rangle + \beta \vert 1 \rangle$, which is
precisely the state we are trying to teleport!

If instead she observed the qubits in state $\vert 01\rangle$, she'd still
know what state Brian has, but it's a little off from the original state. In particular,
we have

$$
\beta \vert 0 \rangle + \alpha \vert 1 \rangle = X \vert \psi \rangle. \tag{7}
$$

Ayanda could tell Brian, after obtaining these results, to simply apply an $X$
gate to his qubit to recover the original state. Similarly, if she obtained
$\vert 10\rangle$, she would tell him to apply a $Z$ gate.

In a more ["traditional" version of
teleportation](https://quantum.country/teleportation), this is, in fact, exactly
what happens. Ayanda would call up Brian on the phone, tell him which state she
observed, and then he would be able to apply an appropriate correction. In this
situation, measurements are happening partway through the protocol, and the
results would be used to control the application of future quantum gates.

Here, we are presenting a slightly different version of teleportation that
leverages the [principle of deferred
measurement](https://en.wikipedia.org/wiki/Deferred_Measurement_Principle). Basically,
we can push all our measurements to the *end* of the circuits.

<img src="images/teleportation-4part.svg" width="500px">

You might have wondered why these two gates were included in the measurement box
in the diagrams; this is why. Ayanda applying controlled operations on Brian's
qubit is performing this same kind of correction *before* any measurements are
made. In the next exercise, you can work this out for yourself!


---

***Exercise I.15.3.*** Using the final state you obtained in the previous
   exercise, evaluate the action of the $CNOT$ and controlled $Z$ on Brian's
   qubit. Has Ayanda's state been successfully teleported?


<details>
  <summary><i>Solution.</i></summary>

Applying the $CNOT$ yields

$$
\begin{equation}
\frac{1}{2} \vert 00\rangle(\alpha\vert 0\rangle + \beta\vert 1\rangle) + \frac{1}{2}\vert 01\rangle (\alpha\vert 0\rangle + \beta\vert 1\rangle) + \frac{1}{2}\vert 10\rangle (\alpha\vert 0\rangle - \beta\vert 1\rangle) + \frac{1}{2}\vert 11\rangle (\alpha\vert 0\rangle - \beta\vert 1\rangle)
\end{equation}.
$$

Then, applying the $CZ$ yields

$$
\begin{equation}
\frac{1}{2} \vert 00\rangle(\alpha\vert 0\rangle + \beta\vert 1\rangle)  + \frac{1}{2}\vert 01\rangle (\alpha\vert 0\rangle + \beta\vert 1\rangle) + \frac{1}{2}\vert 10\rangle (\alpha\vert 0\rangle + \beta\vert 1\rangle) + \frac{1}{2}\vert 11\rangle (\alpha\vert 0\rangle + \beta\vert 1\rangle)
\end{equation}.
$$

When Ayanda measures her two qubits at the end, no matter which outcome she
gets, Brian's qubit will be in the state $\alpha\vert 0\rangle + \beta \vert
1\rangle$.  This means that our protocol has changed the state of Brian's qubit
into the one Ayanda wished to send him, which is truly incredible! ▢

</details>
"""