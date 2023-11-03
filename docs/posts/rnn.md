---
date: 2023-10-29
---

# Recurrent Neural Networks and the Embedded Reber Grammar

Recurrent neural networks are a special type neural network that have been heavily studied for decades towards 
problems involving sequence prediction, which standard feed-forward neural networks tend not to be that great at. 
What makes RNNs different is that it processes input sequentially and generates hidden states 
which are then used in future computations. 

Here, we'll offer an overview of RNNs, present an explicit RNN, and then implement an RNN in Pytorch to learn an artificial grammar 
known as the Reber Grammar. 

<!-- more -->

## Recurrent Neural Networks

The aim of recurrent neural networks is to be able to predict unseen sequences of inputs given training from a set of similar 
sequences of inputs. By a sequence of inputs, we mean a sequence $(x_1, x_2, \dots, x_T)$ of elements, where $x_i \in \mathbb{R}^n$ 
for some dimension $n$. In the typical application of using an RNN for a language model, the $x_i \in \mathbb{R}^n$ can 
represent words, which are one-hot encoded. However, we will keep the discussion regarding the sequence elements general, only
enforcing that each element in the sequence have the same dimension. 

From a very high level, the typical visualization of computing a forward pass through the neural network using such a sequence is via the diagram below. 

<img src="/png/rnn/rnn.png" style="margin: 0 auto; display: block; width: 80%;"/> 

While this diagram is very simple and somewhat useful for visualization, it doesn't mean much mathematically. 
Let us go beyond simply showing a colorful diagram and mathematically explain how RNNs work.

For a sequence 

\[
    (x_1, \dots, x_T) \hspace{1cm} x_t \in \mathbb{R}^n
\]

the recurrent neural network will compute a sequence of **hidden states**

\[
    (h_1, \dots, h_T) \hspace{1cm} h_t \in \mathbb{R}^d
\]

and will output a sequence of **output values**

\[ 
    (y_1, \dots, y_T) \hspace{1cm} y_t \in \mathbb{R}^m
\]

In many applications, $m = n$, and we will assume that here for simplicity in our discussion.
Depending on the type of recurrent neural network one is employing, there are various 
ways to compute $h_t$ and $y_t$. Typically, the equations are given as

\begin{align}
    h_t &= f(x_t, h_{t - 1})\\
    y_t &= g(h_t)
\end{align}

where $f$, and $g$ are some functions.
Usually in an implementation $h_0$ is initialized to the zero vector. 

Now that we have the general mathematical structure of an RNN settled, we can make things concrete and introduce the most basic kind of RNN. In a vanilla RNN, $f$ and $g$ are just feed-forward neural networks. In that case, the computations are given by the equations below. 

\begin{align}
    h_t &= \sigma_i(Wx_{t} + Uh_{t - 1} + b_h)\\
    y_t &= \sigma_o(Vh_t + b_o)
\end{align}

where 

* $W \in \mathbb{R}^{d \times n}$ is a trainable weight matrix that acts on the input $x_{t}$

* $U \in \mathbb{R}^{d \times d}$ is a trainable weight matrix that acts on the hidden states $h_{t - 1}$

* $V \in \mathbb{R}^{d \times n}$ is a trainable weight matrix that acts on the computed hidden state $h_t$

* $b_h$, $b_o$ are trainable biases 

* $\sigma_o$, $\sigma_i$ are activation functions (e.g., sigmoid, tanh, etc.).

In addition, $d$, the dimension of the vector space that the hidden states $h_t$ live in, is a hyperparameter. One can set it 
to 5, 10, 1000, etc, but it ultimately depends on what kind of data the RNN is being trained to predict.

## An RNN, explicitly

Let us offer an explicit RNN with a simple architecture, and demonstrate how input is computed. Once we go over this simple RNN 
we will train it to learn the **reber grammar**, an artificial (fake) grammar invented as a toy sequence prediction problem.
Let us design it is as follows.

* Set $d = 4$. That is, our hidden states $h_t$ live in $\mathbb{R}^4$. 

* Set $n = 7$. Then we expect our sequence elements $x_t$ to live in $\mathbb{R}^7$.

* Set $m = n$.

With these hyperparameter and model parameter choices, what does this RNN look like? We can visualize the network 
computation of one sequence element $x_t$ in a sequence $(x_1, \dots, x_T)$. 
Since $x_t \in \mathbb{R}^7$, let $x_t = (x^t_1, x^t_2, x^t_3, x^t_4, ,x^t_5 , x^t_6, x^t_7)$. 
Then this is what the forward pass on $x_t$ in the RNN would look like:

<img src="/png/rnn/simple_rnn.png" style="margin: 0 auto; display: block; width: 100%;"/> 

For simplicity, we omit the biases in the above image. But from this picture, it is easy to see that a vanilla RNN 
is simply a feed-forward that feeds itself extra values on each forward pass in computing $x_t$. 
Namely, when processing the $t$-th element in the sequence of data,
it feeds itself the current input $x_t$ **and** the hidden layer calculations $h_{t-1}$ from the last time step. 

As we will use this network to learn the Reber grammar, we introduce the grammar. 

## Reber Grammar

The Reber grammar is an artifical grammar introduced in the 1970s by Arthur Reber, a cognitive psychologist, as an example 
of a experimental learning problem. The grammar's rules, which generate legal strings, can be pictured with the following graph.

<img src="/png/rnn/reber_grammar.png" style="margin: 0 auto; display: block; width: 100%;"/> 

How do we generate strings from this grammar? We iteratively build a string by traversing the graph. More specifically,

* For each edge we traverse, we iteratively append the letter corresponding to that edge to our current string in progress.

* The first letter will always be "B", because there is only one edge to traverse at the beginning.

* When we are presented with multiple possible edges to travese, we choose the next edge randomly and uniformly. 

* We eventually traverse the last edge "E" at the end, so the last letter is always "E". 

An example of a string generated by the above example is BTSSSXSE or BTSXXTVVE, but not BPVPS. 
BPVPS is not legal because it doesn't finish traversing the graph. 

As we said before, the Reber grammar was invented as a learning problem. 
The problem, which we will also use the RNN to try to solve, can be stated as follows: 
Suppose we are given many strings which are generated by the Reber grammar. Suppose we do not know how these 
samples are generated. Can we correctly determine whether or not a new, 
unseen string is a legal string generated by the grammar?

Interestingly, while the Reber grammar became widely discussed in the early literature on neural networks, Reber actually 
created this grammar to test on humans. Reber was interested in something called **implicit learning**; the idea is that 
there are certain things we, as humans, learn, but cannot exactly explain how we learned it (e.g. riding a bicycle) or sometimes 
we learn unintentionally or unconsciously (hence "implicitly"). To test this, Reber invented this grammar and gave participants different 
instructions on how to predict if a new, unseen sequence was valid or not; some were given instructions, some were to learn implicitly. 
He demonstrated that participants who were given instructions were worse at predicting valid examples than the ones 
that learned implicitly, giving evidence to the idea that humans can learn implicitly.

While Reber tested humans on this learning problem, we will be doing the same thing but to an RNN. 

In order to do this, we will first implement the graph in code. It is not that hard.

```python
graph = {
    0: [(1, "b")],
    1: [(2, "t"), (3, "p")],
    2: [(2, "s"), (4, "x")],
    3: [(3, "t"), (5, "v")],
    4: [(3, "x"), (6, "s")],
    5: [(4, "p"), (6, "v")],
    6: [(-1, "e")],
}
```

Here, 0 denotes the initial state and -1 denotes the terminal state. 
Next, we'll need to be able to generate strings from this graph. Hence we implement 
a random traversal of this graph (uniform probability when presented with multiple edge options)

```python
import random

def randomly_traverse_graph(graph):
    curr_node = 0
    sentence = ""
    while True:
        next_states = graph[curr_node]
        if len(next_states) > 1:
            next_state = random.choice(next_states)
        else:
            next_state = next_states[0]
        next_node = next_state[0]
        next_letter = next_state[1]

        sentence += next_letter
        curr_node = next_node
        if curr_node == -1:
            break

    return sentence
```