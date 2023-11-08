---
date: 2023-10-29
---

# Understanding LSTMs and Vanishing Gradients in RNNs

In attempting to understand LSTMs, I often find myself reading blog posts, tutorials, explanations which 
all characterize LSTMs in the same way, but they are not very satisfactory in my opinion because they do not 
explain *why* LSTMs were invented or *how* they manage to solve the **vanishing gradient** problem 
in RNNs (which is why they were created). I think this is important to really understanding the 
structure of an LSTM. 

In general, sometimes it's necessary to deeply understand the motivation 
for a concept, especially if (1) the concept is complicated and nontrivial and (2) would lend itself to being
almost arbitrary and random if the motivation were not taken into account. 
Conversely, sometimes this is not necessary, especially if there are other more 
pressing things to deeply understand.

The details of why gradients vanish for RNNs during the BPTT algorithm can be found in certain 90s neural network 
papers, such as [(Hochreiter, 1997)](https://doi.org/10.1162/neco.1997.9.8.1735), a paper in which Hochreiter et. al. 
propose the LSTM model. In that description, the model for the LSTM actually differs from what we know to be an 
LSTM today. Thus the goal of this write up will be to go beyond just 
saying "LSTMs have an input, forget gate", and to offer a modern description of vanishing gradients in 
RNNs and how LSTMs solve this. 

<!-- more -->

In what follows, we will track and analyze the error flow of weights in an RNN, starting from a simple example
and ending with a more general description. We'll observe that the gradient error flow can actually 
be recursively defined, and that a closed form description can be offered. This closed form solution will reveal 
why gradients vanish. 

## RNNs and a concrete example 

Recall that for a vanilla RNN, given an **input sequence** $(x_1, \dots, x_T)$ with $x_t \in \mathbb{R}^n$ for some $n$, we obtain 
a corresponding **output sequence** $(y_1, \dots, y_T)$ with $y_t \in \mathbb{R}^m$ for some $m$ via the following recursive equations.

\begin{align}
    y_t &= \sigma_y(Vh_t + b_y)\\
    h_t &= \sigma_h(Wx_t + Uh_{t-1} + b_h)
\end{align}

where 

* $h_0$ is initialized to $\vec{0}$
* $V \in \mathbb{R}^{m \times d}$, for some $d$, which we refer to as the **output weights**
* $W \in \mathbb{R}^{d \times n}$ which we refer to as the **input weights**
* $U \in \mathbb{R}^{d \times d}$ which we refer to as the **hidden weights**
* $\sigma_y, \sigma_h$ are some activation functions
* $b_y, b_h \in \mathbb{R}^{d}$ are biases

To make things concrete, let us introduce a concrete recurrent neural network. 
Set $n = m = 7$, so that the input and output of the RNN are in $\mathbb{R}^7$, and set $d = 4$, so that the hidden states are in 
$\mathbb{R}^4$. This is what the RNN looks like.
<img src="/png/lstms/simple_rnn.png" style="margin: 0 auto; display: block; width: 95%;"/> 

As a side note, this RNN is capable of perfectly learning the Reber grammar. 

Given this definition of a recurrent neural network, what does it mean to track error flow from the weights throughout 
the network? In order to do this, we need to understand how each weight contributes to the output values of the neural network. 
Specifically, consider an input sequence $(x_1, \dots, x_T)$, and let $(y_1, \dots, y_T)$ be the output of the RNN.
Denote $y_t^k$ to be the $k$-th output node at time $t$ (thus, $k$ ranges over $1$ to $T$). 

Since our trainable parameters are $V$, $W$, $U$, and out biases, we need to calculate the following quantities. 

\begin{align}
    \frac{\partial y_t^k}{\partial v_{ij}} \text{ where } i = 1, 2, \dots, m, j = 1, 2, \dots, d\\
    \frac{\partial y_t^k}{\partial w_{ij}} \text{ where }  i = 1, 2, \dots, d, j = 1, 2, \dots, n\\
    \frac{\partial y_t^k}{\partial u_{ij}} \text{ where } i = 1, 2, \dots, d, j = 1, 2, \dots, d
\end{align}

We also have to calculate the error with respect to our biases, but we'll ignore that for now. 
Calculating these derivatives is a bit tricky. Thus, what we'll do is calculate these 
derivatives in simple cases which will slowly reveal a pattern to the general error propagation. 

## Derivates at the first timestep

Consider the RNN we introduced earlier, and 
suppose we want to run a input sequence $(x_1, x_2, x_3)$ through the RNN. 
What would this look like? It would look like this. 

<img src="/png/lstms/rnn_three_inputs.png" style="margin: 0 auto; display: block; width: 80%;"/> 

Well actually, what it would *really* look like is something like this 

<img src="/png/lstms/rnn_three_inputs_full.png" style="margin: 0 auto; display: block; width: 80%;"/> 

but that is a bit overwhelming to look at. But from this perspective, it is easy to see that an RNN is very similar to a feed 
forward neural network. Additionally, we can see which weights affect what layers. 


## Starting simple: Calculating error at the first step

In order to do this, let us start off simple and calculate the error at the first timestep with $t = 1$. 
In this case we have that 

\begin{align}
    y^k_1 &= \sigma_y\left(\sum_{\alpha = 1}^{4} v_{k\alpha}h^\alpha_1 + b^k_y\right)\\
    h_1^{\alpha} &= \sigma_y\left(\sum_{\beta = 1}^{7} w_{\alpha\beta}x^\beta_1 + \sum_{\gamma = 1}^{4} u_{\alpha\gamma}h^\gamma_0 + b^{\alpha}_h\right)
\end{align}

With respect to the weights $v_{ij}$, which connects hidden node $h^j_1$ to output node $y_1^j$, this weight only has 
an effect on output node $y_1^i$. Thus,

\[
    \frac{\partial y_1^i}{\partial v_{ij}} = h^j_1
\]

Next, observe that, for the weight $w_{ij}$ which connects input node $x^j_1$ to hidden node $h^i_1$, we have that
for any $k =1, 2, \dots, T$

\[
    \frac{\partial y_1^k}{\partial v_{ij}} = \frac{\partial y_1^k}{\partial h^i_1} \frac{\partial h^i_1}{\partial w_{ij}}
    = \sigma_y' v_{ki} \cdot {h^i_1}' x^j_1
\]

where $\sigma_y'$ and ${h^i_1}'$ are scalars values, obtained from evaluating the derivative via the chain rule. 
Our next calculation is very similar:

\[
    \frac{\partial y_1^k}{\partial u_{ij}} = \frac{\partial y_1^k}{\partial h^i_1} \frac{\partial h^i_1}{\partial u_{ij}}
    = \sigma_y' v_{ki} \cdot {h^i_1}' h^j_0
\]

In this simple case, the error propagation looks something like this. It's very similar to the way error in a feedforward neural network 
would flow. 

## Calculating error at the second step

Calculating error at the second step is where things get a bit trickier. In this case, we still have 

\[
    \frac{\partial y_2^i}{\partial v_{ij}} = h^j_2
\]







