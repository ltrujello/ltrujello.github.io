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
In this case the calculations are dead simple. To see why, observe that, for the $k$-th node in the output 
layer for $t = 1$, the influence that the weights have on the value $y_1^k$ is really simple. 

<img src="/png/lstms/rnn_error1.png" style="margin: 0 auto; display: block; width: 60%;"/> 

The explicit equations for this calculation are 

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

which completes our investigation into how the weights influence values in the $y_1$.

## Calculating error at the second step

Calculating error at the second step is where things get a bit trickier. This is because the weights have a more complex effect on the 
final output values $y_2^k$. Here is a diagram that demonstrates what happens. 

<img src="/png/lstms/rnn_error2.png" style="margin: 0 auto; display: block; width: 80%;"/> 

At first, the weights initially have only an effect on $h_1^j$. However, $h_1^j$ is actually used in every single calculation 
for all the nodes in the second hidden state, $h_2$. And each of these nodes in $h_2$ are all used in the calculations for the final 
output value of interest, $y^k_2$. 

We can take into account this information via the chain. Fortunately, the output weights $v_{ij}$ have a rather simple effect on the output nodes. 

\[
    \frac{\partial y_2^i}{\partial v_{ij}} = h^j_2
\]

Next, let us compute the effect that the input weights have. This would be 

\[
    \frac{\partial y_2^k}{\partial w_{ij}} = 
    \sigma_y' 
    \cdot \sum_{\alpha=1}^{4}v_{k\alpha}\frac{\partial h_2^{\alpha}}{\partial w_{ij}}
\]

Therefore, the task at hand is to calculate the above partial derivative in the sum above for each $\alpha= 1, 2, 3, 4$. 
Doing so, observe that 

\[
    \frac{\partial h_2^{\alpha}}{\partial w_{ij}} 
    = 
    \sigma_h' \cdot \left( \sum_{\beta=1}^{7}\frac{\partial w_{\alpha\beta}}{\partial w_{ij}}x_2^{\beta} \right)
    + 
    \sigma_h' \cdot \left( \sum_{\gamma=1}^{4} u_{\alpha\gamma}\frac{\partial h^{\gamma}_{1}}{\partial w_{ij}} \right)
\]

We can simplify this. First, we can factor out $\sigma_h'$. Next, note that the first sum is zero, except in the case where $\alpha=i$ and $\beta=j$. Also note that 
$\frac{\partial h^{\gamma}_{1}}{\partial w_{ij}}$ is zero unless $\gamma=i$. further, because of our previous work,
we actually know the value of $\frac{\partial h^{i}_{1}}{\partial w_{ij}}$, we won't expand it here. Thus 

\[
    \frac{\partial h_2^{\alpha}}{\partial w_{ij}} 
    = 
    \sigma_h' \left( \delta(\alpha, i)x_j
    + 
    u_{\alpha i}\frac{\partial h^{i}_{1}}{\partial w_{ij}}\right)
\]

where $\delta$ is the Kronecker-delta (so it's 1 if the arguments match, zero otherwise).
Hence, we have that 

\[
    \frac{\partial y_2^k}{\partial w_{ij}} = 
    \sigma_y' 
    \cdot \sum_{\alpha=1}^{4}v_{k\alpha}  \sigma_h' \left( \delta(\alpha, i)x_j
    + u_{\alpha i}\frac{\partial h^{i}_{1}}{\partial w_{ij}}\right)
\]

which completes our calculation.

Finally, let us calculate how the hidden weights $u_{ij}$ contribute to the output value $y_2^k$. In this case, we can 
perform a very similar analysis to what we just did: We compute  

\[
    \frac{\partial y_2^k}{\partial w_{ij}} = 
    \sigma_y' 
    \cdot \sum_{\alpha=1}^{4}v_{k\alpha}\frac{\partial h_2^{\alpha}}{\partial u_{ij}}
\]

and then we can calculate that 

\[
    \frac{\partial h_2^{\alpha}}{\partial u_{ij}} 
    =
    \sigma_h' \cdot \left( 
        \sum_{\gamma = 1}^4
        \frac{\partial u_{\alpha\gamma}}{\partial u_{ij}}
        h^{\gamma}_{1}
        + 
        u_{\alpha\gamma}
        \frac{\partial h_{1}^{\gamma}}{\partial u_{ij}}
         \right)
\]

and simplifying as we did similar just before we obtain that 

\[
    \frac{\partial h_2^{\alpha}}{\partial u_{ij}} 
    =
    \sigma_h' \cdot \left( 
    \delta(\alpha,i)
    h^{j}_{1}
    + 
    u_{\alpha i}
    \frac{\partial h_{1}^{i}}{\partial u_{ij}}
    \right)
\]

which leads to 

\[
    \frac{\partial y_2^k}{\partial u_{ij}} = 
    \sigma_y' 
    \cdot
    \sum_{\alpha=1}^{4}v_{k\alpha} \cdot 
    \sigma_h' \left( 
    \delta(\alpha,i)
    h^{j}_{1}
    + 
    u_{\alpha i}
    \frac{\partial h_{1}^{i}}{\partial u_{ij}}
    \right)
\]

which completes our calculation for how the weights affect the outputs values in $y_2$.

## Calculating error, more generally

From what we've seen, the way that each weight in an RNN affects the output values at $y_t$ becomes rather complex.
We can figure out how complex it is by calculating it more generally. To do this, recall that 

\begin{align*}
    y^k_t &= \sigma_y\left( \sum_{\alpha}v_{k\alpha}h^{\alpha}_t + b^k_y \right)\\
    h^\alpha_t &= \sigma_h\left( \sum_{\beta}w_{\alpha\beta}x_t^{\beta} + \sum_{\gamma}u_{\alpha\gamma}h_{t-1}^{\gamma} + b^\alpha_h \right)
\end{align*}

### Calculating the effect of the input weights

First, let us calculate $\frac{\partial y^k_t}{\partial w_{ij}}$. In this case we have that 

\[
   \frac{\partial y^k_t}{\partial w_{ij}} =
   \sigma_y' \cdot 
    \sum_{\alpha}v_{k\alpha}\frac{\partial h^\alpha_t}{\partial w_{ij}}
\]

Next, we can observe that

\begin{align*}
    \frac{\partial h^\alpha_t}{\partial w_{ij}}
    &=
    \sigma_h' 
    \sum_{\beta}\frac{\partial w_{\alpha\beta}}{\partial w_{ij}}x_t^{\beta} 
    + 
    \sigma_h'
    \sum_{\gamma}u_{\alpha\gamma}\frac{\partial h_{t-1}^{\gamma}}{\partial w_{ij}} \\
    &=
    \sigma_h' 
    \delta(\alpha,i)\cdot x_t^{j}
    + 
    \sigma_h'
    \sum_{\gamma}u_{\alpha\gamma}\frac{\partial h_{t-1}^{\gamma}}{\partial w_{ij}} 
\end{align*}

This is a [recurrence relation](https://en.wikipedia.org/wiki/Recurrence_relation).
To see this, let 

\begin{align}
    A(\alpha, t) &= \sigma_h' \cdot \delta(\alpha,i)\cdot x_t^{j}\\
    B(\alpha,\gamma,t) &= \sigma_h' \cdot u_{\alpha\gamma}
\end{align}

Note that in the second equation $\sigma_h'$ is a function of $t$, hence why $B$ is also a function of $t$. 
Using these formulas, we can see that 

\begin{align*}
    \frac{\partial h^\alpha_t}{\partial w_{ij}}
    &= A(\alpha, t) + \sum_{\gamma_1} B(\alpha, \gamma_1, t) \frac{\partial h^{\gamma_1}_{t-1}}{\partial w_{ij}}\\
    &= A(\alpha, t) + \sum_{\gamma_1} B(\alpha, \gamma_1, t) 
    \cdot \left( 
    A(\gamma_1, t) + \sum_{\gamma_2} B(\gamma_1, \gamma_2, t - 1) \cdot \frac{\partial h^{\gamma_2}_{t-2}}{\partial w_{ij}}
    \right)\\
    &= A(\alpha, t) + \sum_{\gamma_1} B(\alpha, \gamma_1, t) 
    \cdot \left( 
    A(\gamma_1, t) + \sum_{\gamma_2} B(\gamma_1, \gamma_2, t - 1) \cdot 
    \left(
        A(\gamma_2, t) + \sum_{\gamma_3} B(\gamma_2, \gamma_3, t - 2) \cdot \frac{\partial h^{\gamma_3}_{t-3}}{\partial w_{ij}}
    \right)
    \right)\\
\end{align*}

Cleaning this up a bit, we see that there's a pattern here:

\begin{align}
    \frac{\partial h^{\alpha}_{t}}{\partial w_{ij}}
    &= 
    A(\alpha, t) + \sum_{\gamma_1} B(\alpha, \gamma_1, t) A(\gamma_1, t - 1)\\
    & +
    \sum_{\gamma_1} \sum_{\gamma_2} B(\alpha, \gamma_1, t) B(\gamma_1,\gamma_2, t - 1)A(\gamma_2, t-2)\\
    & + 
    \sum_{\gamma_1} \sum_{\gamma_2} \sum_{\gamma_2}B(\alpha, \gamma_1, t) 
    B(\gamma_1,\gamma_2, t - 1) B(\gamma_2,\gamma_3, t -2) A(\gamma_3, t-3) \frac{\partial h^{\gamma_3}_{t-3}}{\partial w_{ij}}
\end{align}

Let $\gamma_0 = \alpha$. Then this then leads us to suspect that the general formula is

\[
\frac{\partial h^{\alpha}_{t}}{\partial w_{ij}}
= 
\sum_{k = 0}^{t-1}
\left( 
\sum_{\gamma_1 = 1}^{n}
\cdots
\sum_{\gamma_k = 1}^{n}
\prod_{\ell=0}^{k -1} B(\gamma_\ell, \gamma_{\ell + 1}, t - \ell)A(\gamma_k, t-k)
\right)
\]

Let's prove this by performing induction on the timestep $t$. For our base case, we 
have $t=1$. This reduces to 

\[
\frac{\partial h^{\alpha}_{t}}{\partial w_{ij}} = A(\alpha, t) = \sigma_h' \cdot \sum_{\beta} \frac{\partial w_{\alpha\beta}}{\partial w_{ij}}x_t^{\beta}
= \sigma_h' \cdot x^j_1
\]

which is correct; this matches what we calculated earlier when we were starting simple. Next, suppose that the assertion holds 
for timestep $t \ge 1$; to prove that this equation also holds for timestep $t + 1$, observe that 

\begin{align*}
    \frac{\partial h^\alpha_{t+1}}{\partial w_{ij}}
    &=
    \sigma_h' 
    \sum_{\beta}\frac{\partial w_{\alpha\beta}}{\partial w_{ij}}x_{t+1}^{\beta} 
    + 
    \sigma_h'
    \sum_{\gamma = 1}^{n}u_{\alpha\gamma}\frac{\partial h_{t}^{\gamma}}{\partial w_{ij}} \\
    &=
    \sigma_h' 
    \sum_{\beta}\frac{\partial w_{\alpha\beta}}{\partial w_{ij}}x_{t+1}^{\beta} 
    + 
    \sigma_h'
    \sum_{\gamma =1}^{n} u_{\alpha\gamma}
    \left( 
        \sum_{k = 0}^{t - 1}
        \sum_{\gamma_1 = 1}^{n}
        \cdots
        \sum_{\gamma_k = 1}^{n}
        \prod_{\ell=0}^{k -1} B(\gamma_\ell, \gamma_{\ell + 1}, t - \ell)A(\gamma_k, t-k)
    \right)
     \\
    &= A(\alpha, t+1) + \sum_{\gamma=1}^{n}B(\alpha, \gamma, t + 1)
        \left( 
        \sum_{k = 0}^{t - 1}
        \sum_{\gamma_1 = 1}^{n}
        \cdots
        \sum_{\gamma_k = 1}^{n}
        \prod_{\ell=0}^{k -1} B(\gamma_\ell, \gamma_{\ell + 1}, t - \ell)A(\gamma_k, t-k)
    \right)
\end{align*}

Now if we declare $\gamma_{-1} = \alpha$ and recall that $\gamma_0 = \gamma$, then we have  

\[
    \frac{\partial h^\alpha_{t+1}}{\partial w_{ij}}
    = 
    \sum_{k = -1}^{t - 1}
    \left(
    \sum_{\gamma_0 = 1}^{n}
    \sum_{\gamma_1 = 1}^{n}
    \cdots
    \sum_{\gamma_k = 1}^{n}
    \prod_{\ell=-1}^{k -1} B(\gamma_\ell, \gamma_{\ell + 1}, t - \ell)A(\gamma_k, t-k)
    \right)
\]

by allowing $k$ to range from $-1$ to $t - 1$ and $\ell$ to range from $-1$ to $k - 1$.

Reindexing our sum with new indices $\gamma'$ such that 
$\gamma'_{i + 1} = \gamma_i$,
we can rewrite this as

\[
    \frac{\partial h^\alpha_{t+1}}{\partial w_{ij}}
    = 
    \sum_{k = 0}^{t}
    \left(
    \sum_{\gamma'_1 = 1}^{n}
    \sum_{\gamma'_2 = 1}^{n}
    \cdots
    \sum_{\gamma'_{k} = 1}^{n}
    \prod_{\ell=0}^{k - 1} B(\gamma'_\ell, \gamma'_{\ell + 1}, (t + 1) - \ell)A(\gamma'_{k}, (t + 1)-{k})
    \right)
\]

which asserts our proposition from timestep $t +1$. Since our assertion is true for $t + 1$ whenever $t \ge 1$, 
and we proved the assertion for $t = 1$, this 
proves by induction that our formula holds for all timesteps $t$. 

### Calculating the effect of the hidden weights

Next, we'll calculate $\frac{\partial y^k_t}{\partial u_{ij}}$. In this case we have that 

\[
    \frac{\partial y^k_t}{\partial u_{ij}} = 
    \sigma_y' \cdot 
    \sum_{\alpha}v_{k\alpha}\frac{\partial h^\alpha_t}{\partial u_{ij}}
\]

Thus we need to calculate the above partial derivative in the summation for each $\alpha$. Doing this is 
very similar to the analysis that we performed when calculating how the weights $w_{ij}$ affect the output nodes.
Without repeating ourselves, we can prove by induction that if 

\begin{align*}
A(\alpha, t) &= 
\sigma_h' \cdot \delta(\alpha, i) h^{j}_{t-1}\\
B(\alpha, \gamma, t) &= \sigma_h' \cdot u_{\alpha, \gamma}
\end{align*}

then we have that, if $\gamma_0 = \alpha$, then

\[
\frac{\partial h^{\alpha}_{t}}{\partial u_{ij}}
= 
\sum_{k = 0}^{t-1}
\left( 
\sum_{\gamma_1 = 1}^{n}
\cdots
\sum_{\gamma_k = 1}^{n}
\prod_{\ell=0}^{k -1} B(\gamma_\ell, \gamma_{\ell + 1}, t - \ell)A(\gamma_k, t-k)
\right)
\]

which is actually the same formula we came up with before. 

## Evaluating Error Flow 

At this point, we have achieved our goal of calculating the influence each weight $w_{ij}$ and $u_{ij}$ has 
on each node of our model outputs $y_t^o$, for all applicable $t$ and $o$. 
Substituting in $B$ and $A$ from our formulas before, we obtain that 

\begin{align*}
\frac{\partial y^o_t}{\partial u_{ij}} &= 
\sigma_y' \cdot 
\sum_{\alpha}v_{o\alpha}
\sum_{k = 0}^{t-1}
\left( 
\sum_{\gamma_1 = 1}^{n}
\cdots
\sum_{\gamma_k = 1}^{n}
\prod_{\ell=0}^{k -1} \sigma_{h_{t-\ell}}' \cdot u_{\gamma_{\ell}, \gamma_{\ell+1}} \cdot \sigma_{h_{\gamma_k}}' \cdot \delta(\gamma_k, i) \cdot h_{t-k-1}^j
\right)\\
\frac{\partial y^o_t}{\partial w_{ij}} &= 
\sigma_y' \cdot 
\sum_{\alpha}v_{o\alpha}
\sum_{k = 0}^{t-1}
\left( 
\sum_{\gamma_1 = 1}^{n}
\cdots
\sum_{\gamma_k = 1}^{n}
\prod_{\ell=0}^{k -1} \sigma_{h_{t-\ell}}' \cdot u_{\gamma_{\ell}, \gamma_{\ell+1}} \cdot \sigma_{h_{\gamma_k}}' \cdot \delta(\gamma_k, i) \cdot x_{t-k}^j
\right)
\end{align*}

We are now in a position to understand why gradients tend to vanish or explode in recurrent neural networks. In both formulas, we 
see that we have an interesting factor 

\[
    \prod_{\ell=0}^{k -1} \sigma_{h_{t-\ell}}' \cdot u_{\gamma_{\ell}, \gamma_{\ell+1}} 
\]

which we must compute. The issue with this factor is that we are taking many products. The worst case happens when 
$k = t - 1$, in which case we must compute 

\[
    \prod_{\ell=0}^{t - 2} \sigma_{h_{t-\ell}}' \cdot u_{\gamma_{\ell}, \gamma_{\ell+1}} 
\]

As we can see from this product, if $|\sigma_{h_{t-\ell}}' \cdot u_{\gamma_{\ell}, \gamma_{\ell+1}}| < 1$ for all 
$\ell$, then the above product will vanish, and we eventually won't be able to update our weights during training. 
Conversely, if $|\sigma_{h_{t-\ell}}' \cdot u_{\gamma_{\ell}, \gamma_{\ell+1}}| > 1$, then 
the above product will explode, and our weight updates will thrash around. 



