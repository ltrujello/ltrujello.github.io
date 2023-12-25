---
date: 2023-10-29
---

# Understanding LSTMs and Vanishing Gradients in RNNs

In attempting to understand LSTMs, I often found myself reading blog posts, tutorials, and explanations 
which would describe LSTMs in usually the same way: 
An LSTM has three gates: an input, forget, and output gate. These these gates control the flow of information. 
Further, an LSTM has a linear self-connection.
Also, this architecture addresses the **vanishing gradient** problem. 

However, knowledge of this list of facts about LSTMs alone didn't really translate to a satisfying understanding of 
LSTMs for me. Since I was more curious as to *why* LSTMs work, *how* they address the vanishing gradient problem, 
and *why* the vanishing gradient problem is a significant problem, I decided to go through the math.   

Some excellent resources that did satisfy my curiosity include work by [Graves](https://www.cs.toronto.edu/~graves/preprint.pdf), 
[Gers et. al](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=11540131eae85b2e11d53df7f1360eeb6476e7f4) paper 
on LSTM forget gates,
[Ger's PhD Thesis](http://www.felixgers.de/papers/phd.pdf), and of course the original [1997 paper](https://www.bioinf.jku.at/publications/older/2604.pdf) on LSTMs. 


In what follows, we will understand how gradients vanish in "vanilla" RNNs.
We'll do this by deriving explicit formulae for the error propagation for each weight in a vanilla RNN. 
We'll then discuss error propagation in an LSTM, and how the architecture prevents gradients from vanishing. 

<!-- more -->

## Background

The LSTM model was invented to address an issue that arose from training recurrent neural networks. There are several 
training algorithms, but the simplest method is to perform an extension of backpropagation, similar to what is done for 
feed forward neural networks. Researchers were noticing that RNNs were failing to "recall" sequence features with 
long time lags, and they began to realize this was because RNN backpropgation gradients vanish for longer and longer sequences. 

The details of why gradients vanish for RNNs during the backpropagation algorithm can be found in certain 90s neural network 
papers, such as [(Hochreiter, 1997)](https://doi.org/10.1162/neco.1997.9.8.1735). The full details can be found in 
Hochreiter's PhD thesis, but it is only available in German.  


## Forward Pass of RNNs

Recall that RNNs are an instance of a feed-forward neural network in which hidden states have self-connections. 
Specifically, the forward pass of an RNN can be computed as follows.
Let  $(x_1, \dots, x_T)$ be an **input sequence**, with $x_t \in \mathbb{R}^I$ for some $I$.
We obtain a corresponding **output sequence** $(y_1, \dots, y_T)$ with $y_t \in \mathbb{R}^K$ for some $K$ via the following recursive equations.

\begin{align}
    h_t &= \sigma_h(Wx_t + Uh_{t-1} + b_h)\\
    y_t &= \sigma_y(Vh_t + b_y)
\end{align}

where 

* $h_0$ is initialized to $\vec{0}$
* $V \in \mathbb{R}^{K \times H}$, for some $H$, which we refer to as the **output weights**
* $W \in \mathbb{R}^{H \times H}$ which we refer to as the **input weights**
* $U \in \mathbb{R}^{H \times H}$ which we refer to as the **hidden weights**
* $\sigma_y, \sigma_h$ are some activation functions
* $b_y, b_h \in \mathbb{R}^{H}$ are biases

This describes what is happening at a high-level using matrices. However, we can be more specific and describe what happens at an 
element level during the forward pass. To do this, we borrow a notation system introduced by [(Graves, 2012)](https://doi.org/10.1007/978-3-642-24797-2).

* Let $x_i^{t}$ be the $i$-th element of the $t$-th input $x_t$ 
* Let $a_j$ denote the network input to network node $j$ (in the hidden layer)
* Let $b_j$ be the activation of $a_j$ 

Then we can write

\begin{align}
a_h^{t} &= \sum_{i = 1}^{I}w_{hi}x_i^t + \sum_{h = 1}^{H}w_{hh'}b_{h'}^{t-1}\\
b_h^{t} &= \theta(a_h^{t})
\end{align}

Denote the $k$-th element of $y_t$ as $y_k^{t}$. Then the equation for this output node is 

\begin{align}
a_k^{t} &= \sum_{h = 1}^{H} w_{kh}b_h^{t}\\
y_k^{t} &= \theta(a_k^{t})
\end{align}

## Notation 

At this point, we make a comment on notation. This notation system, introduced by [(Graves, 2012)](https://doi.org/10.1007/978-3-642-24797-2), is technically an abuse of notation, 
but the abuse outweighs the mathematically clarity it empowers us with. 
For example,

* With $w_{kh}$, we are speaking of the weight going from the hidden node $b^{t}_h$ to the output node $y^{t}_k$. 
* With $w_{hi}$, we are speaking of the weight going from the input node $x^{t}_i$ to hidden node $b^{t}_h$. 

While this is an abuse of notation, it is generally obvious from context what time step and nodes we are talking about. The 
alternative to something like this is employing a notation system that perfectly tracks everything, but 
this is unnecessary and undesirable because it severely hinders the readablity of the equations.

## Example RNN

To make things concrete, let us introduce a simple example of a recurrent neural network. 
For our network architecture, set $I = K = 7$, so that the input and output of the RNN are in 
$\mathbb{R}^7$, and set $H = 4$, so that the hidden states are in 
$\mathbb{R}^4$. This is what the RNN looks like.

<img src="/png/lstms/simple_rnn.png" style="margin: 0 auto; display: block; width: 95%;"/> 

As a side note, this RNN is capable of perfectly learning the Reber grammar. 

## RRN Loss 

Given this definition of a recurrent neural network, what does it mean to track error flow from the weights throughout 
the network? 

Let $\mathcal{L}: \mathbb{R}^{I} \times \mathbb{R}^{K} \to \mathbb{R}$ be a (differentiable) loss function for
our recurrent neural network. To calculate network loss with respect to our weights, we need to calculate

\[
    \frac{\partial \mathcal{L}}{\partial w_{ij}}
\]

where $i$, $j$ vary over all connections in our network. In our case, that means we are interested in calculating
three different classes of weights in our network: the weights $w_{kh}$ connecting the hidden nodes to the output nodes, 
the weights $w_{hh'}$ which connect the hidden nodes to themselves, and the weights $w_{hi}$ which connect the input nodes
to the output nodes.G

\begin{align}
    \frac{\partial \mathcal{L}}{\partial w_{kh}} &\text{ where } k = 1, 2, \dots, K, h = 1, 2, \dots, H\\
    \frac{\partial \mathcal{L}}{\partial w_{hh'}} &\text{ where }  h = 1, 2, \dots, H, h' = 1, 2, \dots, H\\
    \frac{\partial \mathcal{L}}{\partial u_{hi}} &\text{ where } h = 1, 2, \dots, H, i = 1, 2, \dots, I
\end{align}

Note that in general, if $w_{ij}$ is the weight connecting node $j$ to node $i$, then 
it is used in the calculation of each network input $a_{j}^t$ for each timestep $t$. In turn, each $a_j^{t}$ contributes to the overall 
network loss. Mathematically, this translates to the equation below. 

\[
    \frac{\partial \mathcal{L}}{\partial w_{ij}} 
    = 
    \sum_{t = 1}^{T} \frac{\partial \mathcal{L}}{\partial a_{i}^{t}} \frac{\partial a_{i}^{t}}{\partial w_{ij}} 
    = 
    \sum_{t = 1}^{T} \delta_i^{t}  \frac{\partial a_{i}^{t}}{\partial w_{ij}} 
\]

where we denote 

\[ 
    \delta_i^{t} =  \frac{\partial \mathcal{L}}{\partial a_{i}^{t}}
\]

We'll continue to employ the above shorthand throughout our work.

If we let $j$ be the $j$-th node in the input layer then 

\[
    \frac{\partial a_{i}^{t}}{\partial w_{ij}} 
    = 
    x_j^{t}
    \implies 
    \frac{\partial \mathcal{L}}{\partial w_{ij}} =
    \sum_{t = 1}^{T} \delta_i^{t} x_j^{t}
\]

while if $j$ is the $j$-th node in the hidden layer then

\[
    \frac{\partial a_{i}^{t}}{\partial w_{ij}} 
    = 
    b_j^{t}
    \implies
    \frac{\partial \mathcal{L}}{\partial w_{ij}} =
    \sum_{t = 1}^{T} \delta_i^{t} b_j^{t}
\]

Thus, in order to calculate the loss with respect of the weights, we need a way to calculate 
$\delta_i^{t}$. Since $a_i^{t}$ is used in the calculation of all the output nodes $y_k^{t}$ *and* 
all the hidden notes $a_h^{t+1}$ in the next layer, we have the following recursive formula

\[
    \delta_i^{t} = \sum_{k = 1}^{K} \delta_k^{t} w_{ki} + \sum_{h = 1}^{H} \delta_{h}^{t + 1} w_{hi}
\]

If we set $\delta_{i}^{T + 1} = 0$, then we can use this recursive formula to calculate all the 
values $\delta_{i}^{t}$ by starting at $t= T$ and decrementing $t$. 
This then completes the calculation for $\frac{\partial \mathcal{L}}{\partial w_{ij}}$. 

Now that we have this formula, we can illustrate calculating the error $\delta_i^{t}$ at different 
timesteps. We'll use the example RNN we introduced earlier for our mental picture of this process.


## Derivates at the first timestep

Consider the example RNN we introduced earlier, and 
suppose we want to run a input sequence $(x_1, x_2, x_3)$ through the RNN. 
What would the forward pass of processing this sequence of inputs look like? It would look like this. 

<img src="/png/lstms/rnn_three_inputs.png" style="margin: 0 auto; display: block; width: 80%;"/> 

Well actually, what it would *really* look like is something like this 

<img src="/png/lstms/rnn_three_inputs_full.png" style="margin: 0 auto; display: block; width: 80%;"/> 

Thinking of an RNN in this way is called "unfolding" the RNN.
From this perspective, it is easy to see that an RNN is very similar to a feed 
forward neural network. Additionally, we can see which weights affect what layers. 


## Starting simple: Calculating error at the first step

In order to calculate the error at the first timestep with $t = 1$, let's compute the forward pass, which looks like 
this.

<img src="/png/lstms/rnn_error1.png" style="margin: 0 auto; display: block; width: 60%;"/> 

The forward pass for this case would be 

\begin{align}
    y_k^{1} &= \theta\left(\sum_{h = 1}^{H} w_{kh}b_h^{1} \right)\\
    b_h^{1} &= \theta\left(\sum_{i = 1}^{I} w_{hi}x_i^{1} + \sum_{h' = 1}^{H} w_{hh'}b_{h'}^{0} \right)
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

We can take into account this information via the chain rule. 
Fortunately, the output weights $v_{ij}$ have a rather simple effect on the output nodes. 

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

## How RNN Backpropagation is Unstable 

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

If the activation function in the hidden layers is the sigmoid function $\sigma$, then we know that 

\begin{align*}
    \sigma' &= \sigma \cdot (1 - \sigma)\\
    \sigma'' &= \sigma \cdot (1 - \sigma)^2 + \sigma \cdot (-\sigma \cdot (1 - \sigma))
    = 
    \frac{\left(e^{-2x}-e^{-x}\right)}{\left(1\ +\ e^{-x}\right)^{3}}
\end{align*}

Setting $e^{-2x} - e^{-x} = 0$, we see that we achieve an extrema of $\sigma'$ when $x = 0$, which turns out to be 
a global maximum of $\sigma'$ with a value of $\sigma'(0) = 0.25$. This implies that 

\[
    \left|\prod_{\ell=0}^{t - 2} \sigma_{h_{t-\ell}}' \cdot u_{\gamma_{\ell}, \gamma_{\ell+1}} \right|
    \le 
    (0.25)^{t-1}
    \left|\prod_{\ell=0}^{t - 2} u_{\gamma_{\ell}, \gamma_{\ell+1}} \right|
\]

Let $u_{\text{max}}$ be the maximum weight that appears in the above product, $u_{\text{min}}$ be the minimum weight, 
and $\sigma'_{\text{min}}$ the smallest value of $\sigma'$ that appears in the above product.
Then we see that 

\[
|\sigma'_{\text{min}} \cdot u_{min}|^{t - 1}
\le
\left|\prod_{\ell=0}^{t - 2} \sigma_{h_{t-\ell}}' \cdot u_{\gamma_{\ell}, \gamma_{\ell+1}} \right|
\le |0.25 \cdot u_{max}|^{t - 1}
\]

This is not great. What this means is that, during training, when the weights tip slightly below the value of $4$, 
the inequality on the right gains control and the above product will rapidly vanish. 
When the weights tip slightly above the value of $4$, which will definitely be greater than $\sigma'_{\text{min}}$, then 
the left inequality takes control and the product will rapidly explode. The reason why they rapidly explode is primarily due 
to the exponential. Thus, perturbing the weights around the value of $4$ can cause 
instability. The chance for instability to occur becomes worse when one takes the limit $t \to \infty$, as this causes the 
left and right inequalities to be more unstable. This is why
issues with training can arise when training on long sequences. 


This analysis can be repeated for other activation functions as well, by simply taking the maximum and minimum values 
of the activation function. Hence, changing the activation function isn't going to help. 
Changing the learning rate isn't going to help either, 
since that would be making a fixed scalar compete with an exponential product (which is a battle it will lose). Our main problem boils 
down to the fact that we are taking too many products. 
