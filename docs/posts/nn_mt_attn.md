---
draft: true 
date: 2023-11-27
---

# Neural Translation with Attention

For many years, machine translation systems relied on statistical phrase-based translation models [(Koehn, 2003)](https://doi.org/10.3115/1073445.1073462), for machine translation. 
During this time, methods in neural translation were researched, although it took some time for performance of these translation 
models to compete with statistical machine translation. The breakthrough that caused many to take neural translation was through the paper 
[(Bahdanau, et. al 2015)](https://arxiv.org/pdf/1409.0473.pdf), in which researchers incorporated existing neural translation models with a concept called *attention*. In this paper, they were able to overcome important issues with existing neural translation methods,
with the worst issue being then fact that previous models would degrade for longer sentences (more than 30 words).  

<!-- more -->

In this post, we'll mathematically introduce the architecture used by Bahdanau et. al. and implement the network in PyTorch. We'll 
also visualize the model's attention on a lexical level that achieves sensible lexical alignment on translation pairs. 

## Background and previous work
Prior to this paper, the state of the art for neural translation was the **encoder-decoder** architecture with recurrent neural networks only. 
This architecture was first porposed in [(Cho et. al, 2014)](https://arxiv.org/pdf/1406.1078.pdf). 
In this architecture, there two components main components are the *encoder*, which encodes the source input sentence in a sequence of vectors, and the decoder, which takes the sequence of vectors and outputs word probabilities. One can search through these word probabilities with a beam search to determine possible 
sentence translations. 

In the encoder-decoder architecture, an input sentence is represented as a sequence of vectors $(x_1, \dots, x_{T_x})$ with $x_i \in \mathbb{R}^n$, and is processed through the encoder via a function $f$ that computes a hidden state for each time step $t$. Specifically,
the forward pass of the encoder is

\[
    h_{t} = f(Ex_t, h_{t-1})
\]

for $t = 1, \dots, T_x$, where $E$ is a word embedding for the source language. This word embedding is learned 
jointly during the training. 

Next, the forward pass of the decoder is 

\[
    s_{t} = f(y_t, s_{t-1})
\]

where in the original set up $s_0$ is initialized to $h_{T_x}$. The hidden states of the decoder $s_t$ are then used to compute the output probability 

\[
    p(y_t | y_1, \dots, y_{t-1}, c) = g(y_{t-1}, s_{t}, c)
\]

where $c = h_{T_x}$, and $g$ is a feedforward network. 

In the first neural translation models, $f$ was a recurrent neural network or an LSTM. Hence, $h_t$ and $s_t$ were hidden states computed by an RNN. However, $f$ in theory can be any function.

## Attention mechanism 

The key modification to this architecture in the  [(Bahdanau, et. al 2015)](https://arxiv.org/pdf/1409.0473.pdf) paper was the modification of the 
context vector $c$. In the original architecture we described, we simply had $c = h_{T_x}$. Thus, the last hidden state of the encoder has to 
be a summary of the entire input sentence. This observation leads to the following question: given a fixed hidden state dimension, what happens when 
we process longer and longer sentences? Intuitively, one would expect that the model would eventually run into difficulty compressing 
so much information in one hidden state, and perhaps performance would degrade for longer sentences. This was empirically verified in the 
Bahdanau et. al. paper, and it was observed that performance of the original encoder-decoder degrades for longer sentences. 

The proposed modification to the existing decoder architecture was simply defining 

\[
    p(y_t | y_1, \dots, y_{t-1}) = g(y_{t-1}, s_{t}, c_t)
\]

where 

\[
    s_t = f(s_{t-1}, y_{t-1}, c_t)
\]

As before, $g$ is a feedforward neural network which outputs a probability vector, and $f$ is an arbitrary function, but most often an RNN or LSTM. 
The vector $c_t$ is known as the **context vector**, and is defined as 

\[
    c_i = \sum_{j = 1}^{T_x}\alpha_{ij}h_j 
\]

where $i = 1, \dots, T_y$, $\alpha_{ij} \in \mathbb{R}$, and 
each $h_j$ is the $j$-th encoder's hidden state. Thus, $\alpha_{ij}$ is a vector that tells us which encoder hidden state is most 
important when computing the $i$-th translated word in the target language. This value is computed via 

\[
    \alpha_{ij} = \frac{\text{exp}(e_{ij})}{\sum_{k=1}^{T_x}\text{exp}(e_{ik})}
\]

where $e_{ij} = a(s_{i-1}, h_j)$ and $a$ is a single layer feedforward neural network, known as the *alignment model*. 
Note that computing all the vectors $c_i$ requires $T_x \times T_y$ many calls to the feedforward alignment model. 

Equivalently, we can define for $j = 1, \dots, T_x$

\[
    \vec{e}_j = (e_{1j}, \dots, e_{T_yj})
\]

in which case 

\[
    \alpha_{ij} = \text{softmax}(\vec{e}_j)_i
\]

This is how the attention calculations were presented in the original paper. However, 
we can go even further to compact the attention mechanism. Since 

\[
    c_i = \sum_{j = 1}^{T_x}\alpha_{ij}h_j = [\alpha_{i1} \cdots \alpha_{iT_x}] \cdot [h_1 \cdots h_{T_x}]^{T}
\]

then we can compute a matrix $C \in \mathbb{R}^{T_x \times T_y}$ where the $i$-th row corresponds to $c_i$ as below:

\begin{align}
    C &= 
    \begin{bmatrix}
    \alpha_{11} & \dots & \alpha_{1T_x}\\
    \vdots & \ddots & \vdots \\
    \alpha_{T_y1} & \dots & \alpha_{T_y T_x}\\
    \end{bmatrix}
    \cdot 
    [h_1, \dots, h_{T_x}]^T
\end{align}

Since the columns of the matrix above correspond to each $e_j$, we have that 

\begin{align}
    C &=
    \text{softmax} (\begin{bmatrix} e_1  & \cdots & e_{T_x} \end{bmatrix})
    \cdot 
    [h_1, \dots, h_{T_x}]^T \\
    &=
    \text{softmax} (E)
    \cdot 
    [h_1, \dots, h_{T_x}]^T 
\end{align}

Viewing it this way, this is similar to the attention mechanism proposed in the famous Attention is All You Need paper that superseded this architecture. 

