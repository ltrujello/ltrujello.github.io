---
date: 2023-11-27
---

# Neural Translation with Attention

For many years, machine translation systems relied on statistical phrase-based translation models [(Koehn, 2003)](https://doi.org/10.3115/1073445.1073462), for machine translation. 
During this time, methods in neural translation were researched, although it took some time for performance of these translation 
models to compete with statistical machine translation. The breakthrough that caused many to take neural translation was through the paper 
[(Bahdanau, et. al 2015)](https://arxiv.org/pdf/1409.0473.pdf), in which researchers incorporated existing neural translation models with a concept called *attention*. In this paper, they were able to overcome important issues with existing neural translation methods,
with the worst issue being then fact that previous models would degrade for longer sentences (more than 30 words).  

In this post, we'll mathematically introduce the architecture used by Bahdanau et. al. and implement the network in PyTorch. We'll 
also visualize the model's attention on a lexical level that achieves sensible lexical alignment on translation pairs. 

## Background and previous work
Prior to this paper, the state of the art for neural translation used an **encoder-decoder** architecture with recurrent neural 
networks only. In such an architecture, there are two components main components including the encoder, which encodes the source input sentence, and the decoder, which outputs the translated sentence in the target language.

In the encoder-decoder architecture, an input sentence is represented as a sequence of vectors $(x_1, \dots, x_{T})$ with $x_i \in \mathbb{R}^n$, and is processed through the encoder via a function $f$ computes a hidden state for each time step $t$. Specifically,
the forward pass of the encoder is

\[
    h_{t} = f(Ex_t, h_{t-1})
\]

where $E$ is a word embedding of choice for the source language.
In the first neural translation models, $f$ was a recurrent neural network. However, $f$ can really be any function. In most implementations an LSTM or GRU is used for $f$. 

Next, the forward pass of the decoder is 
\[
    s_{t} = g(y_t, s_{t-1})
\]
where in the original set up $s_0 = h_{T}$. As one might already suspect in this set up, setting $s_0 = h_{T}$ and running the decoder would probably have degrading performance on longer sentences. This is because $h_T$ is tasked with single-handedly 
carrying all of the information of the previous words in the original source sentence, and the decoder only has this one vector to use for 
translation in its final translations. 

Once the model is trained, one can forward pass an input sentence to obtain a set of possible translations and then employ a beam 
search to obtain the best possible translation. 