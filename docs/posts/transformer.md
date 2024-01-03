---
draft: true 
date: 2023-12-23
---

# Transformer from Scratch in PyTorch

The Transformer architecture, first introduced in [(Vaswani et. al. 2017)](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf), is an encoder-decoder model that 
can be used in many scenarios of supervised sequence learning. In this post, we'll build the Transformer 
model from scratch with an emphasis on modularity and performance. 

## Attention mechanism
We'll first start with the attention.
In the original paper, the Transformer architecture relies heavily on a relatively simple model for 
the concept of attention. Generally, **attention** is a function that takes in a 

- A query $q \in \mathbb{R}^{d_k}$
- A set of key-value pairs $(k_i, v_i)$ where $k_i \in \mathbb{R}^{d_k}$ and $v_i \in \mathbb{R}^{d_v}$ 
for $i = 1, 2, \dots, n$

and returns a vector $\alpha \in \mathbb{R}^{n}$. The interpretation of the vector $\alpha$ 
is that each value in the vector corresponds to an attention weight for each $v_i$. 
In the original Transformer architecture, 
the formula for attention is a matrix computation. If we let $K \in \mathbb{R}^{n \times d_k}$ 
denote the matrix whose rows correspond to the vectors $k_i$, then the attention weights are 

$$
\text{softmax}(
q
\cdot 
K^{T}
)
\in \mathbb{R}^{n}
$$

More generally, if we have $m$-many queries, we can form a matrix $Q \in \mathbb{R}^{m \times d_k}$ 
where each row corresponds to the queries $q_i$. This allows us to parallelize 
the matrix computation to write

$$
\text{softmax}(
Q
\cdot 
K^{T}
)
\in \mathbb{R}^{n \times m}
$$

This then leads us to define attention more generally as 

$$
\text{Attention}(Q, K, V) =
\text{softmax}(
Q
\cdot 
K^{T}
)
V \in \mathbb{R}^{n \times d_v}
$$
