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

and returns a vector $\alpha \in \mathbb{R}^{n}$. 
<!-- more -->
The interpretation of the vector $\alpha$ 
is that each value in the vector corresponds to an attention weight for each $v_i$. 
In the original Transformer architecture, 
attention is modeled via **scaled dot-prodct attention**, and it is a matrix computation. If we let $K \in \mathbb{R}^{n \times d_k}$ 
denote the matrix whose rows correspond to the vectors $k_i$, then the attention weights are 

$$
\text{softmax}\left(
\frac{
q
K^{T}}{\sqrt{d_k}}
\right)
\in \mathbb{R}^{n}
$$

More generally, if we have $m$-many queries, we can form a matrix $Q \in \mathbb{R}^{m \times d_k}$ 
where each row corresponds to the queries $q_i$. This allows us to parallelize 
the matrix computation as

$$
\text{softmax}\left(
\frac{
Q
K^{T}}{\sqrt{d_k}}
\right)
\in \mathbb{R}^{m \times n}
$$

This then leads us to define attention more generally as 

$$
\text{Attention}(Q, K, V) =
\text{softmax}\left(
\frac{
Q
K^{T}}{\sqrt{d_k}}
\right)
V \in \mathbb{R}^{m \times d_v}
$$

The Pytorch code for this would then be as follows. 

```python
import torch
import torch.nn.functional as F
import logging

LOGGER = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def attention(Q, K, V):
    """
    Computes attention given query, keys, values.
    If we have n-many key-value pairs of dimension dk, dv respectively
    and m-many queries of dimension dk, then

    - Q is shape m \\times dk
    - K is shape n \\times dk
    - V is shape n \\times dv
    """
    LOGGER.debug(f"computing attention with dimensions {Q.size()=} {K.size()=} {V.size()=}")
    dk = Q.size(-1)

    # Compute attention
    scale = torch.sqrt(torch.FloatTensor([dk])).to(device)
    attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / scale 
    attention_weights = F.softmax(attention_scores, dim=-1)

    # Calculate the weighted sum of values
    attended_values = torch.matmul(attention_weights, V)

    return attended_values, attention_weights
```
This code allows us to calculate attention for a triplet of 2D tensors. However, this code
is actually more general. The `K.transpose(-2, -1)` call allows this code to compute attention 
for higher dimensional tensors, which is good as we can then parallel-compute 
batches of attention instead of calling this function in a for-loop.


## Multihead attention

In the Transformer architecture, instead of performing a single attention computation on the $Q, K, V$ triplet, 
they proposed performing **multihead attention**. What this does is (1) projects the triplet a certain number of times and (2) computes 
scaled-dot product attention on each projection triplet and then (3) concatenates each attention computation. 
As the projections are learnable model parameters, the idea is that the model can learn to attend to different parts of the input sequence. 

Each attention computation is referred to as an **attention head**, and the $i$-th attention head is equipped with 
learnable projection matrices denoted as
$W^{Q}_i \in \mathbb{R}^{d_k \times d_h}$, 
$W^{K}_i \in \mathbb{R}^{d_k \times d_h}$, 
$W^{V}_i \in \mathbb{R}^{d_v \times d_h}$ 
where $d_h$ is a the **head dimension**. This is usually a fraction of $d_v$, such as $d_v / h$ where $h$ is the number of desired attention heads. 

The $i$-th attention head is computed as 

$$
\text{head}_i = \text{Attention}(QW^{Q}_i, KW^{K}_i, VW^{V}_i) \in \mathbb{R}^{m \times d_h}
$$

These attention heads are then concatenated and applied to a final projection matrix $W^O \in \mathbb{R}^{d_v \times d_v}$, leading us to define 

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1,\text{head}_2, \dots, \text{head}_h) W^{O} \in \mathbb{R}^{m \times d_v}
$$

To write this in code, we could explicitly define $h$-many `torch.nn.Linear` layers. However, note that we can do some parallel computation. 
For example, we can define one matrix $W^Q \in \mathbb{R}^{d_k \times d_v}$ in the code and simply declare that 

\begin{align*}
    Q W^Q = \begin{bmatrix}
    \overbrace{QW_1^Q}^{\text{first } d_h \text{ columns}} & \overbrace{QW_2^Q}^{\text{second } d_h \text{ columns}} & \dots & \overbrace{QW_1^Q}^{\text{last } d_h \text{ 
        columns}}\end{bmatrix}    
\end{align*}

To do this, we write a method called `split_heads` which takes in a Pytorch tensors and splits the 2D matrix columns into $h$-many submatrices

```python
def split_heads(Q, num_heads):
    return torch.stack(Q.split(num_heads, dim=-1))
```

This method will split the 2D matrices, columnwise, into an `num_heads` sized chunks. Each chunk is a view of the original tensor, 
and 
these chunks are then stacked together, horizontally, into individual 2D matrices. The `stack` call copies the chunks
(and so does `torch.cat`), and apparently it is [too complex](https://github.com/pytorch/pytorch/issues/70600) 
for the developers to implement concatenation without copying. In any case, this will transform a tensor like so:
```python 
tensor([[ 1.,  2.,  3.,  4.],
        [ 5.,  6.,  7.,  8.],
        [ 9., 10., 11., 12.],
        [13., 14., 15., 16.]])
```
Into a new tensor 
```python
tensor([[[ 1.,  2.],
         [ 5.,  6.],
         [ 9., 10.],
         [13., 14.]],

        [[ 3.,  4.],
         [ 7.,  8.],
         [11., 12.],
         [15., 16.]]])
```
Using this method, we can now write the Multihead pytorch module. Following the Transformer architecture, we implement this 
by declaring $d_k = d_v = d_{\text{model}}$, and $d_h = d_{\text{model}}/ h$ where $h$ is the number of heads. 

```python
import torch.nn as nn

class MultiheadAttention(nn.Module):
    """
    Class to compute multihead attention with n_heads-many heads.
    """

    def __init__(
        self,
        d_model,
        num_heads,
    ):
        super(MultiheadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Linear projection for the attention heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        # Linear projection for the output layer
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, Q, K, V):
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)

        # Split into multiple heads
        Q = split_heads(Q, self.num_heads)
        K = split_heads(K, self.num_heads)
        V = split_heads(V, self.num_heads)

        # Compute attention
        output, attention_weights = attention(Q, K, V)
        # Concatenate the heads and compute transformation
        output = output.permute(0, 2, 1, 3).reshape(2, 4, -1)
        output = self.W_o(output)
        LOGGER.info(f"{output=} {output.size()=} {output.shape=}")

        return output, attention_weights
```

## Position-wise Networks 

Another relatively simple component of the Transformer architecture are position-wise feed-forward neural networks. These are 
standard feed-forward networks which process each sequence element independently. If our input consists of 
$(x_1, x_2, \dots, x_n)$ with $x_i \in \mathbb{R}^{d_\text{model}}$, then 

$$
    \text{FFN}(x_1, \dots, x_n) = (\text{FFN}(x_1), \dots \text{FFN}(x_n))
$$

where 

$$
\text{FFN}(x_i) = \text{ReLU}(x_iW_1 + b_1)W_2 + b_2
$$ 

where $W_1 \in \mathbb{R}^{d_\text{model} \times h}$ and $W_2 \in \mathbb{R}^{h \times d_{\text{model}}}$. 
In this case, $h$ denotes the number of hidden units in the network, which originally was set to 2048. The 
PyTorch module for this class is as below. 

```python
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionwiseFeedForward, self).__init__()
        self.W_1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.W_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        """
        Computes 
        FFN(x_i) = ReLU(x_iW_1 + b_1)W_2 + b_2.

        - x has shape batch_size \\times seq_length \\times d_model  
        """

        output = self.W_1(x)
        output = self.relu(output)
        output = self.W_2(output)

        return output
```

## Layer Normalization 

Another component of the architecture is **layer normalization**. This is a strategy to stabilize training and reduce training time, introduced 
in [(Lei Be et. al.)](https://arxiv.org/pdf/1607.06450.pdf). In the transformer architecture, this is applied in combination with 
a residual connection in each sublayer in the encoder and decoder. Specifically, the equation is 

$$
\text{LayerNorm}(x + \text{Sublayer}(x))
$$

This is how layer normalization was incorporated into the Transformer architecture as presented in the original paper, and this 
is called **post-layer normalization**. However, 
most if not all implementations of the Transformer will actually perform 
**pre-layer normalization** which we compute as 

$$
x +\text{Sublayer}(\text{LayerNorm}(x))
$$

since as shown in [(Xiong et. al., 2020)](https://arxiv.org/pdf/2002.04745.pdf) it leads to much better training performance. 
In any case, the PyTorch module for layer normalization is given below.  

```python
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        """
        Computes layer normalization. 

        LayerNorm(x) = 
        \\gamma \cdot \\frac{x - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}} + \\beta
        where 
        - \\gamma is a scale parameter
        - \\mu is the mean
        - \\sigma is the standard deviation
        - \\epsilon is an offset for numerical stability 
        - \\beta is a shift parameter.
        For training purposes \\sqrt{\\sigma^2 + \\epsilon} ~= \\sigma + \\epsilon.
        """
        super(LayerNorm, self).__init__()
        self.d_model = d_model
        self.eps = eps

        # Learnable scale and shift parameters
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        # Calculate mean and standard deviation along the last dimension
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)

        # Apply LayerNorm formula
        x_normalized = self.gamma * (x - mean) / (std + self.eps) + self.beta

        return x
```

## Encoder Layer
At this point, we have everything written to now define the encoder layer. The encoder is duplicated 6 times 
before sending its output to the decoder. Each encoder layer consists of layer normalization, multihead self-attention, 
layer normalization again, and a pointwise feed-forward network. We write the 
PyTorch module as below. 
Note that this implements pre-layer normalization, which differs from the original Transformer architecture that implemented 
post-layer normalization. 

```python
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ffn, dropout=0.1):
        super(EncoderLayer, self).__init__()
        
        self.self_attention = MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.feedforward = PositionwiseFeedForward(d_model, d_ffn, dropout=dropout)
        
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Multihead self-attention sub-layer
        x_norm = self.norm1(x)
        attention_output = self.self_attention(x_norm, x_norm , x_norm , mask=mask)
        x = x + self.dropout(attention_output)
        
        # Position-wise feedforward sub-layer
        x_norm = self.norm2(x)
        ff_output = self.feedforward(x_norm)
        output = x + self.dropout(ff_output)
        
        return output
```

