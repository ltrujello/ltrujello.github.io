---
date: 2023-12-23
---

# Transformer From Scratch In PyTorch: Model

The Transformer architecture, first introduced in [(Vaswani et. al. 2017)](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf), is an encoder-decoder model that 
can be used in many scenarios of supervised sequence learning. 
The success of the Transformer is primarily due to its performance, simple architecture, and its 
ability to parallelize input which drastically speeds up training. This is in comparison with previous 
traditional sequence learning models, such as recurrent neural networks, which would 
process elements of a sequence one at a time.

In this post, we'll build the Transformer model from scratch in PyTorch with an emphasis on modularity and performance. 
Note that in our implementation, we will be following the Pre-Layer Normalization version of the Transformer.

<!-- more -->

## Imports 
Here, we summarize the imports and global variables we will be using in our implementation.
```python
import torch
import torch.nn.functional as F
import torch.nn as nn
import logging

LOGGER = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

## Overview

Before diving into the code, we give a high-level overview of the Transformer architecture. The Transformer model follows 
an encoder-decoder architecture, where the source sequence is fed into an encoder, and the target sequence and encoder output is fed into 
a decoder. The decoder then outputs probabilities. 

Below is a diagram of the transformer architecture I made.

<img src="/png/transformer/transformer.png" style="margin: 0 auto; display: block; width: 95%;"/> 

This diagram actually demonstrates the architecture of the Pre-Layer Normalization Transformer, which differs from the 
original transformer from Attention is All You Need, which is a Post-Layer Normalization Transformer. The difference will be explained later in this post. We follow the Pre-Layer Normalization Transformer as it is the transformer model used in most 
applications because it has been shown to be superior in [(Xiong et. al., 2020)](https://arxiv.org/pdf/2002.04745.pdf).

Using the diagram above, we can summarize the forward pass of the Transformer at a high level.

- Given a batch of pairs of input sequences, output sequences, we embed the input and outputs 
using their respective embedding matrices and sum these with positional encoding matrices.
- The input embeddings travel through $N$ encoder layers. Each encoder layer consists of two sublayers, 
including multihead attention (to be defined) and a feed forward network. 
- The output embeddings travel through $N$ decoder layers. Each decoder layer consists of three sublayers, 
including masked (to be defined) multihead attention, an encoder-decoder attention layer, 
and a feed forward network.
- Finally, the output is projected back into a vector space whose dimension is the same as the dimension 
of the target vocabularly space, and softmax is applied element-wise. 

Unlike previous sequence-learning models, the Transformer design allows much parallelization which significantly 
speeds up training. It also avoids many issues with exploding gradients that RNNs and LSTMs were known to suffer from.

As we now understand at a high-level how the architecture operates, we now turn to the individual components that 
are necessary to create the Transformer encoder and decoder. The components of the encoder, decoder, and how the encoder outputs 
are fed into the decoder, are what makes the Transformer successful. 


## Attention mechanism
We'll first start with the attention.
In the original paper, the Transformer architecture relies on a relatively simple model for 
the concept of attention. Generally, **attention** is a function that takes in a 

- A query $q \in \mathbb{R}^{d_k}$
- A set of key-value pairs $(k_i, v_i)$ where $k_i \in \mathbb{R}^{d_k}$ and $v_i \in \mathbb{R}^{d_v}$ 
for $i = 1, 2, \dots, n$

and returns a vector $\alpha \in \mathbb{R}^{n}$. 
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

<!-- python: attention -->
```python
def attention(
    Q: torch.tensor,
    K: torch.tensor,
    V: torch.tensor,
    dropout: Optional[nn.Dropout] = None,
    mask: Optional[torch.tensor] = None,
) -> tuple[torch.tensor, Optional[torch.tensor]]:
    """
    Computes attention given query, keys, values.
    If we have n-many key-value pairs of dimension dk, dv respectively
    and m-many queries of dimension dk, then

    - Q has shape (batch_size, m, dk)
    - K has shape (batch_size, n, dk)
    - V has shape (batch_size, n, dv)
    In the transformer architecture we have
    - m = n = seq_len
    - dk = dv = dmodel

    Thus, the attention_weights has shape (batch_size, seq_len, seq_len)
    and the attended_values has shape (batch_size, seq_len, d_model)
    """
    LOGGER.debug(
        f"computing attention with dimensions {Q.size()=} {K.size()=} {V.size()=}"
        f" with mask.size()={mask.size() if mask is not None else None}"
    )
    dk = float(Q.size(-1))

    # Compute attention
    scale = torch.sqrt(torch.tensor(dk)).to(device)
    attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / scale

    # Apply attention mask (if provided).
    if mask is not None:
        LOGGER.debug(f"Applying {mask.size()=} to {attention_scores.size()=}")
        attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

    attention_weights = F.softmax(attention_scores, dim=-1)
    if dropout is not None:
        attention_weights = dropout(attention_weights)

    # Calculate the weighted sum of values
    attended_values = torch.matmul(attention_weights, V)

    return attended_values, attention_weights
```

We make a few comments on this code. 

- This code allows us to calculate attention for a triplet of 2D tensors. However, this code
can handle higher dimensional tensors, which is good as we can then parallel-compute 
batches of attention instead of calling this function in a for-loop.
- We apply a dropout to the attention calculation as per the original Transformer paper. 
- This attention function optionally takes in a mask, which essentially zeros out the attention calculation 
in certain positions. This is necessary for many attention calculations in the Transformer, which we'll discuss briefly now. 


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

<!-- python: split_heads -->
```python
def split_heads(Q: torch.tensor, num_heads: int):
    """
    Split the last dimension into (num_heads, head_dim).
    Reshape the tensor to (batch_size, seq_length, num_heads, head_dim)
    and then transpose to get (batch_size, num_heads, seq_length, head_dim).
    """
    batch_size, seq_length, d_model = Q.size()
    head_dim = d_model // num_heads

    # Reshape to separate heads
    Q = Q.view(batch_size, seq_length, num_heads, head_dim)

    # Transpose to get (batch_size, num_heads, seq_length, head_dim)
    Q = Q.transpose(1, 2)

    return Q
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

<!-- python: MultiheadAttention -->
```python
class MultiheadAttention(nn.Module):
    """
    Class to compute multihead attention with num_heads-many heads
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super(MultiheadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Linear projection for the attention heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        # Linear projection for the output layer
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        Q: torch.tensor,
        K: torch.tensor,
        V: torch.tensor,
        mask: Optional[torch.tensor] = None,
    ):
        LOGGER.debug(
            f"Computing multihead attention with {Q.size()=} {K.size()=} {V.size()=}"
            f" with mask.size()={mask.size() if mask is not None else None}"
        )
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)
        batch_size = Q.size(0)
        d_model = Q.size(-1)

        # Split into multiple heads. Shape should now be (batch_size, num_heads, seq_length, head_dim)
        Q = split_heads(Q, self.num_heads)
        K = split_heads(K, self.num_heads)
        V = split_heads(V, self.num_heads)

        # Add an extra dimension to the mask to get (batch_size, 1, 1, seq_length)
        if mask is not None:
            mask = mask.unsqueeze(1)

        # Compute attention
        output, attention_weights = attention(Q, K, V, dropout=self.dropout, mask=mask)
        # Concatenate the heads and compute transformation
        output = output.permute(0, 2, 1, 3).reshape(batch_size, -1, d_model)
        output = self.W_o(output)

        # if self.training:
        #     return output, None
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

<!-- python: PositionwiseFeedForward -->
```python
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.W_1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.W_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Computes
        FFN(x_i) = ReLU(x_iW_1 + b_1)W_2 + b_2.

        - x has shape batch_size \\times seq_length \\times d_model
        """

        output = self.W_1(x)
        output = self.dropout(self.relu(output))
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

<!-- python: LayerNorm -->
```python
class LayerNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
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

    def forward(self, x: torch.tensor) -> torch.float:
        # Calculate mean and standard deviation along the last dimension
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)

        # Apply LayerNorm formula
        x_normalized = self.gamma * (x - mean) / (std + self.eps) + self.beta

        return x_normalized
```



## Positional encoding

Before introducing the actual Encoder, we must also implement positional encoding. The purpose of position encoding is 
to inject the model input with information about the $i$-th position. In previous recurrent neural network architectures,
this wasn't necessary because input was fed into networks one at a time. Since the input to a transformer can be parallelized, 
the positional encoding adds information about sequence ordering for the model to learn from.

In the original Transformer paper, they implemented sinusoidal positional encoding, which builds a matrix $\text{PE}$ that 
is precomputed. The elements of this matrix are computed as 

$$
PE_{(pos, i)} = 
\begin{cases}
    \sin(\frac{pos}{10000^{i/d_\text{model}}}) & \text{ if } i \mod 2 = 0\\
    \cos(\frac{pos}{10000^{(i-1)/d_\text{model}}}) & \text{ otherwise } \\
\end{cases}
$$

where $i = 0, \dots, d_\text{model} - 1$ and $pos = 0, \dots, \text{maxlen}$ where maxlen is the 
maximum sequence length we allow for input. 
In code, we can produce the positional encoding matrix as follows.

<!-- python: positional_encoding -->
```python
def positional_encoding(max_len: int, d_model: int):
    """
    Computes positional encoding according to
    PE(pos, 2i) = sin(pos/10000^{2i / dmodel})
    PE(pos, 2i + 1) = cos(pos/10000^{2i / dmodel})
    """
    div_terms = torch.pow(torch.tensor(10_000.0), torch.arange(0, d_model, 2) / d_model)
    pos_enc = torch.arange(max_len, dtype=torch.float32).unsqueeze(1).repeat(1, d_model)

    pos_enc[:, 0::2] = torch.sin(pos_enc[:, 0::2] / div_terms)
    pos_enc[:, 1::2] = torch.cos(pos_enc[:, 1::2] / div_terms)

    return pos_enc
```

## Encoder
At this point, we have everything written to now define the encoder layer. The encoder is duplicated 6 times 
before sending its output to the decoder. Each encoder layer consists of layer normalization, multihead self-attention, 
layer normalization again, and a pointwise feed-forward network. We write the 
PyTorch module as below. 
Note that this implements pre-layer normalization, which differs from the original Transformer architecture that implemented 
post-layer normalization. 

<!-- python: EncoderLayer -->
```python
class EncoderLayer(nn.Module):
    """
    Implements a single Encoder layer with pre-layer normalization.
    """

    def __init__(self, d_model: int, num_heads: int, d_ffn: int, dropout: float = 0.1):
        super(EncoderLayer, self).__init__()

        # Self-attention sub-layer
        self.self_attention = MultiheadAttention(d_model, num_heads, dropout=dropout)

        # Position-wise feedforward sub-layer
        self.feedforward = PositionwiseFeedForward(d_model, d_ffn, dropout=dropout)

        # Layer Normalization
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.tensor, mask: Optional[torch.tensor] = None):
        # Multihead self-attention sub-layer
        x_norm = self.norm1(x)
        attention_output, _ = self.self_attention(x_norm, x_norm, x_norm, mask=mask)
        x = x + self.dropout(attention_output)

        # Position-wise feedforward sub-layer
        x_norm = self.norm2(x)
        ff_output = self.feedforward(x_norm)
        output = x + self.dropout(ff_output)

        return output
```

We can then use this `EncoderLayer` class to define our main `Encoder` class, which can instantiate and 
connect any number of encoder layers together.

<!-- python: Encoder -->
```python
class Encoder(nn.Module):
    "Class for encoder, which consists of N-many EncoderLayers"

    def __init__(
        self,
        num_stacks: int,
        d_model: int,
        num_heads: int,
        d_ffn: int,
        dropout: float = 0.1,
    ):
        super(Encoder, self).__init__()

        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    d_model=d_model, num_heads=num_heads, d_ffn=d_ffn, dropout=dropout
                )
                for _ in range(num_stacks)
            ]
        )

    def forward(self, x: torch.tensor, mask: torch.tensor):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return x
```



## Decoder

Similarly we can define our `DecoderLayer` class to represent one instance of a decoder layer. 
In the Transformer model, the decoder similarly uses 6 decoder layers. 

<!-- python: DecoderLayer -->
```python
class DecoderLayer(nn.Module):
    """
    Implements a single Decoder layer with pre-layer normalization.
    """

    def __init__(self, d_model: int, num_heads: int, d_ffn: int, dropout: float = 0.1):
        super(DecoderLayer, self).__init__()

        # Self-attention sub-layer
        self.self_attention = MultiheadAttention(d_model, num_heads, dropout=dropout)

        # Encoder-Decoder attention sub-layer
        self.encoder_attention = MultiheadAttention(d_model, num_heads, dropout=dropout)

        # Position-wise feedforward sub-layer
        self.feedforward = PositionwiseFeedForward(d_model, d_ffn, dropout=dropout)

        # Layer normalization
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.norm4 = LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.tensor,
        encoder_output: torch.tensor,
        self_mask: Optional[torch.tensor] = None,
        encoder_mask: Optional[torch.tensor] = None,
    ):
        # Self-attention sub-layer
        x_norm = self.norm1(x)
        self_attention_output, _ = self.self_attention(
            x_norm, x_norm, x_norm, mask=self_mask
        )
        x = x + self.dropout(self_attention_output)

        # Encoder-Decoder attention sub-layer
        x_norm = self.norm2(x)
        encoder_output_norm = self.norm3(encoder_output)
        encoder_attention_output, encoder_attention_weights = self.encoder_attention(
            x_norm, encoder_output_norm, encoder_output_norm, mask=encoder_mask
        )
        x = x + self.dropout(encoder_attention_output)

        # Position-wise feedforward sub-layer
        x_norm = self.norm4(x)
        ff_output = self.feedforward(x_norm)
        x = x + self.dropout(ff_output)

        return x, encoder_attention_weights
```

This can then be used analogously in our main `Decoder` class as follows. 

<!-- python: Decoder -->
```python
class Decoder(nn.Module):
    "Class for decoder, which consists of N-many DecoderLayers"

    def __init__(
        self,
        num_stacks: int,
        d_model: int,
        num_heads: int,
        d_ffn: int,
        dropout: float = 0.1,
    ):
        super(Decoder, self).__init__()

        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    d_model=d_model, num_heads=num_heads, d_ffn=d_ffn, dropout=dropout
                )
                for _ in range(num_stacks)
            ]
        )

    def forward(
        self,
        x: torch.tensor,
        encoder_output: torch.tensor,
        self_mask: torch.tensor,
        encoder_mask: torch.tensor,
    ):
        "Pass the input (and mask) through each layer in turn."
        attn_weights = []
        for layer in self.layers:
            x, encoder_decoder_attn_weights = layer(
                x, encoder_output, self_mask, encoder_mask
            )
            attn_weights.append(encoder_decoder_attn_weights)
        return x, attn_weights
```


## Masking 

At this point we can actually introduce the Transformer model. However, we will discuss masking in the architecture 
before we do this, as it is a very critical component of the Transformer model training and 
forward pass computation. 

Generally, masking is the idea of hiding certain fields in a data structure used by an algorithm. Usually, the data structure 
is a matrix, and hiding the fields is achieved by setting them to zero. In this case a mask is a binary-valued 
matrix where `False` values correspond to "please treat this value as zero".

In the transformer architecture, masking is used for two purposes. 

- **Padding.** When running a forward pass on the Transformer, it is generally done in batches. This allows us to 
achieve parallelization by stacking a batch of sequences into a matrix and sending it off into the Transformer. 
However, different sequences will consist of different lengths, an issue we can resolve with padding
the sequences so that they all have the same shape. This is fine, but this causes a problem in our attention calculation. 
Specifically, these padded values will contribute to the attention calculation. 
- **Future Words**. When training the Transformer, it is also done in batches, which consist of pairs 
of input and output sequences. However, the point of the Transformer is to be an autoregressive model, meaning that it 
predicts tokens via (1) a given start token and (2) all previously generated tokens. Therefore, when training to predict an output sequence,
it should not be allowed to see *all* of the tokens in the ground truth output sequence.
That is, when predicting token $i$, it must only use the previously seen tokens $1, 2, \dots, i - 1$.  

These two issues arise because the forward pass and training procedure are heavily parallelized. 
Specifically, they arise because of the attention calculation: 

$$
\text{Attention}(Q, K, V) =
\text{softmax}\left(
\frac{
Q
K^{T}}{\sqrt{d_k}}
\right)
V
$$

The solution to these two issues is simple: We just mask certain columns of the matrix 
below, so as to avoid attention towards specific values in $V$ that might correspond to (1) padding 
or (2) future words that we are not allowed to see

$$
\text{softmax}\left(\frac{QK^T }{\sqrt{d_k}}\right) \in \mathbb{R}^{m \times n}
$$

This explains why our `attention` function above had an optional `mask` argument, and this is exactly 
what the masking functionality achieves. To demonstrate this, here's an example of us using 
the mask in the attention, and what using the mask achieves. 

```python
# Some random Transformer params
batch_size = 2
seq_len = 10
d_model = 5

# Construct Q, K, V matrices
Q = torch.rand(size=(batch_size, seq_len, d_model))
K = torch.rand(size=(batch_size, seq_len, d_model))
V = torch.rand(size=(batch_size, seq_len, d_model))

# Construct a mask, set the 5th and last column to zero
mask = torch.ones(batch_size, 1, seq_len).bool()
mask[:, :, -1] = False
mask[:, :, 5] = False

# Compute attention
attn_vals, attn_weights = attention(Q, K, V, mask=mask)

# The last and 5th attention weights should be zero
assert torch.all(attn_weights[:, :, -1] == 0)
assert torch.all(attn_weights[:, :, 5] == 0)
```

As we can see, the mask allowed us to "zero out" whichever columns we wanted zeroed out in the 
attention calculation. The one question remains; how do we generate these masks in code?  
To answer that, we briefly demonstrate construction of these two types of masks.

Below we construct a padding mask given a batch of sequences with different lengths.

```python
from torch.nn.utils.rnn import pad_sequence

# List of sequences
tensors = [
    torch.tensor([1, 2, 3, 4]),
    torch.tensor([5, 6, 7]),
    torch.tensor([8, 9]),
    torch.tensor([10, 11, 12, 13, 14])
]

# Pad the tensors to the maximum length
padded_tensors = pad_sequence(tensors, batch_first=True, padding_value=0)
print(padded_tensors)
# will print  
tensor([[ 1,  2,  3,  4,  0],
        [ 5,  6,  7,  0,  0],
        [ 8,  9,  0,  0,  0],
        [10, 11, 12, 13, 14]])

# Construct the padding mask
src_mask = (padded_tensor != 0)
print(src_mask)
# will print
tensor([[[ True,  True,  True,  True, False],
         [ True,  True,  True, False, False],
         [ True,  True, False, False, False],
         [ True,  True,  True,  True,  True]]])
```

Constructing the future mask for masking words is also pretty easy. 

```python
def future_mask(sequence_length):
    """
    Creates a lower-triangular n \\times n matrix
    used to mask future positions
    """
    attn_shape = (1, sequence_length, sequence_length)
    future_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return future_mask == 0
```

For example, this function returns output like below.

```python
future_mask(5)
# will print 
tensor([[[ True, False, False, False, False],
         [ True,  True, False, False, False],
         [ True,  True,  True, False, False],
         [ True,  True,  True,  True, False],
         [ True,  True,  True,  True,  True]]])
```

It is important to note that in theory a lot of the padding and future masks would contain much 
duplicate data. We can circumvent this by using [broadcasting](https://pytorch.org/docs/stable/notes/broadcasting.html) 
in conjunction with PyTorch's `masked_fill` method. In this implementation, 

- Padding masks will have shape `(batch_size, 1, sequence_length)` in order to broadcast across the zeroth and second dimension in the attention calculation.
- Future masks will have shape `(1, sequence_length, sequence_length)`, which will broadcast across the zeroth dimension in the attention calculation.

## Transformer

We now introduce the transformer model. Below is our implementation using our previous code. 

<!-- python: Transformer -->
```python
class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        num_encoder_stacks: int = 6,
        num_decoder_stacks: int = 6,
        d_model: int = 512,
        d_ffn: int = 2048,
        num_encoder_heads: int = 8,
        num_decoder_heads: int = 8,
        max_seq_len: int = 100,
        dropout: float = 0.1,
    ):
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.src_embedding = Embeddings(src_vocab_size, d_model)
        self.tgt_embedding = Embeddings(tgt_vocab_size, d_model)
        self.encoder = Encoder(num_encoder_stacks, d_model, num_encoder_heads, d_ffn)
        self.decoder = Decoder(num_decoder_stacks, d_model, num_decoder_heads, d_ffn)
        # Mark positional encoder as not learnable, so that .parameters() doesn't pass it to optimizer
        self.register_buffer(
            "positional_encoder", positional_encoding(max_seq_len, d_model)
        )
        self.output_layer = nn.Linear(d_model, tgt_vocab_size)
        self.src_dropout = nn.Dropout(dropout)
        self.tgt_dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

        # Initialize parameters with Glorot / fan_avg.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src: torch.tensor, src_mask: torch.tensor):
        LOGGER.debug(
            "Computing forward pass of encoder with "
            f"{src.size()=}, {src_mask.size()=}"
        )
        # Embed inputs, add position encoding, apply dropout
        src = self.src_embedding(src)
        src = src + self.positional_encoder[: src.size(1)]
        src = self.src_dropout(src)

        # Encode the source sequence
        enc_output = self.encoder(src, src_mask)
        return enc_output

    def decode(
        self,
        tgt: torch.tensor,
        enc_output: torch.tensor,
        tgt_mask: torch.tensor,
        src_mask: torch.tensor,
    ):
        LOGGER.debug(
            "Computing forward pass of decoder with "
            f"{tgt.size()=}, {enc_output.size()=}, {tgt_mask.size()=}, {src_mask.size()=}"
        )
        # Embed targets, add position encoding, apply dropout
        tgt = self.tgt_embedding(tgt)
        tgt = tgt + self.positional_encoder[: tgt.size(1)]
        tgt = self.tgt_dropout(tgt)

        # Decode the target sequence using the encoder output
        dec_output, encoder_decoder_attn_weights = self.decoder(
            tgt, enc_output, tgt_mask, src_mask
        )
        return dec_output, encoder_decoder_attn_weights

    def forward(
        self,
        src: torch.tensor,
        tgt: torch.tensor,
        tgt_mask: torch.tensor,
        src_mask: torch.tensor,
    ):
        """
        Forward pass of Transformer.

        - src has size (batch_size, src_seq_len)
        - tgt has size (batch_size, tgt_seq_len)
        - src_mask has size (batch_size, 1, seq_len), and
          prevents attention to padding indices
        - tgt_mask has size (1, tgt_seq_len, tgt_seq_len), and
          prevents attention to future positions and padding
        - output has size (batch_size, tgt_seq_len, tgt_vocab_size)
        """
        LOGGER.debug(
            f"computing forward pass with {src.size()=} "
            f"{tgt.size()=} {src_mask.size()=} {tgt_mask.size()=}"
        )

        enc_output = self.encode(src, src_mask)
        dec_output, encoder_decoder_attn_weights = self.decode(
            tgt, enc_output, tgt_mask, src_mask
        )

        # Compute output layer
        logits = self.output_layer(dec_output)
        return logits, encoder_decoder_attn_weights
```