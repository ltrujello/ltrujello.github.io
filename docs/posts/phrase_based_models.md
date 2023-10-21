---
date: 2023-10-19
---

# Phrased-Based Translation Model and Decoder

While IBM models were effective at word alignment, they don't produce the best translation models, since in order to use it for translation one must perform 
a word-by-word approach. However, when it comes to translation, this isn't the best approach. 

For starters, a word-by-word approach does not take into the context that each word appears in. For example, suppose we are machine that has seen a translation of  "Tienes hambre?" and we are now given "Tienes tarea?" Following a naive maximum-likelihood, word-by-word approach, since we know that 
"Tienes hambre?" -> "Are you hungry?", we might end up with a translation "Tienes tarea?" -> "Are you homework?" when it should be "Do you have homework?". Conversely, if we flipped our examples we might end up translating "Tienes hambre?" -> "Do you have hungry?" 

<!-- more -->

It turns out that it is actually easier to translate from a source language to a target language if we perform translation by examining local groups of words--which we call phrases--based on how often we see them together and on what group of words in the target language they tend to be aligned to.

In order to formulate this, consider more generally the fact that what we seek in terms of a translation model is a conditional probabilistic model $p(\mathbf{e} | \mathbf{f})$ where $\mathbf{e}$ represents a target language sentence and $\mathbf{f}$ represents our source language sentence. This is because if we can obtain such a model, we can compute 

\[
    \text{argmax}_{\mathbf{e}} p(\mathbf{e}| \mathbf{f}) 
\]  

in order to find an optimal translation of a given source language sentence $\mathbf{f}$. Note that by a simple application of Bayes theorem, we can write 

\[
\mathrm{argmax}_{\mathbf{e}} p(\mathbf{e}, \mathbf{f}) 
= 
\text{argmax}_{\mathbf{e}} \frac{p(\mathbf{f}| \mathbf{e})p(\mathbf{e}) }{p(\mathbf{f}) }
= 
\text{argmax}_{\mathbf{e}} p(\mathbf{f}| \mathbf{e})p(\mathbf{e}) 
\]

The last expression offers a nice interpretation to our task at hand. It means we must develop a translation model going into the reverse direction (target language to source) and a language model $p(\mathbf{e})$, which from hereon we denote as $p_{LM}$. As a side note, this formulation of $p(\mathbf{e}| \mathbf{f})$ is known as the **Noisy Channel** model, as it can be thought of more generally as attempting to decode a given message over a noisy channel. 

## Phrase-Based Model

At this point, we now introduce the phrased-based translation model 
$p(\mathbf{f}| \mathbf{e})$. Whereas before with IBM models we were partitioning our sentences word-by-word $\mathbf{e} = (e_1, \dots, e_n)$, with the phrase-based approach we partition the sentence by 
groups of words, known as phrases:

\[
    \mathbf{e} = (\overline{e}_1, \dots, \overline{e}_I)
\]

where in the above expression $I$ is the number of phrases we divide our sentence into. Note that phrases are denoted with an overline bar, and that we will write $\mathbf{e}^I$ to emphasize via $I$ that $\mathbf{e}$ is a sentence that has some partition $(\overline{e}_1, \dots, \overline{e}_I)$ relevant to the given context.

With that said, the phrase-based translation model is given by the expression below.

\[ 
    p(\mathbf{f}^I, \mathbf{e}^I) 
    = 
    \prod_{i = 1}^{I}
    \phi(\overline{f}_i | \overline{e}_i)
    d(\text{start}_i - \text{end}_{i-1} - 1)
\]

where 

* $\phi(\overline{e} | \overline{f})$ is a **phrase translation model** that models the probability a French phrase is translated by 
an English phrase (Note that the phi symbol is used for "phrase")
* $\text{start}_i$ denotes the index in the string $\mathbf{f}$ that phrase $\overline{f}_i$ *starts* at
* $\text{end}_i$ denotes the index in the string $\mathbf{f}$ that phrase $\overline{f}_i$ *ends* at
* $d(\text{start}_i - \text{end}_{i-1} - 1)$ is a **distortion model** that takes into account reordering that can occur when performing translation at the phrase level. 

For an initial approach, we can use a simple exponential model $d(x) = \alpha^{|x|}$ to model distortion. 

The above model then leads to our calculation that 

\begin{align}
\mathrm{argmax}_{\mathbf{e}} p(\mathbf{e}, \mathbf{f}) 
&= 
\mathrm{argmax}_{\mathbf{e}}
p(\mathbf{f}^I, \mathbf{e}^I)p_{LM}(\mathbf{e})\\
&= 
\mathrm{argmax}_{\mathbf{e}}
\prod_{i = 1}^{I}
\phi(\overline{f}_i | \overline{e}_i)
d(\text{start}_i - \text{end}_{i-1} - 1)
\prod_{i = 1}^{|\mathbf{e}|}
p_{LM}(e_i| e_1, \dots, e_{i})
\end{align}

The above model is how the first successful phrased-based translation model was proposed by Koehn, Och, and Marcu in their 2003 
paper *Statistical Phrase-Based Translation*. In the paper, they also introduce the idea of weighting the language model, and setting this weight parameter close to 0.25, to achieve their desired performance. It turned out later that the model could be even further customized by weighting different parts of the function as well.

To do this, one can specify parameters $\lambda_{\phi}$, 
$\lambda_{d}$, and $\lambda_{LM}$ to control how the phase translation model, distrotion model, and language model contribute to the overall translation model, as follows.

\[
p(\mathbf{f}^I, \mathbf{e}^I)
=
\prod_{i = 1}^{I}
\phi^{\lambda_{\phi}}(\overline{f}_i | \overline{e}_i)
d^{\lambda_{d}}(\text{start}_i - \text{end}_{i-1} - 1)
\prod_{i = 1}^{I}
p_{LM}^{\lambda_{LM}}(e_i| e_1, \dots, e_{i})
\]

This formulation of the translation model becomes useful once we additionally notice that 
$\mathrm{argmax}_{\mathbf{e}} p(\mathbf{e}, \mathbf{f})
= \mathrm{argmax}_{\mathbf{e}} \log(p(\mathbf{e}, \mathbf{f}))$.
Doing so, we can then see that we actually just need to maximize the expression below.

\begin{align}
\log(p(\mathbf{f}^I, \mathbf{e}^I))
&=
\lambda_{\phi}\sum_{i = 1}^{I}
\phi(\overline{f}_i | \overline{e}_i)\\
& +
\lambda_d
\sum_{i = 1}^{I}
d(\text{start}_i - \text{end}_{i-1} - 1)\\
& + 
\lambda_{LM}
\sum_{i = 1}^{|\mathbf{e}|}
p_{LM}(e_i | e_1, \dots, e_{i-1})
\end{align}

which is actually really nice. The model is now a linear combination of several different features (phrase translation probability, distortion probability, language model). Further, in an implementation context, we can avoid numerical underflow by working in the logarithmic space, which 
is likely to occur because probability can often be very small numbers. And from this perspective, we can additionally tack on 
other statistical features and weight them with different parameters to customize the model even further.

With the model itself introduce, the task at hand is to learn the parameters of such a model given a parallel corpus. Really, the most important task is to learn phrase translation probabilities from a parallel corpus, which is nontrivial. 

## Word alignment for Phrase Extraction


