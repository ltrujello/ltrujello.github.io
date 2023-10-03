---
date: 2023-10-03
---

# IBM Model 1: Statistical Machine Translation and Alignment

In the area of statistical machine translation, the IBM models were a series of models proposed by IBM in the 1990s to perform (1) machine translation
and (2) word alignment
from one source language into another. While these models are now superseded by more modern neural translation models, the ideas and 
algorithms introduced by these models were later utilized in other statistical machine translation models and they're of historical importance. 
Here, we'll discuss the probability theory behind the first model, IBM Model 1, and implement the model in Python.

<!-- more -->

The problem that IBM Model 1 attempted to solve can be stated as follows: Suppose we are given a large corpus of pairs of French and English sentences that 
are translations of each other. Given an French sentence, how can we translate the French sentence into English, and how do we align the words in the pair of translated sentences?

Of course, the actual languages (French, English) are not fixed; ideally we'd like to be able to translate arbitrarily between languages, but of course in 
practice some languages can be translated more easily into others.

## Notation

Let us establish some notation. We'll denote English sentences with a boldface symbol $\mathbf{e}$, and we'll 
denote English words appearing in a sentence as $e_1, \dots, e_n$. 
Similarly, we'll denote French sentences as $\mathbf{f}$ with French words $f_1, f_2, \dots, f_n$. 

## Goal of Machine Translation

Ultimately, what we seek is some kind of conditional probabilistic model that can tell us how likely an English sentence is *given* some French sentence.

\[
    p(\mathbf{e} | \mathbf{f}) = p(e_1, \dots, e_n | f_1, \dots, f_m)
\]

If we had such a model, then we could perform a search over a space of possible English words to maximize the probabilistic model. This 
would lead to our proposed English translation of a given French sentence. 

Defining this model is a very hard. Suppose for a moment we switch to Spanish (because I know Spanish better than French). 
Suppose we have the Spanish sentence "Tienes hambre? Vamos a comer." 
The following are possible translations of the sentence into English.

- Are you hungry? Let's go get something to eat.
- Are you hungry? Let's go eat some food. 
- Are you hungry? Let's go eat.

Even from this simple example, we can immediately see that this is going to be a very hard problem to solve.

- The number of words in each possible valid translation can vary, and it is usually not equal to the number of words in the original foreign sentence.
- Some English words don't correspond to anything in the source language, and vice versa. For example, the word 
"Are" in "Are you hungry" does not align to any Spanish word in "Tienes hambre?". It's technically just "You hungry?" in English but of course that would not be a 
good translation. This is a problem because if you are a machine translating Spanish, how are you supposed to know you need to generate the word "Are" if that English word 
doesn't even correspond to anything in the sentence you're trying to translate?
- We can't really translate one-word-at-a-time; context matters. For example, "Tienes hambre" means "Are you hungry?" but that does not 
mean that "Tienes tarea?" means "Are you homework?"
- Sometimes, languages have conflicting rules. For example "la linea azul" translates to "The blue line", because in Spanish usually the adjective is used 
after the subject, whereas in English it is before. If you're a machine translating this sentence, you might know "azul" means "blue", but how are you supposed to know to 
put "blue" before the subject whereas before, "azul" came after the subject in Spanish? 

Solving this problem in general is extremely difficult. While languages do have structure and coherence, the rules change depending on context like emotion, intention, etc., and they are very far from being well defined in the same way that programming languages are 100% capable of being defined with an explicit grammar, which we can then plug into a parser generator like Lex and Yacc to generate programming language parsers.

IBM Model 1 was created in an attempt to address machine translation, but it turns out that it's actually not a great
translation model. It is, however, pretty good at perform lexical translation, and pretty good at aligning words 
between pairs of translated sentences. 

## Word Alignment

For IBM Model 1, they introduced the concept of a word alignment between translated sentences to make the desired conditional probabilistic model less complex. Suppose $\mathbf{e} = (e_1, \dots, e_n)$ is a translation of $\mathbf{f} = (f_1, \dots, f_m)$. Then an **alignment** of 
$\mathbf{e}$ to $\mathbf{f}$ is a sequence of $n$ values $a_1, \dots, a_n$ such that 
$a_i \in \{0, 1, \dots, m\}$, with the interpretation that 
the word $f_{a_i}$ is a translation of the English word $e_{i}$. 

Thus, each word $e_i$ has an alignment value $a_i$, whose duty is essentially 
to point the French word $f_{a_i}$ that it best corresponds to.


However, it's sometimes the case that a given English word $e_i$ doesn't correspond to any word in the original translation. 
In this case, we assign $a_i = 0$. We saw this earlier; the "Are" in "Are you hungry" translation does not correspond to any spanish word in the sentence "Tienes hambre?"

## IBM Model 1

Now that we have defined what an alignment is between a pair of translated sentences, we can go ahead and offer the definition 
of IBM Model 1. For this model, it is not that ambitious because it actually takes into account word alignment, which is extra information that we usually don't have.
However, making this assumption allows us to achieve a closed-form model which we can directly work with.

Let $\mathbf{e} = (e_1, \dots, e_n)$ be an English sentence, $\mathbf{f} = (f_1, \dots, f_m)$ a French sentence, and let $a_1, \dots, a_n$ be an alignment 
from $\mathbf{e}$ to $\mathbf{f}$. Then 

\[
   p(\mathbf{e}, a_1, \dots, a_m | \mathbf{f}) 
   = 
   p(n | m) \cdot \prod_{i = 1}^{n} p(a_i | m) \cdot t(e_i | f_{a_i}) 
\]

We explain the factors in the above expression.
 
- $p(n | m)$ is the probability that the translated sentence has length $n$ given the input had length $m$. For IBM model one, this is uniformly picked. If $N$ is the maximum 
length possible for an English sentence, then $p(n | m) = 1/N$.
- $p(a_i |m)$ is the probability that $e_i$ aligns to $f_{a_i}$  given the length of the original sentence $m$. Note that this is **not** the same thing as the probability that $e_{i}$ translates the word $f_{a_i}$; that's different. 
For IBM model 1, this is also uniform and is equal to $1/(m + 1)$ (plus one because of the null word).
- $t(e_i | f_{a_i})$ is the probability that $e_i$ is translated, or is *explained*, by the word $f_{a_i}$. 

This leads to the definition of IBM model 1:

\[
   p(\mathbf{e}, a_1, \dots, a_n | \mathbf{f}) 
   = 
   \frac{1/N}{(m + 1)^n} \prod_{i = 1}^{n} t(e_i | f_{a_i}) 
\]

From our simplistic assumptions, we have reduced the model to something very simple. What are the parameters of this model? 
The parameters are the values $t(e | f)$ for all possible pairs of words $e, f$ in our dataset. If we have this, then we can compute $p(\mathbf{e}, a_1, \dots, a_n | \mathbf{f})$ and go on with our day.


## Chicken and Egg problem: Expectation Maximization

So now that we have a conditional probabilistic model, suppose we are given a dataset in which we cant try to find the parameters that best
fit the dataset given the model. In our case, our dataset will be a set of $n$ translation pairs $\{(\mathbf{f}_1, \mathbf{e}), \dots, (\mathbf{f}_nm \mathbf{e}_n)  \}$.

However, we actually can't learn the parameters of our probabilistic model $p(\mathbf{e}, a_1, \dots, a_n | \mathbf{f})$, because
a parallel corpus of translations does not give use 
the alignments on the lexical level. Viewed another way, we are being tasked with learning our model using data that is incomplete (the alignments are "missing").

So, we are in a bit of a chicken and egg problem.

- If we had alignments for all sentences $\mathbf{e}$, $\mathbf{f}$ in our dataset, then we could estimate $t(e | f)$. How? We would just loop over the sentences, keep track of how often a word $e$ is aligned to a word $f$, leading to an estimate $t(e | f)$ for all words in the vocabulary. Then, we could go ahead and happily calculate $p(\mathbf{e}, a_1, \dots, a_m | \mathbf{f})$. 
- If we already had the parameters of the model $t(e | f)$, then we could determine the best possible alignments for every sentence pair $\mathbf{e}$ and $\mathbf{f}$. With the alignments all calculated, we could then happily calculate $p(\mathbf{e}, a_1, \dots, a_n | \mathbf{f})$.
- However, we have neither of these from our parallel corpus dataset. So we're stuck.

The main problem here is that 

- The alignments are a required variable in our model.
- We do not get alignment data from a parallel corpus dataset. 

This is actually a very common scenario in statistics: you have a model, and everything's great, but your dataset lacks any data for some of the variables required in your model. In this case, we refer to the sentences $\mathbf{e}$ as our **observed variables**, because we are given these in our dataset, and we refer to the alignment variables $a_1, \dots, a_n$ as our **unobserved** or **latent** variables. 
This type of problem can be solved with the [**Expectation Maximization**](https://courses.csail.mit.edu/6.867/wiki/images/b/b5/Em_tutorial.pdf).
algorithm.

However, rather than bring out the big statistical guns and call it a day, we are going to solve this intuitively. Then it will become clear how 
this becomes an instance of Expectation Maximization.


