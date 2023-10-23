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

If we had such a model, then we could perform a search over a space of possible English words to maximize the probabilistic model.
The task of doing this given such a probabilistic model as above is known as **decoding**, and is a whole 
other problem on its own that we won't get into. This 
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
fit the dataset given the model. In our case, our dataset will be a set of $k$ translation pairs $\{(\mathbf{f}_1, \mathbf{e}_1), \dots, (\mathbf{f}_k,  \mathbf{e}_k)  \}$.

However, we actually can't learn the parameters of our probabilistic model $p(\mathbf{e}, a_1, \dots, a_n | \mathbf{f})$, because
a parallel corpus of translations does not give use 
the alignments on the lexical level. In fact, not only do we not have the explicit alignment data, we don't know 
what alignments are more "likely".
Viewed another way, we are being tasked with learning our model using data that is incomplete (the alignments are "missing").

So, we are in a bit of a chicken and egg problem.

- If we had alignments for all sentences $\mathbf{e}$, $\mathbf{f}$ in our dataset, then we could estimate $t(e | f)$. How? We would just loop over the sentences, keep track of how often a word $e$ is aligned to a word $f$, leading to an estimate $t(e | f)$ for all words in the vocabulary. Then, we could go ahead and happily calculate $p(\mathbf{e}, a_1, \dots, a_m | \mathbf{f})$. 
- If we already had the parameters of the model $t(e | f)$, then we could determine the best possible alignments for every sentence pair $\mathbf{e}$ and $\mathbf{f}$. With the alignments all calculated, we could then happily calculate $p(\mathbf{e}, a_1, \dots, a_n | \mathbf{f})$.
- However, we have neither of these from our parallel corpus dataset. 

The main problem here is that 

- The alignments are a required variable in our model.
- We do not get alignment data from a parallel corpus dataset. 

This is actually a very common scenario in statistics: you have a model, and everything's great, but your dataset lacks any data for some of the variables required in your model. In this case, we refer to the sentences $\mathbf{e}$ as our **observed variables**, because we are given these in our dataset, and we refer to the alignment variables $a_1, \dots, a_n$ as our **unobserved** or **latent** variables. 
This type of problem can be solved with the [**Expectation Maximization**](https://courses.csail.mit.edu/6.867/wiki/images/b/b5/Em_tutorial.pdf).
algorithm.

However, rather than bring out the big statistical guns and call it a day, we are going to solve this intuitively. Then it will become clear how 
this becomes an instance of Expectation Maximization.


## Estimating alignment probabilities from $(t | e)$

When we were discussing the chicken and egg issue before, we said that if we had the 
model parameters $t(e | f)$, we could determine the best possible alignment. How would that work?

For a specific sentence pair $\mathbf{e}$, $\mathbf{f}$, finding an optimal alignment $a_1, \dots, a_n$ 
would involve calculating the probability
$p(a_1, \dots, a_n | \mathbf{e}, \mathbf{f})$. This quantity is actually a posterior probability, and we can calculate it 
via 

\[
   p(a_1, \dots, a_n | \mathbf{e}, \mathbf{f}) = \frac{ p(\mathbf{e}, a_1, \dots, a_n | \mathbf{f}) }{p(\mathbf{e}|\mathbf{f})}
\]

We already have a formula for $p(\mathbf{e}, a_1, \dots, a_n | \mathbf{f})$, so we can go ahead and focus on calculating $p(\mathbf{e}|\mathbf{f})$. 
Note that 

\[
   p(\mathbf{e}|\mathbf{f}) = \sum_{a_1 = 0}^{m} \dots \sum_{a_n = 0}^{m} p(\mathbf{e}, a_1, \dots, a_n | \mathbf{f})
\]

which we can rewrite as 

\[
   p(\mathbf{e}|\mathbf{f}) = \sum_{a_1 = 0}^{m} \dots \sum_{a_n = 0}^{m} \frac{1/N}{(m + 1)^n} \prod_{i = 1}^{n} t(e_i | f_{a_i}) 
\]

At first, this is a horribly complex summation which would be wildly inefficient in a program implementation.
Fortunately, there is a trick we can perform on this expression to obtain an equivalent but simpler expression.
To demonstrate this trick, let us ignore the factor $\frac{1/N}{(m + 1)^n}$ which we can trivially factor out of the equation. Observe that we can perform a 
simplification to get rid of one of the sigma sums.

\begin{align}
   p(\mathbf{e}|\mathbf{f}) 
   &= 
   \sum_{a_1 = 0}^{m} \dots \sum_{a_{n-1} = 0}^{m} \sum_{a_n = 0}^{m} \prod_{i = 1}^{n} t(e_i | f_{a_i}) \\
   &= 
   \sum_{a_1 = 0}^{m} \dots \sum_{a_{n-1} = 0}^{m} 
   \left(
   \prod_{i = 1}^{n-1} t(e_i | f_{a_i}) \cdot t(e_{n}| f_{0}) + \dots + \prod_{i = 1}^{n-1} t(e_i | f_{a_i})\cdot t(e_{n}| f_{m}) \right)\\
   &= 
   \sum_{a_1 = 0}^{m} \dots \sum_{a_{n-1} = 0}^{m} \left(\prod_{i = 1}^{n-1} t(e_i | f_{a_i}) \sum_{j = 0}^{m}t(e_n | f_j) \right)
\end{align}

Performing this trick again on the resulting above expression, we obtain

\begin{align}
   p(\mathbf{e}|\mathbf{f}) 
   &= 
   \sum_{a_1 = 0}^{m} \dots \sum_{a_{n-2} = 0}^{m} \left(\prod_{i = 1}^{n-2} t(e_i | f_{a_i}) \sum_{j = 0}^{m}t(e_{n-1} | f_j) \sum_{j = 0}^{m}t(e_n | f_j) \right)
\end{align}

Performing this trick on each sigma symbol $n$ times leads to 

\begin{align}
   p(\mathbf{e}|\mathbf{f}) 
   &= 
   \sum_{j = 0}^{m}t(e_{1} | f_j) \cdots \sum_{j = 0}^{m}t(e_{n-1} | f_j) \cdot \sum_{j = 0}^{m}t(e_n | f_j) \\
   &= 
   \prod_{i = 1}^{n} \sum_{j = 0}^{m} t(e_i | f_j)
\end{align}

Of course, adding back our original factor leads to 

\[
   p(\mathbf{e}|\mathbf{f}) 
   = 
   \frac{1/N}{(m + 1)^n}
   \prod_{i = 1}^{n} \sum_{j = 0}^{m} t(e_i | f_j)
\]

Now that we have this calculation, we can state now that 

\begin{align}
   p(a_1, \dots, a_n | \mathbf{e}, \mathbf{f}) 
   &= 
   \frac{ p(\mathbf{e}, a_1, \dots, a_n | \mathbf{f}) }{p(\mathbf{e}|\mathbf{f})} \\
   &= 
   \frac{ \frac{1/N}{(m + 1)^n} \prod_{i = 1}^{n} t(e_i | f_{a_i}) }{ \frac{1/N}{(m + 1)^n} \prod_{i = 1}^{n} \sum_{j = 0}^{m} t(e_i | f_j) }\\
   &= 
   \frac{ \prod_{i = 1}^{n} t(e_i | f_{a_i}) }{ \prod_{i = 1}^{n} \sum_{j = 0}^{m} t(e_i | f_j) }\\
\end{align}

We can then combine the products in the numerator and denominator to obtain that 

\[
   p(a_1, \dots, a_n | \mathbf{e}, \mathbf{f}) 
   = \prod_{i = 1}^{n}
   \frac{  t(e_i | f_{a_i}) }{ \sum_{j = 0}^{m} t(e_i | f_j) }
\]

Therefore, we see that if we know the model parameters $t(e | f)$, then we could estimate the best possible alignments via the above expression.

## Estimating $t(e | f)$ from alignment probabilities

On the other hand, recall again that when we were discussing the chicken and egg problem, we said that if we knew 
the alignment (probabilities) for each sentence then we could estimate the model parameters. How would that work?

Since we want to estimate $t(e | f)$ given knowledge about the probabilities of alignments, let's start with a simpler 
goal of estimating $t(e | f; \mathbf{e}, \mathbf{f})$ where $\mathbf{e}$ is a sentence containing the word $e$ and $\mathbf{f}$ is a 
sentence containing the word $f$. That is, we'll try to estimate this probability given two sentences. 

If one has access to $p(a_1, \dots, a_n | \mathbf{e}, \mathbf{f})$ for all alignments $a_1, \dots, a_n$, 
then we can estimate $t(e | f; \mathbf{e}, \mathbf{f})$ by adding up the probabilities of all alignments where 
$e$ and $f$ are aligned in some way. More formally, we could state that

\[
   t(e | f; \mathbf{e}, \mathbf{f}) \propto \sum_{a_1 = 0}^{m} \dots 
   \sum_{a_n = 0}^{m}
   \left( p(a_1, \dots, a_n | e) 
   \sum_{i = 1}^{n}\delta(e, e_i) \delta(f, f_{a_i})\right)
\]

We state this is *proportional* to $t(e | f)$, because this is not normalized. To normalize this, note that in theory, every 
other word in the sentence $e$ could also be a translation of $f$. Therefore, the correct probability is  

\[
   t(e | f; \mathbf{e}, \mathbf{f}) =
   \frac{\sum_{a_1 = 0}^{m} \dots 
   \sum_{a_n = 0}^{m}
   \left( p(a_1, \dots, a_n | \mathbf{e}, \mathbf{f}) 
   \sum_{i = 1}^{n}\delta(e, e_i) \delta(f, f_{a_i})\right)
   }{
   \sum_{k = 0}^{n}
   \sum_{a_1 = 0}^{m} \dots 
   \sum_{a_n = 0}^{m}
   \left( p(a_1, \dots, a_n | \mathbf{e}, \mathbf{f}) 
   \sum_{i = 1}^{n}\delta(e_k, e_i) \delta(f, f_{a_i})\right)
   }
\]

Another way to see that this is the correct normalization is to observe that 
for this probability distribution, we'd need that
$\sum_{e \in \mathbf{e}} t(e | f; \mathbf{e}, \mathbf{f}) = 1$ to be satisfied, which 
the above expression does in fact satisfy. 

An easier way to understand the above expression is to define

\[
   c(e| f ;\mathbf{e}, \mathbf{f}) =
   \sum_{a_1 = 0}^{m} \dots 
   \sum_{a_n = 0}^{m}
   \left( p(a_1, \dots, a_n | \mathbf{e}, \mathbf{f}) 
   \sum_{i = 1}^{n}\delta(e, e_i) \delta(f, f_{a_i})\right)
\]

This quantity appears more complicated than it actually is: it's just **counting** how many times $e$ and $f$ are aligned in the 
sentence pair, and it is **weighting** this count by the probability of the alignment. 

With this notation, we can more simply write

\[
   t(e | f; \mathbf{e}, \mathbf{f}) =
   \frac{
      c(e| f ;\mathbf{e}, \mathbf{f}) 
   }{
      \sum_{e \in \mathbf{e}}c(e| f ;\mathbf{e}, \mathbf{f}) 
   }
\]

That is, the sum in the denominator indexes over each word in $\mathbf{e}$. 
Alright, so *if* we somehow had access to the probabilities of each alignment for a given sentence pair,
then we could actually estimate the translation probability for each English and French word pair in the sentence. 
However, this is one just estimating $t(e| f)$ relative to one specific sentence pair picked from our 
more large parallel corpus.
Thus how do we generalize out work across all sentences in the corpus to obtain an estimate of $t(e | f )$? 

This is not that much harder than what we did before. Note that we need 

\[
   \sum_{e \in E}t(e | f)= 1.
\]

where $E$ is the (obviously finite) set of all possible English words,
Therefore, we see that 

\[
   t(e | f) = \frac{\sum_{(\mathbf{e}, \mathbf{f})} t(e | f ; \mathbf{e}, \mathbf{f})}{
   \sum_{e \in E} \sum_{(\mathbf{e}, \mathbf{f})} t(e | f ; \mathbf{e}, \mathbf{f})
   }
\]

Note here that $\sum_{\mathbf{e}, \mathbf{f}}$ indicates a sum over all sentence pairs in
our dataset. 

Obviously, we are being pathological and binding ourselves in a contradiction; if $t(e | f)$ is what 
we are estimating, why is it appearing in our formula? 
Note that we can actually do some mathematical simplification here. 

\[
   t(e | f) = \frac{
      \sum_{(\mathbf{e}, \mathbf{f})}   
      \frac{
         c(e| f ;\mathbf{e}, \mathbf{f}) 
         }{
            \sum_{e \in \mathbf{e}}c(e| f ;\mathbf{e}, \mathbf{f}) 
         } 
   }{
   \sum_{e \in E} \sum_{(\mathbf{e}, \mathbf{f})} 
      \frac{
         c(e| f ;\mathbf{e}, \mathbf{f}) 
         }{
            \sum_{e \in \mathbf{e}}c(e| f ;\mathbf{e}, \mathbf{f}) 
         } 
   }
\]

Observe that we can factor out a common factor in both the numerator and denominator: $\sum_{e \in E}c(e | f; \mathbf{e}, \mathbf{f})$.
This then leads to a more simpler formula:

\[
   t(e | f) = \frac{
      \sum_{(\mathbf{e}, \mathbf{f})}   
      c(e| f ;\mathbf{e}, \mathbf{f}) 
   }{
   \sum_{e \in E} \sum_{(\mathbf{e}, \mathbf{f})} 
         c(e| f ;\mathbf{e}, \mathbf{f}) 
   }
\]

Thus, we have an answer to our question. We can estimate $t(e | f)$ using our parallel corpus by basically 
counting how often $e$ and $f$ are aligned in sentence pairs, and by using 
knowledge of how likely any given alignment is between a sentence pair. 

## Alignment algorithm

To summarize at this point:

- If we knew $t(e |f )$ for all words, we could estimate alignment probabilities $p(a_1, \dots, a_n| \mathbf{e}, \mathbf{f})$ 
- If we knew the alignment probabilities $p(a_1, \dots, a_n| \mathbf{e}, \mathbf{f})$, then we could estimate weighted counts
$c(e | f; \mathbf{e}, \mathbf{f})$ across sentence pairs. This then would allow us to estimate $t(e | f)$. 

However, we have neither. This is where the clever idea of the EM algorithm comes in: we simply initialize $t(e | f)$ to be uniform across all word pairs, calculate  $p(a_1, \dots, a_n| \mathbf{e}, \mathbf{f})$, then calculate $c(e | f; \mathbf{e}, \mathbf{f})$ across sentence pairs, and then finally re-estimate $t(e | f)$. Repeating this a few times will allow us to build a table of values $t(e | f)$ which converge on somewhat reasonable word translations. 

