<style>
.md-content {
    max-width: 80em;
}
</style>
#1.2. Permutation Groups.
One of the most important and well-known types of groups are the
permutation groups, which we introduce formally as follows. 

Consider a finite set of elements $X = \{1, 2, \dots, n\}$. We
define a 
**permutation** to be a reordering of the elements of $X$.
More formally, a **permutation** is a bijective mapping
$\sigma: X \to X$, similar to one that follows. 

How can we represent this information? We generally don't use sets
to represent permutations, since sets 
don't care about order. That is, $\{1, 2, 3\} = \{3,2,1\}$, etc.  

Thus for a set $\{1, 2, \dots, n\}$, we can represent a
permutation $\sigma$ of the set of elements as follows:

\[
\sigma =  
\begin{pmatrix}
1 & 2 & \cdots & n\\
\sigma(1) & \sigma(2) & \cdots & \sigma(n)
\end{pmatrix}
\]

where we read this as $1$ is assigned to $\sigma(1)$, 2 is
assigned to $\sigma(2)$. For example, a permutation that just
shifts the elements down the line is 

\[
\sigma = 
\begin{pmatrix}
1 & 2 & \cdots & n\\
2 & 3 & \cdots & 1
\end{pmatrix}.
\]

That is, $\sigma$ sends 1 to 2, 2 to 3 and eventually $n$ to 1. 
Here we'll denote the set of all permutations of the set $\{1, 2,
\dots, n\}$, or more generally a set of $n$ objects (since we can
always enumerate objects with natural numbers) as $S_n$.

\textcolor{NavyBlue}{What is interesting about this is that if we define
"multiplication" of elements of $S_n$ to be function composition,
then the 
**set of permutations
of $X$ form a group** which we show as follows.}

Let $X = \{1, 2, \dots, n\}$. 
\begin{description}
\item[Closed.]
For any $\sigma_1, \sigma_2 \in S_n$, we see that $\sigma_2
\circ \sigma_1$ is (1) a composition of bijective functions
and therefore is bijective and (2) a permutation of $X$. One
way to think about the composition is that $\sigma_1$
rearranges $X$, and
$\sigma_2$ rearranges $X$ again. Thus $\sigma_2 \circ \sigma_1
\in S_n$. 

\item[Associativity.]
Associativity is obvious since function composition is
associative. 

\item[Identity.]
Observe that the permutation $\sigma_e: X \to X$ for which 
$\sigma_e(i) = i$ is technically a permutation of $X$.
Therefore $\sigma_e$ acts as the identity element in $S_n$. 

\item[Inverse.]
Consider a permutation $\sigma \in S_n$. Define $\sigma^{-1}$
to be the function where if $\sigma(i) = j$, then
$\sigma^{-1}(j) = i$. Then by construction, (1) $\sigma^{-1}$ is a permutation
of $X$ and (2) $\sigma \circ \sigma^{-1} = \sigma^{-1} \circ
\sigma = \sigma_e$. Thus $S_n$ contains inverses and
composition of the inverses returns $\sigma_e$, the identity.
\end{description}


<span style="display:block" class="proposition">
For all $n \ge 1$, $|S_n| = n!$ 
</span>


<span style="display:block" class="proof">
This is counting the number of ways to rearrange a set of size
$n$, which we know from combinatorics to simply be $n!$
</span>

Now that we know that $S_n$ is a group, we'll study the properties
of this group. 

Recall earlier our notation for representing a permutation $\sigma
\in S_n$:

\[
\sigma =  
\begin{pmatrix}
1 & 2 & \cdots & n\\
\sigma(1) & \sigma(2) & \cdots & \sigma(n)
\end{pmatrix}
\]

This notation sucks, since it includes more information than we
actually need to. For instance, the top row is always going to be
the same. 

A better way to write this is through
*cycle decomposition*, which we will soon define.


<span style="display:block" class="definition">
Let $\sigma \in S_n$ and suppose $X = \{1, 2, \dots, n\}.$ 
Suppose that there exists a subset $\{n_1, n_2, \dots, n_k\}$
of $X$
such that 

\begin{align*}
\sigma(n_1) = n_2, \sigma(n_2) = n_3, \dots, \sigma(n_k) = n_1.
\end{align*}

Then $\{n_1, n_2, \dots, n_k\}$ is called a 
**cycle**,
and we denote this cycle as 

\[
\sigma = \begin{pmatrix}
n_1 & n_2 & \cdots & n_k
\end{pmatrix}.
\]

We then read this as "$n_1 \to n_2$, $n_2 \to n_3, \dots, n_k
\to n_1$". 

\textcolor{Blue}{**Why do we care about cycles?**} 
\\
Well,
consider an arbitrary cycle $            \sigma = \begin{pmatrix}
n_1 & n_2 & \cdots & n_k
\end{pmatrix}.$ Then again, $\sigma(n_1) = n_2, \sigma(n_2) =
n_3, \dots, \sigma(n_k) = n_1.$ However, what this is really
saying is that 

\[
\sigma(n_1) = n_2, \sigma^2(n_1) = n_3, \dots, \sigma^{{k-1}}(n_1) = n_k, \sigma^k(n_1) = n_1.
\]

However, also take a note to observe that 

\[
\sigma(n_2) = n_3, \sigma^2(n_2) = n_4, \dots, \sigma^{{k-1}}(n_2) = n_1, \sigma^k(n_2) = n_2.
\]

More generally, we see that \textcolor{blue}{the element $\sigma \in S_n$ has
order $n_k$}, which is why the cycle length is $k$. 

\textcolor{Blue}{**We care about cycles**} since, given the fact that $S_n$
is always a finite group, each of its elements will have
finite order. Thus, in some way, we can always represent the
elements of $S_n$ in this form.
\\
\\
**More definitions.**
\\
If $            \begin{pmatrix}
n_1 & n_2 & \cdots & n_k
\end{pmatrix}$ and $            \begin{pmatrix}
n'_1 & n'_2 & \cdots & n'_k
\end{pmatrix}$ 
share no elements in common, i.e., 

\[
\{n_1, n_2, \dots, n_k\} \cap \{n_1', n_2', 
\dots, n_k'\} = \varnothing
\]


then the cycles are defined as
**disjoint cycles**.
\\

Note that if $\sigma(i) = i$ for some $i \in X$, then this is
technically a cycle and we represent the cycle as $            \begin{pmatrix}
i
\end{pmatrix}.$ In this case, we say that $\sigma$ **fixes** $i$.
\\

For example, suppose we have a permutation $\sigma \in S_5$
where $\sigma(1) = 2, \sigma(2) = 4, \sigma(4) = 2$. Then we
have a cycle of length 4 and we denote this as 

\[
\begin{pmatrix}
1 & 2 & 4
\end{pmatrix}.
\]

Since $\sigma \in S_5$, suppose further
that $\sigma(3) = 5$ and $\sigma(5) = 3$. Then we see that we
have another cycle, disjoint with the previous cycle, and we write this one as

\[
\begin{pmatrix}
3 & 5
\end{pmatrix}.
\]

To write the entire permutation, we then can then express
$\sigma$ as 

\[
\sigma =  \begin{pmatrix}
1 & 2 & 4
\end{pmatrix}
\begin{pmatrix}
3 & 5
\end{pmatrix}
\]

which gives us all the information we need to know on how
$\sigma$ rearranges the elements of $X$. Such a representation
of a permutation is called a **disjoint cycle decomposition**. 
It will turn out that
we can actually express *every* permutation $\sigma \in
S_n$ in a product of disjoint cycles.
</span>
**Remark.**
In general, 1-cycles are omitted in the representation of a
disjoint cycle decomposition. Thus if we have a permutation
$\sigma \in S_3$ such that $\sigma(1) = 2$, $\sigma(2) = 1$ and
$\sigma(3) = 3$, then we would write this as 

\[
\sigma = \begin{pmatrix}
1 & 2
\end{pmatrix}.
\]

Such a statement leads us to conclude that $\sigma(3) = 3$. And if
$\sigma \in S_5$, we would furthermore conclude that not only $\sigma(3) =
3$, but also $\sigma(4) = 4$ and $\sigma(5) = 5$.
\\
\\
**Nonuniqueness.**
One thing to note is that cycles are not unique. For example, we
could have written the cycle $\textcolor{ForestGreen}{\begin{pmatrix}
1 & 2 & 4
\end{pmatrix}} $ as $\textcolor{OrangeRed}{\begin{pmatrix}
2 & 4 & 1
\end{pmatrix}}$ or $\textcolor{Cyan}{\begin{pmatrix}
4 & 1 & 2
\end{pmatrix}}$, since the other expressions still capture the fact
that 1 is sent to 2, 2 is sent to 4, and 4 is sent to 1. 

<img src="../../../png/algebra/chapter_1/tikz_code_2_0.png" width="99%" style="display: block; margin-left: auto; margin-right: auto;"/>
Note that the colors correspond to where the cycle starts. Clearly
in the diagram, there are three ways to start the cycle, and hence
why there are three nonunique representations for the cycle. 
\\
More
generally, for any cycle $            \begin{pmatrix}
i_1 & i_2 & \cdots & i_n
\end{pmatrix}$ we have that 

\[
\begin{pmatrix}
i_1 & i_2 & \cdots & i_n
\end{pmatrix}
= \begin{pmatrix}
i_2 & i_3 & \cdots & i_n & i_1
\end{pmatrix}
= 
\cdots 
= 
\begin{pmatrix}
i_n & i_1 & \cdots & i_{n-1}
\end{pmatrix}.
\]









<script src="../../mathjax_helper.js"></script>