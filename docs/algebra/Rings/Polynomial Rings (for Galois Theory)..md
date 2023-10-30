<style>
.md-content {
    max-width: 80em;
}
</style>
#2.8. Polynomial Rings (for Galois Theory).


<span style="display:block" class="definition">
Say $(R, +, \cdot)$ is a ring with identity $1 \ne 0$. We will
always assume $R$ is of this form. 

* **1.** If $x$  is indeterminate (i.e. a variable) a
**polynomial** in $x$ is a formal sum 

$$
a_dx^d + \cdots + a_1x + a_0.
$$

with $a_k \in R$.
* **2.** We say the **degree** $\deg(p) = d$ if $a_d$.
Otherwise, set  $\deg(p) = -\infty$ if $a_d = \cdots = a_0
= 0$.
* **3.** Let $R[x]$ be the collectionn of all such
$p(x)$. This is the **polynomial ring** over $R$.
* **4.** More generally, say $\{x_1, x_2, \dots, x_n\}$
is a collection of $n$ indeterminates. Inductively define
$R[x_1, x_2, \dots, x_n] = S[x_n]$ where $S = R[x_1, x_2,
\dots, x_{n-1}]$. A typical element is in the form 

$$
p(x_1, \cdots, x_n) = \sum_{i_1 =  1}^{d_1}\cdots\sum_{i_n = 1}^{d_n}a(i_1, \cdots, i_n)x_1^{i_1}\cdots x_n^{i_n}.
$$
 
We then say that this $p(x_1, \dots, x_n)$ is a polynomial
in $n$ variables.
* **5.** Assuming that the highest coefficient
\    $a(d_1, \dots, d_n) \ne 0$, we then say the degree is

$$
\deg(p) = \max\{i_1  + \cdots + i_n\} \text{ (fix here)}.                
$$

Otherwise, if all terms are zero, i.e., if all $a(i_1,
\dots, i_n) = 0$, then we set $\deg(p) = -\infty$.

</span>

**Example.**
Consider the polynomial ring $R[x, y]$ as the polynomial ring in
$n = 2$ variables. Let $p(x, y) = 1 + x^3 + y^3$. Then $\deg(p) =
3$. 


<span style="display:block" class="proposition">
$(R[x_1, \cdots, x_n], + , \cdot)$ is a ring with $1 \ne 0$.
</span>


<span style="display:block" class="proof">
We'll showt this by induction:

\begin{align*}
P(n) = "(R[x_1, \dots, x_n]) \text{ is a ring with } 1 \ne 0".
\end{align*}

We have already shown the base case $P(1)$ is true. Assume
that $P(n)$ is true for some $n_0$. We show $P(n)$ is true for
$n = n_0 + 1$.

By definition, $R[x_1, \dots, x_n] = S[x_n]$ where $S = R[x_1,
\dots, x_{n_0}]$. By our inductive hypothesis, we have that
\ $(S, +, \cdot )$ is a ring with $1 \ne 0$. Hence by $P(1)$
we have that $(S[x_n], \cdot, +)$ is also a ring with $1 \ne
0$. Hence $P(n_0 +  1)$ is true.
</span>


<span style="display:block" class="proposition">
Assume $R$ is an integral domain. 

* **1.** Suppose $p, q \in R[x_1, \cdots x_n]$. Then 

$$
\deg(p \cdot q) = \deg(p) + \deg(q).   
$$
* **2.** $R[x_1, \dots, x_n]$ is also an integral
domain.
* **3.** The units of $R[x_1, \dots, x_n]$ is
$R^{\times}$.

</span>


<span style="display:block" class="proof">

* **1.** If either $p = 0$ or $q = 0$, then $p \cdot q =
0$. Then observe that 

$$
\deg(p \cdot q) = \deg(p) + \deg(q)
$$

doesn't make sense unless  we choose  to set $\deg(0) =
-\infty$. Hence we see that

$$
\deg(p \cdot q) = -\infty.   
$$

We could make it positive infinity, but we assume that it
is negative for subtle reasons later on. 

Now assume that $p, q \ne 0$. We write our polynomials 

$$
p = \sum_{i_1, \dots, i_n}^{d_1, d_2, \cdots, d_n}a(i_1, \dots, i_n)x_1^{i_1}\cdots x_n^{i_n}
\qquad 
q = \sum_{j_1, \dots, j_n}^{e_1, e_2, \cdots, e_n}a(j_1, \dots, j_n)x_1^{j_1}\cdots x_n^{j_n}
$$

where $\deg(p) = d_1 + d_2 + \cdots + d_n$ and $\deg(q) =
e_1 + e_2 + \cdots + e_n$. Then we see that 

$$
p \cdot q  
= 
\sum_{k_1, \dots, k_n}^{d_1 + e_1, d_2 + e_2, \cdots, d_n + e_n}c(i_1, \dots, i_n)x_1^{k_1}\cdots x_n^{k_n}  
$$

where 

$$
c(k_1, \dots, k_n) 
= 
\sum_{i_1+j_1 = k_1, \dots, i_n+j_n = k_n.} a(i_1,  \dots i_n)b(j_1, \dots, j_n).
$$

The leading term is  $c(d_1 + e_1, \dots, d_n + e_n) =
a(d_1, \dots, d_n )b(e_1, \dots, e_n)$. Hence, the
leading term is also nonzero, so the degree must be 

$$
d_1 + e_1 +  \dots + d_n + e_n = \deg(p) + \deg(q).  
$$

There are actually many ways to define the degree of a
polynomial in $R[x_1, \dots, x_n]$ when$n \ge 2$.
* **2., 3.** We can show both (2)  and (3) at the same
time via induction. Let 

$$
P(n) = "R[x_1, \dots, x_n] \text{ is an integral domain and } R[x_1, \dots, x_n]^{\times} = R^{\times}."
$$
 
We've done this in the one-variable case, so that $P(1)$
is true. We can invoke our inductive hypothesis to suppose
that $P(n_0)$ is true for some $n_0$. We next show that $n
= n_0 + 1$ is true. 

By definition, our ring $R[x_1, \dots, x_{n_0}] = S[x_n]$ for 

$$
S = R[x_1, \dots, x_{n_0}].
$$

By our inductive hypothesis, we know that (1)  $S$ is an
integral domain. Since $P(1)$ is true, we know that
$S[x_{n_0}]$ is an integral domain. By (2), we know that
$S^\times = R^\times$. By $P(1)$, we see that $S[x_n] =
R^\times$. This show that $P(n)$ is true for all $n$.

</span>


<span style="display:block" class="proposition">
Say $(R, +, \cdots)$ is a ring with $1 \ne 0$ and $I
\subsetneqq R$ is a ideal. Denote $S
= R[x_1, \dots, x_n]$.

* **1.** $J = I[x_1, \dots, x_n]$ is a proper ideal of $S$
* **2.** $S/J = (R/J)[x_1, \dots, x_n]$
* **3.** Moreover, say $R$ is commutative. If $I \subset
R$ is a prime ideal of $R$, then $J$ is a prime ideal of
$S$.

</span>


<span style="display:block" class="proof">

* **1., 2.** Denote $\overline{R} = R/I$ as a ring  with
identity $1 \ne 0$, which holds since  we are working with
a proper ideal. We know both $S$ and $\overline{R}[x_1,
\dots, x_n]$ is also a ring with $1 \ne 0$. 

Consider the following "reduction mod $I$" map:

$$
\phi: R[x_1,\dots, x_n] \to \overline{R}[x_1,\dots, x_n]
$$

where if $\displaystyle p = \sum_{i_1, \dots, i_n}^{d_1, d_2,
\cdots, d_n}a(i_1, \dots, i_n)x_1^{i_1}\cdots x_n^{i_n}$
then 

$$
\overline{p} = \sum_{i_1, \dots, i_n}^{d_1, d_2, \cdots, d_n}\overline{a(i_1, \dots, i_n)}x_1^{i_1}\cdots x_n^{i_n}
$$

where $\overline{a} = a + I$ in $R/I$. \\
**Claim:** This is a ring homomorphism. In fact, this
map is surjective. 
\\
Neither of these are difficult to show. 

Observe that 

$$
\ker(\phi) = \Big\{p = \sum_{i_1, \dots, i_n}^{d_1, d_2,
\cdots, d_n}a(i_1, \dots, i_n)x_1^{i_1}\cdots x_n^{i_n} mid a(i_1, \dots, i_n)\in I \Big\}
= I[x_1, \dots, x_n].
$$

The First Isomorphism Theorem for Rings states that 

\begin{align*}
S/J &\cong R[x_1, \dots, x_n]/\ker(\phi)\\
&\cong \im(\phi)\\
&= (R/I)[x_1, \dots, x_n].
\end{align*}

At this point, we've shown (2). Now observe that $J
\subsetneqq R$ since $(R/I)[x_1, \dots, x_n]$ is a ring
with identity $1 \ne 0$. This proves (1). To show (3), say
$R$ is a commutative ring and $I \subset R$ is a prime
ideal. Since $I$ is prime, we see that $(R/I)$ is an
integral domain. Hence we  see that $(R/I)[x_1, \dots,
x_n]$ is an integral domain. Since $S/J \cong (R/I)[x_1, \dots,
x_n]$, and  we know that $R[x_1, \dots, x_n]$ is also a
commutative ring, we see  that $J \subset S$ must be a
prime ideal as well.

</span>   





<script src="../../mathjax_helper.js"></script>