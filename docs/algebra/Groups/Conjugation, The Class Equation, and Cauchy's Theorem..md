<style>
.md-content {
    max-width: 80em;
}
</style>
#1.10. Conjugation, The Class Equation, and Cauchy's Theorem.
\textcolor{blue}{We now touch on a very deep example of a group action, known as
conjugation.} Let $G$ act on itself "by conjugation", which we define
as follows. Let $g, h \in G$. Then 

\[
g * h = ghg^{-1}
\]

is the group action of conjugation.
Let's show that this is a group action. 
\begin{description}
\item[Composition.] Let $g_1, g_2, h \in G$. Then observe that

\begin{align*}
g_1 * (g_2 * h) = g_1 * (g_2hg_2^{-1}) &= g_1g_2 h g_2^{-1}g_1^{-1} \\
& = (g_1g_2) h (g_1g_2)^{-1} \\
& = (g_1g_2) * h
\end{align*}

so that the first axiom of a group action is satisfied.
\item[Identity.]
Observe also that for $e \in G$, the identity of $G$, 

\[
e * h = ehe^{-1} = h.
\]

Therefore this is a group action.
\end{description}

We'll now show that this group action is very special and
important. Conjugation itself is important in math. In Linear
Algebra, two matrices which are similar (i.e., $A$ is similar to
$B$ if there exists $P$ such that $A = P^{-1}BP$) have \textbf{the same 
rank, determinants, trace, eigenvalues, and much more}. Basically,
they represent the same linear transformation, just in different
bases. To learn more about conjugation, we make a few definitions
with this group action.


<span style="display:block" class="definition">
Let $G$ be a group, and let $G$ act on itself by conjugation.
For any $h \in G$, **the orbit** of this group action

\begin{align*}
Gh & = \{g * h \mid g \in G\} \\
& = \{ghg^{-1} \mid g \in G\}
\end{align*}

is known as a **conjugacy class** of $G$.
</span>

\textcolor{purple}{Previously we discussed how orbits of a group action partition the
set $X$ which is being acted on. Since $G$ acts on itself in this
example, we see that \textbf{the conjugacy classes form a
partition of $G$!}}
\\
\\
**Remark.**
Recall the definition of a centralizer $G$ for a set $A \subset
G$: 

\begin{align*}
C_G(A) & = \{g \in G \mid gs = sg \text{ for all } s \in S\}\\
& = \{g \in G \mid gsg^{-1} = s  \text{ for all } s \in S\}.
\end{align*}

Therefore for a single point $x \in G$, $C_G(x) = \{g \in G \mid
gxg^{-1} = x \} = \{g \in G \mid g * x = x\}$, where in the last
equation we are speaking in terms of group actions. But note that
this last set is exactly the **stabilizer** of $G$ under this
group action. **\textcolor{NavyBlue**{Therefore, $C_G(x) = G_x$ for any $x \in G$ under
this group action.}}

Furthermore, let $x \in Z(G) = \{z \in G \mid z = gzg^{-1} \text{
for all } g \in G\},$ the center of $G$. Then we see that $Gx =
\{gxg^{-1} \mid g \in G\} = \{x\}$. **\textcolor{NavyBlue**{So for any $x \in Z(G)$, the
orbit is of size one. The sole element it contains is just $x$.}} (We can go even further: the conjugacy
classes of an abelian group are all of size one.)
\\
\\
Let's put all of these results together. In general, if $G$
acts on itself via conjugation, then we know its orbits, or
conjugacy classes, partition $G$. Moreover, let $R \subset X$ be a set of
orbit representatives (or conjugacy class representatives, if you
like). Then 

\[
|G| = \sum_{x \in R}|Gx|
\]

Recall that $|Gx| = 1$ if $x \in Z(G)$. Thus we can write this
further as 

\begin{align*}
|G| = 
\sum_{x \in Z(G)}|Gx| + \sum_{x \in R\setminus Z(G)} |Gx|
& = \sum_{x \in Z(G)}1 + \sum_{x \in R\setminus Z(G)} |Gx|\\
& = |Z(G)| + \sum_{x \in R\setminus Z(G)} |Gx|
\end{align*}

By the Orbit-Stabilizer theorem, we can write $|Gx| = |G|/|G_x|$.
Substituting this in, we get 

\begin{align*}
|G| &= |Z(G)| + \sum_{x \in R\setminus Z(G)} |G|/|G_x|
\end{align*}

and since $C_G(x) = G_x$,

\begin{align*}
|G| = |Z(G)| + \sum_{x \in R\setminus Z(G)} |G|/|C_G(x)|
\end{align*}

which is known as the **class equation.** This equation is
pretty badass, as it gives us a way to understand the cardinality
of a group. This equation is also useful in proofs, as we shall
see in the following examples. First, we begin with a lemma. 


<span style="display:block" class="lemma">
Let $G$ be a group. Then $C_G(x) = G$ if and only if $x \in Z(G)$.
</span>


<span style="display:block" class="proof">
Suppose $C_G(x) = G$. Then for all $g \in G$, $gx = xg$.
However, $Z(G)$ is the set of all $G$ which commutes with
every member of $G$, so $x \in Z(G)$. 
Now suppose $x \in Z(G)$. Then $gx = xg$ for all $g \in G$.
Therefore, $C_G(x) = \{g \in G \mid gx = xg\} = G$.
</span>


<span style="display:block" class="theorem">
Let $G$ be a group such that $|G| = p^n$ for some prime $p$
and $n \in \mathbb{N}$. Then $|Z(G)| > 1$. That is, $|Z(G)|
\in \{p, p^2, \dots, p^{n}\}$.
</span>
\textcolor{Purple}{Equivalently, this theorem says that $Z(G)$ is nontrivial.
Moreover, this implies that \textbf{there exists
non identity elements of $\mathbf{G}$ which commute with every
element of $\mathbf{G}$.}}


<span style="display:block" class="proof">
First observe that $Z(G)$ is a subgroup of $G$. Therefore, by
Lagrange's Theorem, we know that $|Z(G)|$ divides $G$. Thus
$|Z(G)| \in \{1, p, p^2, \dots, p^{n}\}$. Our goal is to show
that $|Z(G)|$ cannot equal 1.

\textcolor{NavyBlue}{For the sake of contradicton, suppose $|Z(G)| = 1$}. Then by
the previous lemma, we see that 
there is no nontrivial element $g$ of $G$ such $C_G(g) = G$.

Let $R$ be the set of conjugacy class representatives. 
Then $|G|/|C_G(r)| \in \{p, p^2, \cdots, p^n\}$ for $r \in
R\setminus Z(G)$ (since $|Z(G)| = 1$, $R\setminus Z(G)$ simply
removes $e$, the identtiy, from $R$).

\textcolor{red!40!purple!100}{Why can't $|G|/|C_G(r)| = 1$ for any $r \in
R\setminus Z(G)$? Well, because for such an $r$, $r \not\in
Z(G)$. Therefore $C_G(r) \ne G$, so $|G|/|C_G(r)| \ne 1$.}

Now by the class equation, we see that 

\[
\underbrace{ \hspace{.2cm}|G|\hspace{.2cm}   }_{\text{divisible by } p} \hspace{-.5cm} = |Z(G)| + \overbrace{\sum_{r \in R\setminus Z(G)} |G|/|C_G(r)|}^{\text{divisible by } p}
\]

since $|G|/|C_G(r)| \in \{p, p^2, \dots, p^n\}$ for all $r \in
R\setminus Z(G)$. \textcolor{NavyBlue}{Therefore we see that $|Z(G)|$ must be
divisible by $p$. But this is a contradiction since we said
$|Z(G)| = 1$}. Therefore, we see that $|Z(G)| \in \{p, p^2,
\dots, p^n\}$.

</span>

The above theorem can be used to prove the next theorem, whose
signifiance demonstrates the power of the class equation.
The theorem below is generally
proved by proving the above theorem first in the special case
for when $|G| = p^2$. But it will be helpful to other proofs
later on to consider the more general case as we presented it above.


<span style="display:block" class="theorem">
Let $G$ be a group, and suppose $|G| = p^2$ where $p \ge 2$ is
prime. Then $G$ is abelian. 
</span>


<span style="display:block" class="proof">
\textcolor{green!50!black}{By the previous theorem, we see
that $|Z(G)| \in \{p, p^2\}$.)} We'll proceed by considering two cases.

\begin{description}
\item[$\mathbf{|Z(G)| = p^2}$.] 
In this case $|G| = |Z(G)|$. Since we also have that
$Z(G)$ is a subgroup of $G$, we can conclude that $G =
Z(G)$. 
Therefore, $G$ is abelian. 

\item[$\mathbf{|Z(G)| = p}$.] 
Recall that $Z(G) \normal
G$ from Proposition \ref{normal_center}. Therefore, we can
speak of the quotient group $G/Z(G)$, which has size
$|G|/|Z(G)| = p^2/p = p$. By the corollary to Lagrange's
Theorem, this implies that $G/Z(G)$ is cyclic, since it
has prime order. Thus there
exists a $g \in G$ such that we can represent $G/Z(G)$ as 

\[
G/Z(G) = \{Z(G), Z(G)g, Z(G)g^2, \dots, Z(G)g^{p-1}\}.
\]

As we already know, cosets partition $G$. Therefore, let
$a, b \in G$, and suppose $a \in Z(G)g^i$ and $b \in
Z(G)g^j$. Then there exist $x, y \in Z(G)$ such that 
$a = xg^i$ and $b = yg^j$. Thus observe that 

\begin{align*}
ab = xg^i yg^j = xyg^ig^j = xyg^{i+j} = xyg^jg^i
= yg^jxg^i = ba
\end{align*}

where we used the commutavity of $x,y$ since $x, y \in
Z(G)$. Since $a, b$ were arbitrary members of $G$, this
proves that $G$ is abelian.
\end{description}
</span>

Thus we see that the class equation is useful in proving more
general facts about group theory. The class equation can also be
used to prove the following important theorem in group theory,
known as Cauchy's Theorem. 


<span style="display:block" class="theorem">[ (Cauchy's Theorem)]
Let $G$ be a finite group and $p \ge 2$ be a prime. If $p$ divides
the order of $G$, then $G$ has an element of order $p$. 
</span>

\noindent\textcolor{NavyBlue}{So consider a group $G$ with order $n$,
and suppose 

\[ n = p_1^{i_1}\cdot p_2^{i_2} \cdots
p_n^{i_n}
\]

is its prime factorization. Then there exist elements
$g_1, g_2, \dots, g_n$ such that $|g_i| = p_i$ for $i = 1, 2,
\dots, n$.}

Another way to visualize this as follows. Consider a group $G$
consisting of 10 elements. 

\[
\{e, \hspace{.1cm}g_1,\hspace{.1cm} g_2,\hspace{.1cm} g_3,\hspace{.1cm} g_4,\hspace{.1cm} g_5,\hspace{.1cm} g_6,\hspace{.1cm} g_7,\hspace{.1cm} g_8,\hspace{.1cm} g_{9}\}
\]

By Cauchy's theorem, there exists elements of order $2$ and $5$.
So suppose $g_1$ and $g_2$ are such elements, i.e., $g_1^2 = 3$
and $g_2^5 = e$. Then we can really rewrite this as 

\[
\{e,\hspace{.1cm} \textcolor{red}{g_1},\hspace{.1cm} \textcolor{blue}{g_2},\hspace{.1cm} \textcolor{blue}{g_2^2},\hspace{.1cm} \textcolor{blue}{g_2^3},\hspace{.1cm} \textcolor{blue}{g_2^4},\hspace{.1cm} g_6,\hspace{.1cm} g_7,\hspace{.1cm} g_8,\hspace{.1cm} g_9\}.   
\]

However, we know $\textcolor{red}{g_1}\textcolor{blue}{g_2}, \textcolor{red}{g_1}\textcolor{blue}{g_2^2}, \textcolor{red}{g_1}\textcolor{blue}{g_2^3}$ and $\textcolor{red}{g_1}\textcolor{blue}{g_2^4}$ are all in $G$. Thus
we can really write this as 

\[
\{e, \hspace{.1cm} \textcolor{red}{g_1},\hspace{.1cm} \textcolor{blue}{g_2},\hspace{.1cm} \textcolor{blue}{g_2^2},\hspace{.1cm} \textcolor{blue}{g_2^3},\hspace{.1cm} \textcolor{blue}{g_2^4}, \hspace{.1cm}\textcolor{red}{g_1}\textcolor{blue}{g_2},\hspace{.1cm} \textcolor{red}{g_1}\textcolor{blue}{g_2^2},\hspace{.1cm} \textcolor{red}{g_1}\textcolor{blue}{g_2^3},\hspace{.1cm} \textcolor{red}{g_1}\textcolor{blue}{g_2^4}\}.
\]

Thus we can understand the structure of every single group of
order 10. But this can be done for all finite groups!


<span style="display:block" class="proof">
\textcolor{Plum}{In this proof, we'll prove this in a very
clevery way by letting a subgroup of a permutation group act
on a special set $X$ (both of which we will define). This will then prove the existence of
elements of order $p$.}

Let $p$ be a prime which divides $|G|$. 
Define $H$ to be the cyclic subgroup of $S_p$ generated by
$(1\hspace{.1cm}2\hspace{.1cm}\cdots\hspace{.1cm}p)$. 

We can picture $H$ as the group 

\[
\{(1\hspace{.1cm}2\hspace{.1cm}\cdots\hspace{.1cm}p), (2\hspace{.1cm}3\hspace{.1cm}\cdots\hspace{.1cm}p, \hspace{.1cm}1), \cdots, (p\hspace{.1cm}1\hspace{.1cm}\cdots\hspace{.1cm}p-1)\}.
\]


Now let $H$ act on the set $X$ defined as 

\[
X = \{ (g_1, g_2, \dots, g_p) \mid g_1, g_2, \dots, g_p \in G \text{ and } g_1g_2\cdots g_p = e \}
\]

where the $\sigma \in H$ acts on $g \in X$ as 

\[
\sigma \cdot (g_1, g_2, \dots, g_p) = (g_{\sigma(1)}, g_{\sigma(2)}, \dots, g_{\sigma(p)}).
\]

This $H$ takes a $p$-tuple in $X$ and permutates the elements.
Since $H$ is generated by
$(1\hspace{.1cm}2\hspace{.1cm}\cdots\hspace{.1cm}p)$, it
"pushes" the elements $g_i$ in the tuple over to the right, and the elements
that are pushed out of the right end of the tuple are pushed back in on
the left side.

\textcolor{NavyBlue}{First we'll show that this is a group action.}
\begin{description}
\item[This is a Group Action.] 
Let $x \in X$ and $\sigma \in H$. If $x = (g_1, g_2,
\dots , g_p)$, observe that 

\[
\sigma * x = (g_{\sigma(1)}, g_{\sigma(2)}, \dots, g_{\sigma(p)}).
\]

Suppose $h(1) = n$. Then in general $h(i) = (i + n)
\mbox{ mod }p.$ Therefore, we see that 

\[
(g_{\sigma(1)}, g_{\sigma(2)}, \dots, g_{\sigma(p)}) 
= 
(g_{n}, g_{n+1}, \dots, g_p, g_1, \dots, g_{n-1}).
\]

However, observe that 

\[
g_1g_2\cdots g_p = g_1g_2 \cdots g_{n-1}g_n g_{n+1} \cdots g_p = e
\implies (g_1g_2 \cdots g_{n-1})(g_n g_{n+1} \cdots g_p) = e.
\]

Thus the elements $g_1g_2 \cdots g_{n-1}$ and $g_n
g_{n+1} \cdots g_p$ in $G$ are inverses of each other. But
know that if two group elements are inverses, either order
of their product returns $e$. Therefore 

\[
(g_ng_{n+1} \cdots g_{p})(g_1g_2 \cdots g_{n-1}) 
= g_ng_{n+1} \cdots g_{p}g_1g_2 \cdots g_{n-1} = e.
\]


We therefore see that
$(g_{n}, g_{n+1}, \dots, g_p,
g_1, \dots, g_{n-1}) = \sigma *x \in X$. 

Now we verify associativity. For any $\sigma_1,
\sigma_2 \in H$, we see that 

\begin{align*}
\sigma_1 * \sigma_2*x &= \sigma_1 * (g_{\sigma_2(1)}, g_{\sigma_2(2)}, \dots, g_{\sigma_2(p)})\\
&= (g_{\sigma_1(\sigma_2(1))}, g_{\sigma_1(\sigma_2(2))}, \dots, g_{\sigma_1(\sigma_2(p))})\\
&= (\sigma_1 \sigma_2) * (g_1, g_2, \dots, g_p).
\end{align*}

Thus $*$ is associative. Finally, if $\sigma$ is the
trivial element, 

\[
\sigma * x = (g_{\sigma(1)}, g_{\sigma(2)}, \cdots g_{\sigma(p)}) = (g_1, g_2, \dots, g_p) = x.
\]

Therefore this is a group action.
\end{description}
\textcolor{NavyBlue}{Now that we've shown that this is a group
action, we'll argue that the orbits are either of size 1 or
$p$.}

\begin{description}
\item[The Orbits.] 
For any $x \in X$ such that $x = (g_1, g_2, \dots, g_p)$,
we see that the orbit $Hx$ will simply be all of
the permutations of the $p$-tuple $(g_1, g_2, \dots,
g_p)$. Note however that there are only $p$ many ways to
rearrange this tuple, so that any orbit $Hx$ will be of
size $p$.

Of course, the exception to this is if $g_1 = g_2 = \dots = g_p$. In
this case, there are no other ways to reorganize the
tuple. Hence the orbit will have size 1.
\end{description}

\textcolor{NavyBlue}{Finally, we will show that there exists a
nontrivial orbit of size 1. This is equivalent to show that
there exists a nontrivial element of $G$ of order $p$, which
we'll eloaborate later.}

\begin{description}
\item[Orbit of Size 1.]
First let's count the elements of $X$. Observe that for
any $(g_1, g_2, \dots, g_p) \in
X$, the last element $g_p$ is always determined by the
first $p-1$ elements. This is because if we know the first
$p-1$ elements, then 

\[
g_p = (g_1g_2 \cdots g_{p-1})^{-1}
\]

in order for $g_1g_2\cdots g_p = e$. Since there are
$|G|^{p-1}$ many ways to pick the first $p-1$ elements in
any $p$-tuple of $X$, we see that $X = |G|^{p-1}$. 

Now by hypothesis, $p$ divides $|G|$. Therefore $p$
divides $|X|$ so we may write $|X| = np$ for some integer $n$.

Since the orbits of $X$ form a partition, the orbits
partition a set $np$ elements into orbits of size $1$ or
size $p$. We know one orbit of size 1 exists (namely, the
trivial orbit $He = \{(e, e, \dots, e)\}$), so there must
exist at least $p-1$ nontrivial other orbits of size 1. 

Let $Hx'$ be one of those orbits. Then for some $g \in G$
we have that $Hx = \{(g, g, \dots, g)\}$. However since
$Hx \subset G$, what we have prove is the existence of a
nontrivial
element $g \in G$ such that $gg\cdots g = g^p = e$, which
set out to show.
\end{description}
This completes the proof.
</span>
Cacuhy's Theorem is an incredibly useful tool one can use in
finite group theory. Here's an amazing and useful theorem who's
proof is eased via Cauchy's Theorem.


<span style="display:block" class="theorem">
Let $G$ be a group $p \ge 2$ a prime. If $|G| = p^n$ for some
$n \in \mathbb{N}$, then $G$ has a subgroup of order $p^k$ for
all $0 < k < n$.
</span>
Note we didn't write $ 0 \le k \le n$. We could have, but we already
know that there exists a subgroup of order $p^n$ (namely, $G$
itself) and that there exists a subgrou of order $p^0 = 1$
(namely, the trivial group).

<span style="display:block" class="proof">
\textcolor{NavyBlue}{To prove this, we'll use strong induction
on the statement. Specifically, we'll induct on the powers of
$n$.}

Let us induct on $n$ in the statement above. Then for $n = 1$,
there is no such $k < n$. Hence the statement is vacuously
true. 

Next suppose that the statement is true up to order $p^n$, 
and let $G$ be a group of order $p^{n+1}$. 

By Theorem 1.\ref{center_lemma}, we already note that $|Z(G)| >
1$ and hence is a multiple of $p$. By Cauchy's Theorem, we
then know that $Z(G)$ contains an element $g$ of order $p$.
Note that (1) $\left< g\right>$ is a subgroup of $Z(G)$ and
(2) $h\left< g \right> = \left< g \right>h$ for all $h \in G$
(since, by definition of the center, every element of $Z(G)$
commutes with elements of $G$). Therefore $\left< g \right>
\normal G$. 

Let $H = \left< g \right>$. Since we just showed $H \normal G$
we can appropriately discuss the quotient group $G/H$.

Observe that $|G/H| = |G|/|H| = p^{n+1}/p = p^n$. \textcolor{purple}{Thus by hypothesis, $G/H$ has a
subgroup of order $p^k$ for all $0 < k < n$. Denote these such
subgroups of $G/H$ as}

\[
\{N_1/H, N_2/H, \cdots N_{n-1}/H\}
\]

\textcolor{purple}{where $|N_k/H| = p^k$.}
Since $H \normal G$, we
know by the Fourth Isomorphism Theorem that every subgroup of
$G/H$ is of the form $N/H$ where $H \le N \le G$. Thus we see
that 

\[
H \le N_k \le G
\]

for all $0 < k < n$. But since $|N_k/H| = p^k$, and $|H| = p$,
we see that each such $N_k$ will now have order $p^{k+1}$.
Thus what we have shown is that $G$ itself contains subgroups
of order $k$ for all $1 < k < n+1$. The subgroup $H$ of order
$p$ is the final piece to this puzzle, and allows us to
confirm that $G$ has a subgroup of order $p^k$ for all $0 < k < n$.
By strong induction this holds for all $\mathbb{N}$,
which completes the proof.
</span>






<script src="../../mathjax_helper.js"></script>