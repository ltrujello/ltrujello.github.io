<style>
.md-content {
    max-width: 80em;
}
</style>
#1.12. Fundamental Theorem of Finite Abelian Groups.

Due to Sylow's Theorem, it is now an easy task to classify groups
of small orders up to an isomorphism by hand. However, abelian
groups are even easier to understand. Abelian groups have a simple
enough structure that we can actually generalize the structure of
*every* ablian group with the following theorems.

First we begin with a lemma. 


<span style="display:block" class="lemma">
Let $G$ be a finite abelian group. Then $G$ is isomorphic to a
direct product of its Sylow $p$-subgroups.
</span>


<span style="display:block" class="proof">
Since $G$ is finite, suppose $|G| = p_1^{n_1}p_2^{n_2}\cdots
p_n^{n_k}$ where $p_i$ are distinct primes and  $n_i$ are positive
integers for $i = 1, 2, \dots, k$.

By Sylow's Theorem, there exist Sylow $p_i$-subgroups for each
$i = 1, 2, \dots, k$. Denote these subgroups as $H_i$ (and
hence $|H_i| = p_i^{n_i}$). Observe that $\mbox{gcd}(p_i^{n_i}, p_j^{n_j})
= 1$ for any $i \ne j$. Hence, no $H_i$ is a subgroup of any
other $H_j$ for $i \ne j$. 

\textcolor{Plum}{(Otherwise, Lagrange would tell us
that that's nonsense because the order of subgroup always
divides the order of the bigger group; and in this case, $\mbox{gcd}(p_i^{n_i}, p_j^{n_j})
= 1$.)}

We can equivalently state that $H_i \cap H_j = \{e\}$ for $i
\ne j$, where $e$ is the identity of $G$.

Now observe that (1) $H_i \normal G$ for all $i$ since $G$ is
abelian and (2) $H_i \cap H_j = \{e\}$ and (3) 

\[ 
|G| = |H_1|\cdot|H_2|\cdots|H_k|.
\]

Therefore, we can repeatedly apply Theorem
1.\ref{product_theorem} to conclude that 

\[
G \cong H_1 \times H_2 \times \cdots \times H_k.           
\]

So $G$ is a product of its Sylow subgroups.
</span>


<span style="display:block" class="lemma">
Let $G$ be an abelian group and $p$ a prime. Then if $G = p^n$ for some
positive integer $n$, then $G$ is isomorphic to a direct
product of its cyclic groups.
</span>


<span style="display:block" class="proof">
\textcolor{NavyBlue}{We'll proceed with strong induction.}
Consider our base case with $n = 1$. Then we see that $G = p$,
and by the corollary to Lagrange's Theorem we know that this is
cyclic. 

For the inductive case, suppose this statement holds up to
$p^n$. Let $G$ be a group such that $|G| = p^{n+1}$. Let $g$
be an nontrivial element of $G$, and consider the cyclic
subgroup $\left< g \right>$.





Now define $H$ as follows:

\[
H = (G\setminus\left< g\right>) \cup \{e\} = \{h \in G \mid h \ne g^i \text{ for } i = 1, 2, \dots, m-1\}.  
\]

We will show that this is a subgroup via the subgroup test. 
First
observe that $H$ is nonempty, since we supposed that $|g| \ne
k+1$. Therefore, let $h, h' \in H$. \textcolor{purple}{Suppose
for the sake of contradiction that $h^{-1} \not\in H$.} That
is, 

\[
h^{-1} = g^{j} 
\]

for some $j = 1, 2, \dots, m-1$. \textcolor{purple}{Then
$e  = hg^{j}$.} But since the order of $g$ is $m$, we see that
this implies that $h = g^{m - j} \implies h \in H$. This is
our contradiction so $h^{-1} \in H$. 

Since $h^{-1} \in H$, we see that $h^{-1} \ne g^i$ for any $i
= 1, 2, \dots, m-1$. Since $h' \in H$ we see that 

\[
h'h^{-1} \ne g^{i} \text{ for any } i = 1, 2, \dots, m-1.
\]

Thus $h'h^{-1} \in H$, and by the subgroup test we see that
$H$ is in fact a subgroup of $G$. 

\textcolor{NavyBlue}{The result follows immediately after this, but we will elaborate on why. }

Note that $|\left< g\right>| = m \ne 0$ and $H \cup \left<g \right>
= G$. Therefore, we see that $|H| < |G|$. Since $H$ is a
subgroup of $G$, we know by Lagrange's Theorem that $|H|$
divides $|G| = p^{k+1}$. Hence, $|H| = p^j$ for some $j <
k+1$.

By construction, we see that (1) $\left< g \right> \cap H =
\{e\}$. Therefore 

\[
|H \cdot K| =   
\]


By our inductive hypothesis, we know that $H$ is isomorphic to
a direct product of cyclic groups. 

</span>


<span style="display:block" class="theorem">[ (Fundamental Theorem of Finite Abelian Groups)]
Let $G$ be a finite group. Then $G$ is a direct product of
cyclic groups. (Furthermore, these cyclic groups are Sylow $p$-groups.)
</span>


<span style="display:block" class="proof">
The result follows immediately from the previous two lemmas. 

Note that any finite ablein group $G$ is
isomorphic to a direct product of its Sylow $p$-groups by Lemma \ref{fund_ab_lemma_1}. 
However, each Sylow $p$-group is isomorphic to a product of
cyclic groups by Lemma \ref{fund_ab_lemma_2}. Therefore, we have that $G$ itself is
isomorphic to a product of cyclic groups. 
</span>

The Fundamental Theorem of Finite Abelian Groups is
analagous to the fundamental theorem of arithmetic (hence the
name). While the fundamental theorem of arithmetic allows us to
completely factorize integers, the fundamental theorem of finite
abelian groups allows us to factorize finite abelian groups.
\\
**Example.**
\\
Suppose we have an abelian group $G$ of order 16. Then, up to an
isomophism, $G$ is isomorphic to one of the following:

\begin{gather*}
\ZZ/16\ZZ\\
\ZZ/8\ZZ \times \ZZ2\ZZ\\
\ZZ4\ZZ \times \ZZ4\ZZ\\
\ZZ4\ZZ \times \ZZ2\ZZ \times \ZZ2\ZZ\\
\ZZ2\ZZ \times \ZZ2\ZZ \times \ZZ2\ZZ \times \ZZ2\ZZ
\end{gather*}

\\
**Example.**
\\
Observe that $9000= 9\cdot 5^3 \cdot 2^3$. We know that all abelian groups of order 9000 are going to
be direct products of cyclic subgroups. In this case, we can
represent the isomorphism with $\mathbb{Z}/m\mathbb{Z}$ groups.
Now because of the size of
$G$, we know that there are Sylow 9-, 5- and 2-subgroups of $G$. Thus
we can view $G$ as a
product of $\mathbb{Z}/n\mathbb{Z}$ groups.\\
\\
For the sake of
notation, we'll write that $\mathbb{Z}/n\mathbb{Z} =
\mathbb{Z}_n$. We can then lists the groups as 
\setcounter{equation}{0}

\begin{gather}
\ZZ_9 \times \ZZ_{5^3} \times \ZZ_{2^3}\\
\ZZ_9 \times \ZZ_{5^3} \times (\ZZ_{2^2} \times \ZZ_2)\\
\ZZ_9 \times \ZZ_{5^3} \times (\ZZ_{2} \times \ZZ_2 \times \ZZ_2)\\
\ZZ_9 \times (\ZZ_{5^2} \times \ZZ_5) \times \ZZ_{2^3} \\
\ZZ_9 \times (\ZZ_{5^2} \times \ZZ_5) \times (\ZZ_{2^2} \times \ZZ_2)\\
\ZZ_9 \times (\ZZ_{5^2} \times \ZZ_5) \times (\ZZ_{2} \times \ZZ_2 \times \ZZ_2)\\
\ZZ_9 \times (\ZZ_{5} \times \ZZ_5 \times \ZZ_5) \times \ZZ_{2^3} \\
\ZZ_9 \times (\ZZ_{5} \times \ZZ_5 \times \ZZ_5) \times (\ZZ_{2^2} \times \ZZ_2)\\
\ZZ_9 \times (\ZZ_{5} \times \ZZ_5 \times \ZZ_5) \times (\ZZ_{2} \times \ZZ_2 \times \ZZ_2)\\
(\ZZ_3 \times \ZZ_3) \times \ZZ_{5^3} \times \ZZ_{2^3}
\end{gather}


\begin{gather}
(\ZZ_3 \times \ZZ_3) \times \ZZ_{5^3} \times (\ZZ_{2^2} \times \ZZ_2)\\
(\ZZ_3 \times \ZZ_3) \times \ZZ_{5^3} \times (\ZZ_{2} \times \ZZ_2 \times \ZZ_2)\\
(\ZZ_3 \times \ZZ_3) \times (\ZZ_{5^2} \times \ZZ_5) \times \ZZ_{2^3} \\
(\ZZ_3 \times \ZZ_3) \times (\ZZ_{5^2} \times \ZZ_5) \times (\ZZ_{2^2} \times \ZZ_2)\\
(\ZZ_3 \times \ZZ_3) \times (\ZZ_{5^2} \times \ZZ_5) \times (\ZZ_{2} \times \ZZ_2 \times \ZZ_2)\\
(\ZZ_3 \times \ZZ_3) \times (\ZZ_{5} \times \ZZ_5 \times \ZZ_5) \times \ZZ_{2^3} \\
(\ZZ_3 \times \ZZ_3) \times (\ZZ_{5} \times \ZZ_5 \times \ZZ_5) \times (\ZZ_{2^2} \times \ZZ_2)\\
(\ZZ_3 \times \ZZ_3) \times (\ZZ_{5} \times \ZZ_5 \times \ZZ_5) \times (\ZZ_{2} \times \ZZ_2 \times \ZZ_2)
\end{gather}

(It's a christmas tree!) Recall the fact that
$\mathbb{Z}/mn\mathbb{N} \cong \mathbb{Z}/n\mathbb{Z} \times
\mathbb{Z}/m\mathbb{Z}$
iff $\mbox{gcd}(m, n) = 1$. Thus we see that 

* [1.] $\ZZ_9 \not\cong \ZZ_3 \times \ZZ_3$



* [2.] $\ZZ_{5^3} \not\cong \ZZ_5 \times \ZZ_{5^2}$ and
$\not\cong \ZZ_5 \times \ZZ_5 \times \ZZ_5$



* [3.] $\ZZ_{2^3} \not\cong \ZZ_2 \times \ZZ_{2^2}$ and
$\not\cong \ZZ_2 \times \ZZ_2 \times \ZZ_2$.



Therefore, we see that none of the groups (1) - (18) are
isomorphic to each other, so this exhaustive list of abelian
groups of order 9000 up to isomorphism is complete.
\\
\\
It turns out that our fundamental theorem for finite abelian
groups can actually be strengthened. This strengthened version
isn't that useful, since it is sufficiently useful to know that
every finite abelian group is a product of cyclic groups.
Nevertheless its proof is fun. 


<span style="display:block" class="theorem">
Let $G$ be a finite abelian group. Then there exist integers
$a_1, a_2, \dots, a_k$ such that 

\[
G \cong \ZZ/a_1\ZZ \times \ZZ/a_2\ZZ \times \dots \ZZ/a_k\ZZ
\]

where $a_i \mid a_{i+1}$.
</span>


<span style="display:block" class="proof">

Let $G$ be a finite abelian group and suppose $|G| =
p_1^{k_1}p_2^{k_2} \cdots p_n^{k_n}$. Since $G$ is abelian, 
we know by Lemma
\ref{fund_ab_lemma_1} that it is isomorphic to a 
product of Sylow subgroups. Therefore, we see that 

\[
G \cong H_1 \times H_2 \times \cdots \times H_n  
\]

where for some $H_1, H_2, \dots H_n$ Sylow subgroups, and 
$|H_i| = p_i^{k_i}$. However, observe that for each $i \le n$,  

\[
H_i \cong \underbrace{**(**\mathbb{Z}/p_i\mathbb{Z**)**} \times **(**\mathbb{Z}/p_i\mathbb{Z}**)** \times \cdots \times **(**\mathbb{Z}/p_i\mathbb{Z}**)**.}_{k_i\text{-many times}}
\]

Substituting for each $H_i$, we then have that 

\begin{align*}
G \cong \overbrace{**(**\mathbb{Z}/p_1\mathbb{Z**)**} \times **(**\mathbb{Z}/p_1\mathbb{Z}**)** \times \cdots \times **(**\mathbb{Z}/p_1\mathbb{Z}**)**}^{k_1\text{-many times}} \times 
\overbrace{**(**\mathbb{Z}/p_2\mathbb{Z**)**} \times **(**\mathbb{Z}/p_2\mathbb{Z}**)** \times \cdots \times **(**\mathbb{Z}/p_2\mathbb{Z}**)**}^{k_2\text{-many times}}
\times\\ 
\cdots \times 
\underbrace{**(**\mathbb{Z}/p_n\mathbb{Z**)**} \times **(**\mathbb{Z}/p_n\mathbb{Z}**)** \times \cdots \times **(**\mathbb{Z}/p_n\mathbb{Z}**)**}_{k_n\text{-many times}}.
\end{align*}

Therefore, we can rewrite $G$ as 

\begin{align*}
G \cong  **(**\mathbb{Z}/p_1\mathbb{Z**)**} \times **(**\mathbb{Z}/p_1\mathbb{Z}**)** \times \cdots \times \Big(**(**\mathbb{Z}/p_1\mathbb{Z}**)** \times 
**(**\mathbb{Z}/p_2\mathbb{Z**)**} \times **(**\mathbb{Z}/p_2\mathbb{Z}**)** \times \cdots \Big)\\
\times \Big( **(**\mathbb{Z}/p_2\mathbb{Z}**)** \times **(**\mathbb{Z}/p_3\mathbb{Z}**)**
\times **(**\mathbb{Z}/p_3\mathbb{Z}**)**
\times \cdots \Big) \\
\times \cdots \Big(**(**\ZZ/p_{n-2}\ZZ**)** \times **(**\ZZ/p_{n-1}\ZZ**)** \times \cdots \times **(**\ZZ/p_{n-1}\ZZ**)** \Big)\\ 
\times \Big(**(**\ZZ/p_{n-1}\ZZ**)** \times
**(**\mathbb{Z}/p_n\mathbb{Z**)**} \times **(**\mathbb{Z}/p_n\mathbb{Z}**)** \times \cdots \times **(**\mathbb{Z}/p_n\mathbb{Z}**)** \Big).
\end{align*}

That is, we can factor it into a product where the $i$-th factor
includes one $\ZZ/p_i\ZZ$ factor and $k_{i+1}-1$ many factors of $\ZZ/p_{i+1}\ZZ$.

Let us make the following observation. By Theorem
1.\ref{zmod_iso_thm} we know that 
\setcounter{equation}{0}

\begin{align}
\ZZ/hm\ZZ \cong \ZZ/h\ZZ \times \ZZ/m\ZZ       
\end{align}

since $\mbox{gcd}(h, m) = 1$.
Thus we can collapse the products (in the last equation of $G$) 
back together to observe that 

\begin{align*}
G \cong (\ZZ/p_1^{k_1 - 1}\ZZ)
\times (\ZZ/p_1p_2^{k_2-1}\ZZ) \times \cdots
\times (\ZZ/p_{n-1}p_n^{k_n}\ZZ)
\end{align*}

since by repeated application of equation (1), 

\[
\ZZ/p_ip_{i+1}^{k_{i+1} - 1} \cong \ZZ/p_i\ZZ\times \overbrace{\ZZ/p_{i + 1}\ZZ \times \cdots \times \ZZ/p_{i + 1}\ZZ}^\text{$(k_{i+1} - 1)-$many times}        
\]

for $1 < i < n -2$, and 

\[
\ZZ/p_{n-1}p_n^{k_n}\ZZ \cong **(**\ZZ/p_{n-1}\ZZ**)** \times
\overbrace{**(**\mathbb{Z}/p_n\mathbb{Z**)**} \times **(**\mathbb{Z}/p_n\mathbb{Z}**)** \times \cdots \times **(**\mathbb{Z}/p_n\mathbb{Z}**)**}^\text{$k_n-$many times}.
\]

Since we have that 

\[
G \cong (\ZZ/p_1^{k_1 - 1}\ZZ)
\times (\ZZ/p_1p_2^{k_2-1}\ZZ) \times \cdots
\times (\ZZ/p_{n-1}p_n^{k_n}\ZZ)
\]

if we let $a_1 = p_1^{k_1 - 1}$ and 
$a_i = p_{i-1}p_{i}^{k_{i} - 1}$ for $1 < i < n$ and $a_k =
p_{n-1}p_n^{k_n}$, then we see that 

\[
G \cong  \mathbb{Z}/a_1\mathbb{Z} \times \mathbb{Z}/a_2\mathbb{Z} \times \cdots \mathbb{Z} / a_k \mathbb{Z} 
\]

where $a_i \big| a_{i + 1}$ for $0 < i < n$, as desired.
</span>





<script src="../../mathjax_helper.js"></script>