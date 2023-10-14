<style>
.md-content {
    max-width: 80em;
}
</style>
#2.2. Ring homomorphisms.
After one understand the fundamentals of group theory, they go
on to construct maps between different groups. This is the
same strategy we'll follow here, since we can definitely
define **ring homomorphisms** between rings.

The ring homomorphisms are also useful since they can help us
deduce when two rings $R$ and $S$ are the "same," a concept
which evolves into the concept of isomorphisms.


<span style="display:block" class="definition">
Let $R$ and $S$ be rings, and $f:R \to S$. We define $f$
to be a **ring homomorphism** if it preserves
addition and multiplication; that is, if

\[
f(a + b) = f(a) + f(b) \hspace{0.2cm}\text{ and }\hspace{0.2cm} 
f(ab) = f(a)f(b) 
\]

for all $a, b \in R$. If $f$ is a bijection, then we say
that $f$ is a **ring isomorphism**.
</span>

\textcolor{NavyBlue}{Note that a ring homomorphism is simply a
group homomorphism, with the extra condition of which preserves multiplication of the
ring elements.} Therefore the following proposition, which
hold for group homomorphisms, holds for ring homomorphisms
too. 

<span style="display:block" class="proposition">
Let $R$ and $S$ be rings and $f: R \to S$ a ring homomorphism.
Then 

* [1.] if $0_R \in R$ and $0_S \in S$ are zero
elements, then $f(0_R) = 0_S$. 



* [2.] if $f(-a) = -f(a)$ for all $a \in R$


* [3.] $f(a_1a_2\cdots a_n) = f(a_1)f(a_2)\cdots
f(a_n)$ for all $a_1, a_2, \dots a_n \in R$


* [4.] $f(a_1 + a_2 + \cdots + a_n) = f(a_1) +
f(a_2) + \cdots + f(a_n)$ for all $a_1, a_2, \dots a_n \in R$.



</span>


<span style="display:block" class="proof">
Observe that 

\begin{align*}
\phi(r) + \phi(-r) = \phi(r + (-r))\\
= \phi(r)\\
= 0\\
= \phi(r) +[-\phi(r)].
\end{align*}

Since $(R, +)$ is a group, subtract $\phi(r)$.
</span>


\textcolor{red}{Note that it is not necessarily true that
$f(1_R) = 1_S$.} In group theory, it was always guaranteed
that we could map the identity element from one group to
another. In our case, that's still true: $f(0_R) = 0_S$. Group
identity of $(R,+)$ is still mapped to the identity of $(S, +)$. But
this is mapping the *additive* identity of $R$ to the
*additive* identity of $S$. 

**What we're saying is that
*multiplicative** identities may not always be mapped to each
other. *

Now since we can't always guarantee that
$f(1_R) = 1_S$, \textcolor{red}{we also can't guarantee that $f(a^{-1}) =
f(a)^{-1}$ for some invertible $a \in R$.}
However, there is a clear cut case for when these things do
happen.     
\\


<span style="display:block" class="proposition">
Let $R$ and $S$ be rings and $\phi:R \to S$ a nonzero ring
homomorphism. Denote $1_R \in R$ and $1_S \in S$ to be the
respective multiplicative identities. Then 

* [1.] If $\phi(1_R) \ne 1_S$ then $\phi(1_R)$ is a
zero divisors of $S$. 



* [2.] If $S$ is an integral domain then $\phi(1_R)
= 1_S$. 



* [3.] If $\phi(1_R) = 1_S$ and $u \in R$ is a unit then
$\phi(u)$ is a unit in $S$.
In other words, $\phi(R^*) \subset S^*$



* [4.] If $\phi(1_R) = 1_S$ and if $u \in R$ has an
inverse $u^{-1} \in R$ then $\phi(u^{-1}) = \phi(u)^{-1}$.



</span>

\textcolor{MidnightBlue}{An immediately corollary is this:
$\phi: R \to S$ is a not nonzero ring homomorphism if and only if
$\phi(1_R) \ne 0_S$. Furthermore, 
If
$S$ is an integral domain then $\phi(R^*) \subset S^*$ for any
homomorphism $\phi: R \to S$.
}


<span style="display:block" class="proof">

* [1.] Suppose $\phi(1_R) \ne 1_S$. Since $1_R1_R =
1_R$, we know that

\begin{align*}
\phi(1_R1_R) - \phi(1_R) = 0_S
\implies &
\phi(1_R)\phi(1_R) - \phi(1_R) = 0_S\\
\implies & \big(\phi(1_R) - 1_S\big)\phi(1_R) = 0_S.
\end{align*}


Since $\phi(1_R) \ne 1_S$, either $\phi(1_S) = 0$ or it is a zero
divisor of $S$. 

Suppose $\phi(1_R) = 0_S$ and let $a \in R$. Then

\begin{align*}
\phi(a) = \phi(1_Ra) 
&= \phi(1_R)\phi(a)\\
&= 0_S\phi(a)\\
&= 0_S.
\end{align*}

Thus we see that $\phi$ send every element of $R$ to $0_S$.
However, this cannot be the case since we supposed that $\phi$ is a
nonzero homomorphism. 
Therefore $\phi(1_R)\ne 0$, leaving us with no
choice but to conclude that $\phi(1_R)$ is a zero divisor in $S$
as desired.   



* [2.]
Suppose $S$ is an integral domain, and that $\phi(1_R)
\ne 1_S$ for the sake of contradiction. Then observe
for any $a \in R$

\begin{align*}
\phi(1_R a) - \phi(a) = 0_S \implies \phi(1_R)\phi(a) - \phi(a) = 0_S
\implies (\phi(1_R) - 1_S)\phi(a) = 0_S.
\end{align*}


Since $\phi(1_R) \ne 1_S$, and $\phi$ is a nonzero homomorphism,
this implies that $\phi(a)$ and $\phi(1_R) - 1_S$ are zero
divisors in $S$ for at least one $a \in R$. However, this is a
contradiction since $S$ is an integral domain and hence has no
zero divisors. Thus by contradiction $\phi(1_R) =
1_S$.



* [3.] Suppose $\phi(1_R) = 1_S$ and 
let $u$ be a unit in $R$. Then $uv =
1_R$ for some $v \in R$. So 

\[
\phi(uv) = \phi(1_R) = 1_S \implies \phi(u)\phi(v) = 1_S.        
\]

Therefore, $\phi(u)$ is a unit in $S$. Next, since $uu^{-1} =
1_R$,

\[
\phi(1_R) = 1_S \implies \phi(uu^{-1}) = 1_S \implies 
\phi(u)\phi(u^{-1}) = 1_S \implies \phi(u)^{-1} = \phi(u^{-1})
\]

as desired.



* [4.] Suppose $\phi(1_R) = 1_S$ and that $u \in R$ has
some inverse $u^{-1} \in R$.
Since $uu^{-1} = 1_R$,

\[
\phi(1_R) = 1_S \implies \phi(uu^{-1}) = 1_S \implies 
\phi(u)\phi(u^{-1}) = 1_S \implies \phi(u)^{-1} = \phi(u^{-1})
\]

as desired.




</span>



\noindent**Examples.**\\
Let $n \in \ZZ$, and define the function
$f: \ZZ \to \ZZ$ as 

\[
f(m) = nm.
\]

Then this is a homomorphism if and only if $n = 0$ or 1.
Suppose otherwise. Then observe that the second condition of
the definition of a ring homomorphism specifies that 

\begin{align*}
f(ab) = f(a)f(b) \implies nab &= nanb \\
& = n^2ab.
\end{align*}

This is only true if $n = 0$ or 1, which is our contradiction.

Instead, we can construct the following function to form a
homomorphism between $\ZZ$ and $\ZZ/n\ZZ$, where $n$ is a
positive integer. Let $f: \ZZ \to \ZZ/n\ZZ$ such that 

\[
f(m) = [m]
\]

where $[m] = \{k \in \ZZ \mid k = m \mbox{ mod } n\}$.
\\

Suppose we construct a homomorphism $\phi: \mathbb{R}[x] \to
S$. (Recall that $\RR[x]$ is the set of finite polynomials
with coefficients in $\RR$). Define $\phi$ as  

\[
\phi(p(x)) = p(i).
\]

First, observe that this is surjective, since for any $a + bi
\in \mathbb{C}$ we can send $a + bx \in \RR$ to this element
via $\phi$. Therefore $\im(\phi) = \mathbb{C}$. 

Let us now describe $\ker(\phi)$. First suppose that $p(i) =
0$ for some $p(x) \in \RR[x]$. At this point, we know that $p(x)$ must be at least a
second degree or greater polynomial. Therefore we can express $p(x)$ 
as 

\[
p(x) = q(x)(x^2 + 1) + bx + a
\]

for some $q(x) \in \RR[x]$. Then 

\begin{align*}
p(i) &= q(i)(i^2 + 1) + bi + a \\
&= a + bi
\end{align*}

but this implies that $a + bi = 0 \implies a = b = 0$.
Therefore, $p(i) = 0$ if and only if $p(x) = q(x)(x^2 + 1)$
some $q(x) \in \RR[x]$. In other words, 

\[
\ker(\phi) = \{p(x) \in \RR[x] \mid (x^2 + 1)\big|p(x)\}.
\]

\\
\\
As in group theory, we have the following theorem regarding
isomorphisms. We won't prove this again. 



<span style="display:block" class="theorem">
Let $R$ and $S$ be rings. A ring homomorphism $f: R \to S$ is an
isomorphism if and only if there exists a h             omomorphism $g:
S \to R$ such that $f \circ g$ is the identity map on
$R$ and $g \circ f$ is the identity map on
$S$.
</span>

With the ring homomorphism defined, we again have $\ker(f)$
and $\im(f)$ as valid and important concepts.

<span style="display:block" class="definition">
Let $R$ and $S$ be rings and $f: R \to S$ a ring
homomorphism. Then we define 

\[
\ker(f) = \{a \in R \mid f(a) = 0\}  
\]

and 

\[
\im(f) = \{f(a) \mid a \in R\}.
\]

</span>


<span style="display:block" class="proposition">
Suppose $f: R \to S$ is a ring homomorphism. Then 

* [1.] The kernal $\ker(f)$ is a subring of $R$.


* [2.] The image $\im(f)$ is a subring of $S$.  



</span>
Caveat: Recall that "subrings" are rings that might not
possibly contain $1$, the multiplcative identity.


<span style="display:block" class="proof">

* [1.] We can show this using the Subring Criterion.
As we stated before, $f(0) = 0$. Hence $0 \in
\ker(f)$ so that $\ker(f)$ is nonempty. 

To prove this, observe that 

\begin{align*}
f(0) + f(0) & = f(0 + 0)\\
& = (0)\\
& = f(0) + 0.
\end{align*}

Since $(R, +)$ is a group, we can subtract $f(0)$
from both sides to get $f(0) = 0$.

Next, we want to show that $r_1, r_2 \in \ker(f)
\implies r_1 - r_2 \in \ker(f)$. Since we showed that
$f(-r) = -f(r)$ for all $r \in R$, we know that 

\begin{align*}
f(r_1 - r_2) & = f(r_1) + f(-r_2)\\
& = f(r_1) - f(r_2)\\
& = 0 - 0\\
& = 0.
\end{align*}

Hence, we see that $r_1 - r_2 \in \ker(f)$. 

Now again suppose $r_1r_2 \in \ker(f)$. Then 

\begin{align*}
f(r_1r_2) & = f(r_1)f(r_2)\\
& = 0
\end{align*}

so that $r_1r_2 \in \ker(f)$. By the subring test, we
see that $\ker(f)$ is a subring of $R$. 



* [2.] We can similarly prove this via the Subring
Test. First, observe that $f(0) = 0$, so that $0 \in
\im(f)$. Hence, $\im(f)$ is nonempty. 

Next, suppose $s_1, s_2 \in \im(f)$. Then we want to
show that $s_1 - s_2 \in \im(f)$. Now 

\begin{align*}
s_1 - s_2 & = f(r_1) - f(r_2)\\
& =f(r_1 - r_2).\\
\end{align*}

This shows that $s_1 - s_2 \in \im(f)$. Finally, we'll
show that $s_1s_2 \in \im(f)$. Observe that 

\begin{align*}
s_1 \times s_2 = f(r_1)f(r_2) = f(r_1r_2).
\end{align*}

Hence we see that $s_1\times s_2 \in \im(f)$. Thus
$\im(f)$ is a subring of $R$. 



</span>
Finally, we end this section by noting that two important and
useful mathematical identites continue to hold in the context
of rings. We won't offer their
proofs though since they are a bit tedious. 


<span style="display:block" class="proposition">
Let $R$ be a ring and let $a_1, a_2, \dots, a_m$ and $b_1,
b_2, \dots, b_n$ be elements of $R$. Then 

\[
(a_1 + a_2 + \cdots + a_m)(b_1 + b_2 + \cdots + b_n) 
= \sum_{i = 1}^{m}\sum_{j = 1}^{n}a_ib_j
\]

</span>


<span style="display:block" class="proposition">[Binomial Theorem]
Let $R$ be a ring (with identity) and let $a, b \in R$
with $ab = ba$. Then for any $n \in \mathbb{N}$ 

\[
(a + b)^n = \sum_{k = 0}^{n} {n\choose k} a^kb_{n-k}.
\]

</span>






<script src="../../mathjax_helper.js"></script>