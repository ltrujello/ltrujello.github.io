#3.2. Submodules, Quotient Modules and Isomorphism Theorems.


<span style="display:block" class="definition">
Let $M$ be a $R$-module. Then a set $N \subset M$ is said to
be a **submodule** of $M$ if $N$ is also a $R$-module. 
</span>

\textcolor{MidnightBlue}{What do we need for $N \subset M$ to be a submodule?}
For $N$ to be a submodule,

*  $N$ needs to be an nonnempty abelian group 


*  Axioms (a) - (d) in Definition 1.1 must be satisfied 


*  $N$ needs to be closed under multiplication of $R$. That
is, $\cdot : R \times M|_{N} \to N$, where $M|_N$ is $M$
restricted to $N$ (namely, just $N$).




However, since $N \subset M$, axioms (a) - (d) are already
satisfied for $N$. In addition, if $N$ is a nonempty subgroup of $M$ then
it is automatically abelian. Thus $N$ is a **$R$-submodule** of $M$ if $N$
is a subgroup of $M$ and $N$ is closed under multiplication of
elements from $R$. This leads us to the following submodule test. 


<span style="display:block" class="theorem">[(Submodule Test.)] Let $M$ be an $R$-module and $N
\subset M$ be nonempty. Then $N$ is an $R$-submodule of $M$ if and
only if $an_1 + bn_2 \in N$ for all $n_1, n_2 \in N$ and $a, b \in
R$. 
</span>


<span style="display:block" class="proof">
($\implies$) If $N$ is an $R$-submodule of $N$ then obviously $an_1 + bn_2
\in N$ for all $n_1, n_2 \in N$ and $a, b \in R$. 

($\impliedby$) Suppose $an_1 + bn_2 \in N$ for all $n_1, n_2
\in N$ and $a, b \in R$. 
First observe that $N$ is nonempty. Now
setting $a = 1$ and $b = -1$ we
see that $n_1 - n_2 \in N$ for all $n_1, n_2 \in N$, and thus
by the subgroup test we see that $N$ is a subgroup of $N$. 

Since $an_1 + bn_2 \in N$ for all $a, b \in R$ we see that $N$
is closed under multiplication of elements of $R$. 

Since $N$ is an abelian subgroup of $M$ and is closed under
multiplication of elements of $R$, we see that $N$ is an
$R$-submodule as desired. 
</span>

\noindent **Example.**\\
An immediate example we can create from our previous discussions
the fact that if $f: M \to N$ is an $R$-module homomorphism then 

* [1.] $\ker(f)$ is an $R$-submodule of $M$.


* [2.] $\im(f)$ is an $R$-submodule of $N$. 




As we saw in group and ring theory, arbitrary intersections of
subgroups or subrings resulted in subgroups and subrings. Thus the
following theorem should be of no surprise.


<span style="display:block" class="theorem">
Let $R$ be a ring and $M$ an $R$-module. If $\{N_\alpha\}_{\alpha \in \lambda}$ be a
set of $R$-submodules of $M$, then $N = \bigcap_{\alpha \in
\lambda}N_{\alpha}$ is a submodule of $M$. 
</span>


<span style="display:block" class="proof">
First observe that $N = \bigcap_{\alpha \in
\lambda}N_{\alpha}$ is nonempty, since $0 \in N_\alpha$ (the
identity) for all $\alpha \in \lambda$. Thus for any $n_1, n_2
\in N$ we know that $n_1, n_2 \in N_\alpha$ for all $\alpha
\in \lambda$. Since each such $N_\alpha$ is an $R$-submodule,
we know that $an_1 + bn_2 \in N_\alpha$ for all $\alpha \in 
\lambda$ for any $a, b \in R$. Hence, $an_1 + bn_2 \in N$ for
all $a, b \in R$, proving that $N$ is an $R$-submodule as
desired. 
</span>

Note that what ring $R$ is under discussion, we will
just state a $R$-submodule as simply a submodule.
\\
\\
**Quotient Modules.**
\\
\\
As we discovered quotient groups in group theory and quotient
rings in ring theory, it should again be no surprise that we can
formalize the concept of quotient modules. 

In group theory, a quotient group $G/H$ only made sense if the
group $H$
being quotiened out was **normal** to $G$. This
guaranteed that our desired group operation in the quotient group
worked and made sense as desired. In ring theory, a quotient ring
$R/I$ only made
sense if the ring $I$ being quotiened out was an **ideal** of
$R$. Since we wanted $R/I$ to be a ring, we needed not only
addition but multiplication to be well-defined, but
well-definedness only worked when $I$ was an ideal. 

In both cases, we couldn't quotient out just any subgroup or a
subring to get a quotient group or quotient ring. They had to be
special subsets (e.g. normal groups, ideals). 
However, in
module theory, it does happen to be the case that we can just
quotient out a submodule to get a quotient module. 

\textcolor{purple}{
To define a quotient module, we first consider an $R$-module $M$
and a submodule $N$ of $M$. To turn $R/N$ into an $R$-module, we
first turn this into an abelian group, which we can perfectly do
since $N$ is a subgroup of $M$, an abelian group, so $M/N$ makes
sense. A result from group theory tells us that if $M$ is abelian
then $M/N$ is abelain. 
\\
\indent Next, to turn this into an $R$-module we define scalar
multiplication as 

\[
r(m + N) = rm + N
\]

where $r \in R$, and multiplication of elements as

\[
(m + N)(m' + N) = mm' + N.   
\]

As always, when defining a quotient object we're
worried about the ability of our multiplication to preserve
equivalence of elements. This is usually where we run into trouble
in group theory or ring theory, in which case we modify the set
$N$ which we're quotienting out. In group theory, we'd turn $N$ into
normal group, and in ring theory we'd turn $N$ into an ideal. Here
we'll leave $N$ alone, since it works out in the end.
\\
\indent Thus 
suppose that

\[
m + N = m' + N
\]

that is, $m = m' + n$ for some $n \in N$. 
Then to check if our 
multiplicaton is well-defined, we observe that for $a \in R$

\[
am + N = a(m' + n) + N = am' + an + N
\]

and since $N$ is a submodule, it is closed under scalar
multiplication of elements of $R$. Hence, $an \in N$, so that 

\[
am' + an + N = am' + N.
\]

Thus we see that $am + N = am' + N$, so that our scalar
multiplication is well-defined.
}
This leads to the following definition.


<span style="display:block" class="definition">
Let $R$ be a ring and $M$ an $R$-module. If $N$ is a submodule
of $M$, then we defined $M/N$ to be the \textbf{quotient
$R$-module} of $M$ with respect to $N$. As we showed earlier,
this is in fact an $R$-module.
</span>

As before, it should be no surprise that the Noether Isomorphism
Theorems apply to modules as well. In fact, the Noether
Isomorphism Theorems were first introduced by Emmy Noether for
modules; not through groups or for rings. The Isomorphism Theorems
hold for groups and rings since abelian groups and rings are
special cases of modules. 

First, we introduce two homomorphisms which seem as if they are so stupidly
simple that they don't even deserve a definition; yet, they do. 


<span style="display:block" class="definition">
Let $R$ be a ring and $M$ and $N$ be $R$-module homomorphisms.
Then we define the following $R$-module homomorphisms.

* [1.] The map $\pi: M \to M/N$ given by 

\[
\pi(m) = m + N
\]

is said to be the **projection map**. Note that $\pi$
is
**surjective**, and that $\ker(\pi) = N$ (since $m +
N = N$ if and only if $m \in N$.)



* [2.] The map $i: M/N \to M$ given by 

\[
i(m + N) = m                
\]

is known as the **inclusion map**. More generally, if
$M' \subset M$, the **inclusion map** can also be
defined as $i: M' \to M$ where 

\[
i(m') = m'
\]

for all $m' \in M'$. Note that $i$ is **injective**,
and in the first case $\im(i) = M/N\cup \{0\}$ and in the
second case $\im(i) = M'$.



</span>


<span style="display:block" class="theorem">[(First Isomorphism Theorem)]
Let $R$ be a ring and $M$ and $N$ be $R$-modules. If $f: M \to
N$ is an $R$-module homomorphism, then 

\[
M/\ker(f) \cong \im(f).  
\]

\vspace{-0.8cm}
</span>


<span style="display:block" class="proof">
The proof is the same as before. Define the map $\phi:
M/\ker(f) \to N$ as 

\[ 
\phi(m + \ker(f)) = f(m).
\]

\textcolor{NavyBlue}{We quickly show that this is well-defined.} If $m + \ker(f) =
m' + \ker(f)$ for some $m, m' \in M$, then $m = m' + k$ for
some $k \in K$. Therefore, 

\[
\phi(m + \ker(f)) = f(m) = f(m' + k) = f(m') = \phi(m' + \ker(f)).
\]

\textcolor{NavyBlue}{Next, we show this is in fact an $R$-module homomorphism.}
Linearity is obvious, so we check the second criterion. Now
for any $a \in R$ we see that 

\[
\phi(a(m + \ker(f))) = \phi(am + \ker(f)) = f(am) = af(m) = a(\phi(m + \ker(f)))
\]

where we pulled the $a$ outside from $f(am)$ to make $af(m)$
from the fact that $f$ is an $R$-module homomorphism. 

\textcolor{NavyBlue}{Now we make two observations.} First, we
see that there is a one-to-one correspondence between
$M/\ker(f) \to \im(f)$. Second, this implies that $\phi$ is an
isomorphism between the two modules, so that 

\[
M/\ker(f) \cong \im(f)
\]

as desired.
</span>


<span style="display:block" class="theorem">[(Second Isomorphism Theorem.)]
Let $R$ be a ring and $M$ and $N$ and $P$ be submodules of
$M$. Then 

\[
(N + P)/P \cong N/(N \cap P).
\]

\vspace{-0.8cm}
</span>

\begin{minipage}{0.35 \textwidth}
\begin{figure}[H]
\begin{tikzcd}[column sep=small] 
&  
N + P
\\
N
\arrow[ur, dash]
&&
P
\arrow[ul,swap,"\text{(submodule)}"]
\\
&
N\cap P 
\arrow{ul}{\text{(submodule)}}
\arrow[ur, dash]
\end{tikzcd}
\end{figure}
\end{minipage} \hfill
\begin{minipage}{0.6\textwidth}
The diagram on the left is the same one we used in group
theory and ring theory. That is, the second isomorphism
theorem can still be described using the diamond diagram. 
\end{minipage}  


<span style="display:block" class="proof">
Construct the projection map $\pi : M \to M/P$ and let $\pi'$
be the restriction of $\pi$ to $N$. Then we see that
$\ker(\pi') = N \cap P$, while 

\[
\im(\pi') = \{\pi'(n) \mid n \in N\} = \{n + P \mid n \in N\} = (N + P)/P.
\]

Thus by the First Isomorphism Theorem we have that 

\[
N/\ker(\pi') \cong \im(\pi') \implies (N + P)/P \cong N/(N \cap P)
\]

as desired. 
</span>


<span style="display:block" class="theorem">[(Third Isomorphism Theorem)]
Let $R$ be a ring and $M$ an $R$-module. Suppose $N$ and $P$
submodules such that $P \subset N$. Then 

\[
M/N \cong (M/P)/(N/P).
\]

\vspace{-0.8cm}
</span>


<span style="display:block" class="proof">
Construct the map $f: M/P \to M/N$ by defining $f(m + P) = m +
N$ where $m+P \in M/P$ and $m + N \in M/N$. First observe that
this is a surjective mapping since $P \subset M$, so the
correspondence $m + P \to m + N$ will cover all of $M/N$. 

Now observe that 

\[
\ker(f) = \{m + p \mid m \in N\} = N/P.
\]

Therefore, by the First Isomorphism Theorem

\[
(M/P)/\ker(f) \cong M/N \implies (M/P)/(N/P) \cong M/N
\]

as desired.
</span>


<span style="display:block" class="theorem">[(Fourth Isomorphism Theorem)]
Let $R$ be a ring and $M$ an $R$-module. Suppose $N$ is a
submodule of $M$. Then every submodule of $M/N$ is of
the form $P/N$ where $N \subset P \subset M$. 
</span>
Another way to understand this statement is to realize there is a
one to one correspondence between the submodules of $M$ containing
$N$ and the submodules of $M/N$.


<span style="display:block" class="proof">

</span>





<script src="../../mathjax_helper.js"></script>