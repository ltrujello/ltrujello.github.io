<style>
.md-content {
    max-width: 80em;
}
</style>
#3.1. Definitions.

In group theory, we started with a set $G$ equipped with a bilinear
operation $\cdot : G \times G \to G$ which mapped $G$ to itself.
The operation was required to be associative, and there needed to
be inverses and an identity element. 

In ring theory, we went further to assume $R$ was not only an
abelian group, we placed the group operation with $+: R\times R
\to R$ and then defined a *multiplication* $\cdot: R \times
R \to R$ which is was associative and left- and right-
distributive.

Finally, we reach module theory, which considers again an abelian
group $M$ with operation $+:M \times M \to M$ but lets a ring $R$
act on $M$, whose addition $+R\times R \to R$ agrees with the one
which acts on $M$ but whose multiplication $\cdot: R \times M \to
M$ acts on $R$ and $M$. 

Note how abelian groups and rings are special cases of modules.
This will be more clear once we introduce the axioms. 


<span style="display:block" class="definition">
Let $R$ be a ring with identity, and $M$ an abelian group
equipped with $+:M \times M \to M$. Then $M$ is an
**left $R$-module** if we equip $R \times M$ with
multiplication $\cdot : R \times M \to M$ and for all $m \in
M$ and $a, b \in R$
\begin{enumerate}
\item $a(m_1 + m_2)= am_1 + am_2$
\item $(a + b)m = am + bm$
\item $(ab)m = a(bm)$
\item $1_Rm = m$ where $1_R$ is the identity of $R$.
\end{enumerate} 

Alternatively, an abelian group $M$ is a **right $R$-module** if we equip
$M \times R \to M$ with multiplication $\cdot: M \times R \to
M$ and for all $m \in M$ and $a, b \in R$
\begin{enumerate}
\item $(m + n)a = ma + na$ 
\item $m(a + b) = ma + mb$
\item $m(ab) = (ma)b$
\item $m1_R = m$ where $1_R$ is the identity of $R$.
\end{enumerate}
</span>

Notice that we can think of these products as a group action, or sort of a
"ring action" acting on $M$. That is, an $R$-module $M$ is just an abelian
group that a ring $R$ can act on. If you have an abelian group $N$
that $R$ simply cannot act on and satisfy the above axioms, then
$N$ is not an $R$-module.
\\
For convenience, we will develop the theory of $R$-modules by
solely working with left $R$-modules, since all proofs and
statements will be equivalent up to a swap of variables for right
$R$-modules. 
\\
**Examples.**\\

* [1.] \textcolor{NavyBlue}{Note that if $R$ is commutative, then a left
$R$-module coincides with a right $R$-module. To see this, let $M$
be a left $R$-module. Then construct the right $R$-module by
defining the multiplication as

\[
m\cdot r = rm.
\]

Then we see that for all $m \in M$, $a,b \in R$, 
\begin{enumerate}


*  $(m_1 + m_2)\cdot a = a(m_1 + m_2) = am_1 + am_2 = m_1
\cdot a + m_2 \cdot a$ \checkmark


*  $m(a + b)= (a + b)m = am + bm = m \cdot a + m
\cdot b$ \checkmark 


*  $m\cdot (a b) = (ab)m = (ba)m = b(am) = b(m \cdot a) = (m
\cdot a )\cdot b$ \checkmark 


*  $m \cdot 1_R = 1_Rm = m$. \checkmark
\end{enumerate}
Note that in part $(c)$ is where we used the fact that $R$ is
commutative. So whenever $R$ is commutative, the existence of a
left $R$-module automatically implies that existence of a
right $R$-module, and vice versa.
}



* [2.] Let $R$ be a ring. Then if we substitute $M =R$ in the above
definition, and let the multiplication $
\cdot$ be the multiplication on $R$ then $R$ is a left and a right
$R$-module. This is because $R$ is an abelian group which is
associative and left- and right-distributive. Hence, it satisfies
all of the above axioms. 

So keep in mind that a ring $R$ is just a left- and right-$R$
module that acts on $R$. 




Here's another example which shows that abelian groups are simply
$\ZZ$ modules. 

<span style="display:block" class="proposition">
Let $G$ be an abelian group. Then $G$ is a left and right
$\ZZ$-module. 
</span>


<span style="display:block" class="proof">
Let $\ZZ$ act on $G$ as follows. Define 

\[
ng = 
\begin{cases}
g + g + \cdots + g \text{ ($n$ times)} & \text{ if } n  > 0\\ 
0 & \text{ if } n = 0\\
(-g) + (-g) \cdots (-g) \text{ ($n$ times) } & \text{ if } n < 0
\end{cases}
\]

and 

\[
gn = 
\begin{cases}
g + g + \cdots + g \text{ ($n$ times)} & \text{ if } n  > 0\\ 
0 & \text{ if } n = 0\\
(-g) + (-g) \cdots (-g) \text{ ($n$ times) } & \text{ if } n < 0.
\end{cases}
\]

Then with this definition of multiplication, it is easy to
show that the axioms (a)-(d) are satisfied.
</span>


* [3.] If $R$ is a ring and $I$ is a left (right) ideal of
$R$, then $I$ is a left (right) $R$-module. 



* [4.] Let $V$ be a vector space defined over a field $F$.
Then $V$ is an $F$-module. (Now it is clear why there are a
million axioms in the definition of a vector space!)




With $R$-modules introduced and understood, we can jump right into
homomorphisms. 

<span style="display:block" class="definition">
Let $R$ be a ring and $M$ and $N$ be $R$-modules. We define
$f: M \to N$ to be an **$R$-module homomorphism** if 

* [1.] $f(m_1 + m_2) = f(m_1) + f(m_2)$ for any $m_1,
m_2 \in M$ 


* [2.] $f(am) = af(m)$ for all $a \in R$ and $m \in M$.



If $f$ is a bijective $R$-module homomorphism, then we say
that $f$ is an **isomorphism** and that $M \cong N$.
</span>
Thus we see that $R$-module homomorphisms must not only be linear
over the elements of $M$, but they must also pull out scalar
multiplication by elements of $R$.

Recall earlier that we said a vector space $V$ over a field $F$ is
an $F$-module. Now if $W$ is another vector space and $T: V \to W$
is a linear transformation, then we see that $T$ is also an
$F$-module homomorphism! 

In the language of linear algebra, a
**linear transformation** is usually defined as a function
$T: V \to W$ such that for any $\bf{v}_1, \bf{v}_2, \bf{v} \in V$ and
$\alpha \in F$ we have that 

* [1.] $T(\bf{v}_1 + \bf{v}_2) = T(\bf{v}_1) + T(\bf{v}_2)$


* [2.] $T(\alpha\bf{v}) = $ $\alpha$$T(\bf{v})$. 



As we will see, linear algebra is basically a special case of
module theory. 


<span style="display:block" class="definition">
Let $R$ be a ring and $M$ and $N$ a pair of $R$-modules. Then
$\hom_R(M,N)$ is the set of all $R$-module homomorphisms from
$M$ to $N$. 
</span>

\textcolor{MidnightBlue}{It turns out we can turn $\hom_R(M,N)$
into an abelian group, and under special circumstances it can
actually be an $R$-module itself. It will be the case
that $\hom_R$ will actually be an important functor, but that is
for later.}
\\
\indent To turn this into an abelian group, we define addition of
the elements to be 

\[
(f + g)(m) = f(m) + g(m)
\]

for all $f, g \in \hom_R(M, N)$. We let the identity be the
zero map, and realize associativity and closedness are a given to
conclude that this is in fact an abelian group. 
\\

Suppose we want to make $R$, our ring, act on $\hom_R(M, N)$ in
order for it to be an $R$-module. Then we define scalar
multiplication to be $(af)(m) = a(f(m))$; a pretty reasonable
definition for scalar multiplication. 

\textcolor{Red}{This issue with this is that $\hom_R(M, N)$ will
not be closed under scalar multiplication of elements of $R$
unless $R$ is a commutative ring.
}

We'll demonstrate this as follows. Let $b \in R$ and
$f \in \hom_R(M, N)$. Then the second property of an $R$-module
homomorphism tells us that $f(bm) = bf(m)$ for all $m \in
M$. Now suppose we try to use our definition of scalar
multiplication, and consider $af$ where $a \in R$. Then if we try
to see if $af$ will pass the second criterion for being an
$R$-module homomorphism, we see that 

\[
(af)(bm) = a(f(bm)) = a(bf(m)) = abf(m).
\]

That is, we see that $af$ isn't an $R$-module homomorphism because
$(af)(bm) \ne b(af)(m)$ (which is required for an $R$-module homomorphism); rather, $(af)(bm) = abf(m).$ Now if $R$
is a commutative ring, then 

\[
abf(m) = baf(m)
\]

so we can then say that $(af)(bm) = b(af)(m)$, in which case $af$
passes the test for being an $R$-module homomorphism. 

This proves the following propsition, which will be useful for
reference for later.

<span style="display:block" class="proposition">
Let $M$ and $N$ be $R$-modules. Then $\hom_R(M, N)$ is an
abelian group. Furthermore, it is an 
$R$-module if and only if $R$ is a commutative ring.
</span>
Next, we make the following definitions for completeness. 


<span style="display:block" class="definition">
Let $R$ be a ring and $M$ and $N$ be $R$-modules. If $f: M \to
N$ is an $R$-module homomorphism, then 

* [1.] The set $\ker(f) = \{m \in M \mid f(m) = 0\}$ is
the **kernal** of $f$ 


* [2.] The set $\im(f) = \{f(m) \mid m \in M\}$ is the
**image** of $f$.



</span>









<script src="../../mathjax_helper.js"></script>