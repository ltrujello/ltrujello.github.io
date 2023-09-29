#1.3. Homomorphism and Isomorphisms.

{\color{BlueViolet}As with all mathematical objects, now that we have a well defined
abstract concept (i.e., a group) we'll now be interested
attempting to understand *mappings* between different
groups. Mappings of abstract concepts simply helps mathematicians
get a better sense of what they're dealing with, and most often
provides new insight into understand their objects. 

The most important utility of the following definition is that it
not only leads one to have a better understanding of groups, but it
also helps us understand when two groups are equivalent. For
example, $D_3$ and $S_3$ equivalent, since one could view $D_3$
as simply all the permutations of 1, 2, and 3, if we assigned these
numbers to the vertices of a triangle.
}


<span style="display:block" class="definition">
Let $(G, \cdot)$ and $(G', *)$ be groups. A
**homomorphism** is a mapping $\phi: G \to G'$ such that,
for all $a, b \in G$, 

\[
\phi(a \cdot b) = \phi(a) * \phi(b).
\]

{\color{red}Again, here $*$ is the group operation of $G'$.}

</span>

**Example.** Consider the two groups $GL_n(\mathbb{R})$ and
$\mathbb{R}\setminus\{0\}$. If we define $\phi$ such that, for $A \in
GL_n(\mathbb{R})$ 

\[
\phi(A) = \det(A)
\]

then $\phi$ defines a homomorphism. 

Recall that for for any $n
\times n$ matrices $A, B$ that $\det(AB) = \det(A)\det(B)$.
Therefore 

\[
\phi(AB) = \det(AB) = \det(A)\det(B) = \phi(A)\phi(B).
\]

Since $\phi(AB) = \phi(A)\phi(B)$, we see that $\phi$ satisfies
the condition to be a homomorphism.


<span style="display:block" class="proposition">
Let $\phi: G \to G'$ be a homomorphism. Then all of the
following hold.

* [1.] If $e_G$ is the identity of $G$ and $e_{G'}$ is
the identity of $G'$, then $\phi(e_G) = e_{G'}$.



* [2.] For all $g \in G$, $\phi(g^{-1}) =
\phi(g)^{-1}$. 



* [3.] For $g_1, g_2, \dots, g_n \in G$, then $\phi(g_1
\cdot g_2 \cdot \dots \cdot g_n) =
\phi(g_1)\phi(g_2)\cdots\phi(g_n)$. Consequently, if $g = g_1 =
g_2 = \cdots = g_n$, then $\phi(g^n) = \phi(g)^{n}$.



</span>


<span style="display:block" class="proof">
Let $g \in G$, and suppose $\phi: G \to G'$ is a
homomorphism. 


* [1.] Since $e_G = e_G \cdot e_G$, we have that 

\[
\phi(e_G) = \phi(e_G \cdot e_G) = \phi(e_G)\phi(e_G).
\]

We also know that $\phi(e_G) \in G'$, and becuase $G'$ is a group,
there exists an inverse $\phi(e_G)^{-1} \in G$ of
$\phi(e_G)$. Multiplying this on the left (or right)
yields

\[
e_{G'} = \phi(e_G)  
\]

as desired.


* [2.] Since $gg^{-1} = e_G$, and by (1.) we know that
$\phi(e_G) = e_{G'}$. Hence 

\[
\phi(e_G) = e_{G'} \implies \phi(gg^{-1}) = e_{G'} 
\implies \phi(g)\phi(g^{-1}) = e_{G'}.
\]

Again, $\phi(g) \in G'$, and since $G'$ is a group there
exist an inverse $\phi(g)^{-1} \in G$ of $\phi(g)$.
Multiplying on the left by this inverse, we get 

\[
\phi(g)\phi(g^{-1}) = e_{G'} \implies \phi(g^{-1}) = \phi(g)^{-1}
\]

as desired.



* [3.] This is just repeated application of the
homomorphism property. 
For $g_1, g_2, \dots g_n \in G$, $g_1 \cdot
g_2 \cdot \hspace{0.01mm} \dots \hspace{0.01mm} \cdot g_n = g_1 \cdot (g')$ 
where $g' = g_2
\cdot g_3 \cdot \hspace{0.01mm} \dots \hspace{0.01mm} \cdot g_n$. Applying the
homomorphism property, 

\[
\phi(g_1 \cdot g_2 \cdot \hspace{0.01mm} \dots \hspace{0.01mm} \cdot g_n) = \phi(g_1 \cdot g') = \phi(g_1) \phi(g').
\]

Repeatedly applying the same idea, starting again with the
product $g_2 \cdot g_3 \cdot \hspace{0.01mm} \dots
\hspace{0.01mm} \cdot g_n$ yields the result. The fact
that $\phi(g^n) = \phi(g)^n$ is follows immediately.



</span>
{\color{Plum} 
If $\phi$ is a bijective homomorphism (i.e., one-to-one and
onto) then we say that $\phi$ is an **isomorphism**.
Furthermore, if there exists an isomorphism between two spaces
$G$ and $G'$, then we say these spaces are **isomorphic**
and that $G \cong G'$. As we'll soon see, isomorphisms gives
us really nice results (hence the special terminology and
notation). In addition, it can sometimes be difficult to tell when
two groups $G$ and $G'$ are the same or different. Isomorphisms
can help determine when there *isn't* such an equivalence.

As we'll see, the concept of an isomorphism is very powerful.
However, proving it may not be that simple, and in ceratin cases
the following theorem will be very useful.
}


<span style="display:block" class="theorem">
Let $G$ and $H$ be groups. The homomorphism $\phi: G \to H$ is an
isomorphism if and only if there exists a homomorphism $\psi:
H \to G$ such that $\psi \circ \phi$ is the identity map on
$G$ and $\phi \circ \psi$ is the identity map on
$H$.
</span>


<span style="display:block" class="proof">
($\implies$) Suppose $\phi: G \to H$ is an isomorphism. Since
$\phi$ is bijective, define the inverse map $\phi^{-1}: H \to
G$ such that if $\phi(g) = g'$ then $\phi^{-1}(g') = g$. 

Note that this is a well defined map due to the surjectivity
and injectivity of $\phi$. To show it is a homomorphism, we
need to demonstrate that $\phi^{-1}(h_1\cdot h_2) =
\phi^{-1}(h_1)\phi^{-1}(h_2)$. Thus 
observe that for $h_1, h_2 \in H$ there exist $g_1, g_2 \in G$
such that $\phi(g_1) = h_1$ and $\phi(g_2) = h_2$. Therefore

\[
\phi(g_1 \cdot g_2) = h_1 \cdot h_2 \implies \phi^{-1}(h_1 \cdot h_2) = g_1\cdot g_2
= \phi^{-1}(h_1)\cdot\phi^{-1}(h_2).
\]

Thus $\phi^{-1}$ is a homomorphism.

Now observe that for all $g \in G$ we have that $\phi^{-1}
\circ \phi(g) = g$ and for all $h \in H$, $\phi \circ \phi^{-1}(h) =
h.$ Thus $\phi^{-1} \circ \phi$ is the identity on $G$ while
$\phi \circ \phi^{-1}$ is the identity on $H$, which proves
this direction.

($\impliedby$) Now suppose $\phi: G \to H$ is a homomorphism
and that there exists a homomorphism
$\psi: H \to G$ such that $\psi \circ \phi$ is the identity
map on $G$ and $\phi \circ \psi$ is the identity map in $H$.
In other words, $\psi$ and $\phi$ are inverses of each other. 
Thus $\phi$ is a bijection function from $G \to H$, which
implies that $\phi$ is an isomorphism. 
</span>

We also introduce the following criteria which is frequently used
to evaluate if a homomorphism is one-to-one and/or onto. 


<span style="display:block" class="theorem">
Let $\phi: G \to G'$ be a homomorphism.
Then 

* [1.] $\phi$ is one-to-one if and only if
$\mbox{ker}(\phi)$ is trivial. That is, $\mbox{ker}(\phi) = \{e_G\}$, where $e_G$ is
the identity of $G$.



* [2.] $\phi$ is onto if and only if $\mbox{im}(\phi) = G'$. 



Therefore, $\phi$ is an **isomorphism** if and only if
(1) and (2) hold.
</span>


<span style="display:block" class="proof">

* [1.] Suppose $\phi$ is one-to-one. By proposition
1.1.1, we know that $\phi(e_G) = e_{G'}$. But since $\phi$
is injective we know $e_G$ is the only element in $G$
which is mapped to $e_{G'}$. Therefore $\mbox{ker}(\phi) =
\{e_G\}$.

Now suppose $\mbox{ker}(\phi) = \{e_G\}$. To show $\phi$
is one-to-one, consider
$g, h \in G$ such that

\[
\phi(g) = \phi(h).
\]

Multiplying both sides by $\phi(h)^{-1}$ we get 

\[
\phi(g)\phi(h)^{-1} = e_{G'}.
\]

By proposition 1.1.2, we know that $\phi(h)^{-1} =
\phi(h^{-1})$. Since $\phi$ is a homomorphism, we can then
combine the terms to get 

\[
\phi(gh^{-1}) = e_{G'}.
\]

Since $\mbox{ker}(\phi) = \{e_G\}$, we see that 

\[
gh^{-1} = e_G \implies g = h.                
\]

Therefore $\phi$ is one to one.



* [2.] Suppose $\phi$ is onto. Then $\mbox{im}(\phi) = G'$
is just another way of stating this fact. 

Suppose $\mbox{im}(\phi) = G'$. Then for every element $g'
\in G'$, there exists $g \in G$ such that $\phi(g) = g'$.
That is, $\phi$ covers every value in $G'$ so that it is
onto.



Thus, we have that a function is isomorphic if and only if it
is one to one and onto. Hence, it is isomorphic if and only if
(1) and (2) hold.
</span>

We also make two common definitions for special homomorphisms. 

<span style="display:block" class="definition">
Let $G$ be a group.

* [1.] If $\phi: G \to G$ is a group homomorphism, then
we say that $\phi$ is a **endomorphism**.


* [2.] If $\phi$ is a bijective endomorphism (an
isomophic endomorphism) then we say that $\phi$ is an **automorphism**.



</span>


<span style="display:block" class="theorem">
The set of all automorphisms of a group $G$, denoted as
$\text{Aut}(G)$, forms a group with an operation $\circ$ of
function composition.
</span>


<span style="display:block" class="proof">
We can prove this directly. 
\begin{description}
\item[Closure.] Let $\phi$ and $\psi$ be automorphisms.
Then $\phi \circ \psi$ is (1) a homomorphism from $G \to
G$ and (2) a bijection (as the composition of bijections
is a bijetion).

\item[Associativity.] In general, function composition is
associative. 

\item[Identity.] Let $i:G \to G$ be the identity map. The
(1) $i$ is a group homomorphism and (2) a bijection.
Therefore $i \in \text{Aut}(G)$ and we can set $i$ as the
identity of the group. Note that 

\[
i \circ \phi = \phi = \phi \circ i   
\]

for any $\phi \in \text{Aut}(G)$. 

\item[Inverse.] Let $\phi \in \text{Aut}(G)$. Construct
the function $\phi^{-1}$ as follows. If $\phi(g) = g'$ for
some $g, g' \in G$, then write $\phi^{-1}(g') = g$. Such
an assignment is well-defined since  $\phi$ is a
bijection. Hence we see that 

\[
\phi \circ \phi^{-1} = i = \phi^{-1} \circ \phi.
\]

Finally, observe that $\phi^{-1}$ is (1) a homomorphism
and (2) a bijection, so we see that $\phi^{-1} \in
\text{Aut}(G)$. Therefore this forms a group.
\end{description}
</span>






<script src="../../mathjax_helper.js"></script>