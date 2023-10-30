<style>
.md-content {
    max-width: 80em;
}
</style>
#1.6. Normal subgroups

Normal subgroups are special subgroups which exhibit properties of
interest for when we go on to later define the idea of quotient
groups, a concept we have touched upon slightly in considering
$\mathbb{Z}/2\mathbb{Z}$ and other modulo groups. They are a bit
abstract at first, since they have to do with **cosets**.
Once you work with normal subgroups for a bit, it will s
eventually click and the reasoning behind their definitions
becomes clear. 


<span style="display:block" class="definition">
Let $G$ be a group and suppose $H$ is a subgroup of $G$. We
say that $H$ is **normal** if and only if **for every** $g
\in G$, we have that $Hg = gH$. We denote such a relation as
$H \unlhd G$.
</span>
\noindent We make two remarks here.

* **Commutative Groups.** Note that if $G$ is commutative, then $H$, a subgroup of $G$, is
also commutative. In fact, $H$ commutes with all elements of $G$.
That is, if $H = \{h_1, h_2, \dots \}$ then

$$
gH = \{gh_1, gh_2, \dots\} = \{h_1g, h_2g, \dots\} = Hg
$$

for all $g \in G$. Thus what we're trying to say here is if $G$ is commutative, every
subgroup $H$ of $G$ is normal.
* **Set Equality.** If $H$ is normal to $G$, then $gH = Hg$
all $g \in G$. Be careful with this equation, since what this is
not saying is that $gh=hg$ for all $g\in G$ and $h \in H$; that
would imply commutativity, and it may be the case that $G$ and $H$
are not commutative groups. That is, the above equation is set
equality, not term-by-term equality.

What this does say, however, is if $gH = Hg$, then for each $g\in
G$, and for every $h_1 \in H$, there exists an $h_2 \in H$ such
that 

$$
gh_1 = h_2g.
$$

Note here that commutative groups satisfy this because in their
case, $h_1 = h_2$ satisfies the equation.


Since our current definition of normality would be exhausting to
use directly if we wanted to check if a subgroup is normal, we
have the following theorem that helps us check for normality. 


<span style="display:block" class="theorem">
Let $G$ be a group and $H$ a subgroup of $G$. The following
are equivalent:

* **1.** $H \normal G$ for all $g \in G$
* **2.** $gHg^{-1} = H$ for all $g \in G$
* **3.** $gHg^{-1} \subset H$ for all $g\in G$.
* **4.** $(Hg)(Hh) = H(gh)$ for all $g, h \in G$

</span>


<span style="display:block" class="proof">
We'll prove this by producing a chain of imply statements that
can traverse in both directions.
Let $G$ be a group and $H$ be a subgroup. 

\noindent $\mathbf{(1 \iff 2)}$ If $H \normal G$, then $gH = Hg$
for all $g \in G$. Multiplying on the left by $g^{-1}$, we
then see that $gHg^{-1} = H$ for all $g \in G$.

Proving the reverse direction, if $gHg^{-1} = H$ for all $g \in G$
then $gH = Hg$ for all $g \in G$, which means that $H$ is
normal by defintion. 

\noindent $\mathbf{(2 \iff 3)}$ If $gHg^{-1} = H$ for all $g \in G$
then it is certainly true that $gHg^{-1} \subset H$ for all $g
\in G$. 

Now we prove the other direction. Suppose $gHg^{-1} \subset H$ for
all $g \in G$. Then

$$
gHg^{-1} \subset H \implies gH \subset Hg 
\implies H \subset g^{-1}Hg
$$

by multiplying on the right by $g$ and on the left by
$g^{-1}$. However, since we have assumed (3) is true we know
that 

$$
(g^{-1})H(g^{-1})^{-1} \subset H \implies g^{-1}Hg
\subset H. 
$$
 
By the above equations we then have that $H = g^{-1}Hg$, and
multiplying by $g^{-1}$ on the right and $g$ on the left
yields that $H = gHg^{-1}$ as desired.

\noindent$\mathbf{(2 \iff 4)}$ Suppose (2). Then observe that $gHg^{-1} = H
\implies gH = Hg$ for all $g \in G$.
Therefore for $h \in G$, 

$$
(Hg)(Hh) = H(gH)h = H(Hg)h = H(gh).
$$

In the first step we used associativity and in the
second step we used the fact that $gH = Hg$. 

To prove the other direction, suppose $(Hg)(Hh) = H(gh)$ for
all $g, h \in G$. Let $h = e$. Then 
</span>
To show a subgroup $H$ of $G$ is normal, condition (3) of this
theorem generally the fastest and easy way to take advtange of. It
is usually the least complicated one to show. 
\\
\\
\noindent
**Example.**
\\
Consider the group $GL_n(\mathbb{R})$ and its subgroup
$SL_n(\mathbb{R})$. It turns out that $SL_n(\mathbb{R}) \normal
GL_n(\mathbb{R})$, which we will show using condition (3).

Let $A \in GL_n(\mathbb{R})$ and suppose $T
\in SL_n(\mathbb{R})$. We must show that $ATA^{-1} \in
SL_n(\mathbb{R})$ for all $A \in GL_n(\mathbb{R})$ and $T \in
SL_n(\mathbb{R})$. Observe that 

\begin{align*}
\det(ATA^{-1}) = \det(A)\det(T)\det(A^{-1})
= \det(A)(1)\det(A)^{-1} = 1
\end{align*}

where we used the basic properties of the determinant for the
calculation. Since $\det(ATA^{-1}) = 1$, we have that $ATA^{-1}
\in SL_n(\mathbb{R})$ for all $A$ and $T$ in $GL_n(\mathbb{R})$
and $SL_n(\mathbb{R})$, respectively. Therefore $SL_n(\mathbb{R})$
is normal to $GL_n(\mathbb{R})$.   
\\
\\
**Example.**
\\
One important example is the following: for any group homomorphism
$\phi$ between two groups $G$ and $G'$, recall that
$\mbox{ker}(\phi)$ is a subgroup of $G$. However, we also have
that $\mbox{ker}(\phi) \normal G$, which we'll show as follows.


<span style="display:block" class="proposition">
Let $G, G'$ be groups and $\phi: G \to G'$ be a group
homomorphism. Then $\ker(\phi) \normal G$.
</span>


<span style="display:block" class="proof">
We need to show that for all $g \in G$, $h \in \mbox{ker}(\phi)$
that $ghg^{-1} \in \mbox{ker}(\phi)$. Thus observe that 

$$
\phi(ghg^{-1}) = \phi(g)\phi(h)\phi(g^{-1})
= \phi(g)\cdot 0 \cdot \phi(g^{-1}) = 0.
$$

Since $\phi(ghg^{-1}) = 0$, we thus see that $ghg^{-1} \in
\mbox{ker}(\phi)$ for all $g \in G$ and $h \in \mbox{ker}(\phi)$,
which proves $\mbox{ker}(\phi) \normal G$.    
</span>

Another important example of normality is the fact that the
center of a group $Z(G)$ is normal to $G$ for any group $G$.


<span style="display:block" class="proposition">
Let $G$ be a group. Then $Z(G) \normal G$.
</span>


<span style="display:block" class="proof">
Recall that $Z(G)$ is a subgroup of $G$, consisting of all the
elements of $G$ which commute with every element in $G$. More
precisely, 

$$
Z(G) = \{z \in G \mid gz = zg \text{ for all } g \in G\}.
$$

Now for any $g \in G$ and $z \in Z(G)$, we have that $gzg^{-1}
= gg^{-1}z = z$, since $z$ commutes with all elements of $G$.
Therefore $gzg^{-1} \in Z(G) \implies gZ(G)g^{-1} \subset
Z(G)$. By the previous theorem, we can conclude that $Z(G)
\normal G$ as desired.
</span>

Next, we introduce a small theorem that allows us to quickly and
easily identify if a subgroup $H$ of $G$ is normal. 


<span style="display:block" class="theorem">
If $G$ is a group and $H$ is a subgroup, and $[G:H] = 2$, then
$H \normal G$.
</span>


<span style="display:block" class="proof">
Since $G$ has two right (and equivalently two left) cosets, we
see that they must be of the form $H$ and $Hg$ where $g \in
G\setminus H$ (that is, all of the elements of $G$ which are
not in $H$).   

As we said before, there are equivalently two left cosets $H$
and $gH$ where $g \in G\setminus H$. Since the cosets partition $G$, we see that for any $g \in
G\setminus H$ two partitions of $G$ are 

$$
\{H, Hg\} \hspace{0.2cm}\text{and}\hspace{0.2cm} \{H, gH\}.
$$

Since these partition the same set we see that $gH = Hg$ for
all $g \in G\setminus H$. Note that we already know that for
\    $g \in H$, $Hg = H$ and $gH = H$ so $gH = Hg$. Therefore,
we have all together that $Hg = gH$ for all $g \in G$.
</span>

\noindent In working with normal subgroups, one may form the following
questions. 
\\

\textcolor{ForestGreen}{**Q:** If $K$ is a normal subgroup of $H$ and $H$ is a normal
subgroup of $G$, is $K$ normal to $G$?}
\\

**A:** **Not always**. If $H \normal K$, then $khk^{-1} \in
K$ for all $k \in K$ but there is nothing allowing for us to extend
this further and state that $ghg^{-1} \in K$ for all $g \in G$. 
\\
However, a special case for when this is true involves $Z(G)$. We
know that $Z(G) \normal G$. But if $K \normal Z(G)$ then
it turns out $K \normal G$, 











<script src="../../mathjax_helper.js"></script>