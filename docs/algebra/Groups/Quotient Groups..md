<style>
.md-content {
    max-width: 80em;
}
</style>
#1.7. Quotient Groups.
The work done in the previous section on Normal subgroups now
leads to the formulation of the **Quotient Group**. Up to
this point we've studied groups which have familiar, concerete
objects, but now we're going to get a little bit abstract.
We're going to look at the useful concept of the quotient group, $G/H$,
which is a 
**group whose elements are $H$ cosets**. That is, the elements of
our group are going to be sets themselves. The operation on the
elements of the quotient group can only make sense if the cosets
are from a subgroup $H$ which is normal to $G$.


<span style="display:block" class="theorem">
Let $G$ be a group and $H \normal G$. Define $G/H$ to be the
set consisting of all the possible right (or equivalently
left) $H$ cosets. If we
equip this set with a product $\cdot$ such that 

\[
(Ha)\cdot(Hb) = H(ab)
\]

then $G/H$ forms a group, called the **Quotient Group**.
</span>
Let's review what this is saying. Basically, if we have a normal
subgroup $H$ of $G$, the set of cosets $\{Hg_1, Hg_2, \dots \}$
with the product $Hg_1 \cdot Hg_2 = H(g_1g_2)$ **forms a group**.


<span style="display:block" class="proof">
\begin{description}
\item[Identity.] To show that this set is a group, we first define the identity
element to simply be $H$. This is a "trivial" coset, and for
any $Ha$, where $a \in G$, 

\begin{align*}
(Ha)(H) = Ha \\
(H)(Ha) = Ha
\end{align*}

so $H$ is a natural and apporopriate choice for an identity as
it has the property of an identity element.  

\item[Associativity.] Associativity is derived from the
associativity of our group $G$ itself. Observe that for any
$a, b, c \in G$ we have 

\begin{align*}
(Ha)[ (Hb)(Hc)] = Ha[H(bc)] = H(abc)\\
[(Ha)(Hb)](Hc) = [H(ab)]Hc = H(abc).
\end{align*}

Therefore $(Ha)[ (Hb)(Hc)] = [(Ha)(Hb)](Hc)$ for all $a, b,
c \in G$, so the product relation is associative.

\item[Closedness.] The result of our proposed product is
always a coset itself ($Ha \cdot Hb = H(ab)$), and since 
$G/H$ is a set of all $H$ cosets we see that this set is
closed under $\cdot$.

\item[Inverses.] For any $Ha \in G/H$, where $a \in G$, we
see that the inverse element is $Ha^{-1}$, since 

\begin{align*}
(Ha)(Ha^{-1}) = H(aa^{-1}) = H\\
(Ha^{-1})(Ha) = H(a^{-1}a) = H
\end{align*}

and we already defined $H$ to be our identity element. So
our proposed inverse makes sense.
Note that
$Ha^{-1} \in G/H$ since $a^{-1} \in G$, so an inverse
element not only exists but it also exists in $G/H$
\end{description}
All together, this allows us to observe that we have a group
structure, so long as $H \normal G$.
</span>
{\color{purple}(Why do we need this
the condition that $H \normal G$? Well, because the only way we can make damn sure
that $(Ha)(Hb) = H(ab)$ is by Theorem 1.10, which requires
that $H \normal G$.)
}

{\color{NavyBlue} Note that there is another way to think about $G/H$. The elements
of the quotient group are cosets, right? However, let us not forget
that cosets are simply ***equivalence classes which
respect the following equivalence relation*** }: {\color{Black} if $G$ is a group, $H$ is a
subgroup, then for any $a, b \in G$ we say that $a \sim
b$ if and only if $ab^{-1} \in H$.} {\color{NavyBlue} Thus we can
recast our definition follows:}
\\

\begingroup
\par
\leftskip25pt
\rightskip\leftskip
\noindent Let $H \normal G$. Then the set $G/H$ is defined to consist of all
of the
\sout{right (or left) cosets of $H$ in $G$} equivalence classes of
the elements of $G$ (under the equivalence relation stated in the
previous paragraph). 
\par
\endgroup
\vspace{1cm}

{\color{Violet}We thus have two equivalent ways to interpret the meaning of a
quotient group. One involves equivalence classes, while the other
involves cosets. In our case it seems more complicated to think
about equivalence classes.
However, in different applications of group theory (such
as to algebraic geometry and topology) it will be convenient to
interpret quotient groups as equivalence classes. For now, we'll
stick with the coset interpretation, since it's the easiest way to
understand a quotient group.
}
\\ 

**Example.** Recall that we showed $SL_n(\mathbb{R}) \normal
GL_n(\mathbb{R})$. Thus the quotient group
$GL_n(\mathbb{R})/SL_n(\mathbb{R})$ makes sense by Theorem 1.11,
so let's see what this group looks like.

First, the identity element of our group is $SL_n(\mathbb{R})$.
\\
\\
\indent In dealing with quotient groups, you may be wondering the
following questions:\\
\textcolor{ForestGreen}{**Q:** If $H$ is a normal subgroup of
$G$, and $G$ is abelian, is $G/H$ abelian? If $G/H$ is abelian, is
$G$ abelian?}
\\
**A:** **The answer to the first question is yes**. 
Observe that
by definition, $G/H = \{aH \mid a \in G\}.$ But since $H$ 
is normal, we know that $gH = Hg$ for all $g \in G$. 
Thus observe that for $aH, bH \in G/H$, we have that 

\begin{align*}
(aH)(bH) =(ab)H &= (ba)H \text{ (since } G \text{ is abelian) }\\
&= (bH)(aH).
\end{align*}

Thus the set $G/H$ must be abelian.
\\
\\
**The answer to the second question is \textbf{no, not always}**. If $G/H$ is abelian, 
we know that 
$$
(aH)(bH) = (bH)(aH) \implies (ab)H = (ba)H.
$$ 
for all $a, b \in G$. However, this only guarantees **set equality**, 
not a term-by-term equality (in which case the group would be abelian). 
An example of this is $D_{6}$ with the subgroup $H = \{1, r, r^2\}.$
In this case $H \unlhd D_6$ because all the left cosets are $H, sH$ and therefore 
$[D_{2n}: H] = 2$ (Hence $H \normal G$ by the previous proposition). In addition, 
$H(sH) = sH=  sH(H)$, $sH(sH) = s^2H = (sH)sH$, so $G/H$ is abelian, but the set $D_{2n}$
is itself not an abelian group. Thus, **it is possible for
$G/H$ to be ableian while $G$ itself is not abelian **
\\
\\
Another fun example for when the quotient group $G/H$ is abelian,
even though the group $G$ is abelian, is the following.
\\
\\
**Example.**
Let 

\[
G = \left\{
\begin{pmatrix}
a & b \\
0 & 1    
\end{pmatrix} \mid a, b \in \mathbb{R}, a \ne 0\right\}, 
\quad H = 
\left\{
\begin{pmatrix}
1 & c \\
0 & 1
\end{pmatrix}
\mid c \in \mathbb{R} 
\right\}.   
\]

$G$ is subset of $GL_2(\mathbb{R})$ and $H$ is a subgroup of $G$.
\begin{description}
\item[$\bm{H \normal G}$.] First we'll show that $H$ is normal
to $G$. Thus let $x \in G$, so that 
$
x =         \begin{pmatrix}
a & b \\
0 & 1    
\end{pmatrix}
$
for some $a, b \in \mathbb{R}$ where $a \ne 0$. Now let $h \in
H$ so that 
$
h = \begin{pmatrix}
1 & c \\
0 & 1    
\end{pmatrix}
$
for some $c \in \mathbb{R}$. Then observe that 

\begin{align*}
xhx^{-1} &= 
\begin{pmatrix}
a & b \\
0 & 1    
\end{pmatrix}
\begin{pmatrix}
1 & c \\
0 & 1    
\end{pmatrix}
\begin{pmatrix}
1/a & -b/a \\
0 & 1    
\end{pmatrix}\\
&= \begin{pmatrix}
a & b \\
0 & 1    
\end{pmatrix}
\begin{pmatrix}
1/a & -b/a + c \\
0 & 1    
\end{pmatrix}\\
&=
\begin{pmatrix}
1 & (-b + ca) + b \\
0 & 1    
\end{pmatrix}\\
&= \begin{pmatrix}
1 & ca \\
0 & 1    
\end{pmatrix} \in H.
\end{align*}

Therefore, we have that $xhx^{-1} \in H$ for all $H$, which
implies that $H$ is a normal subgroup of $G$. 

\item[$\bm{G/H}$ is abelian.] Now we'll show that $G/H$ is an
abelian group. Firstly, what does it mean for a quotient group to
abelian? Well, it would mean that for any $x, y \in G$ we have
that 

\[
(Hx)\cdot(Hy) = (Hy)\cdot(Hx).
\]

Or, in other words, 

\[
H(xy) = H(yx).   
\]

Thus we need some kind of set equality to be happening. Thus
consider $h =             \begin{pmatrix}
1 & c \\
0 & 1    
\end{pmatrix}$, where again $x \in \mathbb{R}$, and suppose $x =             \begin{pmatrix}
a_x & b_x \\
0 & 1    
\end{pmatrix}$ and $y =             \begin{pmatrix}
a_y & b_y \\
0 & 1    
\end{pmatrix}$ where $a_x,a_y,b_x,b_y \in \mathbb{R}$ and $a_y,
a_x \ne 0$. Then observe that 

\begin{minipage}{0.40\textwidth}

\begin{align*}
hxy &= 
\begin{pmatrix}
1 & c \\
0 & 1    
\end{pmatrix}
\begin{pmatrix}
a_x & b_x \\
0 & 1    
\end{pmatrix}
\begin{pmatrix}
a_y & b_y \\
0 & 1    
\end{pmatrix}\\
&= 
\begin{pmatrix}
1 & c \\
0 & 1    
\end{pmatrix}
\begin{pmatrix}
a_xa_y & a_xb_y + b_x \\
0 & 1    
\end{pmatrix}\\
&= 
\begin{pmatrix}
a_xa_y & a_xb_y + b_x + c \\
0 & 1    
\end{pmatrix}
\end{align*}

\end{minipage}
\hfill
\begin{minipage}{0.5\textwidth}

\begin{align*}
hyx &= 
\begin{pmatrix}
1 & c \\
0 & 1    
\end{pmatrix}
\begin{pmatrix}
a_y & b_y \\
0 & 1    
\end{pmatrix}
\begin{pmatrix}
a_x & b_x \\
0 & 1    
\end{pmatrix}\\
&= 
\begin{pmatrix}
1 & c \\
0 & 1    
\end{pmatrix}
\begin{pmatrix}
a_ya_x & a_yb_x+ b_y \\
0 & 1    
\end{pmatrix}\\
&= 
\begin{pmatrix}
a_ya_x & a_yb_x + b_y + c \\
0 & 1    
\end{pmatrix}.
\end{align*}

\end{minipage}
\textcolor{purple}{Note that the (1,1) entry in both matrices are
equal; that is, $a_xa_y = a_ya_x$ since they are members of
$\mathbb{R}$.}
Therefore, we see that 

\begin{align*}
Hxy = 
\left\{ 
\begin{pmatrix}
a_xa_y & a_xb_y + b_x + c \\
0 & 1    
\end{pmatrix}
\mid 
a_x,a_y,b_x,b_y, c \in \mathbb{R}, a_x, a_y \ne 0
\right \}\\
Hyx = \left\{
\begin{pmatrix}
a_xa_y & a_yb_x + b_y + c \\
0 & 1    
\end{pmatrix}.
\mid 
a_x,a_y,b_x,b_y, c \in \mathbb{R}, a_x, a_y \ne 0
\right\}.
\end{align*}

Since $b_x, b_y, c$ are arbitrary members of $\mathbb{R}$, we can
replace their sums with another arbitrary $c', c'' \in \mathbb{R}$.
Then we see that 

\begin{align*}
Hxy = 
\left\{ 
\begin{pmatrix}
a_xa_y & a_xb_y +c' \\
0 & 1    
\end{pmatrix}
\mid 
a_x,a_y,b_y, c' \in \mathbb{R}, a_x, a_y \ne 0
\right \}\\
Hyx = \left\{
\begin{pmatrix}
a_xa_y & a_yb_x + c'' \\
0 & 1    
\end{pmatrix}
\mid 
a_x,a_y,b_x,c'' \in \mathbb{R}, a_x, a_y \ne 0,
\right\}.
\end{align*}

After cleaning up the sets, we can now see they are equal, which
wasn't as obvious as it was before. They're equal because their
criteria for set memberships are identical; they just have
different variables, but that of course does not change their
members. Therefore we see that $Hxy = Hyx$ for all $x, y \in G$,
which proves that $G/H$ is an abelian group, even though $G$ nor
$H$ are abelian. 

\end{description}




<script src="../../mathjax_helper.js"></script>