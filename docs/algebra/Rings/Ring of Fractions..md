#2.6. Ring of Fractions.
As rings were modeled based on the behavior of the integers,
we can hypothesize that it is possible to generalize the
construction of $\mathbb{Q}$ from 
$\mathbb{Z}$. Such a construction is possible, and studying it
will utilize all of the concepts introduced up to this point. 
However, there is a very subtle issue that
when we try to do this. The main issue is in defining
$\dfrac{a}{b}$ when $b \ne 0 $ doesn't have an inverse.


<span style="display:block" class="definition">
Say $(R, +, \cdot)$ is a commutative ring with $1 \ne 0$.
We say that $D \subset R$ is a **multiplicative set**
if 

* [1.] $1 \in D$


* [2.] $x \cdot y \in D$ for all $x, y \in D$.  



Condition 1. is not really crucial. We just need to make
sure that $D$ is nonempty. 
</span>
An example of a multiplicative set includes $D = \{1\}$. A
less boring example is $D = \{2n+1 \mid n \in \mathbb{Z}\}$,
since the product of two odd integers is odd. And the maximal
example we can come up with for any ring is to set $D = R$. 

Now we introduce a proposition regarding a relation on
elements. 

<span style="display:block" class="proposition">
Define a relation $\sim$ on $R \times D$ by saying $(a, b) \sim
(c,d)$ whenever we can find $x \in D$ such that 

\[
x\cdot (a\cdot d - b \cdot c) = 0.
\]

Then $\sim$ is an equivalence relation on $R \times D$. 
</span>

<span style="display:block" class="proof">
We must show the three properties of an equivalence
relation. 
\begin{description}
\item[Reflexivity.] $(a, b)\sim(a, b)$. Choosing $x = 1$,
then we see that 

\[
x\cdot(a\cdot b - a\cdot b) = 0.  
\]


\item[Symmetry.] $(a, b) \sim (c, d)$ if and only if $(c,
d) \sim (a, b)$. 

Say $(a, b) \sim (c, d)$. We can find an $x \in D$ such
that 

\[
x\cdot(a\cdot d - b \cdot c) = 0.
\]

Now consider 

\[
x\cdot (c\cdot b - d \cdot a) = (-1)\cdot[x\cdot (a\cdot d - b \cdot c)] = 0
\]

This shows that $(c, d) \sim (a, b)$ 

\item[Transitivity.] If $(a, b) \sim (c, d)$ and $(c, d) \sim
(e, f)$, then $(a, b) \sim (e, f)$. 

We can find $x, y \in D$ such that 

\[
x \cdot (a \cdot d - b \cdot c) = y \cdot (c \cdot f - d \cdot e) = 0.
\]

Denote $z = d \cdot x \cdot y$. This is an element of
$D$ since it is a multiplicative set. Now 

\begin{align*}
z \cdot (a \cdot f - b \cdot e) & = yfxad - yfxbc - xbycf - xbyde\\
& = 
y \cdot f [x \cdot (a \cdot d  - b \cdot c)] + x\cdot b[y \cdot(c \cdot f - d \cdot e)]\\
& = 0.
\end{align*}

\end{description}
</span>
\noindent We'll define the collection of all equivalence classes by the
set 

\[
D^{-1}R = \left\{\frac{a}{b} \mid a \in R, b \in D  \right\}
\]

where 

\[ 
\dfrac{a}{b} = \big\{ (c, d) \in R \times D \mid (a ,
b) \sim (c, d) \big\}.
\]

\textcolor{MidnightBlue}{**Why are we doing this?** Let's say that $R = \mathbb{Z}$, and
$D = \mathbb{Z}\setminus\{0\}$. Then $D$ is a multiplicative
set and $(a, b) \sim (c, d) \text{ if and only if } ab - bc = 0$. As an
example, $(1, 2) \sim (2, 4)$ implies that in our set,
$\dfrac{1}{2} = \dfrac{2}{4}$. In other words, we're basically
saying we don't care whether or not the fraction is in reduced
terms.}


<span style="display:block" class="proposition">
Let $R$ be a ring with identity $1 \ne 0$. 


* [1.] $D^{-1}R$ is a commutative ring 


* [2.] $D^{-1}R = \left\{\dfrac{0}{1}\right\}$ is
the trivial ring if and only if $0 \in D$.


* [3.] The units $(D^{-1}R)^\times$ contains 

\[
D^{-1}D = \left\{ \frac{a}{b} \mid a, b \in D \right\}.
\]




</span>


<span style="display:block" class="proof">

* [1.]
Define addition and multiplication. 

<span style="display:block" class="lemma">
The following is well-defined. 

\begin{align*}
+: D^{-1}R \times D^{-1}R &\longrightarrow D^{-1}R\\
\frac{a}{b}, \frac{c}{d} &\longmapsto \frac{ad - bc}{bd}
\end{align*}

</span>
Suppose that $(a_1, b_1) \sim (a_2, b_2)$ and $(c_1, d_1)
\sim (c_2, d_2)$. We can find $x, y \in D$ such that 

\[
x \cdot (a_1b_2 - a_2b_1) = y \cdot[c_1d_2 - c_2d_1] = 0.
\]

Denote $z = xy \in D$. Then 

\begin{align*}
z[(a_1d_1 + c_1b_1)(b_2d_2) - (a_2d_2 + c_2b_2)(b_1d_1)]
\\=
yd_1d_2[x(a_1b_2 - a_2b_1)]
+ xb_2b_1[y(c_1d_2 - c_2d_1)] = 0.
\end{align*}

Hence $(a_1d_1 + c_1b_1, b_1d_1) \sim (a_2d_2 + c_2d_2,
b_2d_2)$. 


<span style="display:block" class="lemma">
The following is well-defined:

\begin{align*}
\cdot: D^{-1}R \times D^{-1}R &\longrightarrow D^{-1}R\\
\frac{a}{b}, \frac{c}{d} &\longmapsto \frac{ac}{bd}
\end{align*}

</span>
Again, say that $(a_1, b_1) \sim (a_2, b_2)$ and $(c_1, d_1)
\sim (c_2, d_2)$. Denote $z = xy \in D$. Then 

\begin{align*}
z[(a_1c_1) \cdot(b_2d_2) - (a_2c_2)(b_1d_1)]
=
yc_1d_2[x(a_1b_2 - a_2b_1)]
+ 
xa_2b_1[y(c_1d_2 - c_2d_1)]
= 0
\end{align*}

Hence $(a_1c_1, b_1d_1) \sim (a_2c_2, b_2d_2)$. Showing
that $D^{-1}R$ is a commutative ring with these properties
is straightforward. 



* [2.] We show that $D^{-1}R$ is trivial if and only
if $0 \in D$. 

Say $D^{-1}R = \left\{\dfrac{0}{1}\right\}$. Since $1 \in R$, we see that
$\dfrac{1}{1} \in D^{-1}R$ so $\dfrac{1}{1} = \dfrac{0}{1}$. By
definition, we can find an $x \in D$ such that 

\[
x \cdot (1 \cdot 1 - 0 \cdot 1) = 0
\]

so $x = 0$. Hence $0 \in D$. 

Now suppose $0 \in D$. Pick any $\dfrac{a}{b} \in D^{-1}R$.
For $x = 0$, we have that $x \cdot (a \cdot 1 - b \cdot 0) =
0$. Hence we see that $\dfrac{a}{b} = \dfrac{0}{1}$, so that
$D^{-1}R = \left\{\dfrac{0}{1}\right\}$.



* [3.] If $a, b \in D$, then $\dfrac{a}{b} \in
D^{-1}R$ so that $\dfrac{a}{b} \cdot \dfrac{b}{a} =
\dfrac{1}{1}$. Hence $\dfrac{a}{b} \in (D^{-1}R)^{*}$. That
is, $DD^{-1} \subset (D^{-1}R)^*$. 



</span>
$D^{-1}R$ is the ring of fractions of $R$ by $D$. We also
remark that if $D \subset R \setminus \{0\}$, then $D^{-1}R$
is a commutative ring with $\dfrac{1}{1} \ne \dfrac{0}{1}$.
Finally, we also note that the elements of $D$ are invertible
in $D^{-1}R$. We want to construct $D^{-1}R$ as the "smallest"
ring such that $D$ is invetible.

Next we introduce a theorem. 

<span style="display:block" class="theorem">
Let $(R, +, \cdot)$ be a commutative ring with $1 \ne 0$, and $D
\subset R$ be a multiplicative set. Denote $S = D^{-1}R = \{a/b
\mid a \in R, b \in D\}$ as the \textit{ring of fractions of $R$
by $D$}. Then 
\begin{description}
\item[1.] Let $I$ be an ideal of $R$. Then $D^{-1}I =\{x/b
\mid x \in I, b \in D\}$ is an ideal of $S$. (This is called
the *extension* of $I$ to $S$.)

\item[2.] Let $J$ be an ideal of $S$. Define the ring
homomorphism $\pi: R \to S$ which sends $r \mapsto r/1$. Then
the preimage $\pi^{-1}(J)$ is an ideal of $R$. (This is
called the *contraction* of $J$ to $R$.)

\item[3.] For any ideal $J$ of $S$ we have that $D^{-1}[\pi^{-1}(J)] = J$.
(That is, "extension is the left inverse of
contraction.")
\end{description}
</span>


<span style="display:block" class="proof">
\begin{description}
\item[1.] To show this we proceed as follows. \\
\underline{$\bm{D^{-1}I} $ **is nonempty.**}\
\[1.2ex]
Observe that since $D$ is a multiplicative set we have that $1
\in D$. Hence if $I$ is nonempty, then $\dfrac{i}{1} \in
D^{-1}I$ for all $i \in I$. Hence it is nonempty.
\\

\noindent\underline{$\bm{a - b \in D^{-1}I}$ **if** $\bm{a, b
\in D^{-1}I}$.}\\[1.2ex]
Let $a, b \in D^{-1}I$. Then $a = \dfrac{i_1}{d_1}$ and $b =
\dfrac{i_2}{d_2}$. Then observe that 
\[
a - b = \dfrac{i_1d_2 - i_2d_1}{d_1d_2}.
\]

Since $I$ is an ideal, $i_1d_2. i_2d_1 \in I$. Therefore their
difference $i_1d_2 - i_2d_1 \in I$. Since $D$ is a
multiplicative set we have that $d_1d_2 \in D$. Hence we see
that $\dfrac{i_1d_2 - i_2d_1}{d_1d_2} \in D^{-1}I$, so that $a
- b \in D^{-1}I$ whenever $a, b \in D^{-1}I$. 

\noindent\underline{$\bm{s\cdot a\in D^{-1}I}$ **if** $\bm{s
\in S, a \in D^{-1}I}$}\
\[1.2ex]
Let $s \in S$ and $a \in D^{-1}I$. Then $s = \dfrac{r}{d}$ and
$a = \dfrac{i}{d'}$ for $r \in R, i \in I, d,d' \in D$. 
Hence observe that 

\begin{align*}
s\cdot a = \dfrac{r}{d} \cdot \dfrac{i}{d'}
= \dfrac{ri}{dd'}.
\end{align*}

Since $I$ is an ideal, we see that $ri \in I$. Since $D$ is a
multiplicative set, we also see that $dd' \in D$. Hence, we
see that $\dfrac{ri}{dd'} \in D^{-1}I$, so that $s\cdot a \in
D^{-1}I$ if $s \in S$ and $a \in D^{-1}I$.\\
Thus in total we have that $D^{-1}I$ is an ideal of $S$.

\item[2.]
To show this is an ideal, we proceed as follows.\\
\underline{$\bm{\pi^{-1}(J)} $ **is nonempty.**}\\[1.2ex]
Observe that if $J$ is nonempty, then $\dfrac{0}{1} \in J$. This
is because if $J$ is an ideal, then $sj \in J$ for all $s \in
S$ and $j \in J$. Hence let $s = \dfrac{0}{1}$, and $j = \dfrac{r}{d} \in
J$. Then 
\[
sj \in J \implies \dfrac{0}{1}\cdot \dfrac{r}{d} \in J \implies \frac{0}{d} \in J.  
\]

Now observe that $(0, 1) \sim (0, d)$ for any $d \in D$. This is
because 

\[
x \cdot(0 \cdot d - 0 \cdot 1) = x \cdot (0) = 0
\]

is automatically satisfied by any $x \in D$.
Hence, $(0, 1) \sim (0, d)$. Now we argue that

\[
\pi^{-1}(J) = \{r \in R \mid \pi(r) \in J\}.
\]

is nonempty. Observe that $\pi\left(\dfrac{0}{1}\right) =
\dfrac{0}{1}$, which we know is true because $\pi$, as a ring
homomorphism, is also a group homomorphism between the abelian
groups $R$ and $S$ (i.e., the abelian groups we get when we remove
the multiplicative ring stucture on them). We know from group theory
that group homomorphisms map additive zero elements from one group
to the additive zero element of the other group, so that $\pi\left(\dfrac{0}{1}\right) =
\dfrac{0}{1}$.

As we just showed, $\dfrac{0}{1} \in J$. Since
$\pi\left(\dfrac{0}{1}\right) = \dfrac{0}{1}$,
we see that $\dfrac{0}{1} \in \pi^{-1}(J)$, so it is nonempty. 

\noindent\underline{$\bm{a - b \in \pi^{-1}(J)}$ **if** $\bm{a, b
\in \pi^{-1}(J)}$.}\
\[1.2ex]
Suppose $a, b \in \pi^{-1}(J)$. Then $\pi(a) = \dfrac{a}{1}$ and
$\pi(b) = \dfrac{b}{1}$ are both members of $J$. Since $J$ is an
ideal, we know that on one hand,

\begin{align*}
\frac{a}{1}, \frac{b}{1} \in J \implies \frac{a}{1} - \frac{b}{1} \in J
\implies \frac{a - b}{1} \in J.
\end{align*}

On the other hand, observe that $\pi(a - b) = \dfrac{a - b}{1}$,
which we just showed is inside $J$. Therefore $a - b \in
\pi^{-1}(J)$ when $a, b \in \pi^{-1}(J)$, which is what we set out
to show.

\noindent\underline{$\bm{r\cdot a \in \pi^{-1}(J)}$ **if**
$\bm{r \in R, a \in \pi^{-1}(J)}$.}\\[1.2ex]
Now suppose that $r \in R$ and $a \in \pi^{-1}(J)$. Then $\pi(a) =
\dfrac{a}{1} \in J$ by definition. Hence, observe that 
\[
\pi(r \cdot a) = \dfrac{ra}{1} = \frac{r}{1} \cdot \frac{a}{1}.
\]

Observe that since $J$ is an ideal of $S$, $\frac{r}{1} \cdot \frac{a}{1}
\in R$. Hence we see that $\pi(r \cdot a) \in J$, so that $r \cdot
a \in \pi^{-1}(J)$ whenever $r \in R$, $a \in \pi^{-1}(J)$, as
desired. 

\item[3.] We can prove equality between the sets by demonstrating
mutual subset properties. \\
\noindent\underline{$\bm{D^{-1}[\pi^{-1}(J)] \subset J}$}\
\[1.2ex]
Consider any $\frac{a}{b} \in D^{-1}[\pi^{-1}(J)]$ where by definition $a \in
\pi^{-1}(J)$ and $b \in D$. 

Since $a \in \pi^{-1}(J)$ we know that $\pi(a) = \dfrac{a}{1} \in
J$. Consider the element $\dfrac{1}{b} \in S$. Since $J$ is an
ideal of $S$, we know that $sj \in J$ for all $s \in S, j \in J$.
Set $j = \dfrac{a}{1}$ and $s = \dfrac{1}{b}$, and observe that 
\[
sj \in J \implies \frac{1}{b} \cdot \frac{a}{1} = \frac{a}{b} \in J.
\]

Hence we have that $D^{-1}[\pi^{-1}(J)] \subset J$. 
\\
\noindent\underline{$\bm{J \subset D^{-1}[\pi^{-1}(J)]}$}\
\[1.2ex]
Now consider any $j = \dfrac{a}{b} \in J$. To prove this direction,
we just need to show that $a \in \pi^{-1}(J)$ since $b$ is already
a member of $D$. And to prove that, we just need to show that
$\dfrac{a}{1} \in J$. Hence we formalize our claim:

**Claim:** $\dfrac{a}{b} \in J \implies \dfrac{a}{1} \in J$.\\
To show this, observe that $\dfrac{b}{1} \in S$ and since $J$ is an
ideal, 
\[
\frac{b}{1} \cdot \frac{a}{b} \in J \implies \frac{ab}{b} \in J.
\]

Now observe that $(ab, b) \sim (a, 1)$, since 

\[
x \cdot (ab\cdot 1 -b\cdot a) = x \cdot (ab - ab) = 0
\]

is satisfied by any choice of $x \in D$. Therefore, $\dfrac{ab}{b}
= \dfrac{a}{1}$, and since $\dfrac{ab}{b} \in J$ we have that
$\dfrac{a}{1} \in J$. 

Finally, since $\dfrac{a}{1} \in J$, we see that $a \in
\pi^{-1}(J)$. Therefore, $\dfrac{a}{b} \in D^{-1}[\pi^{-1}(J)]$, so
that $J \subset D^{-1}[\pi^{-1}(J)]$ as desired. 

With both directions, we can then conclude that
$D^{-1}[\pi^{-1}(J)] = J$, which is what we set out to show.
\end{description}
</span>


<span style="display:block" class="theorem">
Let $(R, +, \cdot)$ be a commutative ring with $1 \ne 0$, and $D
\subset R$ be a multiplicative set. Let $P$ be an ideal of $R$
with the property that if $d \cdot x \in P$ for $d \in D$ then $x
\in P$. Then the following are equivalent.

* [i.] $P$ is a prime ideal of $R$ with $P \cap D = \varnothing$


* [ii.] $D^{-1}P$ is a prime ideal of the ring of fractions
$D^{-1}R$.  



</span>


<span style="display:block" class="proof">
\begin{description}
\item[$\bm{i \implies ii}$.]
Suppose $P$ is disjoint with $D$. Then we show that $D^{-1}P$
is a prime ideal of $D^{-1}R$.\\
\noindent\underline{$\bm{D^{-1}P}$** is nonempty.**}\
\[1.2ex]
If $P$ is nonempty, then since $1 \in D$, we know that
$\dfrac{p}{1} \in D^{-1}P$ for each $p \in P$. Hence, it is nonempty.

\noindent\underline{$\bm{a, b \in D^{-1}P \implies a - b \in D^{-1}P}$.}\\[1.2ex]
Suppose $a, b \in D^{-1}P$. Write $a = \dfrac{p_1}{d_1}$ and
$b = \dfrac{p_2}{d_2}$. Then we see that 
\[
a - b = \frac{p_1d_2 - p_2d_1}{d_1d_2}.
\]

Observe that $p_1d_2, p_1d_2 \in P$ since $P$ is an ideal of
$R$. Hence, $p_1d_2 - p_2d_1 \in P$, and since $d_1d_2 \in D$,
we have that $a - b \in D^{-1}P$ whenever $a, b \in D^{-1}P$.

\noindent\underline{$\bm{r\cdot a \in D^{-1}P}$ ** for **
$\bm{r \in  D^{-1}R, a \in D^{-1}P}$.}\
\[1.2ex]
Consider an element $a = \dfrac{p}{d} \in
D^{-1}{P}$ and $r = \dfrac{r'}{d'} \in D^{-1}R$. Then 
\[
r \cdot a = \frac{r'}{d'} \cdot \frac{p}{d} = \frac{r'p}{d'd}.
\]

Since $P$ is a prime ideal, we see that $r'p \in P$ and $d'd
\in D$ as it is a multiplicative set. Therefore it is an
ideal. 

\noindent\underline{$\bm{a\cdot b \in D^{-1}P \implies a \in D^{-1}P}$
**or** $\bm{b \in D^{-1}P}$} \
\[1.2ex]
Let $a = \dfrac{r_1}{d_1}$ and $b = \dfrac{r_2}{d_2}$ where
$r_1,r_2 \in R$ and $d_1,d_2 \in D$. Now 
\[
a \cdot b \in D^{-1}P \implies \frac{r_1r_2}{d_1d_2} \in D^{-1}P.
\]

Then we see that $r_1r_2 \in P.$ Since $P$ is a prime ideal
disjoint with $D$, we see that $r_1 \in P$ or $r_2 \in P$ and
$r_1,r_2 \not\in D$ by assumption. Therefore, we see that $a
\in D^{-1}P$ or $b \in D^{-1}P$, so that $D^{-1}P$ is a prime
ideal. 

\item[$\bm{ii \implies i}$.] 
Suppose $D^{-1}P$ is a prime ideal of $D^{-1}R$. 
\
\[1.2ex]
\underline{$\bm{P}$ **is nonempty**.}\\[1.2ex]
If $D^{-1}P$ is nonempty, then since $D$ at least contains $1$, there exists at least one
$\dfrac{a}{1} \in D^{-1}P$ where $a \in P$.
Hence we see that $P$ is nonempty.
\\[1.2ex]
\underline{$\bm{a,b \in P \implies a - b \in P}$.}\\[1.2ex]
Suppose $a, b \in P$. Then we see that 
$\dfrac{a}{1}, \dfrac{b}{1} \in D^{-1}P$. Hence,
\[
\frac{a}{1} - \frac{b}{1} \in D^{-1}P \implies 
\frac{a - b}{1} \in P.
\]

Since $\dfrac{a - b}{1} \in D^{-1}P$, we see that $a - b \in
P$. Hence $a, b \in P \implies a - b \in P$.
\
\[1.2ex]
\underline{$\bm{rp \in P}$ **if** $\bm{r \in R, p \in P}$.}\\[1.2ex]
Consider $\dfrac{p}{1} \in D^{-1}P$ and $\dfrac{r}{1} \in
D^{-1}R$ for any $r \in R$. Then since $D^{-1}P$ is an ideal,
\[
\frac{r}{1} \cdot \frac{p}{1} \in D^{-1}P \implies \frac{rp}{1} \in D^{-1}P.
\]

Thus we see that $rp \in P$. Therefore $r \in R, p \in P
\implies rp \in P$. 
\
\[1.2ex]
\underline{$\bm{D \cap P = \varnothing}.$}\\[1.2ex]
Suppose that $D \cap P \ne \varnothing$. Then there exists a
$p \in P$ where $p \in D$. Hence, observe that $\dfrac{p}{p} =
\dfrac{1}{1} \in D^{-1}P$. Then since $D^{-1}P$ is an ideal of
$D^{-1}R$, we see that
$pr \in D^{-1}P$ for any $p \in D^{-1}P$ and $r \in D^{-1}R$. Thus
see that for any $\dfrac{a}{b} \in D^{-1}R$, 
\[
\frac{1}{1} \cdot \frac{a}{b} \in D^{-1}P \implies \frac{a}{b} \in D^{-1}P
\]

which shows that $D^{-1}P = D^{-1}R$. Hence, if we want
$D^{-1}P$ to be a proper ideal, we need that $D \cap P = \varnothing.$
\
\[1.2ex]
\underline{$\bm{ab \in P \implies a \in P}$ **or** $\bm{b\in P}$}\\[1.2ex]
Suppose that $a = \dfrac{p_1}{d_1}$ and $b =
\dfrac{p_2}{d_2}$ such that 
\[
\frac{p_1}{d_1}\cdot \frac{p_2}{d_2} \in D^{-1}P \implies \frac{p_1}{d_1} \text{ or } \frac{p_2}{d_2} \in D^{-1}P.
\]

Since $\dfrac{p_1}{d_1}\cdot \dfrac{p_2}{d_2} \in D^{-1}P$, we
have that $\dfrac{p_1p_2}{d_1d_2} \in D^{-1}P$, which implies
that $p_1p_2 \in P$. 

The fact that $\dfrac{p_1}{d_1} \in D^{-1}P$ or
$\dfrac{p_2}{d_2} \in D^{-1}P$ implies that $p_1 \in P$ or
$p_2 \in P$. Hence we see that $p_1p_2 \in P \implies p_1 \in
P$ or $p_2 \in P$, which proves that $P$ is a prime ideal.
\end{description}
With both directions proven, we can conclude that the two given
statments are in fact equivalent.
</span>

**Localization.**
The construction we have been implementing relates to a concept
as localization, which we define as follows. 

<span style="display:block" class="definition">
Let $(R, +, \cdot)$ be a commutative ring with identity $1 \ne
0$. Let $S \subset R$ and define $D = R - S$. We define the
**localization** of 
$R$ at $S$ as the ring of fractions 

\[
R_P = D^{-1}R = \left\{ \frac{r}{d} \mid r \in R, d \in D \right\}.
\]

</span>
It turns out that if we localize at a prime ideal, nice things
happen. Specifically, the localization contains a unique maximal
ideally. 

Generally, rings do not have unique maximal ideals, although the
definition of a maximal ideal can often confuse people. For
example, consider the ring $\mathbb{Z}$. Then for any prime $p$,
we see that $p\mathbb{Z}$ is a maximal ideal; given the infinitude
of the primes, we have infinitely many maximal primes. 

For rings that do have a unique, maximal ideal, we give them a
special name. 


<span style="display:block" class="definition">
Let $(R, +, \cdot)$ be a ring. If $R$ has a unique, maximal
ideal $M$, then we say that $R$ is a **local ring**.
</span>


<span style="display:block" class="theorem">
Let $(R, +, \cdot)$ be an integral domain, and $P$ be a prime
ideal of $R$. 
\begin{description}
\item[1.] The set $D = R - P$ is a *multiplicative set*.

\item[2.] The *localization* of $R$ at $P$, $R_P =
D^{-1}R$, is an integral domain. 

\item[3.] The ring $R_P$ is a *local* ring i.e., $R_P$ has
a unique maximal ideal $M_P$.
\end{description}
</span>


<span style="display:block" class="proof">
\begin{description}
\item[1.] First we show that $1 \in D$. 
Suppose $P$ is a prime ideal such that $R \ne P$. Then
observe that $1 \not\in P$. For if $1 \in P$, then for any $r
\in R$, we'd see that 

\[
1 \cdot r = r \in P.
\]

Since $r$ is arbitrary, we'd have that $R = P$, a
contradiction. Therefore, we see that $1 \in R - P = D$, which
proves the first property.

Now we show $x, y \in D \implies xy \in D$. Since $P$ is a prime ideal, we know that
$p_1p_2 \in P \implies p_1 \in P$ or $p_2 \in P$. Hence the reverse
negative of the statement is true: if $p_1 \not\in P$ and $p_2
\not\in P$ then $p_1p_2 \not\in P$. 

Therefore for any $x, y
\in D =R - P$, we see that $xy \not\in P$. Hence $xy \in D$, which
proves the second property.

\item[2.] Since we already know that $R_P$ is a commutative ring,
it suffices to show that there are no zero divisors. Suppose
on the contrary that $\dfrac{a}{b}, \dfrac{c}{d} \in R_P$ are
zero divisors of each other. Hence, $a \ne 0$ and $c \ne 0$.
Then we see that 

\[
\frac{a}{b} \cdot \frac{c}{d} = \frac{0}{1}.
\]

In this case, we see that there exists a $x \in D$ such that 

\[
x \cdot (ac \cdot 1 - bd \cdot 0) \implies x \cdot(ac).            
\]

Since $a, c \ne 0$, and $R$ is an integral domain, we see that
$ac \ne 0$. But $x$ is also nonzero, while $x \cdot (ac) =
0$. This cannot happen since $R$ is an integral domain, so we
have a contradiction. Therefore there are no zero divisors,
and since $R_P$ is a commutative, this makes it an integral
domain. 

\item[3.] Let $M_P = D^{-1}P$. We'll show that this is our unique,
maximal ideal. 

First observe that in the previous theorem, if $P$ is a prime
ideal and $P \cap D = \varnothing$, then $D^{-1}P$ is a prime
ideal of $D^{-1}R$. 

In our case, $D = R - P$. Hence $P \cap D
= \varnothing$, so we may conclude that $M_P$ is a prime
ideal.

Now we show $M_p$ is maximal. Let $I$ be a proper ideal, i.e. $I
\ne D^{-1}R$, and suppose $I
\not\subset M_P$. That is, there exist an element
$\dfrac{a}{b} \in I$ such that $\dfrac{a}{b} \not\in M_P$.

Since $\dfrac{a}{b} \not\in M_P$, we see that $a \not\in P$.
Hence, $a \in R - P = D$, and of course $b \in D$ as well. Now
consider the element $\dfrac{b}{a}$. Observe that
$\dfrac{b}{a} \not\in M_P$, since $b, a \in D$, as shown
earlier. Since $\dfrac{a}{b} \in I$, we have that 

\[
\dfrac{a}{b}\cdot \dfrac{b}{a} \in I \implies = \dfrac{1}{1} \in I.
\]

Since $I$ contains $\dfrac{1}{1}$, we have that $I = D^{-1}R$.
This is because we see that for any
$\dfrac{c}{d} \in D^{-1}R$, 

\[
\dfrac{c}{d} =\dfrac{c}{d}\cdot\dfrac{1}{1} \in I \implies \dfrac{c}{d} \in I.
\]

Hence, $I = D^{-1}R$. But we assumed $I$ was a proper ideal; thus we
have a contradiction, so we see that $M_P = D^{-1}P$ is in fact maxmial.
Since we assumed $I$ was *any* ideal of $D^{-1}R$, this
also proves that $M_P$ is a unique maximal ideal, since what we
showed is that any other ideal is automatically contained in
$M_P$. 
\end{description}
</span>

As an example, consider the prime ideal $\{0\}$ of the ring
$\mathbb{Z}$. Then the localization of $\mathbb{Z}$ at $\{0\}$ is
given by 

\[
\left\{ \dfrac{a}{b} \mid a \in \ZZ, b \in \ZZ - \{0\}  \right\}
\]

which is just the rational numbers. 


<span style="display:block" class="theorem">
Let $(R, +, \cdot)$ be an integral domain, and $P$ be a prime
ideal of $R$. Let $R_P$ be the localization of $R$ at $P$, and
$M_P$ be its unique maximal ideal. Consider the map $\phi: R/P
\to R_P/M_P$ which sends $r + P \mapsto r/1 + M_P$. 
\begin{description}
\item[1.] $\phi$ is a well-defined, injective ring
homomorphism. 
\item[2.] $\phi$ is an isomorphism if $P$ is a
maximal ideal. 
\end{description}
</span>


<span style="display:block" class="proof">
\begin{description}
\item[1.]     \underline{**Well-defined.**}\\ First observe that this function is
well-defined. Suppose that $r + P = r' + P$; that is, $r = r'
+ p$ for some $p \in P$. Observe that 

\begin{align*}
\phi(r + P) = \frac{r}{1} + M_P
&= \frac{r' + p}{1} + M_P\\
&= \frac{r'}{1} + \frac{p}{1} + M_P\\
&= \frac{r'}{1} + M_P\\
&= \phi(r' + P)
\end{align*}

where in the fourth step we used that fact that $\dfrac{p}{1}
\in M_P$ since $p \in P$, $1 \in D$. Since $\phi(r + P) =
\phi(r' + P)$, we see that
this function is well-defined. 
\\
\\
\underline{**Ring homomorphism.**}\\
We demonstrate that $\phi: R/P \to R_P/M_P$ is a ring
homomorphism. 
\begin{description}
\item[$\bm{\phi(a + b) = \phi(a) + \phi(b)}$.]
Let $a = r + P$ and $b = r' + P$ be elements or $R/P$.
Then

\begin{align*}
\phi(a + b) = \phi\big((r + r') + P \big)
& = \frac{r + r'}{1} + M_P\\
& = \frac{r}{1} + \frac{r'}{1} + M_P\\
& = \left( \frac{r}{1} + M_P\right) + \left(\frac{r'}{1} + M_P\right)\\
& = \phi(a) + \phi(b)
\end{align*}

which is what we set out to show.
\item[$\bm{\phi(ab) = \phi(a)\phi(b)}$.] Again, suppose $a
= r + P$ and $b = r' + P$. Then observe that 

\begin{align*}
\phi(ab) = \phi\big( (r + P)(r' + P) \big) & = \phi(rr' + P)\\
& = \frac{rr'}{1} + M_P\\
& = \left(\frac{r}{1} + M_P\right)\left(\frac{r'}{1} + M_P\right)\\
& = \phi(a)\phi(b)
\end{align*}

which is what we set out to show. 
\end{description}
With these two properties, we have that $\phi: R/P \to
R$ is a ring homomorphism.
\\
\\
\underline{**Injectivity.**}\\
Next we show that this is an injective function. Suppose $r +
P,r' + P \in D^{-1}R$ such that 

\[
\phi(r + P)= \phi(r' + P).
\]

Then we have that $\dfrac{r}{1} + M_P = \dfrac{r'}{1} + M_P$. In
other words, we see that 

\[
\frac{r}{1} = \frac{r'}{1} + \frac{a}{b}
\]

for some $\dfrac{a}{b} \in M_P$. This further implies that
$\dfrac{r}{1} = \dfrac{r'b + a}{b}$. Hence, $(r, 1) \sim (r'b
+ a, b)$. For this equivalence to
occur, there must exist an element $x \in D$ such that 

\[
x\cdot(rb - (r'b + a)) = 0.
\]

Since $R$ is an integral domain, and $x \ne 0$, we require
that $rb = r'b + a$. Rearranging, we see that this implies
that 

\[
rb - r'b = a \implies (r - r')b = a.
\]

The above equality implies that $(r - r')b \in P$; recall that
$\dfrac{a}{b} \in M_P = D^{-1}P$, so that $a \in P$
and $b \in D = R - P$. 

Since $P$ is a prime ideal, we thus see that either $r -
r' \in P$ or $b \in P$. Since we just said that $b \not\in P$,
we must have that $r - r' \in P.$ In other words, 

\[
r - r' = p \implies r = r' + p
\]

for some $p \in P$. Hence, 

\[
r + P = r' + p + P = r' + P.   
\]

Therefore we see that $\phi(r + P) = \phi(r' + P) \implies r +
P = r' + P$, so that $\phi$ is injective.

\item[2.] Suppose $P$ is a maximal ideal (in addition to being
prime). To demonstrate that $\phi: R/P \to R_P/M_P$ is an
isomorphism, it suffices to show that $\phi$ is surjective,
since in the previous step we already showed that it was
injective. 

Since $P$ is a maximal ideal, we see
that $R/P$ is a field. Therefore inverse elements exist, so
for the element $b + P$, there exists an element $b' + P$
where $b' \not\in P$ such that 

\[
(b + P)(b' + P) =  1 + P.
\]

That is, $bb' = 1 + p$ for some $p \in P$. 

Let $r = ab'$, and consider the elements $\dfrac{a}{b} + M_P$
and $\dfrac{r}{1} + M_P$. Since we want $\dfrac{a}{b} + M_P$
to be nontrivial, we let $a, b \in D$ (that is, $a \not\in P$
or else in that case $\dfrac{a}{b} \in M_P$).

Observe that $\dfrac{r}{1} \not\in
M_P$ since $a \in D, b' \in D$, and because $D$ is a
multiplicative set, $r = ab \in D$. Therefore $r \not\in P$,
so that $\dfrac{r}{1} \not\in M_P$. Hence, $\dfrac{r}{1} + M_P
\ne \dfrac{0}{1} + M_P$. 

Now observe that 

\begin{align*}
\left(\frac{a}{b} + M_p\right) - \left(\frac{r}{1} + M_P\right)
& = \left( \frac{a}{b} - \frac{r}{1} \right) + M_P\\
& = \frac{a - br}{b} + M_P\\
& = \frac{a - b(ab')}{b} + M_P\\
& = \frac{a - abb'}{b} + M_P \text{ (by commutativity)}\\
& = \frac{a - a(1 + p)}{b} + M_P\\
& = \frac{-ap}{b} + M_P.
\end{align*}

Since $P$ is an ideal, we see that $-ap \in P$. Therefore,
$\dfrac{-ap}{b} \in M_P$, so that
$\dfrac{-ap}{b} + M_P = \dfrac{0}{1} + M_P$. Thus what we've shown is that 

\[
\left(\frac{a}{b} + M_P\right) - \left( \frac{r}{1} + M_P \right) = \frac{0}{1} + M_P.
\]

which implies that 

\[
\frac{a}{b} + M_P = \frac{r}{1} + M_P.
\]

Thus we see that $\phi(r + P) = \dfrac{r}{1} + M_P =
\dfrac{a}{b} + M_P$. However, $\dfrac{a}{b} + M_P$ was an
arbitrary element of $R_P/M_P$. Hence, we've shown that for
any $\dfrac{a}{b} + M_P$, there exists an $r + P\in R/P$ such that 
$\phi(r + P) = \dfrac{a}{b} + M_P$. In particular, $\phi(0 + P) =
\dfrac{0}{1} + M$. Therefore $\phi$ is surjective,
and as we already showed it is injective, this makes it an
isomorphism. 
\end{description}
</span>





<script src="../../mathjax_helper.js"></script>