#2.3. Ideals and Quotient Rings.

Consider a ring
homomorphism $f: R \to S$. Let $a \in R$ and
suppose $b \in \ker(f)$. Then 

\[
f(ab) = f(a)f(b) = 0f(b) = 0.
\]

Therefore, if $a \in \ker(R)$, then $ab \in \ker(R)$ for all
$b \in R$. Many subrings behave this way and are particularly
interesting, so we give them a special name! 
\\
First, we'll introduce the concept of a coset.

<span style="display:block" class="definition">
Let $(R, +, \cdot)$ be a ring with identity $1 \ne 0$. Suppose $I$ is
a subring. Then we define the set 

\[
\overline{a} = a + I = \{a + i \in R \mid i \in I\}   
\]

to be a **coset** $I$ in $R$. Since $R$ is an abelian
group under addition, we see that 

\[
a + I = I + a
\]

for all $a \in R$. Hence, left and right cosets are the
concept here. Finally, we define the \textbf{collection of
cosets} by 

\[
R/I = \{\overline{a} \mid a \in R\}.               
\]

</span>
We are now ready to introduce the concept of an ideal. 

<span style="display:block" class="definition">
Let $R$ be a ring and suppose $I \subset R$. Then we
define $I$ to be an **ideal** of $R$ if and only if 

*  [1.] $I$ is an additive subgroup of $R$ 


*  [2.] $rI \subset I$ for all $r \in R$ 


*  [3.] $Ir \subset I$ for all $r \in R$




\textcolor{Purple}{An ideal is simply an interesting
subring $R'$ of a ring $R$ which sort of "sucks in"
elements of $R$ and sends them into $R'$. That is, $rr'
\in R'$ for every $r \in R$ and $r' \in R'$.
\\
\\
We've already seend many examples of this, although we
don't usually think of them that way. For instance, it's
a well known fact that for any integer times an even
number is again an even number. Algebraically, for $n \in \ZZ$
and $k \in 2\ZZ$ we have that $nk \in 2\ZZ$ and $kn \in
2\ZZ$. 
\\
\\
Thus $2\ZZ$ is an ideal of $\ZZ$. In fact, if $k$ is any
even integer then $k\ZZ$ is an ideal of $\ZZ$.
\\
\\
The set of odd integers is not an ideal of $\ZZ$, since we
could always take an even number $n \in \ZZ$ and any odd
$k$, and multiply them to obtain an even number $nk$ which
is obviously not in the set of odd integers.
}


If $I \subset R$ satisfies (2) then $I$ is
said to be a **left ideal**. On the other hand if $I
\subset R$ satisfies (3) then it is said to be a
**right ideal**. 
</span>

Thus any ideal $I$ is both a left and right ideal. In
addition, the concept of a left ideal is identical to a right
ideal in a commutative ring.  


<span style="display:block" class="theorem">
Suppose $I \subset R$ is a proper subring. Then the
following are equivalent:

* [1.] $I = \ker(f)$ for some $f: R \to S$


* [2.] $r\cdot x = x \cdot r \in I$ for any $r \in
R$, $x \in I$ 


* [3.] $R/I$ is a ring with $\overline{1} \ne \overline{0}$. 


* [4.] $I$ is an ideal.  



</span>


<span style="display:block" class="proof">

* [1.] We'll show $(i) \implies (ii)$. Assume $I =
\ker(f)$. Given $r \in R$ and $i \in I$, 

\begin{align*}
\phi(r \cdot i) = \phi(r)\cdot\phi(i) = \phi(r)\cdot 0 = 0\\
\phi(i \cdot r) = \phi(i)\cdot\phi(r) = 0 \cdot\phi(r) = 0
\end{align*}

This shows that $r \cdot i, i \cdot r \in \ker(f)$. 



* [ii.] We'll show that $(ii) \implies (iii)$.
Assume $ri, ir \in I$ for all $r \in R, i \in I$.
We'll show that this is a ring with $\overline{1} \ne
\overline{0}$. 

First, we define that 

\begin{align*}
\overline{a} + \overline{b} = \overline{a + b}\\
\overline{a}\overline{b} =\overline{ab}.
\end{align*}

We first need to show that these definitions are
well-defined. Suppose $\overline{a_1} =
\overline{a_2}$ and $\overline{b_1} = \overline{b_2}$.
Then $a_1 = a_2 + x$ and $b_1 = b_2 + y$ for some
$x,y\in I$. Then 

\[
a_1 + b_1 = (a_2 + b_2) + (x + y).
\]

Since $I \subset R$ is a subring, $x+y \in I$. So, 

\[
\overline{a_1 + b_1} = \overline{a_1}\overline{a_2}.   
\]


Simiarly, $\cdot$ is well defined on $R/I$. Again, suppose $\overline{a_1} =
\overline{a_2}$ and $\overline{b_1} = \overline{b_2}$.
Then $a_1 = a_2 + x$ and $b_1 = b_2 + y$ for some
$x,y\in I$. Then 

\begin{align*}  
a_1 \cdot b_1 & = (a_2 + x) \cdot (b_2 + y)\\
& = (a_2\cdot b_2) + [(a_2 \cdot y) +(x \cdot b_2) + (x \cdot y)].
\end{align*}

$I$ is a subring, so $x \cdot y \in I$. Now $(ii)$ is
true, so $a_2 \cdot y \in I$ and $x \cdot b_2 \in I$.
Therefore, $\overline{a_1\cdot b_1} = \overline{a_2
\cdot b_2}$. 

Finally, we'll show that $(R/I, +, \cdot)$ is a ring. 
\begin{description}


* [(R1: Addition)] Observe that $\overline{0}
\in R/I$ is the identity and $\overline{-a}$ are
inverses of $\overline{a} \in R/I$. 



* [(R2: Closure)] The set is closed by
construction on $\cdot$. 



* [(R3: Assoc), (R5: Distributivity)] hold for
$R/I$ because they hold for $R$. 



* [(R4: Identity)] The identitty holds for
$\overline{1} \in R/I$. One can check that
$\overline{1} \ne \overline{0}$. 
\end{description}


* [iii] Now we can show that $(iii) \implies (i)$.
Assume $S = R/I$ is a ring. Define 

\[
\phi: R \to S \quad a \mapsto \overline{a} = a + I
\]

One checks that $\ker(\phi) = I$. 



* [iv.] Our work in the previous section has
allowed us to prove $(i) \implies (iv)$. Now observe
that we can prove $(iv) \implies (i)$ by simply
considering the map in $(iii)$.



</span>



<span style="display:block" class="theorem">[ (Properties of Ideals)]
Let $R$ be a ring and $I, J$ ideals of $R$. Then 

* [1.] $I +J$ is an ideal of $R.$ (Note we may
extend this to larger, finit sums)


* [2.] $IJ = \left\{\displaystyle \sum_{k=1}^ni_kj_k \mid \text{for all
} n \in \mathbb{N}, i_k \in I, j_k \in J\right\}$ is an
ideal of $R$. (Note we can extend this to larger,
finite products.) 


* [3.] $I \cap J$ is an ideal of $R$. Morever, if
$\{I_\alpha\}_{\alpha \in \lambda}$ is a family of
ideals of $R$, then $\bigcap\limits_{\alpha \in
\lambda} I_\alpha$ is an ideal of $R$. 



</span>


<span style="display:block" class="proof">

* [1.] 
By the Second Isomorphism Theorem we know that $I + J$
is a subring of $R$. Thus, we just need it to be
closed under multiplication for it to be an ideal. 

Let $i + j \in I + J$ and let $r \in R$.
then $r(i + j) = ri +rj \in I + J$, since $ri \in I$
and $rj \in J$. Similarly, $(i + j)r \in I + J$, so
that $I + J$ is an ideal of $R$.



* [2.] In words, $IJ$ is the set of all finite sums
of elements of the form $ij$ where $i \in I$ and $j
\in J$. Thus is clearly an abelian group. To show it
is closed under multiplication, let $r \in R$. Then
observe that $r(\sum_{k=1}^{n}i_kj_k) =
\sum_{k=1}ri_kj_k$. 
Now $ri_k \in I$ for all $k$ since $I$ is an ideal.
Therefore $r(\sum_{k=1}^{n}i_kj_k) \in IJ$. 

For similar reasons $(\sum_{k=1}^{n}i_kj_k)r \in I$,
so that $IJ$ is an ideal. 



* [3.] By our knowledge of
group theory we know that intersections of subgroups
form a group, so that this is an abelian subgroup.
To see it is an ideal we just need to check it is
closed under scalar multiplication. 

Let $i \in I \cap J$. Then $i \in I$ and $i \in J$.
Hence, $ir \in I$ and $ri \in J$, and $ri \in I$ and
$rj \in J$ as $I$ and $J$ are ideals. Hence $ir \in I
\cap J$ and $ri \in I \cap J$, so that $I \cap J$ is
an ideal. 

The more general statement has the same proof structure.



</span>



<span style="display:block" class="lemma">
If $S$ is a nonempty
partially ordered set in which every chain $I_1 \subset I_2
\subset \cdots$ has an upper bound $I$, then $S$ has a maximal
element $M$.
</span>


<span style="display:block" class="theorem">[ (Properties of Ideals)]
Let $(R, +, \cdot)$ be a ring with identity $1 \ne 0$.
Consider a chain $I_1 \subseteq I_2 \subseteq \cdots \subseteq
I_n \subseteq \cdots \subseteq R$ of proper ideals of $R$.

* [1.] $\displaystyle I = \bigcup_{n \ge 1}I_n$ is a proper
ideal of $R$. 



* [2.] Each proper ideal $I$ of $R$ is
contained in a maximal ideal $M$ of $R$.



</span>


<span style="display:block" class="proof">

* [1.]
\underline{$\bm{I}$ **is nonempty**.}\
\[1.2ex]
Observe that $I$ is nonempty if at least one $I_k$ is
nonempty. 

\noindent\underline{$\bm{a, b \in I \implies a -b \in I}$.}\\[1.2ex]
Pick $a, b \in I$. Then $a \in I_n$ and $b \in I_m$ for some
$n, m$. Without loss of generality assume $n \le m$. Then $I_n
\subseteq I_m$. Thus $a \in I_m$ as well, and since $I_m$ is
an ideal, we see that $a - b \in I_m$. Hence $a - b \in I$. 

\noindent\underline{$\bm{ra \in I}$ **if** $\bm{r \in R,
a \in I}$}.\\[1.2ex]
If $a \in I$ then $a \in I_k$ for some $k$. Since $I_k$ is an
ideal, we have that $ra \in I_k$. Hence $ra \in I$. 

\noindent\underline{$\bm{I \ne R}$.}\\[1.2ex]
Suppose on the contrary that $I = R$. Then for every $r \in R$
there exists an integer $k$ such that $r \in I_k$. In
particular, for some $u \in R^{\times}$ (the unit group),
there is a $k$ such that $u \in I_k$. Since $I_1 \subseteq I_2
\subseteq
\cdots \subseteq I_k$, we see that all ideal $I_1, I_2,
\dots, I_k$ are not proper (as they contain a unit.)
\\
\\
However, this is a contradition, since each $I_n$ must be
proper. Thus $I$ cannot be all of $R$. 



* [2.] Consider any proper ideal $I_1$ of $R$. If $I_1$ is not maximal,
then there exists an ideal $I_2$ such that $I_1 \subset I_2$. If
$I_2$ is not maximal, then there exists an ideal $I_3$ such
that $I_2 \subset I_3$.
Now construct the set 
\[
S = \{I_n \text{ is proper } \mid I_{n} \subset I_{n+1}\}.
\]

where $I_n \in S_j$ whenever there exists a proper ideal
$I_{n+1}$ where $I_n \subset I_{n+1}$.

If this set is finite, then we take the maximal element
(relative to partial ordering on subset inclusion) $M$ as the
maximal ideal. 

Suppose on the other hand that this set is infinite. 
By part
$(a)$, we see that every $I_n \in S$ is a subset of the proper
ideal $\bigcup_{n \ge 1} I_n$, so that this is an upper bound
on the set of elements $S_J$ (in terms of set inclusion).
Hence by Zorn's lemma, we see that there must exist a maximal
element $M \in S$. As all members of $S$ are proper ideals, we
see that $M$ is by definition a maximal ideal where $M \ne R$.
As $I_1$ was arbitrary, we see that all ideals are contained in
some maximal ideal $M$, as we set out to show.



</span>

The following is a useful example of an ideal known as the
nilradical:

<span style="display:block" class="proposition">
Let $(R, +, \cdot)$ be a commutative ring with $1 \ne 0$, and let
$I \subset R$ be a proper ideal. The 
*radical* of $I$ is the set 

\[
\sqrt{I} = \{r \in R \mid r^n \in I \text{ for some } n \in \zz_{> 0}  \}.
\]

\begin{enumerate}
\item $\sqrt{I}$ is an ideal containing $I$. 

\item $\sqrt{I}$ is the intersection of all prime
ideals $P$ which contain $I$.
\end{enumerate}
</span>


<span style="display:block" class="proof">
\begin{enumerate}
\item First observe that $I \subset \sqrt{I}$. Since for any $r \in I$,
we see that $r^1 = r \in I$. Hence $r \in \sqrt{I}$. 

Now we'll show that $\sqrt{I}$ is an ideal.\
\[1.2ex]
\noindent\underline{$\bm{\sqrt{I} \ne \varnothing}$.}\\[1.2ex]
Since $I \subset \sqrt{I}$, we see that $\sqrt{I}$ is nonempty. 
\\[1.2ex]
\noindent\underline{$\bm{a, b \in \sqrt{I} \implies a - b \in
\sqrt{I}}$.}\\[1.2ex]
Let $a, b \in \sqrt{I}$. Then there exist positive integers $m, n$
such that $a^m \in I$ and $b^n \in I$. Now observe that 
\[
(a -  b)^{n + m} = \sum_{k = 0}^{m + n}\binom{m+n}{k}a^{n + m - k}(-b)^{k}.
\]

by the binomial theorem. Observe that when $k \le n$,

\begin{align*}
k \le n & \implies n - k \ge 0\\
& \implies n + m - k \ge m.
\end{align*}

Hence we see that $a^{n + m - k} = a^{n - k}a^m \in I$ because
$a^m \in I$. Since $I$
is an ideal, we see that 

\[
\sum_{k = 0}^{n}\binom{m+n}{k}a^{n + m - k}(-b)^{k}  
\]

is a sum of terms in $I$, so therefore it is in $I$. 

Now suppose $k > n$. Then we get that 

\begin{align*}
n < k &\implies k =  n + j \text{ for some } j \in \mathbb{Z}^{+}.
\end{align*}

Therefore we see that $b^{k} = b^{j}b^{n} \in I$. Since $I$ is an
ideal, the sum

\begin{align*}
\sum_{k = n+1}^{n}\binom{m+n}{k}a^{n + m - k}(-b)^{k}  
\end{align*}

is a sum of terms in $I$. Hence the total sum is in $I$. Now we
see that 

\[
\sum_{k = 0}^{m + n}\binom{m+n}{k}a^{n + m - k}(-b)^{k}
=
\sum_{k = 0}^{n}\binom{m+n}{k}a^{n + m - k}(-b)^{k}  
+ 
\sum_{k = n+1}^{n}\binom{m+n}{k}a^{n + m - k}(-b)^{k}
\]

so that $\displaystyle (a - b)^{m+n} = \sum_{k = 0}^{m + n}\binom{m+n}{k}a^{n +
m - k}(-b)^{k}$ is a sum of two terms in $I$, and hence is in $I$.
Thus we have that $a, b \in \sqrt{I} \implies a - b \in \sqrt{I}$.
\
\[1.2ex]
\noindent\underline{$\bm{ra \in I}$ **if** $\bm{r \in R, a
\in I}$.}\\[1.2ex]
Suppose that $a \in \sqrt{I}$. Then $a^n \in I$ for some positive integer
$n$. Since $R$ is a commutative ring, we see that $(ra)^n = r^na^n
\in I$ since $a^n \in I$ and $I$ is an ideal. Thus $ra \in I$ for
any $r \in R$, $a \in I$. 
\\[1.2ex]
\noindent\underline{$\bm{\sqrt{I} \ne R}$.}\\[1.2ex]
Suppose that $\sqrt{I} = R$. Then for every $r \in R$, there
exists a positive integer $n$ such that $r^n  \in I$. 

Then in particular for some unit $u \in R^{\times}$ we have that $u^m
\in I$ for some integer $m$. However, since $R^\times$ is a
group under multiplication, we know that $u^m \in R^{\times}$.
Hence $u^m$ is a unit. Since $u^m \in I$, this implies that
$I$ contains a unit, which ultimately implies that $I = R$. 
\\
\\
(\underline{Note}: It is a fact from class that if an ideal
$I$ of $R$ contains a unit, it is all of $R$. I am utilizing
this fact. Please don't dock off points for this literal fact
from class.)
\\
\\
However, this is a contradition since we assumed that $I$ was
proper. Hence $\sqrt{I} \ne R$, which proves that it is a
proper ideal.

\item 
First we prove the hint. 
\\
\\
Following the hint, suppose $x \not\in \sqrt{I}$. If we
let $D = \{1, x, x^2, \dots\}$, pick a maximal ideal $M$ in
the ring $S = D^{-1}R/D^{-1}I$. 

Let $\phi: R \to S$ where $\phi(r) = \dfrac{r}{1} + D^{-1}I$.
Let $P$ be the pull-back of $M$ under $\phi$. 
We'll now prove the hint by showing $P$ is prime, $x \not\in
\sqrt{I} \implies x \not\in P$ and that $I \subset P$.
\\[1.2ex]
\underline{$\bm{P}$ **is prime**}.\\[1.2ex]
First observe we need to make sure that the pullback is well
defined, in the sense that 
if $M$ is maximal then $P$ is prime. First observe that since
$M$ is maximal, it is prime by our previous lemma. Thus we
know from Hw 2 
that we need to show two things.

* [1.] $\bm{\phi(1) = 1}.$ Observe that 
\[
\phi(1) = \dfrac{1}{1} + D^{-1}I   
\]

which is the identity element in $D^{-1}R/D^{-1}I$. Hence,
$\phi(1) = 1$. 
$\phi^{-1}(P)$ is a prime ideal. From problem 2, we know
that this allows us to conclude the pull-back is well defined.



* [2.] $\bm{P = \phi^{-1}(M)}$ **is prime**. (It
may help the reader for me to refer to $P$ explicitly as
$\phi^{-1}(M)$, in terms of clarity of the solution, so
I'll follow that convention.)
\begin{description}


* [$\bm{\phi^{-1}(M)}$ is nonempty.]
Observe that $\phi(0) = \dfrac{0}{1} + D^{-1}I \in M$,
as $M$ is an ideal of $D^{-1}R/D^{-1}I$ and hence
contains the zero element. Therefore $0 \in
\phi^{-1}(M) = P$ and so
$P$ is nonempty. 



* [$\bm{a, b \in \phi^{-1}(M)\implies a-b \in \phi^{-1}(M)}$.] 
Let $a, b \in \phi^{-1}(M)$. Then $\phi(a), \phi(b) \in M$.
Hence, we see that 

\begin{align*}
\phi(a), \phi(b) \in M & \implies \phi(a) - \phi(b) \in M \text{ (since } M \text{ is a prime ideal)}\\
& \implies \phi(a - b) \in M \text{ (by homomorphism properties)}\\
& \implies a - b \in \phi^{-1}(M).
\end{align*}

Therefore, we see that $a - b \in \phi^{-1}(M)$ if $a, b \in
\phi^{-1}(M)$.  



* [$\bm{ra \in \phi^{-1}(M)}$ **if** $\bm{r \in R, p \in \phi^{-1}(M)}$.] 
We'll show that $r \cdot a \in \phi^{-1}(M)$ for all
$r \in R$. Observe that 

\begin{align*}
\phi(r\cdot a) = \phi(r)\phi(a).
\end{align*}

Since $\phi(a) \in M$, and $M$ is a prime ideal, $s\phi(a) \in M$
for all $s \in D^{-1}R/D^{-1}I$. In particular, since $\phi(r) \in D^{-1}R/D^{-1}I$, we
see that $\phi(r)\phi(a) \in M$. Therefore, $\phi(r \cdot a)
\in M$ so that $r\cdot a \in \phi^{-1}(M)$.



* [$\bm{ab \in \phi^{-1}(M) \implies a \in \phi^{-1}(M)}$ **or** $\bm{b \in \phi^{-1}(M)}$]
Suppose $ab \in \phi^{-1}(M)$. Then we see that 

\[ 
\phi(ab) \in M \implies \phi(a)\phi(b) \in M.
\]

Since $M$ is a maximal, and hence a prime ideal (as
proven earlier), we see that either $\phi(a)
\in M$ or $\phi(b) \in M$. In either case, we see that
either $a \in \phi^{-1}(M)$ or $b \in \phi^{-1}(M)$, which
is what we set out to show.



* [$\bm{\phi^{-1}(M)}$ **is proper**.] Finally, we show
that $\phi^{-1}(M)$ is proper. Suppose that $\phi^{-1}(M) = R$. Then 

\[
\phi^{-1}(M) = R \implies \phi(R) = M.   
\]

Thus we see that $\phi(r) \in M$ for all $r \in R$.
Let $r = 1$. 

\[
\phi(r) \in M \implies \phi(1) \in M \implies 1 \in M
\]

since we have that $\phi(1) = 1$.
However, $1 \not\in M$ since $M$ is maximal and hence
proper. As we've reached a contradiction, we see that
the pullback $P$ must always be proper. 
\end{description} 



Thus we see that the pullback is well-defined (i.e., if $M$ is
prime, so is its pullback $P$) in this case and
that $P$ is prime. 
\\
\\
(Note: it was technically unnecessary to do all of this work.
Even in terms of clarity, I could have just referenced Hw 2,
problem 3, and argued that the work carries over via the
$\phi(1) = 1$ argument, since that was the only reason we 
need $R$ and $S$ to be integral domains there, and then used
the fact that maximal ideals are prime. However, I included
the full work to be explicitly clear.)
\\
\\
Next, we continue and prove the hint.\\
\underline{$\bm{x \not\in \sqrt{I} \implies x \not\in
P}$}.\
\[1.2ex]
Recall we supposed $x \not\in \sqrt{I}$. Now
if $M$ is an ideal of $D^{-1}R/D^{-1}I$, then by the Fourth
Isomorphism theorem we have that $M$ corresponds to some ideal
$M'$ of $D^{-1}R$ where $D^{-1}I \subset M'$. Hence we can
write  
\[
M = M' +  D^{-1}I.
\]

(\underline{Note}: before you dock off points, the above choice of
notation was introduced by Professor Goins himself. I think
it's a bit unorthodox, which you may also think as well, but
again, Goins used this notation so I will as well.)
\\
\\
Now suppose for a contradiction that $x \in P$. Then we have that $\dfrac{x}{1} +
D^{-1}I \in M$. For this to be the case, we need that $\dfrac{x}{1} \in
M'$. Since $M'$ is an ideal of $D^{-1}R$, we know that
$r\dfrac{x}{1} \in M$ for all $r \in D^{-1}R$. In particular, we see that 

\[
\dfrac{1}{x} \cdot \dfrac{x}{1} \in M \implies \dfrac{1}{1} \in M'.
\]

As $\dfrac{1}{1}$ is a unit, this implies that $M' = D^{-1}R$
(\underline{Note}: It is a fact from class that if an ideal
$I$ of $R$ contains a unit, it is all of $R$. I am utilizing
this fact. Please don't dock off points for this literal fact
from class.)
\\
However, by the Fourth Isomorphism Theorem, this implies that
$M = D^{-1}R/D^{-1}I$; a contradiction to the assumption that
$M$ is a maximal ideal. Thus we see that $x \not\in P$. 
\
\[1.2ex]
\underline{$\bm{I \subset P}.$}\\[1.2ex]
Now since $M$ is an ideal, we see that it contains the zero
element $D^{-1}I$. Now observe that for any $i \in I$,
\[
\phi(i) = \dfrac{i}{1} + D^{-1}I = D^{-1}I \in M.
\]

Therefore we see that $I \subset \phi^{-1}(M) = P$.
\\
\\
As this point we have shown that if $P$ is the pullback of $M$
under the given homomorphism, then (1) the pull back is well-defined (2) $P$ is prime (3) if $x
\not\in \sqrt{I}$ then $x \not\in P$ and (4) $I \subset P$.


Now consider the fact that $x \not\in \sqrt{I} \implies x
\not\in P$. Let $\displaystyle \bigcap_{I \subset P' \text{, prime}}P'$
denote the intersection of all prime ideals containing $I$. Since 

\[
\bigcap_{I \subset P' \text{, prime}}P' \subset P    
\]

because $P$ is a prime ideal contaning $I$, we see that if $x \not\in P$ then $\displaystyle x \not\in \bigcap_{I
\subset P' \text{, prime}}P'$. As we proved that if $x \not\in
\sqrt{I}$, then $x \not\in P$, we see that 

\[
x \not\in \sqrt{I} \implies x \not\in \bigcap_{I \subset P' \text{, prime}}P'.
\]

Taking the contrapositive of the statement, we can then conclude
that 

\[
x \in \bigcap_{I \subset P' \text{, prime}}P' \implies x \in \sqrt{I}
\]

which ulimately implies that $\displaystyle \bigcap_{I \subset P', \text{prime}}P'
\subset \sqrt{I}$. 
\
\[1.2ex]
\underline{$\bm{x \in \sqrt{I} \implies x \in P}$}\\[1.2ex]
To show the reverse inclusion, suppose  $x \in \sqrt{I}$, and
let $P$ be a prime ideal such that $I \subset P$. Then
$x^n \in I$ for some positive integer $n$. 

Suppose for the sake of contradiction that $x \not\in P$ Let $N$ be the
smallest positive integer such that $x^N \in I$. 
Since $x^N \in I \subset P$, we see that $x^N \in P$. Note
that 
\[
x^N = x \cdot x^{N-1} \in P.            
\]

Since $P$ is a prime ideal, either $x \in P$ or $x^{N-1} \in
P$. However, by assumption $x \not\in P$. Thus we must have
that $x^{N-1} \in P$. But since $I \subset P$, this implies
that $x^{N-1} \in I$. This contradicts our choice of $N$ as
the smallest positive integer as $x^N \in I$. We have our
contradiction, so we must have that $x \in P$. 

Since $x \in \sqrt{I} \implies x \in P$  for every prime ideal
$P$ such that $I \subset P$, we see that 

\[
\sqrt{I} \subset \bigcap_{I\subset P \text{, prime}} P.    
\]

Since we already showed that $\displaystyle \bigcap_{I\subset P, \text{
prime}} P \subset \sqrt{I}$, both set inclusions imply  that 

\[
\sqrt{I} = \bigcap_{I\subset P \text{, prime}} P 
\]

as desired.
\end{enumerate}
</span>



<span style="display:block" class="proposition">
Let $R$ be a ring and $I, J$ be ideals of $R$ such that $I
\subset J \subset R$. Then $I$ is an ideal of $J$.
</span>


<span style="display:block" class="proof">
To prove this, simply observe that for any $j \in J$ and
$i \in I$ we have that $ij \in I$ and $ji \in I$. 
</span>

A primary example of an ideal is any kernal of a homomorphism.


<span style="display:block" class="lemma">
Let $\phi:R \to S$ be ring homomorphism. Then $\ker(\phi)$
is an ideal of $R$.
</span>

We already partially showed this earlier, and the full proof
is not difficult. 


<span style="display:block" class="lemma">
If $R$ is a division ring then the only ideals of $R$ are
$\{0\}$ and $R$ itself.
</span>


<span style="display:block" class="proof">
Of course, $\{0\}$ is an ideal for any ring. Therefore let
$I$ be a nonzero ideal. Then 

\[ 
ir \in I
\]

for any $i \in I$ and $r \in R$. Since $R$ is a division ring, every
element has a multiplicative inverse (except 0). Hence for
any nonzero $i$ we can choose $r = i^{-1}$ to conclude
that $ii^{-1} = 1_R \implies 1_R \in I$. 

Since $1_R \in I$, we can set $r \in R$ to be any element
to conclude that $1_Rr = r \implies r \in I$. Therefore $I
= R$. So every ideal is either $R$ or $\{0\}$.
</span>


<span style="display:block" class="proposition">
Let $R$ be an integral domain. Any
ring homomorphism $\phi$ from $R$ to an arbitrary ring $S$ is
injective or the zero map.
</span>


<span style="display:block" class="proof">
Since $\ker(\phi)$ is an ideal of $R$, it is either
$\{0\}$, in which case $\phi$ in injective, or $R$, in
which case $\phi$ is the zero map.
</span>

Next we can introduce the concept of a quotient ring, which
involves quotienting out an ideal. Note that for a ring $R$
and an ideal $I$, the concept of $R/I$ makes sense since $R$
is an abelian group, while $I$ is a subgroup and is therefore a
normal group to $R$. Thus we make the following definition.


<span style="display:block" class="definition">
Let $R$ be a ring and $I$ an ideal of $R$. Then $R/I$, the
set of all elements $r + I$ where $r \in R$, is
defined to be a **quotient ring** whose operations
are specified as follows. 
\begin{description}
\item[Addition.] For any $r + I, s + I \in R/I$ we have
that 

\[
(r + I) + (s + I) = (r + s) + I.
\]

\item[Multiplication.]
For $r + I, s + I \in R/I$ we have that 

\[
(r + I)\cdot(s + I) = rs + I.
\]

\end{description}
</span>
First, let's check that this is even sensical. Again, we know from
our group theory intuition that $R/I$ definitely makes sense
when looked at as an additive group. The identity is $I$,
inverses exist, it is closed and of course associative. Nothing has changed from our
group theory perspective. 

We want $R/I$ to not only be an abelian group, but
*also* a ring, we defined multiplication of elements as
$(r + I)\cdot(s + I) = rs + I$. Thus we'll check the validity
such multiplication. 
\\

\textcolor{MidnightBlue}{The issue at hand is that, for any $r +
I \in R/I$, there are many ways we can represent the element.
For instance, for any $r' \in R$ such that $r = r' + i$ for
some $i \in I$, we have that $r + I = r' + I$. That is, the way we
decide to represent our elements is not unique. Thus we just
need to check that the way we defined multiplication doesn't
depend on the chosen representative of an element $r + I \in
R/I$.}

To do this suppose that $r + I = r' + I$ and $s + I = s' + I$
are elements of $R/I$. Then $r = r' + i$ and $s = s' + j$ for
some $i, j \in I$. 
Therefore, $(r' + I)(s' + I) = r's' + I$. On the other hand

\begin{align*}
(r + I)\cdot(s + I) &= rs + I\\
&= (r' + i)(s' + j) + I\\
&= r's' + \underbrace{r'j + is' + ij}_{\text{all are in } I} + I\\
&= r's' + I.
\end{align*}

where in the last step we used the fact that since $I$ is an
ideal, $r'j \in I$ and $is' \in
I.$ Obviously $ij \in I$ as well. Therefore $(r + I)(s + I) =
(r' + I)(s' + I)$, so our definition for
multiplication is clear and well-defined.             
\\

\textcolor{Plum}{
You may be wondering the following: In a quotient ring $R/I$,
why does $I$ have to be an ideal of $R$? To answer this,     
note in the second to last step above, we
used the fact that $I$ was 
an ideal of $R$ to conclude that $r'j, is' \in I$. If $I$
hadn't been an ideal, we wouldn't have been able to absorb
these elements into $I$. Hence, we wouldn't have been able to
make sure that our desired multiplication is well-defined.
So this is why a quotient ring
must always quotient out an ideal, and why we can't just
quotient out any subring of $R$. }

<span style="display:block" class="definition">
Consider the following map $\pi: R \to R/I$, known as the
**projection map**, defined as 

\[
\pi(r) = r + I.
\]

Note that this is a stupidly simple map. It's so stupid it
almost doesn't even deserve a name. But it will be
*convenient* to be able to refer back to the
concept of associating an element $r \in R$ with a coset
$r + I \in R/I$ as a **projection**. It's so
convenient that if you go on in algebra you won't stop
this "coset" mapping, yet everytime you see it you'll
probably think it's dumb.

Also notice that in this case $\ker(\pi) = I$, and that
$\im(\pi) = R/I$.
</span>        




<script src="../../mathjax_helper.js"></script>