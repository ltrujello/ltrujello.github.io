<style>
.md-content {
    max-width: 80em;
}
</style>
#2.1. Definitions.

While many mathematical objects come in the form of groups, we
also know that there are objects and spaces which require more
than one operation. Can we generalize them?

For example, we know that the integers $\mathbb{Z}$ form a group
under addition. But don't we also know that multiplication of
elements of $\mathbb{Z}$ also yield elements of $\mathbb{Z}$?
Isn't this another type of group-like structure we would like to
generalize?

We could do this on $\mathbb{R}$ too. It's a group under addition,
but we know it's closed under multiplication and has some identity
element.

This is where rings come into play, which we define as follows. 


<span style="display:block" class="definition">
Let $R$ be a set. We define $(R, +, \cdot)$ to be a **ring** if there
exist binary operations $+: R\times R \to R$ and $\cdot: R
\times R \to R$ (referred to as addition and multiplication)
such that 

* [(**R1**)] **Group addition.** $(R, +)$ is an
**abelian group**, with $0$ denoted as the
identity. (In this group, the additive inverse of
an element $a$ is always denoted $-a$.)


* [(**R2**)] **Closure.** For all $a$, $b \in R$, we have that $a \cdot b \in R$.


* [(**R3**)] **Associativity.** For all $a$, $b$, $c \in R$, we have that $a \cdot (b \cdot c) = (a \cdot b) \cdot c$


* [(**R4**)] **Distributivity.** Similarly, we have that $a \cdot (b + c) = a\cdot b + a \cdot c$ and $(b
+ c) \cdot a = b \cdot a + c \cdot a$.


* [(**R4**)] There exists an element $1 \ne 0$ in $R$ such that 
$1 \cdot a = a \cdot 1 = a$ for all $a \in R$. This is the **unit of the ring**. 



</span>


<span style="display:block" class="remark">

*  As usual, if the multiplication operation $\cdot$ is specified and
well-understood, then we will drop $\cdot$ and write
multiplication of ring elements as $gh$ instead of $g \cdot h$.



*  Axioms (**R5**) is technically optional. However, we don't really 
care about rings without unity, so we just add it to ou defintion.



</span>



<span style="display:block" class="proposition">
Suppose $R$ is a ring with identity $1 \ne 0$. Then 

* [1.] $0 \cdot a = a \cdot 0$ for all $a \in R$ 


* [2.] $-(a \cdot b) = (-a) \cdot b = a \cdot (-b)$ for
all $a, b \in R$ 


* [3.] $-a = a \cdot (-1) = (-1) \cdot a$


* [4.] $(-a) \cdot (-b) =  a \cdot b$ 


* [5.] The multiplicative identity is unique.   



</span>
This is just the stuff you would expect from a ring $R$ based
on the fact that many domains you've seen are in fact rings,
and some of these facts are obvious in those domains.


<span style="display:block" class="proof">

* [1.] Observe that 

\begin{align*}
(0 \cdot a) + (0 \cdot a) &= (0 + 0) \cdot a \text{ (by R4 )}\\
& = (0 \cdot a) + 0 \text{  (since 0 + 0 = 0)}
\end{align*}

where we added
$0$ to the righthand side (which of course 
does not change the value of the equation.) Subtracting
$(0 \cdot a)$ from both sides, we get that 

\[
0 \cdot a  = 0.
\]

Similarly, observe that 

\begin{align*}
(a \cdot 0) + (a \cdot 0) & = a \cdot (0 + 0) \text{ (by R4)}\\
& = (a \cdot 0)+ 0 \text{ (since 0 + 0 = 0)}
\end{align*}

where again, we added $0$ to both sides. Subtracting $-(a
\cdot 0)$ from both sides, we get 

\[
a \cdot 0 = 0
\]

as desired. 



* [2.] First we'll show that $-(a \cdot b) = (-a) \cdot
b$. To prove this, observe that 

\begin{align*}
(a \cdot b) - [(a \cdot b)] & = 0 \\
& = a \cdot 0 \text{ (which we just proved)}\\
& = a \cdot [b + (-b)]\\
& = a \cdot b + a \cdot (-b) \text{ (by R4)}
\end{align*}

and adding $-(a \cdot b)$ to both sides yields 

\begin{align*}
-(a \cdot b) = a \cdot (-b)
\end{align*}

as desired. Now we'll show that $-(a \cdot b) = (-a) \cdot
b$. Observe that 

\begin{align*}
(a \cdot b) - [(a \cdot b)] & = 0 \\
& = 0 \cdot b \text{ (which we just proved)}\\
& = [a + (-a)] \cdot b\\
& = a \cdot b + (-a) \cdot b \text{ (by R4)}
\end{align*}

and adding $-(a \cdot b)$ gives that 

\begin{align*}
-(a \cdot b) = (-a) \cdot b
\end{align*}

which proves the asserition.



* [3.] Simply let $b = 1$ in the previous statements. 


* [4.] To prove that $(-a) \cdot (-b) = a \cdot b$,
first observe that for any $c \in  R$ we already proved
that 

\[
(-a) \cdot c = a \cdot (-c).
\]

Thus let $c = -b$. Then observe that 

\[
(-a) \cdot (-b) = a \cdot [-(-b)]
\]

and from group theory, we know that $-(-b) = b$.
Therefore, we see that 

\[
(-a) \cdot (-b) = a \cdot b
\]

as desired. 



* [5.] To prove that uniqueness of the multiplicative
identity, first suppose that it is not unique. That is,
there exists elements $1_1$ and $1_2$ such that 

\[
1_1 \cdot a = a \cdot 1_1 = a \hspace{1cm}   1_2 \cdot a = a \cdot 1_2 = a.
\]

for all $a \in R$. Then observe that 

\[
1_1 = 1_1 \cdot 1_2 = 1_2
\]

so that the uniqueness must hold.





</span>
\textcolor{NavyBlue}{An example of a ring is of course $\mathbb{Z}$, but that's boring. 
Is $(\mathbb{Z}/n\mathbb{Z}, + , \cdot)$, where $n$ is a positive
integer, a ring? Let's check if it is. }
\begin{description}
\item[Abelian.] Since addition is commutative, we already know
that $\mathbb{Z}/n\mathbb{Z}$ is abelian (in fact, it is
cyclic.)

\item[Associativity.] Let $a, b$ and $c \in \ZZ/n\ZZ$. Now
obviously, $a(bc) = (ab)c$ under *standard* or "normal"
multiplication of integers. Therefore we see that   

\begin{align*}
a\cdot(b \cdot c) &= a(bc) \mbox{ mod } n \\
& = (ab)c \mbox{ mod }n \\
& =(a \cdot b ) \cdot c.
\end{align*}


\item[Distributivity.] Let $a, b$ and $c$ be defined as before.
Again, we know that $a(b + c) = ab + ab$ in $\mathbb{Z}$.
Therefore 

\begin{align*}
a\cdot(b + c) &= a(bc) \mbox{ mod } n \\
& = (ab + ac) \mbox{ mod }n \\
& = ab \mbox{ mod }n + ac \mbox{ mod }n\\
& = a \cdot b + a \cdot c.
\end{align*}

The argument is exactly the same to prove left distributivity.
Altogether, we see that $\ZZ/n\ZZ$ satisfies the axioms of a
ring when endowed with modulo addition for $+$ and modulo
multiplication for $\cdot$. 
\end{description}


\noindent**Multiplication yielding zeros.**\\
For our ring $\mathbb{Z}$, we know that the only way to ever
obtain $0$ by multiplication is to just take $0$ itself and multiply
it by an integer. Thus in this ring, if $n, m$ are nonzero
then we always know that $n \cdot m$ is nonzero.

However, note that in $\ZZ/n\ZZ$, we have
that $a \cdot b = 0$ if and only if $a \cdot b$ is a multiple
of $n$.

\textcolor{Plum}{If $n$ is prime, then there are no elements
in $\ZZ/n\ZZ = \{0, 1, 2, \dots, n-1\}$ whose product will be
a multiple of $n$. This is just because nothing divides $n$.
\\
\\
But if $n$ is composite, then there exist
integers $pq$ such that $n = pq$, and since $p < n$ and $q <
n$, you can be certain that $p, q \in \ZZ/n\ZZ$. Then we'd see
that $pq = 0$ in $\ZZ/n\ZZ$. If $p$ or $q$ are also composite, then
there are even more combinations of integers in $\ZZ/n\ZZ$
whose product yields 0 in $\ZZ/n\ZZ$. 
}

So in the ring $\ZZ$, multiplication of nonzero elements will
be nonzero. But in the ring $\ZZ/n\ZZ$ there are many ways one
one can multiply elements to get zero (if $n$ is not prime).
Obviously these are both rings, but they're behaving
differently! Hence we introduce the following definitions. 


<span style="display:block" class="definition">
Let $(R, +, \cdot)$ be a ring and suppose $a \ne 0$ and $b
\ne 0$ are elements of $R$,
while 

\[
a \cdot b = 0.
\]

Then $a$ and $b$ are *both* called
**zero divisors** of the ring ${R}$. Note that $0$ is
not a zero divisor.
Meanwhile, if $R$ has an identity, and for some $a \in R$
there exists a $b \in R$ such that 

\[
ab = 1 = ba
\]

then we call *both* $a$ and $b$ **units** in $R$.
It turns out the set of units of a ring $R$ form an
abelian group, which we denote as $R^*$.
</span>

Note that $\ZZ$ has no zero divisors, and its unit group $R^*$
is just $\{1, -1\}$. We can see that since if $ab = 1$ for $a, b \in \ZZ$, then we
know that $a = b = 1$ or $-1$.

On the other hand, $\ZZ/n\ZZ$ can have a more interesting unit group.
Observe that if there exists integers $p, q \in \ZZ/n\ZZ$ such that 

\[
pq = n +1
\]

then we see that $p \cdot q = pq \mbox{ mod } n = n + 1
\mbox{ mod } n = 1 $ in $\ZZ/n\ZZ$. If either $p$ or $q$ are composite,
then $R^*$ becomes even more interesting.

As a more specific example, observe that the ring $\ZZ/10\ZZ$
has units $\{1, 3, 7, 9\}$ and zero divisors $\{2, 4, 6, 8\}$.


<span style="display:block" class="lemma">
A zero divisor can never be a unit.
</span>


<span style="display:block" class="proof">
Let $R$ be a ring and suppose $a \in R$ is a zero divisor.
Then there exists an element $b \in R$ where $b \ne 0$ and $ab = 0.$
Now suppose that $a$ is also a unit, so that there exists
a $c \in R$ such tha $ac = ca = 1$. Then observe that 

\begin{align*}
1 = ca \implies b &= (ca)(b)\\
&= c(ab)\\
&= c(0)\\
& = 0
\end{align*}

which is a contradiction since we said $b \ne 0$. Hence
$a$ cannot be a unit.
</span>

We'll next prove another useful lemma which is commonly known
as the cancellation law. 


<span style="display:block" class="lemma">
Let $R$ be a ring, and $a \in R$ such that $a \ne 0$. If
$a$ is not a zero divisor, then for any $b, c \in R$ such
that $ab = ac$ we have that $b = c$. In addition, if $ba =
ca$ then $b = c$.
</span>


<span style="display:block" class="proof">
Suppose $ac = ab$ for some elements $a, b, c \in R$ where
$a$ is not a zero divisor. Then observe that 

\[
ab = ac \implies ac - ab = 0 \implies a(b - c) = 0.
\]

Since $a$ is not a zero divisor, the only way for the
above equation to hold is if $b - c = 0 \implies b = c$.
Proving the analagous statement is identical to this
proof. 
</span>

\textcolor{NavyBlue}{Now that we have identified terms and can
describe the specific elements of a ring $R$ based on their
properties, we again return to our observation that $\ZZ$ and
$\ZZ/n\ZZ$ behaved differnetly. This is not uncommon in ring
theory, so we can divide rings into specific classes as
follows.}


<span style="display:block" class="definition">
Let $R$ be a ring. 

* [1.] If $R$ is commutative ring with identity and has
no zero divisors, then $R$ is said to be an
**integral domain.**


* [2.] The ring $R$ is a said to be a **division ring**
if every element of $R$ has a multiplicative inverse.
An equivalent condition is if $R^* = R\setminus
\{0\}$.


* [3.] If $R$ is a commutative division ring, then
$R$ is said to be a **field**.



</span>

You've probably read textbooks that called $\mathbb{R}$ or
$\mathbb{C}$ fields. This is what they're talking about. 


<span style="display:block" class="proposition">

* [1.] If $(R, +, \cdot)$ is an integral domain, then the
cancellation law holds for all elements of $R$. 



* [2.] $(R, +, \cdot)$ is an integral domain if and
only if for $a, b \in R$, the equation $a\cdot b = 0$
implies either $a = 0$ or $b = 0$. 



* [3.] $(R, +, \cdot)$ is a division ring if and only if $ax =
b$ and $ya = b$ are solvable in $R$ for every $a, b \in R$
where $a \ne 0 $.



</span>

\textcolor{MidnightBlue}{Consider again the ring $\ZZ/p\ZZ$
where $p$ is a positive integer. We noted that if $p$ is
prime then there are no zero divisors. Thus we could state
that this an integral domain. However, we can strengthen
this even further and state that this is a field, as follows.
\\
\indent Let $a \in \ZZ/p\ZZ$ be nonzero. We know that there exists an
inverse $a^{-1}$ such that 

\[ 
aa^{-1} = 1 \mbox{ mod } p 
\]

if $a$ is coprime with $p$, which is of course true. Since
every element has a multiplicative invesrse we see that
$\ZZ/p\ZZ$ is a division ring. Since this is a commutative
division ring, we have that $\zz/p\zz$ is a field.
}
Observe that we could have more easily proved thiss tatemen
with the following theorem.


<span style="display:block" class="theorem">
Any finite integral domain is a field.
</span>


<span style="display:block" class="proof">
Let $R$ be a finite integral domain and let $a \in R$ be nonzero.
Construct a function $\phi_a: R \to R$ by $\phi_a(b) = ab$
for $b \in R$. 

Suppose $\phi_a(b) = \phi_a(c)$ for $b, c \in R$. Then $ab
= ac \implies b = c$ since $R$ is an integral domain.
Therefore $\phi_a$ is injective, and it is clearly
surjective so that $\phi_a(R) = R$.

Since $\phi_a$ is bijective for each $a \in R$, we know 
there always exists a $b \in R$ such that $\phi_a(b) = 1
\implies ab = 1$. In other words, each $a \in R$ has an
inverse, proving $R$ is a division ring. Since $R$ is an
integral domain and thus a commutative we have that it is a
commutative divison ring, and hence a field.
</span>

\textcolor{MidnightBlue}{As we said before $\ZZ/p\ZZ$ is an
integral domain. Since it's also finite, this allows us to
conclude it is a field, which we proved before we proved the
above theorem.}

The following is apparently too difficult for any
introductory algebra book to prove in terms of elementary language.

<span style="display:block" class="theorem">
Any finite division ring is a field.
</span>

With the integral domain, division ring and field introduced,
we have a solid footing in the fundamentals of ring theory. We
move forward by introducing the concept of a subring.

\subsection*{Subrings.}


<span style="display:block" class="definition">
Let $R$ be a ring and $S$ be a nonempty subset of $R$.
Then $S$ is a **subring** of $R$ if $S$ is a ring under the
addition and multiplication equipped on $R$. 

Specifically, $S$ is a subring if $S$ is an abelain group
under addition and is closed under multiplication.
</span>
\noindent
**Examples.**\\
We already have an example from our previous work. We know
that $\ZZ$ and $\ZZ/n\ZZ$, where $n$ is a positive integer,
are both rings. Since $\ZZ/n\ZZ \subset \ZZ$, we see that
$\ZZ/n\ZZ$ is a subring of $\ZZ$. 
\\

\textcolor{Blue!80!White}{Define the set $\ZZ[i] = \{m + ni: m, n \in \ZZ\}$. 
Then this is a ring.} This is
clearly an abelian group under addition (0 is the identity, associativity is
obvious, closedness is clear, inverse of any given element is
the same element with coefficients of opposite sign).
Multiplicative associativity and left and right distributions
are clear. However, since $\ZZ[i] \subset \mathbb{C}$, we see
that $\ZZ[i]$ is a subring of $\mathbb{C}$.
\\

\textcolor{Green!80!White}{Let $R$ be a ring. Then the set of $n \times n$ matrices with
entries $R$, denoted $M_{n}(R)$, forms a ring.} Addition on
this set forms an abelian group. And we know from linear
algebra that matrix multiplication is associative and left and
right distributive.
It turns out this ring has many interesting subrings, which
we'll list here 
\begin{description}
\item[Diagonals.]

\[
D_n(R) = \{A \in \mathbb{R}: a_{ij} = 0 \text{ if } i \ne j\}
\]

\item[Upper Triangulars.]

\[
T^n(R) = \{A \in M_n(R): a_{ij} = 0 \text{ if } i > j\}
\]

\item[Lower Triangulars.]  

\[
T_n(R) = \{A \in M_n(R): a_{ij} = 0 \text{ if } i < j\}.
\]

\end{description}
These are all subrings of $M_n(R)$.
\\

\textcolor{Purple!80!White}{Let $G$ be abelian. Then $\mbox{End}(G)$, the set of
endomorphism (homomorphisms from $G$ to itself), forms a
ring under addition as function addition and multiplication as
function composition.} 
\begin{description}
\item[Abelian Group.] First observe that this is a
commutative structure since $G$ is abelian. We just have
to show that this is a group.
\begin{description}
\item[Identity.] Let $0_G$ be the identity element of
$G$. Construct the identity element for $\mbox{End}(G)$ to
be the zero map $0$
defined as $0: G \to G$ such that $0(g) = 0_G$ for all
$g \in G$. 

\item[Associativity.] Since $G$ is associative, and
the images of elements in $\mbox{End}(G)$ are in $G$,
associativity is inherited. 

\item[Closedness.] Let $f, g \in \mbox{End}(G)$ and
define $h = f + g$. Then $h: G \to G$, and is
obviously a homomorphism, so that $h \in
\mbox{End}(G)$. 

\item[Inverses.] Let $f \in \mbox{End}(G)$. Then
construct the function $f^{-1} : G 
\to G$ such that $f^{-1}(g) = -h$ whenever $f(g) =
h$. (Note that $-h$ is the inverse of $h$.) Then we
see that $f^{-1}(g) + f(g) = 0$ for all $g \in G$, and
that $f^{-1}(g) \in \mbox{End}(G)$, so that $f^{-1}$
is an inverse of $f$. 
\end{description}

\item[Multiplicatively Closed.] Observe that if $h: f
\circ g$, then $h: G \to G$, and it is a homomorphism.
Hence $h \in \mbox{End}(G)$.
Therefore our multiplcative operator is closed.

\item[Multiplicative Associativity.] This holds in our
case since function composition is in general associative
for homomorphisms.

\item[Distributivity.] Let $f, g, h \in \mbox{End}(G)$.
Then observe that 

\[
f(g + h) = f \circ (g + h) = f \circ g + f \circ h = fg + fh
\]

and 

\[
(g + h)f = (g + h) \circ f = g \circ f + h \circ f = gf + hf
\]

by linearity of $f, g$ and $h$ (since homomorphisms in
general are linear functions).
\end{description}
Therefore, we have that $\mbox{End}(G)$ forms a ring under
function additon and composition.
\\

\textcolor{Red!70!Blue}{
**Polynomial Rings.**
Polynomials are an interesting example of a ring, which we
construct as follows.}

\indent Let $R[x]$ be the set of all functions
$f: \ZZ^+ \to R$ such that $f(n) = 0$ for all but finitely
many $n$. These functions will be the coefficients to our
polynomials, and we want them to be finite, so we request that
only finitely many of our cofficients are nonzero.
That is, $f(n)$ represents the $n$-th coefficient.
\\

\noindent Define addition and multiplication for two $f, g \in R[x]$ as 

\[
(f + g)(n) = f(n) + g(n) \hspace{0.2cm}\text{ and }\hspace{0.2cm}  (f \cdot g)(n) =
\sum_{i = 0}^{n}f(i)g(n - i).
\]

This last formula is the formula for the
$n$-th coefficient from the product of two polynomials. We'll
show this is a ring.
\begin{description}
\item[Abelian.] First we'll show this is an abelian group
under addition.
\begin{description}
\item[Identity.]Let $0_R \in R$ be the 0 element of $R$.
If we define 0 to be
the map $0(n) : \mathbb{Z} \to R$ such that $0(n) = 0_R$
for all $n \in \mathbb{Z}$, then clearly $0 \in R[x]$ and
$0 + f = f + 0 = f$ for any $f \in R[x]$. It is our
additivite identity. 

\item[Associativity.] Associativity is derived from
the fact that $R$ is associative under addition.

\item[Closedness.] 
To show this is closed we, show that $f + g$ is nonzero
for at most finitely many elements for any $f, g \in
R[x]$. Simply observe if $f$ is nonzero for $k$-many
elements and $g$ is nonzero 
for $l$-many elements then $(f + g)$ is nonzero for at
most $(l + k)$-many elements. Therefore $(f + g) \in R[x]$.

\item[Inverses.] 
For any $f \in R[x]$, define $f^{-1}$ to be $f^{-1}(n)
= -f(n)$ for all $n \in \ZZ^{+}$. Obviously $f^{-1}$
is nonzero for at most finitely many elements if $f$
is, so $f^{-1} \in R[x]$, and $f^{-1}(n) + f(n) = 0$
for any $n \in \ZZ^{+}$. Therefore $R[x]$ contains inverses.  
\end{description} 

\item[Multiplicatively Closed.]
Observe now that this is closed under multiplication. For
any $f, g \in R[x]$, we can simply observe that since $f,
g$ are nonzero for at most finitely many values of $n \in
ZZ^{+}$, we note that 

\[
fg(n) = \sum_{i = 1}^{n}f(i)g(n - i)
\]

is a function which is nonzero for at most finitely many
values, since it is always a finite sum of $f$ and $g$.

\item[Multiplicative Associativity.]
Let $f, g, h \in R[x]$. Then observe that 

\begin{align*}
(fg)h(n) = \sum_{i = 0}^{n}(fg)(i)h(n - i) &= 
\sum_{i = 0}^{n}\left( \sum_{j = 0}^{i}f(j)g(i - j) \right)h(n - i)\\
&= f(0)g(0)h(n) + \Big(f(0)g(1) + f(1)g(0)\Big)h(n-1)\\ 
&+ \Big(f(0)g(2) + f(1)g(1) + f(2)g(0)\Big)h(n-2) + \cdots \\
&= \sum_{i = 0}^{n}f(n)\sum_{j = 0}^{n-i}g(j)h(n-j-i)\\
&= \sum_{i = 0}^{n}f(n)(gh)(n - i)\\
&= f(gh)(n).
\end{align*}

Therefore multiplicative associativity is satisfied.

\item[Distributivity.] 
Since the image of our functions are elements in $R$, distributivity is inherited from the ring $R$, which must
be left and right distributed. 
\end{description}

Therefore we see that $R[x]$ forms a rings. We'll now realize
that this is the set of polynomials by describing the function
a stupidly simple function: 

\[
x^n(m) = 
\begin{cases}
1 & \text{ if } n = m\\
0 & \text{ otherwise }
\end{cases}.
\]

Then observe that for any $f \in R[x]$, we may uniquely associate
with it the following object: 

\[
f = \sum_{n = 0}^{\infty}f(n)x^n.
\]

The $\infty$ in the upper limit is there to allow us to define
any polynomial of an arbitrary degree. We know it will always
a finite polynomial since we said that $f(n) \ne 0$ for at
most finitely many $n$. 

Thus what we've shown is that the space $R[x]$, constructed by
focusing on the coefficients, defining their rules for
polynomial multiplication, and realizing the polynomial
structure we wanted, is in fact a ring!
\\

Note that if we don't assume that $f(n)$ is nonzero for
finitely many $n$, then we'll end up constructing a different
ring, know as the **formal power series** ring denoted $R[[x]]$. This has
the same rules of addition and multiplication, so the ring
structure doesn't change. The only thing that changes is that
$R[[x]]$ includes infinitely long polynomials. 

Thus, we see that $R[x] \subset R[[x]]$ and therefore $R[x]$
is a subring of $R[[x]]$.

In group theory there was a Subgroup Test which simplified the
task of determine whether or not a subspace form a group or
not. Fortunately, such a tool is available in ring theory.


<span style="display:block" class="theorem">[Subring Test.]
Let $R$ be a ring and $S \subset R$. Then $S$ is a
**subring** of $R$ if and only if, for all $x, y \in
S$ we have that $x - y \in S$ and $xy \in S$.
</span>


<span style="display:block" class="proof">
($\implies$) Suppose $S \subset R$ is a subring. Then
certain $x - y \in S$ and $xy \in S$. 

($\impliedby$) Suppose now that $x - y \in S$ and $xy \in
S$ for all $x, y \in S$. The first condition immediately
$(S, +)$ is a subgroup of $(R, +)$, since $x - y \in S$
for all $x, y \in S$ is just the subgroup test. Now observe that $xy \in S$ for all $x, y \in S$.
Since $R$ is an abliean group under addition and is closed
under multiplication of its elements, we have that $S$ is
a subring of $R$ as desired.
</span>

It turns out that arbitrary intersections of subrings produce
a subring, an important result we include here. 


<span style="display:block" class="theorem">
Let $R$ be a ring and $\{S_\alpha\}_{\alpha \in \lambda}$
be a family of subrings of $R$. Then $S = \bigcap_{\alpha
\in \lambda} S_\alpha$ is a subring of $R$. 
</span>


<span style="display:block" class="proof">
From group theory, we know that the arbitrary intersection
of subgroups is again a group. So $S = \bigcap_{\alpha \in
\lambda} S_\alpha$ is an abelian subgroup of $R$. 
Therefore, we just need to
check that $S$ is
closed under multiplication. 

From group theory, we know that the arbitrary
intersection of a family of subgroups is a group. Thus $S$
is an abelian group, and we just need to check that it is
closed under multiplication. 

For any $s, s' \in S$ we know that $s, s' \in S_\alpha$
for all $\alpha \in \lambda$. Since each subring is
obviously closed under multiplication we see that $ss' \in
S_\alpha$ for all $\alpha \in \lambda$. Hence, $ss' \in S$
as desired.
</span>




<script src="../../mathjax_helper.js"></script>