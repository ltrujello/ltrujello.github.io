<style>
.md-content {
    max-width: 80em;
}
</style>
#1.11. Sylow Theorems.
Lagrange's Theorem states that $H \le G$, then $|H|$ divides
$|G|$. However, you may wonder if there is some kind of converse.
If $k$ divides $|G|$, is there a subgroup of order $k$? 

By Cauchy's Theorem, we know that if $p$ is a prime which divides
then there exists an element of order $p$. Can we generalize this
result further (for example, state *how* many such elements
satisfy this)? 

The answer to both questions is yes and is achieved through
Sylow's Theorem. It's a foundational theorem in finite group
theory, as it
strengthens our two most power theorems for finite groups:
Lagrange's Theorem and Cauchy's Theorem.


<span style="display:block" class="definition">
$H$ is a **$p$-subgroup** of a group $G$ if $H$ is a
subgroup of $G$ and $|H| = p^n$ for some $n \ge 1$.
</span>


<span style="display:block" class="definition">
Let $G$ be a group and let $p$ be a prime such that $p\mid
|G|$. Suppose $p^k$ is the largest power such that $p^k \mid
|G|$. That is, $|G| = p^km$ for some integer $m \in G$,
$\mbox{gcd}(p, m) = 1$. Then any subgroup $H$ of $G$ with $|H| = p^k$ is called a
**Sylow $p$-subgroup**.
\\
\\
An equivalent definition is the following: $H$ is a 
**Sylow-$p$ subgroup** if $H$ is a $p$-subgroup where
$|H| = p^k$.
</span>

\textcolor{purple}{A Sylow $p$-subgroup is nothing more than a subgroup $H$ where
$|H| = p^k$ and $|G| = p^km$ where $\mbox{gcd(p, m)} = 1$.}


<span style="display:block" class="definition">
Let $G$ be a group and $P$ and $Q$ be subgroups of $G$. If
there exists an element $g \in G$ such that 

$$
gPg^{-1} = Q
$$

then $P$ is **conjugate** to $Q$.
</span>
Recall that if $H \normal G$, then for any $g \in G$ we see that
$gHg^{-1} = H$. Thus $H$ is conjugate to itself.
Also note that if $P$ is a subgroup then so is $gPg^{-1}$.


<span style="display:block" class="theorem">[ (Sylow Theorem)] Let $G$ be a finite group and $p$ a
prime such that $p \mid |G|$. Suppose further that $|G| = p^km$
where $\mbox{gcd}(p, m) = 1$. Then 

* **1.** There exists a Sylow $p$-subgroup (equivalently, there
exists a subgroup $H$ of $G$ where $H = p^k$) and every
$p$-subgroup of $G$ is contained in some Sylow $p$-subgroup
* **2.** All Sylow $p$-subgroups are conjugate to each other,
and the conjugate of any Sylow $p$-subgroup is also a Sylow
$p$-subgroup
* **3.** If $n_p$ is the number of Sylow $p$-subgroups, then 

$$
n_p \mid m \hspace{0.5cm}\text{and}\hspace{0.5cm} n_p = 1 \mbox{ mod } m.
$$

\vspace{-.4cm}
</span>


<span style="display:block" class="proof">
\textcolor{NavyBlue}{We can prove the first part by letting
$G$ act on a special set $\Omega$. It will turn out that the
stabilizer of our action will be the desired Sylow
$p$-subgroup.}


* **1.** Define 

$$
\Omega = \{ X \subset G \mid |X| = p^k\}
$$
 
and let $G$ act on $\Omega$ from the
left. Observe that for $X \in \Omega$, $g * X = \{gx \mid
\text{ for all } x \in X\} = gX.$ Since $|gX| = |X| = p^k$, we
see that $gX \in \Omega$. Associativity and identity
applications are trivial, so we get that this is a group
action.

\textcolor{NavyBlue}{Now that we have shown that this is a
group action, we will consider the orbits of the group
aciton.}

Since $|G| = p^km$, there are $\displaystyle
\binom{p^km}{p^k}$ many ways for us to choose a subset $X$
of $G$ with size $p^k$. Hence $|\Omega| = \displaystyle
\binom{p^km}{p^k}$. Note that since this is a group action, the
orbits form a partition of $X$. Now from number theory, we
know that 

$$
\binom{p^km}{p^k} = m \mbox{ mod } p.
$$

Since the orbits must partition $\Omega$, the above result
tells us that we cannot partition $G$ with sets which are
divisible by $p$. In other words, there must exist some
orbit $\mathcal{O}$
such that $|\mathcal{O}|$ is not divisible by $p$. 

\textcolor{NavyBlue}{Now that we know that there exists an
orbit not divisible by $p$, we will anaylze the
corresponding stabilizer of this orbit. This stabilizer
will turn out to be our Sylow $p$-subgroup. }

Let $H$ be the orbit corresponding to $\mathcal{O}$. Then
by the Orbit-Stabilizer Theorem 

$$
|G| = |\mathcal{O}||H| \implies p^km = |\mathcal{O}||H|.                
$$

By the last equation, we see that $p^k$ must divide both
sides. However, $|\mathcal{O}|$ is not divisible by $p$.
Hence $|H|$ must be divisible by $p^k$.

However, by Lagrange's Theorem, $|H|$ divides $|G| =
p^km$. Therefore $|H| = m$ or $|H| \in \{1, p, p^2, \dots,
p^k\}$. In either case $|H| \le p^k$ (since $m \le p$).
But we just showed that $p^k$ divides $|H|$, which proves
that $|H| = p^k$. 

Since $H$ is a stabilizer, $H \le G$, so
we have effectively proved the existence of a subgroup of
order $p^k$; or, in other words, a Sylow $p$-subgroup.
* **2.** Suppose $H$ and $K$ are Sylow $p$-subgroups of $G$. 
Then observe that
* **3.**

</span>

The consequences of this theorem are immediate. 

<span style="display:block" class="proposition">
Let $G$ be a finite group and suppose $|G| = p^km$ for some
prime $p$ where $\mbox{gcd}(p, m) = 1$. Then $G$ has a normal
subgroup of order $p^k$ if and only if $n_p = 1$.
</span>


<span style="display:block" class="proof">
($\implies$) Suppose $G$ has a normal subgroup $H$ of order $p^k$. By
Sylow's Theorem, we know that all 
other Sylow $p$-subgroups are conjugate to $H$. Thus let $g
\in G$ and observe that 

$$
gHg^{-1} = H
$$

since $H$ is normal. Therefore, there are no other Sylow
$p$-subgroups so $n_p = 1.$

($\impliedby$) Now suppose that $n_p = 1$. Let $H$ be a sole
Sylow $p$-subgroup of $G$. Since it is the only Sylow
$p$-subgroup, we see that 

$$
gHg^{-1} = H
$$

for all $g \in G$. However this exactly the definition for
$H$ to be a normal subgroup of $G$. This proves the result.
</span>

Once you use Sylow's Theorem and study finite groups more, you'll
realize that some groups aren't that complicated. For example,
consider *any* subgroup of order 4. This can be any wild
group you want, but at the end of the day, it turns out one of the
following options is true:

$$
G \cong \mathbb{Z}/4\mathbb{Z} \hspace{0.2cm}\text{ or }\hspace{.2cm} 
G \cong \mathbb{Z}/2\mathbb{Z} \times \mathbb{Z}/2\mathbb{Z}.
$$

The process leading to such a conclusion is known as
**classifying groups up to an isomorphism**. That is, you
start with a group with a fixed order, and then determine much
simpler groups that your group could be isomorphic to. In our
example, we say that any group of order 4 can only be two things
up to an isomorphism.

The cool thing about Sylow's Theorem is that it is so strong that
it allows us to classify groups up to an isomorphism. 

In general, when classifying groups up to an isomorphism, it is
convenient to do in terms of integer groups $\mathbb{Z}$ or
modulo integer groups, as we saw above. This isn't always
possible, but when it is, the following theorem comes in handy.


<span style="display:block" class="theorem">
Let $m, n$ be positive integers. Then 

$$
\mathbb{Z}/mn\mathbb{Z} \cong \mathbb{Z}/m\mathbb{Z} \times \mathbb{Z}/n\mathbb{Z}
$$

if and only if $m$ and $n$ are coprime.
</span>


\noindent
**Example.**
\\
\textcolor{NavyBlue}{Suppose we want to classify all groups of order 1225 up to an
isomorphism.}
\\
Let $G$ be a group such that $|G| = 1225 = 5^27^2$. Then observe
$\mbox{gcd}(5, 7^2) = 1$. By Sylow's theorem, we know that 
if $n_5$ is the number of Sylow $5$-subgroups of $G$, then 

$$
n_5 \big| 7^2 \quad \text{ and }\quad n_5 \equiv 1 \mbox{ mod } 5.
$$

Observe that $n_5$ can only equal 1. Since $n_5 = 1$, we know by
Propsition \ref{sylow_normal} that
for the unique Sylow 5-subgroup $H$ that $H \unlhd G$. Also note
that $|H| = 5^2$.
\\
\\
Now observe that $\mbox{gcd}(7, 5^2) = 1$. By Sylow's Theorem, we
know that if $n_7$ is the number of Sylow 7-subgroups of $G$ that 

$$
n_7 \big| 5^2 \quad \text{ and }\quad n_7 \equiv 1 \mbox{ mod } 7.
$$

Note that $n_7$ must also equal 1. Thus again for the unique Sylow
7-subgroup $K$, we must have that $K \unlhd G$ and $|K| = 7^2$. Now we can observe
that (1) $\mbox{gcd}(|H|, |K|) = 1$ and (2) $|G| = |H||K|$ so that

$$
G \cong H \times K     
$$

by Theorem 1.\ref{product_theorem}. 
Now observe that since $|H| = 5^2$, $H \cong
\mathbb{Z}/25\mathbb{Z}$ and $H \cong \mathbb{Z}/5\mathbb{Z}
\times \mathbb{Z}/5\mathbb{Z}$. Since $K = 7^2$, $K \cong 
\mathbb{Z}/49\mathbb{Z}$ and $H \cong (\mathbb{Z}/7\mathbb{Z}
\times \mathbb{Z}/7\mathbb{Z})$.
Therefore, we see that the groups of order 1225 are, up to
isomorphism, 

* **(1)** $(\mathbb{Z}/25\mathbb{Z}) \times (\mathbb{Z}/49\mathbb{Z})$
* **(2)** $(\mathbb{Z}/25\mathbb{Z}) \times
(\mathbb{Z}/7\mathbb{Z} \times \mathbb{Z}/7\mathbb{Z})$
* **(3)** $(\mathbb{Z}/5\mathbb{Z} \times \mathbb{Z}/5\mathbb{Z})
\times \mathbb{Z}/49\mathbb{Z}$
* **(4)** $(\mathbb{Z}/5\mathbb{Z} \times \mathbb{Z}/5\mathbb{Z})
\times (\mathbb{Z}/7\mathbb{Z} \times
\mathbb{Z}/7\mathbb{Z})$.

\textcolor{purple}{We suspect that these are all the groups of
order 1225 up to an isomorphism. However, we double check that
none of these groups are actually equivalent to each other, i.e.,
that we have no redundancies.}

Observe that (1)
is not isomorphic to any of the the other groups, since $(1, 1)
\in (\mathbb{Z}/25\mathbb{Z}) \times (\mathbb{Z}/49\mathbb{Z})$,
has order 1225 but none of the other groups have an element of
order 1225.
\\
\\
In addition, (3) is not isomorphic to (2) or (3) since $(0, 1) \in
(\mathbb{Z}/5\mathbb{Z} \times \mathbb{Z}/5\mathbb{Z})
\times \mathbb{Z}/49\mathbb{Z}$ and has
order 49
but no element of either (2) or (3) has an element of either 49. 
\\
\\
Finally, we see that (2) is not isomorphic to (4) because $(1, 0)
\in (\mathbb{Z}/25\mathbb{Z}) \times
(\mathbb{Z}/7\mathbb{Z} \times \mathbb{Z}/7\mathbb{Z})$ is an
element of order 25 but there is no element of order 25 in (4).
Thus we see that (1) these subgroups are isomorphic to $G$ and (2)
none of them are isomorphic to each other. Therefore, this an
exhaustive list of all the groups of order 1225 up to isomorphism.

Here's another example in which Sylow's Theorem helps us classify
a specific type of group.


<span style="display:block" class="theorem">
Let $p,q$ be primes with $p<q$ and suppose $p$ does not divide
$q-1$.  If $G$ is a group such that $|G| = pq$, then 
$G \cong \mathbb{Z}/pq\mathbb{Z}$. 
</span>


<span style="display:block" class="proof">
Let $G$ be a group and $|G| = pq$. Since $\mbox{gcd}(p, q) = 1,$
by the Sylow Theorem, 
there exists a Sylow $p$-subgroup and Sylow $q$-subgroup of $G$. 
\\
Now let $n_p$ and $n_q$ be the number of Sylow $p$ and $q$-subgroups,
respectively. Then observe that 

$$
n_p \big|q \qquad n_p \equiv 1 \mbox{ mod } p   
$$

so that $n_p = 1$ and 

$$
n_q \big|p   \qquad n_q \equiv 1 \mbox{ mod } q.
$$

Now observe that $n_p = 1$ or $q$. However, since $p$ does not divide
$q - 1$, we know that 

$$
q \not\equiv 1 \mbox{ mod } p.  
$$

Thus $n_p = 1$. Again, either $n_p = 1$ or $p$ but $p < q$ so 

$$
n_q \not\equiv 1 \mbox{ mod } q 
$$

unless $n_q = 1$. Thus there is one and only one Sylow $p$-subgroup
and Sylow $q$-subgroup, which we can call $H$ and $K$ respectively.
By proposition \ref{sylow_normal}, 

$$
H \unlhd G \qquad K \unlhd G.  
$$

Note that (1)
$\mbox{gcd}(|H|, |K|) = \mbox{gcd}(p, q) = 1$ and (2) $|G| = |H||K| =
pq$. Thus $G \cong H \times K$ by Theorem 1.\ref{product_theorem}. Now observe that $H$ and $K$ are of
prime order, so that $H \cong \mathbb{Z}/p\mathbb{Z}$ and $K \cong
\mathbb{Z}/q\mathbb{Z}$. We then see that 

$$
G \cong  \mathbb{Z}/p\mathbb{Z} \times \mathbb{Z}/q\mathbb{Z}.
$$

From theorem ???, we know that if $m, n$ are positive
integers and $\mbox{gcd}(m, n) = 1$, then 

$$
\mathbb{Z}/m\mathbb{Z} \times \mathbb{Z}/n\mathbb{Z} 
\cong \mathbb{Z}/mn\mathbb{Z}.
$$

Obviously, $\mbox{gcd}(p, q) = 1$, so that 

$$
\mathbb{Z}/p\mathbb{Z} \times \mathbb{Z}/q\mathbb{Z}
\cong \mathbb{Z}/pq\mathbb{Z}.
$$
 
Now isomorphic relations are transitive, so we can finally state that 

$$
G \cong \mathbb{Z}/pq\mathbb{Z}
$$

as desired.
</span>




<script src="../../mathjax_helper.js"></script>