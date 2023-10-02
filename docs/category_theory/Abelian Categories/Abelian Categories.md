<style>
.md-content {
    max-width: 80em;
}
</style>
#8.5. Abelian Categories

Let $\cc$ be a preabelian category, and consider an arbitrary morphism 
$\phi: A \to B$. Then, since we are in an abelian category, we can calculate 
the kernel and cokernel of this morphism, which both have their familiar 
universal properties. 

<img src="../../../png/category_theory/chapter_8/tikz_code_5_0.png" width="99%" style="display: block; margin-left: auto; margin-right: auto;"/>
One thing we can do is examine both the kernel and the cokernel *of* these two morphisms.
Specifically, we can calculate the kernel $\ker(c)$ of $c$ and the cokernel 
$\coker(e)$ of $e$. However, since we have a map $\phi: A \to B$ such that 
$c \circ \phi = 0$, we see that there exists a unique map $u: A \to \ker(\coker(f))$
such that $\phi = e' \circ u$. 
Dually, since $\phi \circ e = 0$, there exists a unique map $v: \coker(\ker(f)) \to B$. 
such that $\phi = v \circ c'$. 
\
<img src="../../../png/category_theory/chapter_8/tikz_code_5_1.png" width="99%" style="display: block; margin-left: auto; margin-right: auto;"/>















<script src="../../mathjax_helper.js"></script>