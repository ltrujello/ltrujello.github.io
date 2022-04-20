<!-- title: Context Free Grammar For Polynomials -->
<!-- syntax_highlighting: on -->
<!-- date: 2022-04-15 -->
<!-- mathjax: on -->

I finally figured out how to write a set of productions to correctly characterize polynomial arithmetic in a non-ambiguous fashion. I didn't really see anything like this on the internet in general. However, I've needed this since I am hand-writing a parser for polynomial arithmetic (in C, so yeah, that's been a lot of... fun).

The first thing to do is to realize that a polynomial can be thought of as an expression of monoimals. In such an expression, each monomial is separated by either a plus or minus sign. Additionally, a polynomial is itself a monomial. This leads to characterizing a polynomial with the following three productions.
\begin{align}
Polynomial \to &Polynomial + Monomial\\\\
&| Polynomial - Monomial \\\\
&| Monomial
\end{align}
The next step is to define productions for a monomial. To do this there are a couple of things to consider.

First, you need to decide on what field $\mathbb{F}$ you're taking the coefficients of your polynomial over. For example, you can consider coefficients over the complex numbers, real numbers, rational numbers, etc. In most cases, in particular for a computer, you simply consider floating point real numbers. 

Thus one stab at defining productions for a monomial might be:
\begin{align}
Monomial \to &Number\, x^{Int} \\\\
&| x^{Int} \\\\
&| Number \, x\\\\
&| x\\\\
&| -Monomial \\\\
&| Number \\\\
\end{align}
Here, a $Number$ would be anything in the set of (floating point) real numbers, and an $Int$ would be any integer.

However, you'll also probably include *expressions* in your coefficients. For example, the following string should be parsed as a valid monomial, but isn't by the above productions:

\begin{align}
((4 + 3\cdot 5.5/1.3) \cdot 2)x^2
\end{align}

Thus a more appropriate set of productions would be
\begin{align}
Monomial \to &Factor\, x^{Int} \\\\
&| x^{Int} \\\\
&| Factor\, x\\\\
&| x\\\\
&| -Monomial \\\\
&| Factor \\\\
\end{align}
Here, a $Factor$ is a variable that we can describe using one of the classic examples of a context free grammar for basic calculator arithmetic, which appears in every compiler book.

\begin{align}
Factor \to &(Expr)\\\\
&| Number \\\\
&| -Factor 
\end{align}
Here, a $Number$ is again anything in our base field, and an $Expr$ would be
\begin{align}
Expr \to &Expr + Term\\\\
&|Expr - Term\\\\
&| Term \\\\
\end{align}
with a $Term$ being defined as 
\begin{align}
Term \to &Term * Factor\\\\
&| Term / Factor\\\\
&| Factor \\\\
\end{align}
which completes the definition for a monomial. At this point, all that is left is to describe operations that one can perform with a polynomial. The typical operations you can perform with a polynomial are multiplication and exponentiation, which you could begin to describe productions with
\begin{align}
PolyExpr \to &PolyExpr * PolyTerm \\\\
&|PolyTerm^{Int} \\\\
&|PolyTerm \\\\
\end{align}
with a $PolyTerm$ being defined as 
\begin{align}
PolyTerm \to &(Polynomial) \\\\
&| -(Polynomial)\\\\
&| Monomial\\\\
\end{align}
which completes the definition of a $PolyExpr$. 

This grammar could continued to be improved, since one thing that isn't parsed is 
\begin{align}
x^{(2 + 2)}
\end{align}
although this would require writing separate $Factor$, $Term$, and $Expr$ productions for integers, separate from floating point numbers (since you probably wouldn't want floating point numbers appearing in the exponents).




