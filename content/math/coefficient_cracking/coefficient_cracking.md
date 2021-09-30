<!-- title: Integer Coefficient Cracking -->
<!-- syntax_highlighting : on -->
<!-- date: 2021-09-15  -->
<!-- mathjax: on -->

After learning about the applied math techniques behind rational interpolation, I realized that one could use Cosmin-Ionita's work to actually find the integer coefficients of a rational curve that interpolates a sequence of points.

This is useful if you are given a set of data and would like to know not only if can be interpolated by a rational function, but a rational function with integer coefficients. And I think this could be useful for brute forcing searches that seek new mathematical series or formulas, because such formulas usually involve a rational function with integer coefficients. Such formulas are of interest because they are generally really fast and easy to compute.

## Background

To do this, we apply the original interpolation technique which interpolates a sequence of points $(x_i, f_i)$ with $i = 0, 1, 2, \dots, N$. The rational interpolation algorithm produces a rational function 
$$
\frac{\sum_{i=0}^N \dfrac{a_iw_i}{x - \lambda_i}}{ \sum_{i=0}^{N} \dfrac{a_i}{x - \lambda_i} }
$$
Here the $a_i$ values are taken from one of the columnn vectors of the nullspace of the calculated Loewner matrix, and $\lambda_i$ and $w_i$ values are subsets of the $x$ and $y$ data, respectively. 

The above form of the rational interpolater is the **Barycentric** form, and it is used for further calculations, such as evaluating the function at a point, because it is more numerically stable than using the expanded form. However, we are interested in the actual coefficients of the rational function, so we can expand it using Vieta's formula. Specifically, we write 
$$
\frac{\sum_{i=0}^N \dfrac{a_iw_i}{x - \lambda_i}}{ \sum_{i=0}^{N} \dfrac{a_i}{x - \lambda_i} }
=
\frac{\sum_{i=0}^N a_iw_i \cdot \prod_{j = 0, j \ne i}^{N}(x -\lambda_i)  }{ \sum_{i=0}^{N} a_i \cdot \prod_{j=0, j \ne i}^{N}(x - \lambda_i) }
$$

We can then write a simple script to compute the above numerator and denominator polynomials. First, we need a function that implements Vieta's formula. That is, it takes in a list of roots $\lambda_i$ and returns the coefficients of the polynomial $(x - \lambda_0)\cdots (x - \lambda_n)$. 

```python
def linear_product_coeffs(roots: list) -> np.array:
    """Return the coefficients (in descending degree) corresponding to the 
    polynomial (x - r_0)...(x - r_n), r_i a root."""
    num_roots = len(roots)
    coeffs = np.zeros(num_roots + 1) # A n-deg polynomial has (n+1) many coeffs
    coeffs[0] = 1  # The polynomial is monic
    
    for i in range(len(roots)):
        coeff = 0
        subset_size = i + 1
        # Generate ascending subsets of the indices of the roots 
        indices = itertools.combinations(range(num_roots), subset_size)
        for inds in indices:
            prod = 1 
            for elem in [roots[i] for i in inds ]:  # Get the roots corresponding to the subset
                prod *= elem
            coeff += (-1)**(i+1)*prod
        coeffs[i + 1] = coeff
        
    return coeffs
```
We can test the function on the polynomial $(x - 1)(x - 3)(x - 4) = x^3 - 8x^2 + 19x + 12$. Running our function, we get
```python
>>> linear_product_coeffs([1, 3, 4])
array([  1.,  -8.,  19., -12.])
```
and so our output makes sense. We can now use this function to calculate the actual coefficients of the rational function that interpolates our data via the following function.
```python
def approx_coeffs(nullspace: np.array, lambda_hat: list, w_hat: list) -> tuple:
    """Return the approximate coefficient values given the nullspace, lambda_hat, and w_hat
    calculations."""
    nullspace = nullspace.flatten() # Make nullspace a 1D vector
    
    numer = np.zeros(len(nullspace))
    denom = np.zeros(len(nullspace))
    
    for i in range(len(nullspace)):
        a_i = nullspace[i]
        w_i = w_hat[i]
        roots = lambda_hat[:i] + lambda_hat[i + 1:]  # Omit the i-th root
        numer += a_i*w_i*linear_product_coeffs(roots)
        denom += a_i*linear_product_coeffs(roots)
            
    return numer, denom
```
This function takes in the parameters used in the rational interpolation algorithm, including the nullspace vector obtained from calculating the Loewner matrix, and the $\hat{\lambda}$ and $\hat{w}$ quantities. 

## An example

Let us test this on the rational function with integer coefficients
$$
\frac{4}{8x + 1} - \frac{2}{8x + 4} - \frac{1}{8x + 5} - \frac{1}{8x + 6}.
$$
Then we can perform the rational interpolation algorithm for 100 equidistant points in $[-1, 1]$ to obtain the nullspace, $\hat{\lambda}$, and $\hat{w}$ vectors.
```python
x_data = [val for val in np.linspace(-1, 1, 100)]
def f(x):
    return 4/(8*x + 1) - 2/(8*x + 4) - 1/(8*x + 5) - 1/(8*x + 6)
y_data = [f(val) for val in x_data]

nullspace, lambda_hat, w_hat = loewner_matrix(x_data, y_data)
```
Running our function `approx_coeffs` then yields
```
numer, denom = approx_coeffs(nullspace, lambda_hat, w_hat)
print(numer, denom)
[ 1.17961196e-16  1.38777878e-16 -9.56549972e-05 -1.20365871e-04 -3.74648739e-05] 
[-4.08127988e-04 -8.16255976e-04 -5.67552983e-04 -1.54642245e-04 -1.19568747e-05]
```
Now this doesn't really mean much, it looks like a lot of random quantities. However, the correct answer is here, it is just corrupted by a lot of noise. What we can do now is interpret the quantities to be some kind of signal that indicates how much that power of $x$ participates in the interpolation. As we can see, $x^4, x^3$ in the numerator contribute very little *relative* to the other powers of $x$. Each power in the denominator contributes roughly equally.

Thus what we can do is eliminate $x^4$ and $x^3$ as possible candidates for integer coefficients
```python
numer[0] = 0
numer[1] = 0
```
and then divide the numerator and denominator by the smallest element (in absolute value) that we have. In this case, that is the last element of denominator. 
```python
numer /= denom[-1]
denom /= denom[-1]
print(numer)
print(denom)
[-0.         -0.          8.         10.06666667  3.13333333] 
[34.13333333 68.26666667 47.46666667 12.93333333  1.        ]
```
Now we clearly have rational coefficients, and turning them into integer coefficients simply requires multiplying both the numerator and denominator by primes until we achieve integers. In this case, that can be achieved by multiplying the top and bottom by 3 and 5. 
```python
numer *= 3 * 5
denom *= 3 * 5
print(numer)
print(denom)
[ -0.  -0. 120. 151.  47.] 
[ 512.         1023.99999999  712.          194.           15.        ]
```
And those are the correct integer coefficients since 
$$
\frac{4}{8x + 1} - \frac{2}{8x + 4} - \frac{1}{8x + 5} - \frac{1}{8x + 6}
=
\frac{120x^2 + 151x + 47}{512x^4 + 1024x^3 + 712x^2 + 194x + 15}.
$$
In general extracting exact mathematical solutions to any problem via computers is very challenging due to the fact that computers cannot represent all real numbers and so they are limited to floating point arithmetic. What we have is kind of nice since the steps we performed can be easily programmed, and we can use the rank of the Loewner matrix to tell if we should even bother trying to find integer coefficients.

## Making a script to do the work
