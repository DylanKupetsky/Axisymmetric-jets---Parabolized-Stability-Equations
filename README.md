# Recreation of results from Yen & Messersmith 1998

Here, we recreate the results from [Application of Parabolized Stability Equations to the Prediction of Jet Instabilities](https://arc.aiaa.org/doi/abs/10.2514/2.552?journalCode=aiaaj). 

The parabolized stability equations (PSE) offer an effecient approach for locating instabilities in boundary layer-type flows. Linear stability approaches assume the flow is locally parallel, whereas PSE assumes the flow is locally non-parallel. 

For the inviscid case, the initial condition is given by the most unstable mode for the generalized eigenvalue problem

$$ \frac{u'}{r} + \partial_r u' + \partial_z w' = 0 $$

$$ \partial_t u' + W\partial_z u' + \partial_r p' = 0$$

$$ \partial_t w' + W\partial_z w' + u'\partial_r W + \partial_z p' = 0$$

with

$$q'(r,z,t) = \hat{q}(r)\text{exp}\left(i(\alpha z + \omega t \right)) + c.c. $$

(c.c. is complex conjugate)

with $\alpha \in C$ and $\omega \in R$, and $u'$ the radial velocity, $w'$ the axial (so $z$ is the direction of the jet). With $\omega$ an input, the most negative imaginary part of $\alpha$, corresponding to the fastest downstream growth, is chosen as the solution for the problem. Associated eigenfunctions $\hat{q}(r)$ are chosen as initial conditions for the PSE algorithm.
You can read more about the PSE algorithm [here](https://www.annualreviews.org/content/journals/10.1146/annurev.fluid.29.1.245). The _ansatz_ is then updated to

$$q'(r,z,t) = \hat{q}(r,z)\text{exp}\left(i(\alpha(z) - \omega t \right)$$

so that now, z-derivatives of both $\hat{q}$ and $\alpha$ must be included in the numerical procedure.

Code requires MATLAB engine in Python for generalized eigenvalue problem as the initial condition for the instabilities, but can be modified to work with SciPy as well.
