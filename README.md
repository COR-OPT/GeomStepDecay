# GeomStepDecay
This repository contains a reference implementation for the Algorithms
appearing in the paper \[1\] for a selection of statistical recovery
problems, namely robust phase retrieval and blind deconvolution.

### Dependencies
The code has been tested in `Julia 1.0.2` and depends on a number of Julia
packages. For the core implementation, found under `src/`:

* `Distributions.jl`: for generating streaming versions of phase retrieval / blind
  deconvolution with arbitrary measurement vectors.
* `Polynomials.jl`: for solving the proximal subproblems in robust blind
  deconvolution, which reduce to finding the roots of a quartic polynomial.
* `MLDatasets`: for obtaining the `MNIST` dataset, for the sparse logistic
  regression problem.

For the remaining scripts which are aimed to reproduce some of the experimental
results found in the paper and can be found in the root directory of this repo,
the following packages are required:

* `ArgParse.jl`: for providing an `argparse`-like prompt for the command line.
* `PyPlot`: for access to the Matplotlib backend for plotting. Please refer to
  the installation instructions of `PyPlot` for details.
* `JLD`: for loading the solution found by full proximal gradient for the MNIST
  experiment.


### Quick Tour

We offer implementations for both the fixed and high probability variants of
our algorithms for robust phase retrieval and blind deconvolution.

The first step is to include the two libraries:

```julia
include("src/RobustPR.jl")   # for phase retrieval
include("src/RobustBD.jl")   # for blind deconvolution
```

The user can generate problem instances under both finite-sample and
streaming settings. For the former, measurement vectors are assumed Gaussian.
Let us generate such a problem with dimension `d = 100`, number of measurements
`m = 8 * d` and 15% of the entries corrupted with noise:

```julia
d = 100
probPR = RobustPR.genProb(8 * d, d, 0.15)  # 15% corrupted measurements
probBD = RobustBD.genProb(8 * d, d, 0.15)  # similarly
```

In the streaming setting, the user can pass arbitrary distributions which
satisfy the interface designated by [`Distributions.jl`](https://github.com/JuliaStats/Distributions.jl).
Below, we generate a blind deconvolution instance where the left measurement
vector is sampled from a normal distribution and the right measurement vector
is sampled from a *truncated* normal distribution in the range [-5, 5].

```julia
using Distributions

Ldist = Normal(0, 1); Rdist = TruncatedNormal(0, 1, -5, 5)
probBDStream = RobustBD.genProb(Ldist, Rdist, d)  # no noise by default
```

We can now proceed to solving the above problems. Both libraries expose two
generic optimization methods. We will look at blind deconvolution:

```julia
RobustBD.opt(prob, δ, K, T, λ; method)
RobustBD.sOpt(prob, δ, ρ0, α0, K, ε0, M, T; method)
```

The method `opt` is the constant probability variant, while the method `sOpt`
is the high-probability variant of the proposed algorithms.
In the above, `T` is the number of "outer" loops, `K` is the number of "inner"
loops, and `δ` is a specified initial distance from the solution set. The
parameter `λ` can be either a callable accepting a single argument
corresponding to the iteration counter, or a constant. For the rest of the
arguments appearing above, please refer to Section 3 of \[1\]. Finally, `method`
can be one of `:subgradient, :proximal, :proxlinear, :clipped`, corresponding
to different local models.

We demonstrate solving one of the above problems using the prox-linear method
with exponential step decay and initial step size of `.01`:

```julia
λSched = k -> 0.01 * 2.0^(-k)  # callable implementing step schedule
(wSol, xSol), ds, totalEv =
	RobustBD.opt(probBDStream, 0.1, 5000, 15, λSched, method=:proxlinear)
```

In the above, `wSol` and `xSol` are the final iterates found by the algorithm,
`ds` is a history of distances from the solution set (one for each outer loop)
and `totalEv` is the total number of inner iterations performed.

#### Running the MNIST experiment
In order to reproduce the sparse logistic regression experiment on the MNIST
dataset, we offer the script `plot_mnist.jl` together with a `JLD` file
`mnist_full.jld`. The latter contains the approximate solution found by running
the full proximal gradient method. To evaluate the performance of the RMBA
method, make sure `mnist_full.jld` is in the same directory as `plot_mnist.jl`
and run

```bash
julia -O3 plot_mnist.jl plot_rmba
```

To evaluate the RDA algorithm with `γ = 0.1`, (see \[2\] for details), type:

```bash
julia -O3 plot_mnist.jl plot_rda --gamma 0.1
```


\[1\]: Damek Davis, Dmitriy Drusvyatskiy, Vasileios Charisopoulos. *Stochastic Algorithms with geometric step decay converge linearly on sharp functions.*

\[2\]: Lee, Sangkyun, and Stephen J. Wright. *Manifold identification in dual averaging for regularized stochastic online learning.* Journal of Machine Learning Research 13.Jun (2012): 1705-1744.
