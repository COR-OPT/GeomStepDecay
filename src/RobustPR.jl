#! /usr/bin/env julia

module RobustPR

using Distributions
using LinearAlgebra
using Random
using Statistics

struct PhaseRetrievalProblem
  A :: Union{Matrix{Float64}, Distribution}
  y :: Vector{Float64}
  x :: Vector{Float64}
  pfail :: Float64
end

"""
  distance_to_solution(problem::PhaseRetrievalProblem, x::Vector{Float64})

Compute the distance of `x` to the solution of a phase retrieval `problem`.
"""
function distance_to_solution(problem::PhaseRetrievalProblem, x::Vector{Float64})
  return min(norm(x - problem.x), norm(x + problem.x))
end


"""
  sample_vectors(problem::PhaseRetrievalProblem, num_samples::Int) -> (A, y)

Sample `num_samples` design vectors and measurements from a phase retrieval problem.
In the streaming setting, newly sampled vectors are corrupted independently with
probability `problem.pfail` by large noise noise.

Returns:
- `A`: a `num_samples × d` matrix of design vectors, where `d` is problem dimension.
- `y`: a `num_samples × 1` matrix of the corresponding measurements.
"""
function sample_vectors(problem::PhaseRetrievalProblem, num_samples::Int)
  if isa(problem.A, Matrix)
    indices = rand(1:size(problem.A, 1), num_samples)
    return problem.A[indices, :], problem.y[indices]
  else
    d = length(problem.x); p = problem.pfail
    vectors = rand(problem.A, num_samples, d)
    measurements = (vectors * problem.x).^2
    # Replace a `p` fraction with large noise.
    replace!(x -> (rand() <= p) ? 10 * randn().^2 : x,
             measurements)
    return vectors, measurements
  end
end

"""
  subgradient(x::Vector{Float64}, a::Vector{Float64}, b::Float64) -> g

Compute a subgradient of the function `|b - (a'x)^2|`.
"""
function subgradient(
  x::Vector{Float64}, a::Vector{Float64}, b::Float64,
)
  return 2 * a * (sign((a'x)^2 - b) * (a'x))
end


"""
  mba(problem::PhaseRetrievalProblem, x₀::Vector{Float64}, α::Float64, K::Int, is_conv::Bool, prox_step::Function) -> x_out

Run the MBA algorithm with step size `α` for `K` iterations from `x₀`.
The update used in each step is given by `prox_step`, which is a callable
accepting a design vector, a measurement, the current point and a step size
and returns the next iterate.
"""
function mba(
  problem::PhaseRetrievalProblem, x₀::Vector{Float64}, α::Float64, K::Int,
  is_conv::Bool, prox_step::Function,
)
  # Generate one sample per iteration up-front.
  A, y = sample_vectors(problem, K)
  if is_conv
    running_sum = x₀[:]
    for k in 1:K
      x₀ = prox_step(A[k, :], y[k], x₀, α)
      running_sum .+= x₀
    end
    return (1 / (K + 1)) * running_sum
  else
    stop_time = rand(0:K)
    for k in 1:stop_time
      x₀ = prox_step(A[k, :], y[k], x₀, α)
    end
    return x₀
  end
end


"""
  rmba(problem::PhaseRetrievalProblem, x₀::Vector{Float64}, α::Float64, K::Int, T::Int, is_conv::Bool, prox_step::Function) -> g

Run the RMBA algorithm with initial step size `α` for `T` outer iterations,
each comprising of `K` inner iterations. The step sizes are geometrically
decreasing after each outer iteration.
The update used in each step is given by `prox_step`, which is a callable
accepting a design vector, a measurement, the current point and a step size
and returns the next iterate.
"""
function rmba(
  problem::PhaseRetrievalProblem, x₀::Vector{Float64}, α::Float64, K::Int,
  T::Int, is_conv::Bool, prox_step::Function,
)
  for t in 0:(T-1)
    step = α * 2^(-t)
    x₀ = mba(problem, x₀, step, K, is_conv, prox_step)
  end
  return x₀
end


function pmba(
  problem::PhaseRetrievalProblem, x₀::Vector{Float64}, ρ::Float64, α::Float64,
  K::Int, prox_step::Function,
)
  # Proximal center for the updates.
  x_base = x₀[:]
  stop_time = rand(0:K)
  A, y = sample_vectors(problem, stop_time)
  for k in 1:stop_time
    x₀ = prox_step(A[k, :], y[k], x₀, x_base, ρ, α)
  end
  return x₀
end


"""
  pairwise_distance(V::Matrix{Float64})

Given a `d × k` matrix, return a `k × k` matrix such that the (i, j)
element holds the distance between `V[:, i]` and `V[:, j]`.
"""
function pairwise_distance(V::Matrix{Float64})
  col_norms = sum(V.^2, dims=1)
  return col_norms .+ col_norms' - 2 * V'V
end


"""
  epmba(problem::PhaseRetrievalProblem, x₀::Vector{Float64}, ρ::Float64, α::Float64, K::Int, M::Int, ϵ::Float64, prox_step::Function)

Run the ensemble method with `M` independent trials of `pmba(problem, x₀, ρ, α, K, prox_step)`.
Among all points returned by each call to `pmba`, return the one that is the center of a ball
of radius `2 * ϵ` containing the most other points.
"""
function epmba(
  problem::PhaseRetrievalProblem, x₀::Vector{Float64}, ρ::Float64, α::Float64,
  K::Int, M::Int, ϵ::Float64, prox_step::Function,
)
  Ys = zeros(length(x₀), M)
  for i in 1:M
    Ys[:, i] = pmba(problem, x₀[:], ρ, α, K, prox_step)
  end
  pairwise_dists = pairwise_distance(Ys)
  most_idx = argmax(vec(sum(pairwise_dists .<= 2 * ϵ, dims=2)))
  return Ys[:, most_idx]
end


function rpmba(
  problem::PhaseRetrievalProblem, x₀::Vector{Float64}, ρ₀::Float64, α₀::Float64,
  K::Int, M::Int, T::Int, ϵ₀::Float64, prox_step::Function,
)
  for t in 0:(T - 1)
    ρ = 2^t * ρ₀
    α = 2^(-t) * α₀
    ϵ = 2^(-t) * ϵ₀
    x₀ = epmba(problem, x₀[:], ρ, α, K, M, ϵ, prox_step)
  end
  return x₀
end

# phase retrieval distance
dist(x, y) = min(norm(x - y), norm(x + y))
setup_step(v) = isa(v, Function) ? v : (_ -> v)

# mapping function for big corrupted entries
_bigCorr(p) = (x -> (rand() <= p) ? 10 * randn().^2 : x)


#= _pSgdInner: inner loop with constant step size =#
function _pSgdInner(prob::PrProb, xCurr, iters, λ; ρ=0.0, x0, ϵ)
  stopTime = rand(1:iters); As, ys = sampleVecs(prob, samples=stopTime)
  @inbounds for i = 1:stopTime  # pick unif. at random
    g = _subgrad(prob, xCurr, As[i, :], ys[i])
    # incorporate stabilization with x0, if ρ <> 0
    xCurr[:] = (xCurr + λ * ρ * x0 - λ * g) / (1 + λ * ρ)
    (i % 10 == 0) && (dist(xCurr, prob.x) <= ϵ) && return xCurr, i
  end
  return xCurr, stopTime
end


"""
  pSgd(prob::PrProb, δ, iters, inSched, λ=nothing)

Run the subgradient method consisting of an outer loop which adjust the
step size and an inner loop which runs a number of iterations with constant
step size. Initialize at a point ``\\delta``-close to the optimum, with
`inSched` being either a number or a callable implementing the number of
inner steps, which can adapt to the outer iteration index. λ is either a
number or callable implementing the inner step size schedule.
"""
function pSgd(prob::PrProb, δ, iters, inSched, λ; ρ=0.0, ϵ=1e-16)
  inIters = setup_step(inSched); λSched = setup_step(λ)
  ρSched = setup_step(ρ)
  d, m = length(prob.x), length(prob.y)
  rDir = randn(d); rDir /= norm(rDir)
  xinit = prob.x + norm(prob.x) * δ * rDir; x0 = copy(xinit)
  dists = fill(0.0, iters); totalEvals = 0
  for i = 1:iters
    dists[i] = dist(xinit, prob.x) / norm(prob.x)
    (dists[i] <= ϵ) && return xinit, dists[1:i], totalEvals
    ρi = ρSched(i)
    xinit[:], stopTime = _pSgdInner(prob, xinit, inIters(i),
                     λSched(i), ρ=ρi, x0=x0, ϵ=ϵ)
    totalEvals += stopTime
  end
  return xinit, dists, totalEvals
end


"""
  proximal_point_step(a::Vector{Float64}, b::Float64, x::Vector{Float64}, λ::Float64)

Compute the proximal point of the function |(a'x)^2 - b| with proximal penalty
`λ`.
"""
function proximal_point_step(
  a::Vector{Float64}, b::Float64, x::Vector{Float64}, λ::Float64,
)
  ip = a'x; a_norm = norm(a)
  # Possible stationary points.
  Xs = hcat((
    x - ( (2 * λ * ip) / (2 * λ * a_norm + 1) ) * a,
    x - ( (2 * λ * ip) / (2 * λ * a_norm - 1) ) * a,
    x - ( (ip + sqrt(b)) / a_norm ) * a,
    x - ( (ip - sqrt(b)) / a_norm ) * a)...)
  # Index yielding the minimum function value.
  min_idx = argmin(
    (abs.((a' * Xs).^2 .- b) .+ (1 / (2 * λ)) .* sum((Xs .- x).^2, dims=1))[:])
  return Xs[:, min_idx]
end


"""
  proximal_point_step(a::Vector{Float64}, b::Float64, x::Vector{Float64}, x_base::Vector{Float64}, ρ::Float64, λ::Float64)

Like `proximal_point_step(a, b, x, λ)`, but with an additional penalty of the
form `(ρ / 2) * |x - x_base|^2`.
"""
function proximal_point_step(
  a::Vector{Float64}, b::Float64, x::Vector{Float64}, x_base::Vector{Float64},
  ρ::Float64, λ::Float64,
)
  weight = (1 / (1 + λ * ρ))
  return proximal_point_step(a, b, weight * (x + λ * ρ * x_base) , weight * λ)
end

# Projection to interval [-1, 1].
proj_one(x) = min(abs(x), 1) * sign(x)


"""
  prox_linear_step(a::Vector{Float64}, b::Float64, x::Vector{Float64}, λ::Float64)

Compute a prox-linear step for the function `|(a'x)^2 - b|` and prox parameter
`λ`.
"""
function prox_linear_step(
  a::Vector{Float64}, b::Float64, x::Vector{Float64}, λ::Float64,
)
  ip = a'x
  γ = λ * (ip^2 - b)
  ζ = 2 * λ * ip .* a
  Δ = proj_one(-γ / (norm(ζ)^2)) .* ζ
  return x + Δ
end

function prox_linear_step(
  a::Vector{Float64}, b::Float64, x::Vector{Float64}, x_base::Vector{Float64},
  ρ::Float64, λ::Float64,
)
  weight = 1 / (1 + λ * ρ)
  return prox_linear_step(a, b, weight * (x + λ * ρ * x_base), weight * λ)
end

trunc(x, lb, ub) = max(min(x, ub), lb)

"""
  truncated_step(a::Vector{Float64}, b::Float64, x::Vector{Float64}, λ::Float64)

Perform one proximal step using a truncated first-order model for the function
|(a'x)^2 - b| using prox-parameter `λ`.
"""
function truncated_step(a::Vector{Float64}, b::Float64, x::Vector{Float64}, λ::Float64)
  res = (a'x)^2 - b
  (abs(res) ≤ 1e-15) && return x
  c = 2 * (a'x) * sign(res) * a
  return x - λ * trunc(abs(res) / (λ * norm(c)^2), 0, 1.0) * c
end

"""
  truncated_step(a::Vector{Float64}, b::Float64, x::Vector{Float64}, x_base::Vector{Float64}, ρ::Float64, λ::Float64)

Like `truncated_step(a, b, x, λ)`, but with an additional quadratic penalty of
the form `(ρ / 2) * |x - x_base|^2`.
"""
function truncated_step(
  a::Vector{Float64}, b::Float64, x::Vector{Float64}, x_base::Vector{Float64},
  ρ::Float64, λ::Float64,
)
  weight = 1 / (1 + λ * ρ)
  return truncated_step(a, b, weight * (x + ρ * λ * x_base), weight * λ)
end

"""
  subgradient_step(a::Vector{Float64}, b::Float64, x::Vector{Float64}, λ::Float64)

Perform one step of the subgradient method with step size `λ`.
"""
function subgradient_step(a::Vector{Float64}, b::Float64, x::Vector{Float64}, λ::Float64)
  return x - λ * subgradient(x, a, b)
end

"""
  subgradient_step(a::Vector{Float64}, b::Float64, x::Vector{Float64}, x_base::Vector{Float64}, ρ::Float64, λ::Float64)

Like `subgradient_step(a, b, x, λ)`, but with an additional quadratic penalty
`(ρ / 2) * |x - x_base|^2`.
"""
function subgradient_step(
  a::Vector{Float64}, b::Float64, x::Vector{Float64}, x_base::Vector{Float64},
  ρ::Float64, λ::Float64,
)
  weight = 1 / (1 + ρ * λ)
  return subgradient_step(a, b, weight * (x + λ * ρ * x_base), weight * λ)
end


function _trunc(x, a, b)
  return max(min(x, b), a)
end



#= proximal point update by taking the point with lowest prox value =#
function _prox_update_sampled(prob::PrProb, xk, ai, b, λ)
  ip = ai' * xk; aiNrmSq = norm(ai)^2
  Xs = hcat((
    xk - ( (2 * λ * ip) / (2 * λ * aiNrmSq + 1) ) * ai,
    xk - ( (2 * λ * ip) / (2 * λ * aiNrmSq - 1) ) * ai,
    xk - ( (ip + sqrt(b)) / aiNrmSq ) * ai,
    xk - ( (ip - sqrt(b)) / aiNrmSq ) * ai)...)
  # index
  minIdx = argmin(
    (abs.((ai' * Xs).^2 .- b) .+ (1 / (2 * λ)) .* sum((Xs .- xk).^2, dims=1))[:])
  return Xs[:, minIdx]
end

projOne(x) = min(abs(x), 1) * sign(x)

#= update for the prox-linear problem with parameter λ =#
function _proxlin_update_sampled(prob::PrProb, xk, ai, b, λ)
  ip = ai' * xk
  γ = λ * (ip^2 - b); ζ = 2 * λ * ip .* ai
  Δ⁺ = projOne(-γ / (norm(ζ)^2)) * ζ
  return xk + Δ⁺
end

function _inProx(prob::PrProb, xCurr, iters, λ; ρ, x0, ϵ)
  stopTime = rand(1:iters); As, ys = sampleVecs(prob, samples=stopTime)
  dnm = 1 + λ * ρ
  @inbounds for i = 1:stopTime
    xCurr[:] = _prox_update_sampled(prob, (xCurr + λ * ρ * x0) / dnm,
                    As[i, :], ys[i], λ / dnm)
    (i % 10 == 0) && (dist(xCurr, prob.x) <= ϵ) && return xCurr, i
  end
  return xCurr, stopTime
end


function _inProxlin(prob::PrProb, xCurr, iters, λ; ρ, x0, ϵ)
  stopTime = rand(1:iters); As, ys = sampleVecs(prob, samples=stopTime)
  dnm = 1 + λ * ρ
  @inbounds for i = 1:stopTime
    xCurr[:] = _proxlin_update_sampled(prob, (xCurr + λ * ρ * x0) / dnm,
                       As[i, :], ys[i], λ / dnm)
    (i % 10 == 0) && (dist(xCurr, prob.x) <= ϵ) && return xCurr, i
  end
  return xCurr, stopTime
end


"""
  sProx(prob::PrProb, delta, iters, inSched, λ, method=:proximal)

Run the stochastic proximal point method for the phase retrieval problem
for `iters` outer iterations, each of which consists of a number of inner
iterations with constant prox-parameter. Both `inSched` and `λ` can be
either scalars or callables implementing the inner step and prox-parameter
schedule, respectively. Start at an iterate with normalized distance
``delta`` from the ground truth.
"""
function sProx(prob::PrProb, delta, iters, inSched, λ; method=:proximal,
                 ρ=0.0, ϵ=1e-16)
  innerFn = (method == :proximal) ? _inProx : _inProxlin
  d, m = length(prob.x), length(prob.y)
  rDir = randn(length(prob.x)); rDir /= norm(rDir)
  xinit = prob.x + norm(prob.x) * delta * rDir; x0 = copy(xinit)
  dists = fill(0.0, iters); totalEvals = 0
  inIters = setup_step(inSched); λSched = setup_step(λ)
  ρSched = setup_step(ρ)
  for i=1:iters
    dists[i] = dist(xinit, prob.x) / norm(prob.x)
    (dists[i] <= ϵ) && return xinit, dists[1:i], totalEvals
    ρi = ρSched(i)  # apply regularization, if ρ <> 0
    xinit[:], stopTime = innerFn(prob, xinit, inIters(i), λSched(i),
                   ρ=ρi, x0=x0, ϵ=ϵ)
    totalEvals += stopTime
  end
  return xinit, dists, totalEvals
end


function _truncUpdate(prob::PrProb, xCurr, λ, a, y)
  r = (a' * xCurr)^2 - y
  if (abs(r) <= 1e-15)
    return xCurr
  end
  c = 2 * (a' * xCurr) * sign(r) * a
  return xCurr - λ * _trunc(abs(r) / (λ * norm(c)^2), 0, 1.0) * c
end


function _inTrunc(prob::PrProb, xCurr, iters, λ; ρ, x0, ϵ)
  stopTime = rand(1:iters); As, ys = sampleVecs(prob, samples=stopTime)
  dnm = 1 + λ * ρ
  @inbounds for i = 1:stopTime
    xCurr[:] = _truncUpdate(prob, (xCurr + λ * ρ * x0) / dnm,
                λ / dnm, As[i, :], ys[i])
    (i % 10 == 0) && (dist(xCurr, prob.x) <= ϵ) && return xCurr, i
  end
  return xCurr, stopTime
end


"""
  sTrunc(prob::PrProb, δ, iters, inSched, λ)

Run the truncated model method on a phase retrieval problem for a total
of `iters` outer iterations, each of which consists of a number of inner
iterations with constant prox-parameter. `inSched` and `λ` can be either
numbers or callables implementing the inner step and prox-parameter
schedules, respectively. Start at an iterate with normalized distance
`δ` from the ground truth.
"""
function sTrunc(prob::PrProb, δ, iters, inSched, λ; ρ=0.0, ϵ=1e-16)
  rDir = randn(length(prob.x)); normalize!(rDir)
  xinit = prob.x + norm(prob.x) * δ * rDir; x0 = copy(xinit)
  dists = fill(0.0, iters); totalEvals = 0
  inIters = setup_step(inSched); λSched = setup_step(λ)
  ρSched = setup_step(ρ)
  for i = 1:iters
    dists[i] = dist(xinit, prob.x) / norm(prob.x)
    (dists[i] <= ϵ) && return xinit, dists[1:i], totalEvals
    ρi = ρSched(i)
    xinit[:], stopTime = _inTrunc(prob, xinit, inIters(i), λSched(i),
                    ρ=ρi, x0=x0, ϵ=ϵ)
    totalEvals += stopTime
  end
  return xinit, dists, totalEvals
end


"""
  opt(prob::PrProb, δ, T, K, λ; ϵ=1e-16, method) -> (xSol, ds, tEv)

Run a low-probability version of one of the available optimization
`method`s on a phase retrieval problem for `T` outer iterations and `K`
inner iterations, starting from an estimate initialized at normalized
distance `δ` from the ground truth. The search is terminated when a
normalized accuracy of `ϵ` is achieved.
Return:
- `xSol`: the solution found by the method
- `ds`: a history of normalized distances to the solution set
- `tEv`: the total number of oracle calls
"""
function opt(prob::PrProb, δ, T, K, λ; ϵ=1e-16, method)
  if method == :subgradient
    return pSgd(prob, δ, T, K, λ, ϵ=ϵ)
  elseif method == :proximal
    return sProx(prob, δ, T, K, λ, ϵ=ϵ, method=:proximal)
  elseif method == :proxlinear
    return sProx(prob, δ, T, K, λ, ϵ=ϵ, method=:proxlinear)
  elseif method == :clipped
    return sTrunc(prob, δ, T, K, λ, ϵ=ϵ)
  else
    throw(ArgumentError("method $(method) not recognized"))
  end
end


#= determine optimization method =#
innerMethod(meth) = begin
  if meth == :subgradient
    return _pSgdInner
  elseif meth == :proxlinear
    return _inProxlin
  elseif meth == :proximal
    return _inProx
  else
    return _inTrunc
  end
end

# pairwise distance matrix
pairwiseDist(Y) = begin
  nMat = sum(Y.^2, dims=1);
  return nMat .+ nMat' - 2 * Y' * Y
end


function pgmba(prob::PrProb, y0, ρ, α, K, innerFn)
  return innerFn(prob, copy(y0), K, α, ρ=ρ, x0=y0, ϵ=1e-16)
end


function epgmba(prob::PrProb, y0, ρ, α, K, M, ϵ;
        method=:subgradient, Ys)
  innerFn = innerMethod(method); d = length(prob.x)
  if (Ys == nothing)
    Ys = fill(0.0, d, M)
  end
  for i = 1:M
    Ys[:, i], _ = pgmba(prob, y0, ρ, α, K, innerFn)
  end
  pDists = pairwiseDist(Ys)
  mostIdx = argmax(vec(sum(pDists .<= 2 * ϵ, dims=2)))
  return Ys[:, mostIdx]
end


"""
  sOpt(prob::BDProb, x0::Array{Float64, 1}, ρ0, α0, K, ϵ0, M, T; method) -> (xSol, ds)

Run the proximally regularized stochastic optimization algorithm using one
of the available `method`s (`:subgradient, :proximal, :proxlinear, :clipped`)
with initial estimate `x0` and parameters `ρ0` and `α0`, controlling the
proximal regularization and initial step size, respectively, as well as
clustering parameter `ϵ0`, using `K` inner and `T` outer
iterations and `M` independent trials per iteration. Return:
- `xSol`: the solution found by the method
- `ds`: a history of normalized distances to the solution set
"""
function sOpt(prob::PrProb, x0::Array{Float64, 1}, ρ0, α0, K, ϵ0, M, T;
        method)
  xCurr = copy(x0); dists = fill(0.0, T)
  d = length(prob.x); Ys = fill(0.0, d, M)
  for t = 0:(T-1)
    dists[t+1] = dist(xCurr, prob.x)
    ρ = 2.0^t * ρ0; ϵ = 2.0^(-t) * ϵ0; α = 2.0^(-t) * α0
    xCurr[:] = epgmba(prob, xCurr, ρ, α, K, M, ϵ, method=method, Ys=Ys)

  end
  return xCurr, dists
end


"""
  sOpt(prob::PrProb, δ::Float64, ρ0, α0, K, ϵ0, M, T; method) -> (xSol, ds)

Run the proximally regularized stochastic optimization algorithm using one
of the available `method`s (`:subgradient, :proximal, :proxlinear, :clipped`)
with an initial estimate `δ`-close to the ground truth and parameters `ρ0`
and `α0`, controlling the proximal regularization and initial step size,
respectively, as well as a clustering parameter `ϵ0`, using `K` inner and `T`
outer iterations and `M` independent trials per iteration.
Return:
- `xSol`: the solution found by the method
- `ds`: a history of normalized distances to the solution set
"""
function sOpt(prob::PrProb, δ::Float64, ρ0, α0, K, ϵ0, M, T; method)
  rndDir = randn(length(prob.x)); normalize!(rndDir)
  x0 = prob.x + rndDir * norm(prob.x) * δ
  return sOpt(prob, x0, ρ0, α0, K, ϵ0, M, T, method=method)
end


"""
  genProb(m::Int, d; pfail=0.0)

Generate a phase retrieval problem with `m` measurements and dimension
`d`. Optionally, corrupt a fraction of `pfail` measurements.
"""
function genProb(m::Int, d; pfail=0.0, outliers=:big)
  x = randn(d); A = randn(m, d); normalize!(x)
  num_corr = trunc(Int, pfail * m)
  y = (A * x).^2;
  if outliers == :big
    # big sparse outliers
    y[randperm(m)[1:num_corr]] = 10 * (randn(num_corr)).^2
  else
    y[randperm(m)[1:num_corr]] = (1 / sqrt(d)) * (randn(num_corr)).^2
  end
  return PrProb(A, y, x, pfail)
end


"""
  genProb(A::Distribution, d; pfail=0.0)

Generate a phase retrieval problem with dimension `d`, the measurement
vectors of which are generated from a distribution `A` (a `Distributions.jl`
object). Optionally, corrupt a fraction of `pfail` measurements.
"""
function genProb(A::Distribution, d; pfail=0.0)
  x = randn(d); normalize!(x)
  return PrProb(A, [0.0], x, pfail)
end


end
