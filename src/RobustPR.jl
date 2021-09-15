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
  mba(problem::PhaseRetrievalProblem, x₀::Vector{Float64}, α::Float64, K::Int, is_conv::Bool, prox_step::Function)

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
  rmba(problem::PhaseRetrievalProblem, x₀::Vector{Float64}, α::Float64, K::Int, T::Int, is_conv::Bool, prox_step::Function)

Run the RMBA algorithm with initial step size `α` for `T` outer iterations,
each comprising of `K` inner iterations. The step sizes are geometrically
decreasing after each outer iteration.
The update used in each step is given by `prox_step`, which is a callable
accepting a design vector, a measurement, the current point and a step size
and returns the next iterate.

Returns the final iterate as well as a vector of iterate distances to the
solution set.
"""
function rmba(
  problem::PhaseRetrievalProblem, x₀::Vector{Float64}, α::Float64, K::Int,
  T::Int, is_conv::Bool, prox_step::Function,
)
  distances = zeros(T)
  for t in 1:T
    step = α * 2^(-(t - 1))
    x₀ = mba(problem, x₀[:], step, K, is_conv, prox_step)
    distances[t] = distance_to_solution(problem, x₀)
  end
  return x₀, distances
end


"""
  pmba(problem::PhaseRetrievalProblem, x₀::Vector{Float64}, ρ::Float64, α::Float64, K::Int, prox_step::Function)

Run the PMBA algorithm, which is a version of `MBA(problem, x₀, α, K, prox_step)`
with an additional proximal penalty around `x₀` in each iteration.

Return the final iterate of the algorithm.
"""
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


"""
  rpmba(problem::PhaseRetrievalProblem, x₀::Vector{Float64}, ρ₀::Float64, α₀::Float64, K::Int, M::Int, T::Int, ϵ₀::Float64, prox_step::Function)

Run the restarted EPMBA algorithm, which comprised of `T` calls to `epmba(...)`
with geometrically decaying proximal penalty and geometrically increasing
proximal stabilization parameter.

Return the final iterate as well as a vector of iterate distances from the
solution set.
"""
function rpmba(
  problem::PhaseRetrievalProblem, x₀::Vector{Float64}, ρ₀::Float64, α₀::Float64,
  K::Int, M::Int, T::Int, ϵ₀::Float64, prox_step::Function,
)
  distances = zeros(T)
  for t in 0:(T - 1)
    ρ = 2^t * ρ₀
    α = 2^(-t) * α₀
    ϵ = 2^(-t) * ϵ₀
    x₀ = epmba(problem, x₀[:], ρ, α, K, M, ϵ, prox_step)
    distances[t+1] = distance_to_solution(problem, x₀)
  end
  return x₀, distances
end

# TODO: Add a parameter ε to terminate inner loop with dist(x, x₀) ≤ ε.

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

"""
  generate_synthetic_problem(m::Int, d::Int, pfail::Float64, noise_stdev::Float64)

Generate a synthetic phase retrieval problem with `m` measurements in `d`
dimensions, with a `pfail` fraction of measurements corrupted by noise of
magnitude approximately `noise_stdev`.
"""
function generate_synthetic_problem(m::Int, d::Int, pfail::Float64, noise_stdev::Float64)
  x = normalize(randn(d)); A = randn(m, d); y = (A * x).^2;
  num_corrupted = trunc(Int, pfail * m)
  y[randperm(m)[1:num_corrupted]] = noise_stdev .* ((randn(num_corrupted)).^2)
  return PhaseRetrievalProblem(A, y, x, pfail)
end


"""
  generate_synthetic_problem(A::Distribution, d::Int, pfail::Float64)

Generate a synthetic phase retrieval problem with design vectors sampled i.i.d.
from a distribution `A` in `d` dimensions, with a `pfail` fraction of
measurements corrupted.
"""
function generate_synthetic_problem(A::Distribution, d::Int, pfail::Float64)
  return PhaseRetrievalProblem(A, [0.0], normalize(randn(d)), pfail)
end

end
