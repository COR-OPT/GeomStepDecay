module LowRankMatrixSensing

import GeomStepDecay

using Distributions
using LinearAlgebra
using Random
using Statistics

const OptProblem = GeomStepDecay.OptProblem

struct PhaseRetrievalProblem <: OptProblem
  A::Union{Matrix{Float64}, Distribution}
  y::Vector{Float64}
  x::Vector{Float64}
  pfail::Float64
end

struct BilinearSensingProblem <: OptProblem
  L::Union{Matrix{Float64}, Distribution}
  R::Union{Matrix{Float64}, Distribution}
  y::Vector{Float64}
  w::Vector{Float64}
  x::Vector{Float64}
  pfail::Float64
end

"""
  distance_to_solution(problem::PhaseRetrievalProblem, x::Vector{Float64})

Compute the distance of `x` to the solution of a phase retrieval `problem`.
"""
function distance_to_solution(
  problem::PhaseRetrievalProblem, x::Vector{Float64},
)
  return min(norm(x - problem.x), norm(x + problem.x)) / norm(problem.x)
end

"""
  distance_to_solution(problem::BilinearSensingProblem, w::Vector{Float64}, x::Vector{Float64})

Compute the distance of `(w, x)` to the solution of a bilinear sensing `problem`.
"""
function distance_to_solution(
  problem::BilinearSensingProblem, w::Vector{Float64}, x::Vector{Float64},
)
  return norm(w .* x' .- problem.w .* problem.x') / (norm(problem.w) * norm(problem.x))
end


"""
  sample_vectors(problem::PhaseRetrievalProblem, num_samples::Int) -> (A, y)

Sample `num_samples` design vectors and measurements from a phase retrieval problem.
In the streaming setting, newly sampled vectors are corrupted independently with
probability `problem.pfail` by large noise.

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
  draw_sample(problem::PhaseRetrievalProblem)

Like `sample_vectors(problem, num_samples)` but with `num_samples = 1`.
"""
function draw_sample(problem::PhaseRetrievalProblem)
  A, y = sample_vectors(problem, 1)
  return A[1, :], y[1]
end


"""
  sample_vectors(problem::BilinearSensingProblem, num_samples::Int) -> (L, R, y)

Sample `num_samples` design vectors and measurements from a bilinear sensing problem.
In the streaming setting, newly sampled vectors are corrupted independently with
probability `problem.pfail` by large noise.

Returns:
- `L`: a `num_samples × d_1` matrix of design vectors, where `d_1` is the left dimension.
- `R`: a `num_samples × d_2` matrix of design vectors, where `d_2` is the right dimension.
- `y`: a `num_samples × 1` matrix of the corresponding measurements.
"""
function sample_vectors(problem::BilinearSensingProblem, num_samples::Int)
  if isa(problem.L, Matrix) && isa(problem.R, Matrix)
    indices = rand(1:length(problem.y), num_samples)
    return problem.L[indices, :], problem.R[indices, :], problem.y[indices]
  elseif isa(problem.L, Distribution) && isa(problem.R, Distribution)
    d1 = length(problem.w); d2 = length(problem.x); p = problem.pfail
    vectors_left = rand(problem.L, num_samples, d1)
    vectors_right = rand(problem.R, num_samples, d2)
    measurements = (vectors_left * problem.w) .* (vectors_right * problem.x)
    replace!(x -> (rand() ≤ p) ? 10 * randn() : x,
             measurements)
    return vectors_left, vectors_right, measurements
  else
    throw(ErrorException("problem.L and problem.R must be both matrices " *
                         "or distributions"))
  end
end


function draw_sample(problem::BilinearSensingProblem)
  L, R, y = sample_vectors(problem, 1)
  return L[1, :], R[1, :], y[1]
end

#=============================================#
#====== Phase retrieval implementation =======#
#=============================================#

function proximal_point_step(
  problem::PhaseRetrievalProblem, x::Vector{Float64}, step_size::Float64,
)
  a, b = draw_sample(problem); λ = step_size
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

# Projection to interval [-1, 1].
proj_one(x) = min(abs(x), 1) * sign(x)

function prox_linear_step(
  problem::PhaseRetrievalProblem, x::Vector{Float64}, step_size::Float64,
)
  λ = step_size; a, b = draw_sample(problem)
  ip = a'x
  γ = λ * (ip^2 - b)
  ζ = 2 * λ * ip .* a
  Δ = proj_one(-γ / (norm(ζ)^2)) .* ζ
  return x + Δ
end

_trunc(x, lb, ub) = max(min(x, ub), lb)

function truncated_step(
  problem::PhaseRetrievalProblem, x::Vector{Float64}, step_size::Float64,
)
  a, b = draw_sample(problem); λ = step_size
  res = (a'x)^2 - b
  (abs(res) ≤ 1e-15) && return x
  c = 2 * (a'x) * sign(res) * a
  return x - λ * _trunc(abs(res) / (λ * norm(c)^2), 0, 1.0) * c
end


function subgradient_step(
  problem::PhaseRetrievalProblem, x::Vector{Float64}, step_size::Float64,
)
  a, b = draw_sample(problem); λ = step_size
  g = 2 * a * (sign((a'x)^2 - b) * (a'x))
  return x - λ * g
end

function solution_distance(problem::PhaseRetrievalProblem, x::Vector{Float64})
  return min(norm(problem.x + x), norm(problem.x - x))
end


function mba(
  problem::PhaseRetrievalProblem, x₀::Vector{Float64}, step_size::Float64,
  num_iterations::Int, prox_step::Function; ϵ::Float64 = 1e-15,
)
  stop_condition = (problem, x, _) -> (solution_distance(problem, x) ≤ ϵ)
  return GeomStepDecay.mba_template(problem, x₀, step_size, num_iterations,
                                    false, prox_step,
                                    stop_condition=stop_condition)
end


function rmba(
  problem::PhaseRetrievalProblem, x₀::Vector{Float64}, initial_step_size::Float64,
  outer_iterations::Int, inner_iterations::Int, prox_step::Function;
  ϵ::Float64 = 1e-15,
)
  callback = (problem, x, _) -> solution_distance(problem, x)
  stop_condition = (problem, x, _) -> (solution_distance(problem, x) ≤ ϵ)
  return GeomStepDecay.rmba_template(problem, x₀, initial_step_size,
                                     outer_iterations, inner_iterations,
                                     false, prox_step, callback,
                                     stop_condition=stop_condition)
end


function pmba(
  problem::PhaseRetrievalProblem, x₀::Vector{Float64}, step_size::Float64,
  prox_penalty::Float64, num_iterations::Int, prox_step::Function;
  ϵ::Float64 = 1e-15,
)
  stop_condition = (problem, x, _) -> (solution_distance(problem, x) ≤ ϵ)
  return GeomStepDecay.pmba_template(problem, x₀, step_size, prox_penalty,
                                     num_iterations, prox_step,
                                     stop_condition=stop_condition)
end


function epmba(
  problem::PhaseRetrievalProblem, x₀::Vector{Float64}, step_size::Float64,
  prox_penalty::Float64, ensemble_radius::Float64, num_repeats::Int,
  num_iterations::Int, prox_step::Function; ϵ::Float64 = 1e-15,
)
  stop_condition = (problem, x, _) -> (solution_distance(problem, x) ≤ ϵ)
  return GeomStepDecay.epmba_template(problem, x₀, step_size, prox_penalty,
                                      ensemble_radius, num_repeats,
                                      num_iterations, prox_step,
                                      stop_condition=stop_condition)
end


function rpmba(
  problem::PhaseRetrievalProblem, x₀::Vector{Float64}, initial_step_size::Float64,
  initial_prox_penalty::Float64, initial_ensemble_radius::Float64,
  inner_iterations::Int, outer_iterations::Int, num_repeats::Int,
  prox_step::Function; ϵ::Float64 = 1e-15,
)
  callback = (problem, x, _) -> solution_distance(problem, x)
  stop_condition = (problem, x, _) -> (solution_distance(problem, x) ≤ ϵ)
  return GeomStepDecay.rpmba_template(problem, x₀, initial_step_size,
                                      initial_prox_penalty, initial_ensemble_radius,
                                      inner_iterations, outer_iterations,
                                      num_repeats, prox_step, callback,
                                      stop_condition=stop_condition)
end

"""
  generate_synthetic_problem(m::Int, d::Int, pfail::Float64, noise_stdev::Float64)

Generate a synthetic phase retrieval problem with `m` measurements in `d`
dimensions, with a `pfail` fraction of measurements corrupted by noise of
magnitude approximately `noise_stdev`.
"""
function generate_synthetic_phase_retrieval_problem(
  m::Int, d::Int, pfail::Float64, noise_stdev::Float64)
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
function generate_synthetic_phase_retrieval_problem(
  A::Distribution, d::Int, pfail::Float64
)
  return PhaseRetrievalProblem(A, [0.0], normalize(randn(d)), pfail)
end


#=============================================#
#====== Bilinear sensing implementation ======#
#=============================================#


end
