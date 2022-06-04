import LinearAlgebra: norm, normalize
import Distributions

import GeomStepDecay

struct PhaseRetrievalProblem <: GeomStepDecay.OptProblem
  A::Distributions.Distribution
  x::Vector{Float64}
  pfail::Float64
end

struct BilinearSensingProblem <: GeomStepDecay.OptProblem
  L::Distributions.Distribution
  R::Distributions.Distribution
  w::Vector{Float64}
  x::Vector{Float64}
  pfail::Float64
end

"""
  generate_phase_retrieval_problem(D::Distributions.Distribution, pfail::Float64 = 0.0)

Generate a phase retrieval problem with measurement vectors sampled from a
distribution `D` and a `pfail` fraction of corrupted measurements.
"""
function generate_phase_retrieval_problem(
  D::Distributions.Distribution,
  pfail::Float64 = 0.0,
)
  d = length(D)
  return PhaseRetrievalProblem(D, normalize(randn(d)), pfail)
end

"""
  generate_bilinear_sensing_problem(L::Distributions.Distribution, R::Distributions.Distribution,
                                    pfail::Float64 = 0.0)

Generate a bilinear sensing problem with measurement vectors sampled from a
pair of distributions `(L, R)` and a `pfail` fraction of corrupted
measurements.
"""
function generate_bilinear_sensing_problem(
  L::Distributions.Distribution,
  R::Distributions.Distribution,
  pfail::Float64 = 0.0,
)
  d₁ = length(L)
  d₂ = length(R)
  return BilinearSensingProblem(
    L,
    R,
    normalize(randn(d₁)),
    normalize(randn(d₂)),
    pfail,
  )
end

"""
  distance_to_solution(problem::PhaseRetrievalProblem, x::Vector{Float64})

Compute the distance of a vector `x` to the solution of `problem`.
"""
function distance_to_solution(
  problem::PhaseRetrievalProblem,
  x::Vector{Float64},
)
  return min(
    norm(x - problem.x),
    norm(x + problem.x),
  )
end

"""
  distance_to_solution(problem::BilinearSensingProblem, z::Vector{Float64})

Compute the distance of a vector `z = (w, x)` to the solution of `problem`.
"""
function distance_to_solution(
  problem::BilinearSensingProblem,
  z::Vector{Float64},
)
  d₁ = length(problem.L)
  w = z[1:d₁]
  x = z[(d₁+1):end]
  return norm(w .* x' .- problem.w .* problem.x')
end

"""
  generate_samples(problem::PhaseRetrievalProblem, num_samples::Int) -> (vectors, measurements)

Generate `num_samples` measurement vectors and measurements for a
`PhaseRetrievalProblem`.

Returns:
- `vectors::Matrix{Float64}`: a `num_samples × d` matrix of measurement vectors.
- `measurements::Vector{Float64}`: a vector holding the `num_samples` resulting measurements.
"""
function generate_samples(
  problem::PhaseRetrievalProblem,
  num_samples::Int,
)
  p = problem.pfail
  vectors = rand(problem.A, num_samples)'
  measurements = (vectors * problem.x) .^ 2
  # Replace a `p` fraction with large noise.
  num_rep = trunc(Int, p * num_samples)
  measurements[1:num_rep] .= 10 * randn(num_rep) .^ 2
  return vectors, measurements
end

"""
  generate_samples(problem::PhaseRetrievalProblem, num_samples::Int) -> (vectors_left, vectors_right, measurements)

Generate `num_samples` measurement vectors and measurements for a
`BilinearSensingProblem`.

Returns:
- `vectors_left::Matrix{Float64}`: a `num_samples × d₁` matrix of measurement vectors.
- `vectors_right::Matrix{Float64}`: a `num_samples × d₂` matrix of measurement vectors.
- `measurements::Vector{Float64}`: a vector holding the `num_samples` resulting measurements.
"""
function generate_samples(
  problem::BilinearSensingProblem,
  num_samples::Int,
)
  p = problem.pfail
  vectors_left = rand(problem.L, num_samples)'
  vectors_right = rand(problem.R, num_samples)'
  measurements = (vectors_left * problem.w) .* (vectors_right * problem.x)
  # Replace a `p` fraction with large noise.
  num_rep = trunc(Int, p * num_samples)
  measurements[1:num_rep] .= 10 .* randn(num_rep)
  return vectors_left, vectors_right, measurements
end

"""
  subgradient_step(problem::PhaseRetrievalProblem, x::Vector{Float64}, step_size::Float64;
                   batch_size::Int = 1)

Take one step of the subgradient method for `problem` starting at `x` with a
given `step_size`. Use a batch of size `batch_size` when computing the
subgradient.
"""
function subgradient_step(
  problem::PhaseRetrievalProblem,
  x::Vector{Float64},
  step_size::Float64;
  batch_size::Int = 1,
)
  A, y = generate_samples(problem, batch_size)
  # Subgradient for given batch.
  subgrad = (2 / batch_size) * A' * (sign.((A * x).^2 .- y) .* A * x)
  return x - step_size * subgrad
end

"""
  subgradient_step(problem::BilinearSensingProblem, z::Vector{Float64}, step_size::Float64;
                   batch_size::Int = 1)

Take one step of the subgradient method for `problem` starting at `z = (w, x)`
with a given `step_size`. Use a batch of size `batch_size` when computing the
subgradient.
"""
function subgradient_step(
  problem::BilinearSensingProblem,
  z::Vector{Float64},
  step_size::Float64;
  batch_size::Int = 1,
)
  d₁ = length(problem.L)
  w = z[1:d₁]
  x = z[(d₁+1):end]
  L, R, y = generate_samples(problem, batch_size)
  # Subgradient for the given batch.
  Lw = L * w
  Rx = R * x
  s = (1 / batch_size) .* sign.(Lw .* Rx .- y)
  return z - step_size * [
    (Rx .* L)' * s;
    (Lw .* R)' * s
  ]
end

# Callback functions for distance to solution.
distance_callback(problem::PhaseRetrievalProblem, x::Vector{Float64}, _) =
  distance_to_solution(problem, x)

distance_callback(problem::BilinearSensingProblem, z::Vector{Float64}, _) =
  distance_to_solution(problem, z)
