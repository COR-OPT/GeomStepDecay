import LinearAlgebra: norm, normalize
import Distributions

import GeomStepDecay

struct PhaseRetrievalProblem <: GeomStepDecay.OptProblem
  A::Distributions.Distribution
  x::Vector{Float64}
  pfail::Float64
end

function generate_problem(D::Distributions.Distribution, pfail::Float64 = 0.0)
  d = length(D)
  return PhaseRetrievalProblem(D, normalize(randn(d)), pfail)
end

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
  vectors = Matrix(rand(problem.A, num_samples)')
  measurements = (vectors * problem.x) .^ 2
  # Replace a `p` fraction with large noise.
  replace!(
    x -> (rand() <= p) ? 10 * randn().^2 : x,
    measurements,
  )
  return vectors, measurements
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

# Callback function for distance to solution.
distance_callback(problem::PhaseRetrievalProblem, x::Vector{Float64}, _) =
  distance_to_solution(problem, x)
