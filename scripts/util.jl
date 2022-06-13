import LinearAlgebra: norm, normalize
import Distributions
import Polynomials
import Random

import Hadamard: hadamard, ifwht

import GeomStepDecay

struct NormalBatch <: Distributions.Sampleable{
  Distributions.Multivariate, Distributions.Continuous}
  vectors::Matrix{Float64}
end

Base.length(s::NormalBatch) = size(s.vectors, 1)

# Sample a single vector from a `NormalBatch` distribution.
function Distributions._rand!(
  rng::Random.AbstractRNG,
  s::NormalBatch,
  x::AbstractVector{T}
) where T <: Real
  copyto!(x, s.vectors[:, rand(rng, 1:size(s.vectors, 2))])
  return x
end

# Sample multiple vectors from a `NormalBatch` distribution.
function Distributions._rand!(
  rng::Random.AbstractRNG,
  s::NormalBatch,
  A::AbstractMatrix{T}
) where T <: Real
  num_vectors = size(s.vectors, 2)
  num_samples = size(A, 2)
  if mod(num_vectors, num_samples) == 0
    num_blocks = num_vectors ÷ num_samples
    block_idx = rand(rng, 1:num_blocks)
    A[:, 1:num_samples] .=
      s.vectors[:, ((block_idx - 1) * num_samples + 1):(block_idx * num_samples)]
  else
    inds = rand(rng, 1:num_vectors, num_samples)
    A[:, 1:num_samples] .= s.vectors[:, inds]
  end
  return A
end

struct PhaseRetrievalProblem <: GeomStepDecay.OptProblem
  A::Distributions.Sampleable
  x::Vector{Float64}
  pfail::Float64
end

struct HadamardPhaseRetrievalProblem <: GeomStepDecay.OptProblem
  sign_patterns::Matrix{Int64}
  x::Vector{Float64}
  pfail::Float64
end

struct BilinearSensingProblem <: GeomStepDecay.OptProblem
  L::Distributions.Sampleable
  R::Distributions.Sampleable
  w::Vector{Float64}
  x::Vector{Float64}
  pfail::Float64
end

struct HadamardBilinearSensingProblem <: GeomStepDecay.OptProblem
  sign_patterns_left::Matrix{Int64}
  sign_patterns_right::Matrix{Int64}
  w::Vector{Float64}
  x::Vector{Float64}
  pfail::Float64
end

"""
  generate_phase_retrieval_problem(D::Distributions.Sampleable, pfail::Float64 = 0.0)

Generate a phase retrieval problem with measurement vectors sampled from a
distribution `D` and a `pfail` fraction of corrupted measurements.
"""
function generate_phase_retrieval_problem(
  D::Distributions.Sampleable,
  pfail::Float64 = 0.0,
)
  d = length(D)
  return PhaseRetrievalProblem(D, normalize(randn(d)), pfail)
end

"""
  generate_hadamard_phase_retrieval_problem(dim::Int, num_patterns::Int, pfail::Float64 = 0.0)

Generate a phase retrieval problem with measurement vectors drawn from the
random Hadamard ensemble with `num_patterns` random sign patterns and a `pfail`
fraction of corrupted measurements.
"""
function generate_hadamard_phase_retrieval_problem(
  dim::Int,
  num_patterns::Int,
  pfail::Float64 = 0.0,
)
  return HadamardPhaseRetrievalProblem(
    rand([-1, 1], dim, num_patterns),
    normalize(randn(dim)),
    pfail,
  )
end

"""
  generate_bilinear_sensing_problem(L::Distributions.Sampleable, R::Distributions.Sampleable,
                                    pfail::Float64 = 0.0)

Generate a bilinear sensing problem with measurement vectors sampled from a
pair of distributions `(L, R)` and a `pfail` fraction of corrupted
measurements.
"""
function generate_bilinear_sensing_problem(
  L::Distributions.Sampleable,
  R::Distributions.Sampleable,
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
  generate_hadamard_bilinear_sensing_problem(d₁::Int, d₂::Int, num_patterns::Int, pfail::Float64 = 0.0)

Generate a bilinear sensing problem with measurement vectors sampled from the
randomized Hadamarad ensemble and a `pfail` fraction of corrupted measurements.
"""
function generate_hadamard_bilinear_sensing_problem(
  d₁::Int,
  d₂::Int,
  num_patterns::Int,
  pfail::Float64 = 0.0,
)
  return HadamardBilinearSensingProblem(
    rand([-1, 1], d₁, num_patterns),
    rand([-1, 1], d₂, num_patterns),
    normalize(randn(d₁)),
    normalize(randn(d₂)),
    pfail,
  )
end

"""
  distance_to_solution(problem::Union{PhaseRetrievalProblem, HadamardPhaseRetrievalProblem}, x::Vector{Float64})

Compute the distance of a vector `x` to the solution of `problem`.
"""
function distance_to_solution(
  problem::Union{PhaseRetrievalProblem, HadamardPhaseRetrievalProblem},
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
  problem::Union{BilinearSensingProblem, HadamardBilinearSensingProblem},
  z::Vector{Float64},
)
  d₁ = length(problem.w)
  d₂ = length(problem.x)
  w = view(z, 1:d₁)
  x = view(z, (d₁+1):(d₁+d₂))
  return sqrt(abs(
    norm(w)^2 * norm(x)^2 + norm(problem.w)^2 * norm(problem.x)^2 -
      2 * (w' * problem.w) * (x' * problem.x)))
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

"""
  proximal_point_step(problem::PhaseRetrievalProblem, x::Vector{Float64}, step_size::Float64)

Take one step of the proximal point method for `problem` starting at `x`
with a given `step_size`.
"""
function proximal_point_step(
  problem::PhaseRetrievalProblem,
  x::Vector{Float64},
  step_size::Float64,
)
  A, y = generate_samples(problem, 1)
  a = A[1, :]
  b = y[1]
  ip = a'x
  a_norm = norm(a)
  # Possible stationary points.
  Xs = reshape(
    [
      x - ( (2 * step_size * ip) / (2 * step_size * a_norm + 1) ) * a;
      x - ( (2 * step_size * ip) / (2 * step_size * a_norm - 1) ) * a;
      x - ( (ip + sqrt(b)) / a_norm ) * a;
      x - ( (ip - sqrt(b)) / a_norm ) * a
    ],
    length(x),
    4,
  )
  # Index yielding the minimum function value.
  min_idx = argmin(
    (abs.((a' * Xs).^2 .- b) .+ (1 / (2 * step_size)) .* sum((Xs .- x).^2, dims=1))[:])
  return Xs[:, min_idx]
end

function proximal_point_step(
  problem::BilinearSensingProblem,
  z::Vector{Float64},
  step_size::Float64,
)
  d₁ = length(problem.w)
  d₂ = length(problem.x)
  w = view(z, 1:d₁)
  x = view(z, (d₁ + 1):(d₁ + d₂))
  L, R, y = generate_samples(problem, 1)
  ℓ = L[1, :]
  r = R[1, :]
  b = y[1]
  ℓ_w = ℓ'w
  r_x = r'x
  nrmw_sq = norm(ℓ)^2
  nrmx_sq = norm(x)^2
  denom = 1 - step_size^2 * nrmw_sq * nrmx_sq
  # Case 1: ℓ_w * r_x ≠ b.
  # Put resulting vectors into separate columns.
  Ws = hcat(
    w - step_size * (( r_x - step_size * nrmx_sq * ℓ_w ) / denom) * ℓ,
    w - step_size * ((-r_x - step_size * nrmx_sq * ℓ_w ) / denom) * ℓ,
  )
  Xs = hcat(
    x - step_size * (( ℓ_w - step_size * nrmw_sq * r_x ) / denom) * r,
    x - step_size * ((-ℓ_w - step_size * nrmw_sq * ℓ_w ) / denom) * r,
  )
  # Case 2: ℓ_w * r_x = b.
  # Find roots of quartic.
  p = Polynomials.Polynomial(
    [-b^2 * nrmw_sq, b * nrmw_sq * r_x, 0.0, nrmx_sq * ℓ_w, nrmx_sq]
  )
  ηs = real.(filter(isreal, Polynomials.roots(p)))
  # Get coefficients.
  if length(ηs) > 0
    γs = (ηs .* ℓ_w - ηs.^2) / (b * nrmw_sq)
    Ws = hcat(Ws, w .- (ℓ .* (γs .* (b ./ ηs))'))
    Xs = hcat(Xs, x .- (r .* (γs .* ηs)'))
  end
  costs = abs.((ℓ' * Ws) .* (r' * Xs) .- b) .+
    (1 / (2 * step_size)) .*
      sum((Ws .- w).^2, dims=1) + sum((Xs .- x).^2, dims=1)
  min_idx = argmin(costs[:])
  return [Ws[:, min_idx]; Xs[:, min_idx]]
end

# Projection to interval [-1, 1].
proj_one(x) = min(abs(x), 1) * sign(x)

"""
  prox_linear_step(problem::PhaseRetrievalProblem, x::Vector{Float64}, step_size::Float64)

Take one step of the prox-linear method for `problem` starting at `x` with a
given `step_size`.
"""
function prox_linear_step(
  problem::PhaseRetrievalProblem,
  x::Vector{Float64},
  step_size::Float64,
)
  A, y = generate_samples(problem, 1)
  a = A[1, :]
  b = y[1]
  ip = a'x
  γ = step_size * (ip^2 - b)
  ζ = 2 * step_size * ip .* a
  Δ = proj_one(-γ / (norm(ζ)^2)) .* ζ
  return x + Δ
end

"""
  prox_linear_step(problem::BilinearSensingProblem, z::Vector{Float64}, step_size::Float64)

Take one step of the prox-linear method for `problem` starting at `z = (w, x)`
with a given `step_size`.
"""
function prox_linear_step(
  problem::BilinearSensingProblem,
  z::Vector{Float64},
  step_size::Float64,
)
  d₁, d₂ = length(problem.w), length(problem.x)
  w = view(z, 1:d₁)
  x = view(z, (d₁ + 1):(d₁ + d₂))
  L, R, y = generate_samples(problem, 1)
  ℓ = L[1, :]
  r = R[1, :]
  b = y[1]
  γ = step_size * ((ℓ'w) * (r'x) - b)
  g = step_size .* [ℓ * (r'x); r * (ℓ'w)]
  return z .+ proj_one(-γ / (norm(g)^2)) .* g
end

_trunc(x, lb, ub) = max(min(x, ub), lb)

"""
  truncated_step(problem::PhaseRetrievalProblem, x::Vector{Float64}, step_size::Float64)

Take one truncated step for `problem` starting at `x` with a given `step_size`.
"""
function truncated_step(
  problem::PhaseRetrievalProblem,
  x::Vector{Float64},
  step_size::Float64,
)
  A, y = generate_samples(problem, 1)
  a = A[1, :]
  b = y[1]
  res = (a'x)^2 - b
  (abs(res) ≤ 1e-15) && return x
  c = 2 * (a'x) * sign(res) * a
  return x - step_size * _trunc(abs(res) / (step_size * norm(c)^2), 0, 1.0) * c
end

"""
  truncated_step(problem::BilinearSensingProblem, z::Vector{Float64}, step_size::Float64)

Take one truncated step for `problem` starting at `z = (w, x)` with a given
`step_size`.
"""
function truncated_step(
  problem::BilinearSensingProblem,
  z::Vector{Float64},
  step_size::Float64,
)
  d₁ = length(problem.w)
  d₂ = length(problem.x)
  w = view(z, 1:d₁)
  x = view(z, (d₁ + 1):(d₁ + d₂))
  L, R, y = generate_samples(problem, 1)
  ℓ = L[1, :]
  r = R[1, :]
  b = y[1]
  res = (ℓ'w) * (r'x) - b
  (abs(res) <= 1e-15) && return z
  g = sign(res) * [(r'x) * ℓ; (ℓ'w) * x]
	g_fact = _trunc(abs(res) / (step_size * norm(g)^2), 0, 1.0)
  return z - (step_size * g_fact) .* g
end

opA(S::AbstractMatrix{Int64}, v::AbstractVector{Float64}) = vec(ifwht(S .* v, 1))

opAT(S::AbstractMatrix{Int64}, v::AbstractVector{Float64}) = begin
  d, k = size(S)
  return (S .* ifwht(reshape(v, d, k), 1)) * ones(k)
end

function opAT(
  S::AbstractMatrix{Int64},
  v::AbstractVector{Float64},
  mask::BitVector,
)
  d, k = size(S)
  return (S .* ifwht(reshape(v .* mask, d, k), 1)) * ones(k)
end

"""
  subgradient_step(problem::HadamardPhaseRetrievalProblem, x::Vector{Float64}, step_size::Float64)

Take one step of the subgradient method for `problem` starting at `x` with a
given `step_size`.
"""
function subgradient_step(
  problem::HadamardPhaseRetrievalProblem,
  x::Vector{Float64},
  step_size::Float64;
  batch_size::Int = length(problem.x)
)
  d = length(x)
  num_patterns = size(problem.sign_patterns, 2)
  S = problem.sign_patterns[:, rand(1:num_patterns, 1)]
  y = opA(S, problem.x) .^ 2
  Ax = opA(S, x)
  # Take into account batch sizes that are not d.
  # If the batch size is smaller than d, we find a subblock to use
  # in the subgradient calculation instead.
  if batch_size ≠ d
    block_index = rand(1:(d ÷ batch_size))
    ind_lo = (block_index - 1) * batch_size + 1
    ind_hi = block_index * batch_size
    # Binary mask for subrows.
    mask = ind_lo .≤ (1:d) .≤ ind_hi
    return x - (2 * step_size / batch_size) *
      opAT(S, sign.(Ax .^ 2 .- y) .* Ax, mask)
  else
    return x - (2 * step_size / batch_size) *
      opAT(S, sign.(Ax.^2 .- y) .* Ax)
  end
end

"""
  subgradient_step(problem::HadamardBilinearSensingProblem, z::Vector{Float64}, step_size::Float64)

Take one step of the subgradient method for `problem` starting at `z = [w; x]`
with a given `step_size`.
"""
function subgradient_step(
  problem::HadamardBilinearSensingProblem,
  z::Vector{Float64},
  step_size::Float64;
  batch_size::Int = length(problem.w)
)
  d₁ = length(problem.w)
  d₂ = length(problem.x)
  w = view(z, 1:d₁)
  x = view(z, (d₁ + 1):(d₁ + d₂))
  num_patterns = size(problem.sign_patterns_left, 2)
  block_idx = rand(1:num_patterns, 1)
  S₁ = problem.sign_patterns_left[:, block_idx]
  S₂ = problem.sign_patterns_right[:, block_idx]
  y = opA(S₁, problem.w) .* opA(S₂, problem.x)
  # Subgradient for the given batch.
  Lw = opA(S₁, w)
  Rx = opA(S₂, x)
  s = sign.(Lw .* Rx .- y)
  if batch_size ≠ d₁
    block_index = rand(1:(d₁ ÷ batch_size))
    ind_lo = (block_index - 1) * batch_size + 1
    ind_hi = block_index * batch_size
    # Binary mask for subrows.
    mask = ind_lo .≤ (1:d₁) .≤ ind_hi
    return z - (step_size / batch_size) * [
      opAT(S₁, Rx .* s, mask);
      opAT(S₂, Lw .* s, mask)
    ]
  else
    return z - (step_size / batch_size) * [
      opAT(S₁, Rx .* s);
      opAT(S₂, Lw .* s)
    ]
  end
end


# Callback functions for distance to solution.
distance_callback(
  problem::Union{
    PhaseRetrievalProblem,
    HadamardPhaseRetrievalProblem,
    BilinearSensingProblem,
    HadamardBilinearSensingProblem
  },
  z::Vector{Float64},
  _,
) = distance_to_solution(problem, z)
