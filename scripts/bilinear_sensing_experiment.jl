using ArgParse
using CSV
using DataFrames
using Distributions
using LinearAlgebra
using Random
using Printf
using PyPlot

import GeomStepDecay

# enable latex printing
rc("text", usetex=true)

include("util.jl")

LBLUE="#519cc8"
MBLUE="#1d5996"
HBLUE="#908cc0"
TTRED="#ca3542"

struct TrialResult
  dist_real::Float64
  dist_calc::Float64
  iter_ind::Int
  sample_ind::Int
end

function main(d, pfail, δ, ϵ_stop=(sqrt(2 * d) * 1e-15))
  batch_size = trunc(Int, sqrt(d))
  μ = 1 - 2 * pfail
  L = sqrt(d / batch_size)
  η = 1.0
  δ_fail = 1 / 3
  ϵ = 1e-5
  T = trunc(Int, ceil(log2(2 * δ / ϵ)))
  K = trunc(Int, T * (L / (δ_fail * μ))^2)
  @info "T = $T, K = $K, d = $d"
  R = sqrt(δ) * μ
  α₀ = (R / L) * (1 / sqrt(K + 1))
  callback(problem::BilinearSensingProblem, z::Vector{Float64}, t::Int) =
    (dist_real = distance_to_solution(problem, z),
     dist_calc = 2.0^(-t) * R,
     iter_ind = t * K * batch_size)
  # Standard normal distribution
  DL = Distributions.MultivariateNormal(d, 1.0)
  DR = Distributions.MultivariateNormal(d, 1.0)
  problem = generate_bilinear_sensing_problem(DL, DR, pfail)
  step_fn = (p, z, α) -> subgradient_step(p, z, α, batch_size=batch_size)
  w₀ = problem.w + δ * normalize(randn(d))
  x₀ = problem.x + δ * normalize(randn(d))
  _, callback_results = GeomStepDecay.rmba_template(
    problem,
    [w₀; x₀],
    α₀,
    5 * T,
    K,
    false,
    step_fn,
    callback,
    stop_condition = (p, x, _) -> (distance_to_solution(p, x) ≤ ϵ_stop),
  )
  CSV.write(
    "bilinear_sensing_$(d)_$(@sprintf("%.2f", pfail)).csv",
    DataFrame(callback_results),
  )
end

settings = ArgParseSettings(
  description="Run the stochastic subgradient algorithm on bilinear sensing.",
)
@add_arg_table! settings begin
  "--d"
    help = "Problem dimension"
    arg_type = Int
  "--p"
    help = "Corruption probability"
    arg_type = Float64
    default = 0.2
  "--delta"
    help = "Initial normalized distance"
    arg_type = Float64
    default = 0.25
  "--seed"
    help = "The random generator seed."
    arg_type = Int
    default = 999
end

parsed = parse_args(settings)
Random.seed!(parsed["seed"])
main(parsed["d"], parsed["p"], parsed["delta"])
