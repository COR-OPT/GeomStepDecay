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

function main(d, pfail, δ, batch_size, streaming, ϵ_stop=(sqrt(d) * 1e-15))
  μ = 1 - 2 * pfail
  L = sqrt(d / batch_size)
  δ_fail = 0.45
  ϵ = 1e-5
  T = trunc(Int, ceil(log2(2 * δ / ϵ)))
  K = trunc(Int, T * (L / (δ_fail * μ))^2)
  @info "T = $T, K = $K, d = $d"
  R = sqrt(δ) * μ
  α₀ = (R / L) * (1 / sqrt(K + 1))
  callback(problem::PhaseRetrievalProblem, x::Vector{Float64}, t::Int) =
    (dist_real = distance_to_solution(problem, x),
     dist_calc = 2.0^(-t) * R,
     iter_ind = t * K * batch_size,
     passes_over_dataset = streaming ? 0 : (t * K * batch_size / (8 * d)))
  # Distribution with finite support.
  D = streaming ?
    Distributions.MultivariateNormal(fill(1.0, d)) :
    NormalBatch(randn(d, 8 * d))
  problem = generate_phase_retrieval_problem(D, pfail)
  step_fn = (p, x, α) -> subgradient_step(p, x, α, batch_size=batch_size)
  x₀ = problem.x + δ * normalize(randn(d))
  _, callback_results = GeomStepDecay.rmba_template(
    problem,
    x₀,
    α₀,
    5 * T,
    K,
    false,
    step_fn,
    callback,
    stop_condition = (p, x, _) -> (distance_to_solution(p, x) ≤ ϵ_stop),
  )
  fname = "phase_retrieval_$(d)_$(batch_size)_$(@sprintf("%.2f", pfail))"
  if streaming
    fname *= "_streaming"
  end
  CSV.write("$(fname).csv", DataFrame(callback_results))
end

settings = ArgParseSettings(
  description="Run the stochastic subgradient algorithm on phase retrieval.",
)
@add_arg_table! settings begin
  "--d"
    help = "Problem dimension"
    arg_type = Int
  "--p"
    help = "Corruption probability"
    arg_type = Float64
    default = 0.0
  "--batch-size"
    help = "Batch size used in each random sample."
    arg_type = Int
  "--delta"
    help = "Initial normalized distance"
    arg_type = Float64
    default = 0.25
  "--streaming"
    help = "Set to use streaming instead of finite-sample measurements."
    action = :store_true
  "--seed"
    help = "The random generator seed."
    arg_type = Int
    default = 999
end

parsed = parse_args(settings)
Random.seed!(parsed["seed"])
main(
  parsed["d"],
  parsed["p"],
  parsed["delta"],
  parsed["batch-size"],
  parsed["streaming"],
)
