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

function main(d, pfail, δ, num_patterns, batch_size, ϵ_stop=(sqrt(d) * 1e-10))
  μ = 1 - 2 * pfail
  L = sqrt(d / batch_size)
  δ_fail = 0.45
  ϵ = 1e-5
  T = trunc(Int, ceil(log2(2 * δ / ϵ)))
  K = trunc(Int, T * (L / (δ_fail * μ))^2)
  @info "T = $T, K = $K, d = $d"
  R = sqrt(δ) * μ
  α₀ = (R / L) * (1 / sqrt(K + 1))
  callback(
    problem::HadamardPhaseRetrievalProblem,
    x::Vector{Float64},
    k_elapsed::Int,
    t::Int) =
    (dist_real = distance_to_solution(problem, x),
     dist_calc = 2.0^(-t) * R,
     iter_ind = k_elapsed * batch_size,
     passes_over_dataset = (k_elapsed * batch_size / (num_patterns * d)))
  stop_condition(
    problem::HadamardPhaseRetrievalProblem,
    x::Vector{Float64},
    _::Int) = distance_to_solution(problem, x) ≤ ϵ_stop
  problem = generate_hadamard_phase_retrieval_problem(d, num_patterns, pfail)
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
    stop_condition = stop_condition,
  )
  fname = "hadamard_phase_retrieval_$(d)_$(num_patterns)"
  CSV.write(
    "$(fname)_$(@sprintf("%.2f", pfail)).csv",
    DataFrame(callback_results),
  )
end

settings = ArgParseSettings(
  description="Run the stochastic subgradient algorithm on hadamard phase retrieval.",
)
@add_arg_table! settings begin
  "--d"
    help = "Problem dimension"
    arg_type = Int
  "--p"
    help = "Corruption probability"
    arg_type = Float64
    default = 0.0
  "--num-patterns"
    help = "Number of random sign patterns."
    arg_type = Int
  "--batch-size"
    help = "The size of each random batch of samples."
    arg_type = Int
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
main(
  parsed["d"],
  parsed["p"],
  parsed["delta"],
  parsed["num-patterns"],
  parsed["batch-size"],
)
