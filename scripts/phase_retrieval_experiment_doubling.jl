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

function main(d, pfail, δ, batch_size, method, streaming, ϵ_stop=(sqrt(d) * 1e-15))
  @assert (batch_size > 1 && method == "subgradient") || (batch_size == 1)
  μ = 1 - 2 * pfail
  ϵ = 1e-5
  T = trunc(Int, ceil(log2(2 * δ / ϵ)))
  @info "T = $T, d = $d"
  R = sqrt(δ) * μ
  callback(
    problem::PhaseRetrievalProblem,
    x::Vector{Float64},
    k_elapsed::Int,
    t::Int) =
  begin
    @info "Distance = $(distance_to_solution(problem, x))"
    return (
        dist_real = distance_to_solution(problem, x),
        dist_calc = 2.0^(-t) * R,
        iter_ind = k_elapsed * batch_size,
        passes_over_dataset = streaming ? 0 : (k_elapsed * batch_size / (8 * d)))
  end
  # Distribution with finite support.
  D = streaming ?
    Distributions.MultivariateNormal(fill(1.0, d)) :
    NormalBatch(randn(d, 8 * d))
  problem = generate_phase_retrieval_problem(D, pfail)
  step_fn = begin
    if method == "subgradient"
      (p, x, α) -> subgradient_step(p, x, α, batch_size=batch_size)
    elseif method == "proximal_point"
      proximal_point_step
    elseif method == "prox_linear"
      prox_linear_step
    else
      truncated_step
    end
  end
  x₀ = problem.x + δ * normalize(randn(d))
  _, callback_results = GeomStepDecay.rmba_template_doubling(
    problem,
    x₀,
    T,
    false,
    step_fn,
    callback,
    stop_condition = (p, x, _) -> (distance_to_solution(p, x) ≤ ϵ_stop),
    initial_κ = 1.0,
  )
  fname = "phase_retrieval_$(d)_$(batch_size)_$(@sprintf("%.2f", pfail))_doubling"
  if streaming
    fname *= "_streaming"
  end
  CSV.write("$(fname)_$(method).csv", DataFrame(callback_results))
end

settings = ArgParseSettings(
  description="Run the RMBA algorithm on synthetic phase retrieval.",
)
method_choices = [
  "subgradient",
  "proximal_point",
  "prox_linear",
  "truncated"
]
@add_arg_table! settings begin
  "--d"
    help = "Problem dimension"
    arg_type = Int
    required = true
  "--p"
    help = "Corruption probability"
    arg_type = Float64
    default = 0.0
  "--batch-size"
    help = "Batch size used in each random sample."
    arg_type = Int
    default = 1
  "--method"
    help = "The method used to form the stochastic models."
    arg_type = String
    range_tester = (x -> x ∈ method_choices)
  "--delta"
    help = "Initial normalized distance"
    arg_type = Float64
    default = 0.5
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
  parsed["method"],
  parsed["streaming"],
)