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

function slow_method(
  problem::LeastAbsoluteDeviationProblem,
  x₀::Vector{Float64},
  T::Int,
  K::Int,
  step_fn::Function,
  η::Float64,
)
  x = x₀[:]
  dist_history = []
  total_rounds = T * K
  elapsed = 0
  for _ in 1:T
    for _ in 1:K
      elapsed += 1
      step_size = (1 / total_rounds)^η
      x_new = step_fn(problem, x, step_size)
      x = (elapsed / (elapsed + 1)) * x + (1 / (elapsed + 1)) * x_new
    end
    push!(dist_history, norm(problem.x - x))
  end
  return dist_history
end

function main(d, inner_iters, pfail, δ, batch_size, method, streaming, η, ϵ_stop=(sqrt(d) * 1e-15))
  @assert (batch_size > 1 && method == "subgradient") || (batch_size == 1)
  μ = 1 - 2 * pfail
  L = sqrt(d / batch_size)
  ϵ = 1e-8
  T = trunc(Int, ceil(log2(2 * δ / ϵ)))
  K = inner_iters
  @info "T = $T, K = $K, d = $d"
  R = sqrt(δ) * μ
  α₀ = (R / L) * (1 / sqrt(K + 1))
  callback(
    problem::LeastAbsoluteDeviationProblem,
    x::Vector{Float64},
    k_elapsed::Int,
    t::Int) =
    (dist_real = distance_to_solution(problem, x),
     dist_calc = 2.0^(-(t - 1)) * R,
     dist_fake = R / sqrt((t - 1) * inner_iters + k_elapsed),
     iter_ind = k_elapsed * batch_size,
     passes_over_dataset = streaming ? 0 : (k_elapsed * batch_size / (8 * d)))
  # Distribution with finite support.
  D = streaming ?
    Distributions.MultivariateNormal(fill(1.0, d)) :
    NormalBatch(randn(d, 8 * d))
  problem = generate_least_absolute_deviation_problem(D, pfail)
  step_fn = (p, x, α) -> subgradient_step(p, x, α, batch_size=batch_size)
  x₀ = problem.x + δ * normalize(randn(d))
  _, callback_results = GeomStepDecay.rmba_template(
    problem,
    x₀[:],
    α₀,
    5 * T,
    K,
    true,
    step_fn,
    callback,
    stop_condition = (p, x, _) -> (distance_to_solution(p, x) ≤ ϵ_stop),
  )
  slow_results = slow_method(problem, x₀[:], 5 * T, K, step_fn, η)
  fname = "lad_$(d)_$(batch_size)_$(@sprintf("%.2f", pfail))"
  if streaming
    fname *= "_streaming"
  end
  CSV.write("$(fname)_$(method).csv", DataFrame(callback_results))
  fname_slow = "lad_$(d)_$(batch_size)_$(@sprintf("%.2f", pfail))_slow"
  CSV.write("$(fname_slow)_$(method).csv", DataFrame(history=Float64.(slow_results)))
  @show slow_results
end

settings = ArgParseSettings(
  description="Run the RMBA algorithm on synthetic phase retrieval.",
)
method_choices = [
  "subgradient",
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
  "--inner-iters"
    help = "Number of inner iterations."
    arg_type = Int
    default = 1
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
    default = 0.25
  "--decay-exponent"
    help = "The exponent for the slowly decaying step schedule."
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
  parsed["inner-iters"],
  parsed["p"],
  parsed["delta"],
  parsed["batch-size"],
  parsed["method"],
  parsed["streaming"],
  parsed["decay-exponent"],
)
