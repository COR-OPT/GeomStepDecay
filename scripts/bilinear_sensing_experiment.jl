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

function main(d, pfail, δ, batch_size, method, streaming, ϵ_stop=(sqrt(2 * d) * 1e-15))
  @assert (batch_size > 1 && method == "subgradient") || (batch_size == 1)
  μ = 1 - 2 * pfail
  L = sqrt(2d / batch_size)
  δ_fail = 0.45
  ϵ = 1e-5
  T = trunc(Int, ceil(log2(2 * δ / ϵ)))
  K = trunc(Int, T * (L / (δ_fail * μ))^2)
  @info "T = $T, K = $K, d = $d"
  R = sqrt(δ) * μ
  α₀ = (R / L) * (1 / sqrt(K + 1))
  callback(
    problem::BilinearSensingProblem,
    z::Vector{Float64},
    k_elapsed::Int,
    t::Int) =
    (dist_real = distance_to_solution(problem, z),
     dist_calc = 2.0^(-t) * R,
     iter_ind = k_elapsed * batch_size,
     passes_over_dataset = streaming ? 0 : (k_elapsed * batch_size / (16 * d)))
  # Distributions with finite support.
  DL = streaming ?
    Distributions.MultivariateNormal(fill(1.0, d)) :
    NormalBatch(randn(d, 8 * d))
  DR = streaming ?
    Distributions.MultivariateNormal(fill(1.0, d)) :
    NormalBatch(randn(d, 8 * d))
  problem = generate_bilinear_sensing_problem(DL, DR, pfail)
  step_fn = begin
    if method == "subgradient"
      (p, z, α) -> subgradient_step(p, z, α, batch_size=batch_size)
    elseif method == "proximal_point"
      proximal_point_step
    elseif method == "prox_linear"
      prox_linear_step
    else
      truncated_step
    end
  end
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
  fname = "bilinear_sensing_$(d)_$(batch_size)_$(@sprintf("%.2f", pfail))"
  if streaming
    fname *= "_streaming"
  end
  CSV.write("$(fname)_$(method).csv", DataFrame(callback_results))
end

settings = ArgParseSettings(
  description="Run the RMBA algorithm on synthetic bilinear sensing.",
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
    default = 0.2
  "--batch-size"
    help = "The batch size of each random sample."
    arg_type = Int
    default = 1
  "--method"
    help = "The method used to form the stochastic models."
    arg_type = String
  "--delta"
    help = "Initial normalized distance"
    arg_type = Float64
    default = 0.25
  "--streaming"
    help = "Set to use a streaming instead of finite-sample measurement model."
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
