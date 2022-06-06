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

function main(d, pfail, δ, num_patterns, ϵ_stop=(sqrt(2d) * 1e-10))
  μ = 1 - 2 * pfail
  L = 1
  δ_fail = 0.45
  ϵ = 1e-5
  T = trunc(Int, ceil(log2(2 * δ / ϵ)))
  K = trunc(Int, T * (L / (δ_fail * μ))^2)
  @info "T = $T, K = $K, d = $d"
  R = sqrt(δ) * μ
  α₀ = (R / L) * (1 / sqrt(K + 1))
  callback(problem::HadamardBilinearSensingProblem, z::Vector{Float64}, t::Int) =
    (dist_real = distance_to_solution(problem, z),
     dist_calc = 2.0^(-t) * R,
     iter_ind = t * K * d,
     passes_over_dataset = (t * K / num_patterns))
  problem = generate_hadamard_bilinear_sensing_problem(d, d, num_patterns, pfail)
  step_fn = (p, x, α) -> subgradient_step(p, x, α)
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
  fname = "hadamard_bilinear_sensing_$(d)_$(num_patterns)"
  CSV.write(
    "$(fname)_$(@sprintf("%.2f", pfail)).csv",
    DataFrame(callback_results),
  )
end

settings = ArgParseSettings(
  description="Run the stochastic subgradient algorithm on hadamard bilinear sensing.",
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
)
