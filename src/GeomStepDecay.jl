#! /usr/bin/env julia

module GeomStepDecay

using Distributions
using LinearAlgebra
using Random
using Statistics

abstract type OptProblem end

function mba_template(
  problem::OptProblem, x₀::Vector{Float64}, step_size::Float64,
  num_iterations::Int, is_conv::Bool, prox_step::Function;
  stop_condition::Function = nothing
)
  # If no condition provided, set to always false.
  stop_condition = (stop_condition == nothing) ? ((_...) -> false) : stop_condition
  if is_conv
    running_sum = x₀[:]
    for k in 1:num_iterations
      x₀ = prox_step(problem, x₀, step_size)
      running_sum .+= x₀[:]
      if stop_condition(problem, x₀, k)
        @debug "Stopping early at iteration: $(k)"
        return (1 / (k + 1)) * running_sum
      end
    end
    return (1 / (num_iterations + 1)) * running_sum
  else
    stop_time = rand(0:num_iterations)
    for k in 1:stop_time
      x₀ = prox_step(problem, x₀, step_size)
      if stop_condition(problem, x₀, k)
        @debug "Stopping early at iteration: $(k)"
        return x₀
      end
    end
    return x₀
  end
end


function rmba_template(
  problem::OptProblem, x₀::Vector{Float64}, initial_step_size::Float64,
  outer_iterations::Int, inner_iterations::Int, is_conv::Bool,
  prox_step::Function, callback::Function;
  stop_condition::Function = nothing
)
  callback_results = zeros(outer_iterations)
  for t in 1:outer_iterations
    step = initial_step_size * 2^(-(t - 1))
    x₀ = mba(problem, x₀[:], step, inner_iterations, is_conv, prox_step,
             stop_condition=stop_condition)
    callback_results[t] = callback(problem, x₀, t)
  end
  return x₀, callback_results
end


function pmba_template(
  problem::OptProblem, x₀::Vector{Float64}, step_size::Float64,
  prox_penalty::Float64, num_iterations::Int, prox_step::Function;
  stop_condition::Function = nothing
)
  # If stop_condition is unset, make it always false.
  stop_condition = (stop_condition == nothing) ? ((_...) -> false) : stop_condition
  # Proximal center for the updates.
  x_base = x₀[:]
  stop_time = rand(0:num_iterations)
  weight = 1 / (step_size + prox_penalty)
  for k in 1:stop_time
    # Including the proximal penalty is equivalent to evaluating
    # the proximal operator at a nearby point with a slightly different
    # proximal step.
    x₀ = prox_step(problem,
                   weight * (step_size * x₀ + prox_penalty * step_size * x_base),
                   weight * step_size)
    if stop_condition(problem, x₀, k)
      @debug "Stopping early at iteration: $(k)"
      return x₀
    end
  end
  return x₀
end


function epmba_template(
  problem::OptProblem, x₀::Vector{Float64}, step_size::Float64,
  prox_penalty::Float64, ensemble_radius::Float64, num_repeats::Int,
  num_iterations::Int, prox_step::Function;
  stop_condition::Function = nothing
)
  ϵ = ensemble_radius
  Xs = zeros(length(x₀), num_repeats)
  for i in 1:num_repeats
    Xs[:, i] = pmba_template(problem, x₀[:], step_size, prox_penalty,
                             num_iterations, prox_step,
                             stop_condition=stop_condition)
  end
  pairwise_dists = pairwise_distance(Xs)
  most_idx = argmax(vec(sum(pairwise_dists .<= 2 * ϵ, dims=2)))
  return Xs[:, most_idx]
end


"""
  rpmba_template(problem::OptProblem, x₀::Vector{Float64}, initial_step_size::Float64,
                 initial_prox_penalty::Float64, initial_ensemble_radius::Float64,
                 inner_iterations::Int, outer_iterations::Int, num_repeats::Int,
                 prox_step::Function, callback::Function; stop_condition::Function = nothing) -> (x, callback_results)

Run the RPMBA algorithm on a problem starting from `x₀` and return the final
iterate as well as a vector of callback results, one per outer iteration.

The `prox_step` argument is a callable that implements a single inner iteration
for the problem at hand, with signature
`prox_step(problem, inner_iteration_index, current_point, step_size)`.

On the other hand, `callback(problem, current_point, outer_iteration_index)`
should produce a single scalar (for example, the distance to the solution).
"""
function rpmba_template(
  problem::OptProblem, x₀::Vector{Float64}, initial_step_size::Float64,
  initial_prox_penalty::Float64, initial_ensemble_radius::Float64,
  inner_iterations::Int, outer_iterations::Int, num_repeats::Int,
  prox_step::Function, callback::Function;
  stop_condition::Function = nothing
)
  callback_results = zeros(outer_iterations)
  for t in 0:(outer_iterations - 1)
    ρ = 2^t * initial_prox_penalty
    α = 2^(-t) * initial_step_size
    ϵ = 2^(-t) * initial_ensemble_radius
    x₀ = epmba_template(problem, x₀[:], α, ρ, ϵ, num_repeats, inner_iterations,
                        prox_step, stop_condition=stop_condition)
    callback_results[t+1] = callback(problem, x₀, t)
  end
  return x₀, callback_results
end


"""
  pairwise_distance(V::Matrix{Float64})

Given a `d × k` matrix, return a `k × k` matrix such that the (i, j)
element holds the distance between `V[:, i]` and `V[:, j]`.
"""
function pairwise_distance(V::Matrix{Float64})
  col_norms = sum(V.^2, dims=1)
  return col_norms .+ col_norms' - 2 * V'V
end

end
