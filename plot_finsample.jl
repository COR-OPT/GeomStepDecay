using ArgParse
using Distributions
using LinearAlgebra
using PyPlot
using Random
using Statistics

# enable latex printing
rc("text", usetex=true)


include("src/RobustPR.jl")
include("src/RobustBD.jl")

# lightGreen: 66c2a4
# heavyGreen: 2ca25f
# lblue;
# \definecolor{lred}{HTML}{cb5501}
# \definecolor{mred}{HTML}{f1885b}
# \definecolor{hred}{HTML}{b3001e}

LBLUE="#519cc8"
MBLUE="#1d5996"
HBLUE="#908cc0"
TTRED="#ca3542"

#= low probability algorithm =#
function runOpt(d, pfail, δ, repeats;
                problem=:phase_retrieval, ϵ_stop=1e-16)
	μ = 1 - 2 * pfail; L = sqrt(d); η = 1.0; δ2 = 1 / sqrt(10); γ = 1
	ϵ = 1e-5; T = trunc(Int, ceil(log2(2 * δ / ϵ)))
	K = trunc(Int, T^2 * (L / (δ2 * μ))^2)
	R0 = sqrt(δ) * γ * μ; α0 = sqrt(R0^2 / (L^2 * (K + 1)))
	pLib = (problem == :phase_retrieval) ? RobustPR : RobustBD
	λ = (k -> α0 * 2.0^(-k))
	println("Running with T = $T, K = $K")
	prob = pLib.genProb(8 * d, d, pfail=pfail)
	dists = fill(0.0, 4, T, repeats)
	@inbounds for j = 1:repeats
		@info("Running repeat $j...")
		_, dists[1, :, j], evals = pLib.opt(prob, sqrt(δ) * γ, T, K, λ,
											ϵ=ϵ_stop, method=:subgradient)
		_, dists[2, :, j], evals = pLib.opt(prob, sqrt(δ) * γ, T, K, λ,
											ϵ=ϵ_stop, method=:proxlinear)
		_, dists[3, :, j], evals = pLib.opt(prob, sqrt(δ) * γ, T, K, λ,
											ϵ=ϵ_stop, method=:proximal)
		_, dists[4, :, j], evals = pLib.opt(prob, sqrt(δ) * γ, T, K, λ,
											ϵ=ϵ_stop, method=:clipped)
	end
	dsMeans = reshape(mean(dists, dims=3), 4, T)
	dsStds = max.(reshape(std(dists, dims=3, corrected=true), 4, T), 1e-16)
	# generate plots - subgradient
	subplot(221)
	@show size(dsMeans), size(dsStds)
	errorbar((0:(T-1)) .* K, dsMeans[1, :], yerr=dsStds[1, :],
			 linewidth=2, color=LBLUE, label="Subgradient", marker="o")
	plot((0:(T-1)) .* K, dsMeans[1, 1] .* (2.0.^(-(0:(T-1)))),
		 color=TTRED, label=L" R_0 \cdot 2^{-k} ", linestyle="dashed")
	xlabel(L" k "); yscale("log"); legend()
	# proxlinear
	subplot(222)
	errorbar((0:(T-1)) .* K, dsMeans[2, :], yerr=dsStds[2, :],
			 linewidth=2, color=MBLUE, label="Proxlinear", marker="o")
	plot((0:(T-1)) .* K, dsMeans[2, 1] .* (2.0.^(-(0:(T-1)))),
		 color=TTRED, label=L" R_0 \cdot 2^{-k} ", linestyle="dashed")
	xlabel(L" k "); yscale("log"); legend()
	# proximal
	subplot(223)
	errorbar((0:(T-1)) .* K, dsMeans[3, :], yerr=dsStds[3, :],
			 linewidth=2, color=HBLUE, label="Proximal", marker="o")
	plot((0:(T-1)) .* K, dsMeans[3, 1] .* (2.0.^(-(0:(T-1)))),
		 color=TTRED, label=L" R_0 \cdot 2^{-k} ", linestyle="dashed")
	xlabel(L" k "); yscale("log"); legend()
	# clipped
	subplot(224)
	errorbar((0:(T-1)) .* K, dsMeans[4, :], yerr=dsStds[4, :],
			 linewidth=2, color="black", label="Clipped", marker="o")
	plot((0:(T-1)) .* K, dsMeans[4, 1] .* (2.0.^(-(0:(T-1)))),
		 color=TTRED, label=L" R_0 \cdot 2^{-k} ", linestyle="dashed")
	xlabel(L" k "); yscale("log"); legend()
	# show everything
	show();
end


s = ArgParseSettings(description="""
	Compare stochastic subgradient and proximal point algorithms on
	phase retrieval under the fixed budget setting with a finite number
    of samples.""")
@add_arg_table s begin
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
	"--repeats"
		help = "Number of repeats for each run"
		arg_type = Int
		default = 25
	"--problem"
		help = "Choose problem to solve - one of {phase_retrieval, blind_deconvolution}."
		range_tester = (x -> lowercase(x) in [
											  "phase_retrieval", "blind_deconvolution"])
end

Random.seed!(999)
parsed = parse_args(s); d, p = parsed["d"], parsed["p"]
δ, probType = parsed["delta"], parsed["problem"]
repeats = parsed["repeats"]
runOpt(d, p, δ, repeats, problem=Symbol(probType), ϵ_stop=1e-16)
