using ArgParse
using Distributions
using LinearAlgebra
using PyPlot
using Random
using Statistics

include("src/RobustPR.jl")
include("src/RobustBD.jl")

LBLUE="#519cc8"
MBLUE="#1d5996"
HBLUE="#908cc0"
TTRED="#ca3542"

# set up pretty printing
rc("text", usetex=true)


# lightGreen: 66c2a4
# heavyGreen: 2ca25f

# runFixBudget: run using a fixed budget, report avg last accuracy
# ============
function runFixBudget(d, pfail, δ, repeats; problem, method)
	D = Normal(0, 1); ϵ = 1e-5; γ = 1; δ2 = 1 / sqrt(10)
	μ = 1 - 2 * pfail; L = sqrt(d); η = 1.0;
	T = trunc(Int, ceil(log2(2 * δ / ϵ))); K = trunc(Int, 0.1 * T^2 * (L / (δ2 * μ))^2)
	R0 = sqrt(δ) * γ * μ / η; α0 = sqrt(R0^2 / (L^2 * (K + 1)))
	pLib = (problem == :phase_retrieval) ? RobustPR : RobustBD
	@info("Running with T = $T, K = $K, method=$(method)...")
	prob = pLib.genProb(D, d, pfail=pfail)
	dists = fill(0.0, 20, repeats); evals = fill(0, 20, repeats)
    αs = 2.0.^(-10:9)
	for i = 1:length(αs)
		λ = k -> αs[i] * α0 * 2.0^(-k)
		@info("Running with c_λ = $(αs[i])")
		@inbounds for j = 1:repeats
			@info("repeat $j")
			_, ds, evals[i, j] = pLib.opt(prob, sqrt(δ) * γ, T, K, λ, ϵ=1e-20,
										  method=method)
			dists[i, j] = ds[end]
		end
	end
	# get means and standard deviations for distances
	dsMean = mean(dists, dims=2)[:]; dsStd = max.(std(dists, dims=2)[:], 1e-17)
	uEnv = dsMean .+ dsStd; lEnv = dsMean .- dsStd
	# generate plot
	loglog(αs, dsMean, linewidth=2, color=MBLUE, aa=true, label="$(method)",
		   marker="o")
	loglog(αs, uEnv, linewidth=2, color=MBLUE, alpha=0.3, aa=true)
	loglog(αs, lEnv, linewidth=2, color=MBLUE, alpha=0.3, aa=true)
	fill_between(αs, uEnv, lEnv, color=MBLUE, alpha=0.25)
	xlabel(L" \lambda "); title("Avg. final distance"); legend(); show()
end


# runOpt: run using a fixed accuracy, report avg evals to accuracy
# ======
function runOpt(d, pfail, δ, repeats; problem, method)
	D = Normal(0, 1); ϵ = 1e-5; γ = 1; δ2 = 1 / sqrt(10)
	μ = 1 - 2 * pfail; L = sqrt(d); η = 1.0;
	T = trunc(Int, ceil(log2(2 * δ / ϵ))); K = trunc(Int, T^2 * (L / (δ2 * μ))^2)
	R0 = sqrt(δ) * γ * μ / η; α0 = sqrt(R0^2 / (L^2 * (K + 1)))
	pLib = (problem == :phase_retrieval) ? RobustPR : RobustBD
	println("Running with T = $T, K = $K, method=$(method)...")
	prob = pLib.genProb(D, d, pfail=pfail)
	dists = fill(0.0, 20, repeats); evals = fill(0, 20, repeats)
	αs = 2.0.^(-10:9)
	for i = 1:length(αs)
		λ = k -> αs[i] * α0 * 2.0^(-k)
		@info("Running with c_λ = $(αs[i])")
		@inbounds for j = 1:repeats
			@info("repeat $j")
			_, ds, evals[i, j] = pLib.opt(prob, sqrt(δ) * γ, T, K, λ, ϵ=ϵ, method=method)
			dists[i, j] = ds[end]
		end
	end
	# get means and standard deviations for distances
	evalMean = mean(evals, dims=2)[:]; evalStd = std(evals, dims=2)[:]
	uEnv = evalMean .+ evalStd; lEnv = evalMean .- evalStd
	# generate plot
	loglog(αs, evalMean, linewidth=2, color=MBLUE, aa=true, label="$(method)",
		   marker="o")
	loglog(αs, uEnv, linewidth=2, color=MBLUE, alpha=0.3, aa=true)
	loglog(αs, lEnv, linewidth=2, color=MBLUE, alpha=0.3, aa=true)
	fill_between(αs, uEnv, lEnv, color=MBLUE, alpha=0.25)
	xlabel(L" \lambda "); title("Avg. evals to accuracy"); legend(); show()
end


s = ArgParseSettings(description="""
	Compare stochastic subgradient and proximal point algorithms on
	statistical problems under the fixed budget / fixed target accuracy
	setting.""")
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
		help = """
		Choose problem to solve - one of
		{phase_retrieval, blind_deconvolution}."""
		range_tester = (x -> lowercase(x) in [
			"phase_retrieval", "blind_deconvolution"])
	"--method"
		help = """
		Choose optimization method to use - one of
		{subgradient, proximal, proxlinear, clipped}."""
		range_tester = (x -> lowercase(x) in [
			"subgradient", "proximal", "proxlinear", "clipped"])
	"--type"
		help = "Choose between checking evals or accuracy"
		range_tester = (x -> lowercase(x) in ["accuracy", "evals"])
end

Random.seed!(999)
parsed = parse_args(s); d, p = parsed["d"], parsed["p"]
δ, probType = parsed["delta"], parsed["problem"]
repeats, method = parsed["repeats"], parsed["method"]
if lowercase(parsed["type"]) == "evals"
	runOpt(d, p, δ, repeats, problem=Symbol(probType),
		   method=Symbol(method))
else
	runFixBudget(d, p, δ, repeats, problem=Symbol(probType),
				 method=Symbol(method))
end
