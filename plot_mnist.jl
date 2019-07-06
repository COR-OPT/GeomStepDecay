using ArgParse
using JLD
using LinearAlgebra
using PyPlot
using Random
using Statistics

# enable latex printing
rc("text", usetex=true)

include("src/SparseLogReg.jl")

LBLUE="#519cc8"
MBLUE="#1d5996"
HBLUE="#908cc0"
TTRED="#ca3542"

# util for computing R0
dist(wFull, bFull, wInit, bInit) = begin
    norm(vcat(wFull - wInit, bFull - bInit))
end


#= Run Regularized Dual Averaging from (Lee & Wright, '12) on the
#  MNIST logistic regression problem. =#
function plot_sparse_logreg_rda(γ::Real, μ::Real, Kmax::Int)
	data = load("mnist_full.jld"); prob = SparseLogReg.genProb(0.0)
	ϵ = 1e-5; L = SparseLogReg.getLipConstant(prob)
    wCurr = fill(0.0, 784); bCurr = 0.0
	for key in filter(x -> startswith(x, "p_"), keys(data))
		p, wFull, bFull = data[key]; τ = 2.0^(-p); prob.τ = τ
		@info("Testing τ = 2.0^(-$(p))...")
		R0 = dist(wFull, bFull, wCurr, bCurr)
		T, K, _ = SparseLogReg.getRunParams(prob, μ, ϵ, R0, 0.3)
		itTotal = trunc(Int, K) * T
		_, dsFun, dsOut, dsSig, nnz = SparseLogReg.rda(
			prob, itTotal, γ, wFull, bFull, wCurr, bCurr)
		tRun = length(dsFun)
        # plot function values
        subplot(211)
        semilogy((0:(tRun-1)) .* K, dsFun, linewidth=2, color=MBLUE,
                 marker="o", label=L"f(x) - \min f")
        xlabel(L" k "); legend()
        # plot iterate distances
        subplot(212)
        semilogy((0:(tRun-1)) .* K, dsOut, linewidth=2, color=MBLUE,
                 marker="o", label=L" \| w_{S^c} \|_2 ")
        semilogy((0:(tRun-1)) .* K, dsSig, linewidth=2, color=HBLUE,
                 marker="o", label=L" \| x - \tilde{x} \| / \| \tilde{x} \|")
        xlabel(L" k "); legend()
        # show everything
        suptitle("Sparse logistic regression - RDA"); show()
	end
end


#= Run the stochastic algorithm on the sparse logistic regression
#  problem described in (Lee, Wright '12). =#
#  best mu found: 0.65 - prob: 0.3
function plot_sparse_logreg(μ::Float64)
	data = load("mnist_full.jld"); prob = SparseLogReg.genProb(0.0)
    ϵ = 1e-5; wCurr = fill(0.0, 784); bCurr = 0.0
    for key in filter(x -> startswith(x, "p_"), keys(data))
        p, wFull, bFull = data[key]; τ = 2.0^(-p); prob.τ = τ
		@info("Testing τ = 2.0^(-$(p))...")
        R0 = dist(wFull, bFull, wCurr, bCurr)
		T, K, α0 = SparseLogReg.getRunParams(prob, μ, ϵ, R0, 0.3)
		K = trunc(Int, K); α0 *= 10  # scale up step slightly
		_, dsFun, dsOut, dsSig, nnz =
			SparseLogReg.sProx(prob, T, K, k -> α0 * 2.0^(-k),
							   wFull, bFull, copy(wCurr), bCurr)
		tRun = length(dsFun)
        # plot function values
        subplot(211)
        semilogy((0:(tRun-1)) .* K, dsFun, linewidth=2, color=MBLUE,
                 marker="o", label=L"f(x) - \min f")
        xlabel(L" k "); legend()
        # plot iterate distances
        subplot(212)
        semilogy((0:(tRun-1)) .* K, dsOut, linewidth=2, color=MBLUE,
                 marker="o", label=L" \| w_{S^c} \|_2 ")
        semilogy((0:(tRun-1)) .* K, dsSig, linewidth=2, color=HBLUE,
                 marker="o", label=L" \| x - \tilde{x} \| / \| \tilde{x} \|")
        xlabel(L" k "); legend()
        # show everything
        suptitle("Sparse logistic regression - RMBA"); show()
    end
end


s = ArgParseSettings(description="""
	Evaluate the performance of stochastic proximal gradient on logistic
	regression, applied to a pair of MNIST digits.""")
@add_arg_table s begin
	"--mu"
		help = "The sharpness coefficient μ"
		arg_type = Float64
		default = 0.3
	"plot_rmba"
		help = "Plot the performance of the RMBA algorithm"
		action = :command
	"plot_rda"
		help = "Plot the performance of RDA from (Lee & Wright, 2012)"
		action = :command
end


@add_arg_table s["plot_rda"] begin
	"--gamma"
		help = "The step size multiplier γ for RDA"
		arg_type = Float64
		default = 1.0
	"--K"
		help = "Adjust to increase/decrease max budget of RDA method"
		arg_type = Int
		default = 1000000000
end


Random.seed!(123)
parsed = parse_args(s)
if parsed["%COMMAND%"] == "plot_rmba"
    plot_sparse_logreg(parsed["mu"])
elseif parsed["%COMMAND%"] == "plot_rda"
	subp = parsed["plot_rda"]
	plot_sparse_logreg_rda(subp["gamma"], parsed["mu"], subp["K"])
end
