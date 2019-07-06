#! /usr/bin/env julia

module RobustBD

	using Distributions
	using LinearAlgebra
	using Polynomials
	using Random
	using Statistics

	struct BDProb
		L :: Union{Array{<:Number, 2}, Distribution}
		R :: Union{Array{<:Number, 2}, Distribution}
		y :: Array{<:Number, 1}
		w :: Array{<:Number, 1}
		x :: Array{<:Number, 1}
		pfail :: Float64
	end

	# mapping function for big corrupted entries
	_bigCorr(p) = (x -> (rand() <= p) ? 10 * randn() : x)

	"""
		sampleVecs(prob::PrProb; samples=1)

	A sampling function to draw a number of measurement vectors and measurements
	from a blind deconvolution instance.
	"""
	function sampleVecs(prob::BDProb; samples=1)
		if isa(prob.L, Array) && isa(prob.R, Array)
			idx = rand(1:length(prob.y), samples)
			return prob.L[idx, :], prob.R[idx, :], prob.y[idx]
		elseif isa(prob.L, Distribution) && isa(prob.R, Distribution)
			d1 = length(prob.w); d2 = length(prob.x); p = prob.pfail
			measL = rand(prob.L, samples, d1)
			measR = rand(prob.R, samples, d2)
			yMeas = (measL * prob.w) .* (measR * prob.x)
			replace!(_bigCorr(p), yMeas)
			return measL, measR, yMeas
		end
	end


	"""
		subgrad(prob::BDProb, wCurr, xCurr, ℓᵢ, rᵢ, yᵢ)

	Compute a stochastic subgradient for the robust blind deconvolution
	objective using measurement vectors ``\\ell_i, r_i`` and measurement
	``y_i``.
	"""
	function subgrad(prob::BDProb, wCurr, xCurr, ℓᵢ, rᵢ, yᵢ)
		Lw = ℓᵢ' * wCurr; Rx = rᵢ' * xCurr
		s = sign.(Lw * Rx - yᵢ)
		return ℓᵢ * (Rx * s), rᵢ * (Lw * s)
	end


    # _initGuess: initial guess for an estimate, located δ-far from solution
    _initGuess(prob, δ) = begin
        rwDir = randn(length(prob.w)); rxDir = randn(length(prob.x))
        normalize!(rwDir); normalize!(rxDir)
        return prob.w + norm(prob.w) * δ * rwDir, prob.x + norm(prob.x) * δ * rxDir
    end


	# blind deconv distance
	dist(w, x, w̄, x̄) = norm(w .* x' .- w̄ .* x̄') / (norm(x̄) * norm(w̄))

	setup_step(v) = isa(v, Function) ? v : (_ -> v)


	_checkTerm(prob::BDProb, idx, wCurr, xCurr, ϵ) = begin
		(idx % 10 == 0) && (dist(wCurr, xCurr, prob.w, prob.x) <= ϵ)
	end

	#= _pSgdInner: inner loop with constant step size =#
	function _pSgdInner(prob::BDProb, wCurr, xCurr, iters, λ;
						ρ, w0, x0, ϵ=1e-16)
		stopTime = rand(1:iters); Ls, Rs, ys = sampleVecs(prob, samples=stopTime)
		for i = 1:stopTime
			gw, gx = subgrad(prob, wCurr, xCurr, Ls[i, :], Rs[i, :], ys[i])
			wCurr[:] = projBall((wCurr + λ * ρ * w0 - λ * gw) / (1 + λ * ρ), 5.0)
			xCurr[:] = projBall((xCurr + λ * ρ * x0 - λ * gx) / (1 + λ * ρ), 5.0)
			_checkTerm(prob, i, wCurr, xCurr, ϵ) && return wCurr, xCurr, i
		end
		return wCurr, xCurr, stopTime
	end


	"""
		pSgd(prob::BDProb, δ, iters, inSched, λ=nothing)

	Run the subgradient method consisting of an outer loop which adjust the
	step size and an inner loop which runs a number of iterations with constant
	step size. Initialize at a point ``\\delta``-close to the optimum, with
	`inSched` being either a number or a callable implementing the number of
	inner steps, which can adapt to the outer iteration index. λ is either a
	number or callable implementing the inner step size schedule.
	"""
    function pSgd(prob::BDProb, δ, iters, inSched, λ; ρ=0.0, ϵ=1e-16)
		inIters = setup_step(inSched); λSched = setup_step(λ)
        winit, xinit = _initGuess(prob, δ)
        w0 = copy(winit); x0 = copy(xinit)
		dists = fill(0.0, iters); totalEvals = 0
		for i = 1:iters
			dists[i] = dist(winit, xinit, prob.w, prob.x)
			(dists[i] <= ϵ) && return (winit, xinit), dists[1:i], totalEvals
			winit[:], xinit[:], stopTime = _pSgdInner(prob, winit, xinit,
													  inIters(i), λSched(i),
													  ρ=ρ, w0=w0, x0=x0, ϵ=ϵ)
			totalEvals += stopTime
		end
		return (winit, xinit), dists, totalEvals
	end


	#= proximal point cost function =#
	proxPtCost(wCurr, xCurr, wk, xk, ℓ, r, b, λ) = begin
		return abs((ℓ' * wCurr) * (r' * xCurr) - b) + (1 / (2 * λ)) * (
			norm(wCurr - wk)^2 + norm(xCurr - xk)^2)
	end

	# norm squared of columns
	colNormSq(A) = sum(A.^2, dims=1)

	#= proximal point update by taking the point with lowest prox value =#
	function _prox_update(prob::BDProb, wk, xk, ℓ, r, b, λ)
		ℓw = ℓ' * wk; rx = r' * xk; nrmW = norm(ℓ); nrmX = norm(r)
		denom = 1 - λ^2 * nrmW^2 * nrmX^2
		# two cases to consider - case 1: ℓw * rx ≠ b
		# put resulting vectors into separate columns
		Ws = hcat((wk - λ * ((rx - λ * nrmX^2 * ℓw) / denom) * ℓ,
				   wk - λ * ((-rx - λ * nrmX^2 * ℓw) / denom) * ℓ)...)
		Xs = hcat((xk - λ * ((ℓw - λ * nrmW^2 * rx) / denom) * r,
				   xk - λ * ((-ℓw - λ * nrmW^2 * rx) / denom) * r)...)
		# case 2 : ℓw * rx = b
		# Find roots of quartic
		p = Poly([-b^2 * nrmW^2, b * nrmW^2 * rx, 0.0, nrmX^2 * ℓw, nrmX^2])
		ηs = []
		try
			append!(ηs, map(real, filter(isreal, roots(p))))
		catch ArgumentError
			ηs = []
		end
		# get coefficients γ
		if (length(ηs) > 0)
			γs = ((ηs .* ℓw - ηs.^2) / (b * nrmW^2))
			hcat(Ws, wk .- (ℓ .* (γs .* (b ./ ηs))'))
			hcat(Xs, xk .- (r .* (γs .* ηs)'))
		end
		costs = (abs.((ℓ' * Ws) .* (r' * Xs) .- b) .+ ((1 / (2 * λ)) .* (
			colNormSq(Ws .- wk) + colNormSq(Xs .- xk))))
		minIdx = argmin(vec(costs))
		return Ws[:, minIdx], Xs[:, minIdx]
	end


	projOne(x) = min(abs(x), 1) * sign(x)

	#= update for the prox-linear problem with parameter λ =#
	function _proxlin_update(prob::BDProb, wk, xk, ℓi, ri, b, λ)
		γ = λ * ((ℓi' * wk) * (ri' * xk) - b)
		ζw = λ * ℓi * (ri' * xk); ζx = λ * ri * (ℓi' * wk)
		ζnsq = norm(ζw)^2 + norm(ζx)^2
		return (wk, xk) .+ (projOne(-γ / ζnsq) .* (ζw, ζx))
	end

	projBall(x, γ) = (norm(x) <= γ) ? x : normalize(x) * γ

	#= proximal / proxlinear update incorporating projection to ball,
	#  if maxIt > 0. =#
	#= uses ADMM internally =#
	function _prox_update_proj(prob::BDProb, wk, xk, ℓ, r, b, λ; maxIt=0,
							   in_ϵ=1e-10, method=:proximal)
		proxFn = (method == :proximal) ? _prox_update : _proxlin_update
		sW, sX = copy(wk), copy(xk)
		yW, yX = proxFn(prob, sW, sX, ℓ, r, b, λ)
		uW, uX = zero(sW), zero(sX)
		for i = 1:maxIt
			yW, yX = proxFn(prob, sW .- uW, sX .- uX, ℓ, r, b, λ)
			sW = projBall(yW .- uW, 5.0); sX = projBall(yX .- uX, 5.0)
			broadcast!(+, uW, uW, yW - sW); broadcast!(+, uX, uX, yX - sX)
			if (norm(yW - sW) <= in_ϵ) && (norm(yX - sX) <= in_ϵ)
				return yW, yX
			end
		end
		return yW, yX
	end


	check_nan(wCurr, xCurr) = begin
		any((isnan.(wCurr) .| isinf.(wCurr)) .| (isnan.(xCurr) .| isinf.(xCurr)))
	end

	#= inner iteration for proxlinear method =#
	function _inProxlin(prob::BDProb, wCurr, xCurr, iters, λ;
						maxIt=0, in_ϵ=1e-10, ρ, w0, x0, ϵ=1e-16)
		stopTime = rand(1:iters); Ls, Rs, ys = sampleVecs(prob, samples=stopTime)
		dnm = 1 + λ * ρ
		@inbounds for i = 1:stopTime
			wCurr[:], xCurr[:] = _prox_update_proj(prob, (wCurr + λ * ρ * w0) / dnm,
												   (xCurr + λ * ρ * x0) / dnm, Ls[i, :],
												   Rs[i, :], ys[i], λ / dnm, in_ϵ=in_ϵ,
												   maxIt=maxIt, method=:proxlinear)
			# stop if desired accuracy achieved
			_checkTerm(prob, i, wCurr, xCurr, ϵ) && return wCurr, xCurr, i
		end
		return wCurr, xCurr, stopTime
	end


	#= inner iteration for proximal method =#
	function _inProx(prob::BDProb, wCurr, xCurr, iters, λ;
					 maxIt=0, in_ϵ=1e-10, ρ, w0, x0, ϵ=1e-16)
		stopTime = rand(1:iters); Ls, Rs, ys = sampleVecs(prob, samples=stopTime)
		dnm = 1 + λ * ρ
		@inbounds for i = 1:stopTime
			wCurr[:], xCurr[:] = _prox_update_proj(prob, (wCurr + λ * ρ * w0) / dnm ,
												   (xCurr + λ * ρ * x0) / dnm, Ls[i, :],
												   Rs[i, :], ys[i], λ / dnm, in_ϵ=in_ϵ,
												   maxIt=maxIt, method=:proximal)
			check_nan(wCurr, xCurr) && return wCurr, xCurr, stopTime
			# stop if desired accuracy achieved
			_checkTerm(prob, i, wCurr, xCurr, ϵ) && return wCurr, xCurr, i
		end
		return wCurr, xCurr, stopTime
	end


	"""
		sProx(prob::BDProb, δ, iters, inSched, λ; method=:proxlinear)

	Run the stochastic proximal point or prox-linear methods for the blind
	deconvolution problem for `iters` outer iterations, each of which consists
	of a number of inner iterations with constant prox-parameter. Both `inSched`
	and `λ` can be either scalars or callables implementing the inner step and
	prox-parameter schedule, respectively. Start at an iterate with normalized
	distance `δ` from the ground truth.
	"""
	function sProx(prob::BDProb, δ, iters, inSched, λ; method=:proxlinear,
                   maxIt=0, in_ϵ=1e-10, ρ=0.0, ϵ=1e-16)
		innerFn = (method == :proxlinear) ? _inProxlin : _inProx
        wCurr, xCurr = _initGuess(prob, δ); w0 = copy(wCurr); x0 = copy(xCurr)
		inIters = setup_step(inSched); λSched = setup_step(λ)
		ρSched = setup_step(ρ)
		dists = fill(0.0, iters); totalEvals = 0
		for i = 1:iters
			dists[i] = dist(wCurr, xCurr, prob.w, prob.x)
			(dists[i] <= ϵ) && return (wCurr, xCurr), dists[1:i], totalEvals
			ρi = ρSched(i)
			wCurr[:], xCurr[:], stopTime = innerFn(prob, wCurr, xCurr, inIters(i),
												   λSched(i), maxIt=maxIt, in_ϵ=in_ϵ,
												   ρ=ρi, w0=w0, x0=x0, ϵ=ϵ)
			totalEvals += stopTime
            # if diverged, stop at current iteration and set dist to NaN
            if (dists[i] >= 1e10)
                dists[i] = NaN; return (wCurr, xCurr), dists[1:i], totalEvals
            end
		end
		return (wCurr, xCurr), dists, totalEvals
	end


	function _trunc(x, a, b)
		return max(min(x, b), a)
	end


	function _truncUpdate(prob::BDProb, wCurr, xCurr, ℓ, r, y, λ)
		d = length(wCurr); res = (ℓ' * wCurr) * (r' * xCurr) - y
		if (abs(res) <= 1e-16)
			return wCurr, xCurr
		end
		c = sign(res) * vcat((r' * xCurr) * ℓ, (ℓ' * wCurr) * r)
		cFact = _trunc(abs(res) / (λ * norm(c)^2), 0, 1.0)
		return wCurr - λ * cFact * c[1:d], xCurr - λ * cFact * c[(d+1):end]
	end


	function _inTrunc(prob::BDProb, wCurr, xCurr, iters, λ; ρ, w0, x0, ϵ=1e-16)
		stopTime = rand(1:iters); Ls, Rs, ys = sampleVecs(prob, samples=stopTime)
		dnm = 1 + λ * ρ
		@inbounds for i = 1:stopTime
			wCurr[:], xCurr[:] = _truncUpdate(prob, (wCurr + λ * ρ * w0) / dnm,
											  (xCurr + λ * ρ * x0) / dnm, Ls[i, :],
			                                  Rs[i, :], ys[i], λ / dnm)
			_checkTerm(prob, i, wCurr, xCurr, ϵ) && return wCurr, xCurr, i
		end
		return wCurr, xCurr, stopTime
	end


	"""
		sTrunc(prob::BDProb, δ, iters, inSched, λ)

	Run the truncated method on a blind deconvolution problem for `iters` outer
	iterations, each of which consists of a number of inner iterations with
	constant prox-parameter. `inSched` and `λ` can be either numbers or callables
	implementing the inner step and prox-parameter schedule, respectively. Start
	at an iterate with normalized distance `δ` from the ground truth.
	"""
    function sTrunc(prob::BDProb, δ, iters, inSched, λ; ρ=0.0, ϵ=1e-16)
        wInit, xInit = _initGuess(prob, δ); w0 = copy(wInit); x0 = copy(xInit)
		inIters = setup_step(inSched); λSched = setup_step(λ)
		ρSched = setup_step(ρ); dists = fill(0.0, iters); totalEvals = 0
		for i = 1:iters
			dists[i] = dist(wInit, xInit, prob.w, prob.x)
			(dists[i] <= ϵ) && return (wInit, xInit), dists[1:i], totalEvals
			ρi = ρSched(i)
			wInit[:], xInit[:], stopTime = _inTrunc(prob, wInit, xInit, inIters(i),
												    λSched(i), ρ=ρi, w0=w0, x0=x0,
													ϵ=ϵ)
			totalEvals += stopTime
		end
		return (wInit, xInit), dists, totalEvals
	end


	"""
		opt(prob::BDProb, δ, T, K, λ; ϵ=1e-16,
			method, in_ϵ=1e-10, maxIt=0) -> ((wCurr, xCurr), ds, tEv)

	Run a low-probability version of one of the available optimization
	`method`s on a blind deconvolution problem for `T` outer iterations and
	`K` inner iterations,  starting from an estimate initialized at normalized
	distance `δ` from the ground truth. The search is terminated when reaching
	a normalized distance of at most `ϵ`. Return:
	- `wCurr, xCurr`: final estimates found by the method
	- `ds`: a history of normalized distances from the solution set
	- `tEv`: the total number of oracle calls
	"""
	function opt(prob::BDProb, δ, T, K, λ; ϵ=1e-16,
				 method, in_ϵ=1e-10, maxIt=0)
		if method == :subgradient
			return pSgd(prob, δ, T, K, λ, ϵ=ϵ)
		elseif method == :proximal
			return sProx(prob, δ, T, K, λ, ϵ=ϵ,
						 method=:proximal, in_ϵ=in_ϵ, maxIt=maxIt)
		elseif method == :proxlinear
			return sProx(prob, δ, T, K, λ, ϵ=ϵ,
						 method=:proxlinear, in_ϵ=in_ϵ, maxIt=maxIt)
		elseif method == :clipped
			return sTrunc(prob, δ, T, K, λ, ϵ=ϵ)
		else
			throw(ArgumentError("method $(method) not recognized"))
		end
	end


	innerMethod(meth) = begin
		if meth == :subgradient
			return _pSgdInner
		elseif meth == :proxlinear
			return _inProxlin
		elseif meth == :proximal
			return _inProx
		elseif meth == :clipped
			return _inTrunc
		else
			throw(ArgumentError("""
				Unrecognized method $meth. Choose one of :subgradient, :proxlinear,
				:proximal or :clipped."""))
		end
	end

	# pairwise distance matrix
	pairwiseDist(Y) = begin
		nMat = sum(Y.^2, dims=1);
		return nMat .+ nMat' - 2 * Y' * Y
	end


	function pgmba(prob::BDProb, w0, x0, ρ, α, K, innerFn)
		return innerFn(prob, copy(w0), copy(x0), K, α, ρ=ρ, w0=w0, x0=x0)
	end


	function epgmba(prob::BDProb, w0, x0, ρ, α, K, M, ϵ; method=:subgradient)
		innerFn = innerMethod(method); d1 = length(w0); d2 = length(x0)
		Zs = fill(0.0, d1 + d2, M)
		@inbounds for i = 1:M
			Zs[:, i] = vcat(pgmba(prob, w0, x0, ρ, α, K, innerFn)[1:2]...)
		end
		pDists = pairwiseDist(Zs)
		mostIdx = argmax(sum(pDists .<= 2 * ϵ, dims=2)[:])
		return Zs[1:d1, mostIdx], Zs[(d1+1):end, mostIdx]
	end


	"""
		sOpt(prob::BDProb, w0::Array{Float64, 1}, x0::Array{Float64, 1}, ρ0,
		α0, K, ϵ0, M, T; method) -> ((wCurr, xCurr), ds)

	Run the proximally regularized stochastic optimization algorithm using one
	of the available `method`s (`:subgradient, :proximal, :proxlinear, :clipped`)
	with initial estimates `w0, x0` and parameters `ρ0` and `α0`, controlling
	the regularization term and initial step size, respectively, as well as
	clustering parameter `ϵ0`, using `K` inner and `T` outer iterations and `M`
	independent trials per iteration. Return:
	- `wCurr, xCurr`: estimates found by the method
	- `ds`: a history of normalized distances to the solution set
	"""
	function sOpt(prob::BDProb, w0::Array{Float64, 1}, x0::Array{Float64, 1},
				  ρ0, α0, K, ϵ0, M, T; method)
		wCurr = copy(w0); xCurr = copy(x0); dists = fill(0.0, T)
		for t = 0:(T-1)
			dists[t+1] = dist(wCurr, xCurr, prob.w, prob.x)
			ρ = 2.0^t * ρ0; ϵ = 2.0^(-t) * ϵ0; α = 2.0^(-t) * α0
			wCurr[:], xCurr[:] = epgmba(prob, wCurr, xCurr, ρ, α, K, M, ϵ,
										method=method)
		end
		return (wCurr, xCurr), dists
	end


	"""
		sOpt(prob::BDProb, δ::Float64, ρ0, α0, K, ϵ0, M, T; method) -> ((wCurr, xCurr), ds)

	Run the proximally regularized stochastic optimization algorithm using one
	of the available `method`s (`:subgradient, :proximal, :proxlinear, :clipped`)
	with an initial estimate `δ`-close to the ground truth and parameters `ρ0`
	and `α0`, controlling the regularization term and initial step size,
	respectively, as well as clustering parameter `ϵ0`, using `K` inner and `T`
	outer iterations and `M` independent trials per iteration. Return:
	- `wCurr, xCurr`: estimates found by the method
	- `ds`: a history of normalized distances to the solution set
	"""
	function sOpt(prob::BDProb, δ::Float64, ρ0, α0, K, ϵ0, M, T; method)
        wInit, xInit = _initGuess(prob, δ); w0 = copy(wInit); x0 = copy(xInit)
		return sOpt(prob, wInit, xInit, ρ0, α0, K, ϵ0, M, T, method=method)
	end



	"""
		genProb(m, d; pfail=0.0)

	Generate a blind deconvolution problem with `m` measurements and dimension
	`d`. Optionally, corrupts a fraction of `pfail` measurements.
	"""
	function genProb(m::Int, d; pfail=0.0, outliers=:big)
		L = randn(m, d); R = randn(m, d); w = randn(d); x = randn(d)
		normalize!(w); normalize!(x); num_corr = trunc(Int, pfail * m)
		y = (L * w) .* (R * x)
		if outliers == :big
			# big sparse outliers
			y[randperm(m)[1:num_corr]] = 10 * (randn(num_corr))
		else
			y[randperm(m)[1:num_corr]] = (1 / sqrt(d)) * (randn(num_corr))
		end
		return BDProb(L, R, y, w, x, pfail)
	end


	"""
		genProb(L::Distribution, R::Distribution, d; pfail=0.0)

	Generate a blind deconvolution problem with dimension `d`, with `L`
	and `R` being two `Distributions.jl` objects giving the generating
	distribution of the left and right measurement vectors. Optionally, corrupt
	a fraction of `pfail` measurements.
	"""
	function genProb(L::Distribution, R::Distribution, d; pfail=0.0)
		w = randn(d); x = randn(d); normalize!(w); normalize!(x)
		return BDProb(L, R, [0.0], w, x, pfail)
	end


	function genProb(A::Distribution, d; pfail=0.0)
		w = randn(d); x = randn(d); normalize!(w); normalize!(x)
		return BDProb(A, A, [0.0], w, x, pfail)
	end


end
