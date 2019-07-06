module SparseLogReg

	using MLDatasets
	using LinearAlgebra
	using Random
	using Statistics


    #= make mutable to play with τ =#
    mutable struct RegProb
		xData :: Array{<:Number, 2}
		yData :: Array{<:Number, 1}
		τ :: Float64  # ℓ₁-penalty weight
    end


    #= helper for normalized distance =#
    _distance(wCurr, bCurr, wFull, bFull) = begin
        norm(vcat(wCurr - wFull, bCurr - bFull)) / norm(vcat(wFull, bFull))
    end


	"""
		getLipConstant(prob::RegProb)

	Compute an approximate Lipschitz constant, which is the square root
	of the average squared norm of the images, or just 1.0 if unnormalized.
	"""
    function getLipConstant(prob::RegProb)
        return sqrt(mean(sum(prob.xData.^2, dims=2)))
	end


	"""
		getGradConstant(prob::RegProb)

	Compute the Lipschitz constant of the gradient, which is equal to
	`λmax((1 / n) * X^T X)`.
	"""
	function getGradConstant(prob::RegProb)
		n, d = size(prob.xData)
        # lipschitz constant: (1 / 4) * \| (X^T X)/n \|_2^2
        return eigmax((1 / (4 * n)) * Symmetric(prob.xData' * prob.xData))
	end


	"""
		getRunParams(prob::RegProb, μ, ϵ, R0, δ; maxK=nothing)

	Given a problem instance, the sharpness ``\\mu``, accuracy level
	``\\epsilon``, distance to solution `R0` and failure probability
	``\\delta``, compute the number of outer and inner iterations,
	`T` and `K`, as well as the initial step size ``\\alpha_0``.
	Optionally, set the maximum number of inner iterations `maxK`.
	"""
	function getRunParams(prob::RegProb, μ, ϵ, R0, δ; maxK=nothing)
		T = trunc(Int, log2(R0 / ϵ)); L = getLipConstant(prob)
		K = trunc(Int, 8 * T^2 * (L / (δ * μ))^2)
		K = isa(maxK, Int) ? min(K, maxK) : K
		α0 = sqrt(R0^2 / (2 * L^2 * (K + 1)))
		return T, K, α0
	end


    _nfeats(x) = Float64.(x)


	"""
		genProb(τ::Float64)

	Generate a sparse logistic regression problem on MNIST digits 6 and 7
	with an ℓ₁ penalty weighted by `τ`.
	"""
	function genProb(τ::Float64)
		train_x, train_y = MNIST.traindata()
		train_x = MNIST.convert2features(train_x)'  # 60000x784 array
		inds = (train_y .== 6) .| (train_y .== 7)   # keep 6 and 7
		# map y to +1 for 6, -1 for 7
		return RegProb(
			mapslices(_nfeats, train_x[inds, :], dims=2),
			map(x -> (x == 6) ? 1.0 : -1.0, train_y[inds]), τ)
	end


	setup_step(v) = (isa(v, Function)) ? v : (_ -> v)


	#= proximal mapping of ℓ₁ norm =#
	ℓ₁prox(x, η) = sign.(x) .* max.(abs.(x) .- η, 0)


	#= stochastic gradient given index `idx`. =#
	_grad(prob, wCurr, bCurr, idx) = begin
		xImg = prob.xData[idx, :]; yImg = prob.yData[idx]
		cExp = exp(yImg * (wCurr' * xImg + bCurr))
		return (-yImg * 1 / (1 + cExp)) * xImg, (-yImg * 1 / (1 + cExp))
	end


	#= full gradient =#
	_gradFull(prob, wCurr, bCurr) = begin
		m = length(prob.yData)
		cExp = exp.(prob.yData .* (prob.xData * wCurr .+ bCurr))
		cFac = -prob.yData .* (1 ./ (1 .+ cExp))
		return (1 / m) * prob.xData' * cFac, (1 / m) * sum(cFac)
	end


	#= full proximal gradient step for ℓ₁-reg. least squares =#
	function _proxGradFull(prob::RegProb, wCurr, bCurr, λ)
		gw, gb = _gradFull(prob, wCurr, bCurr)
		return ℓ₁prox(wCurr - λ * gw, λ * prob.τ), bCurr - λ * gb
	end



	"""
		_proxGrad(prob::LsProb, wCurr, bCurr, K, λ)

	Inner loop for stochastic proximal gradient, running for `K` iterations
	with a constant step size `λ`.
	"""
	function _proxGrad(prob::RegProb, wCurr, bCurr, K, λ)
		wAvg = fill(0.0, length(wCurr)); bAvg = 0.0
		wGrad = fill(0.0, length(wCurr)); m = length(prob.yData)
		sInd = collect(1:length(prob.yData))  # array of indices
		elapsed = 0;
		for s = 1:div(K, length(prob.yData))  # full passes
			shuffle!(sInd)  # shuffle indices
			# run a full iteration over the dataset
			for idx in sInd
				wGrad[:], bGrad = _grad(prob, wCurr, bCurr, idx)
				wCurr[:] = ℓ₁prox(wCurr - λ * wGrad, λ * prob.τ)
				bCurr -= λ * bGrad
				elapsed += 1  # update elapsed iterations count
				wAvg += (wCurr - wAvg) / elapsed
				bAvg += (bCurr - bAvg) / elapsed
			end
		end
		# run remaining iters
		shuffle!(sInd)
		for idx in sInd
			wGrad[:], bGrad = _grad(prob, wCurr, bCurr, idx)
			wCurr[:] = ℓ₁prox(wCurr - λ * wGrad, λ * prob.τ)
			bCurr -= λ * bGrad
			elapsed += 1   # update elapsed iters count
			wAvg += (wCurr - wAvg) / elapsed
			bAvg += (bCurr - bAvg) / elapsed
			(elapsed > K) && break
		end
		return wAvg, bAvg
	end


	"""
		_proxGradClassic(prob::LsProb, K, λ, wFull, bFull, wCurr, bCurr) -> ((wCurr, bCurr), dsFun, dsOut, dsSol)

	Loop for stochastic proximal gradient, running for `K` iterations
    with a step size schedule `λ` given a full-gradient guess `(wFull, bFull)`
    and an initial guess `(wCurr, bCurr)`.
    Return:
    - `wCurr, bCurr`: final iterates found
    - `dsFun`: function value gap
    - `dsOut`: norm on the complement of the support set
    - `dsSol`: normalized distance to full solution
	"""
	function _proxGradClassic(prob::RegProb, K, λ,
                              wFull::Array{Float64, 1}, bFull::Float64,
                              wCurr::Array{Float64, 1}, bCurr::Float64)
        λs = setup_step(λ)
		wAvg = fill(0.0, length(wCurr)); bAvg = 0.0
		wGrad = fill(0.0, length(wCurr)); m = length(prob.yData)
		sInd = collect(1:length(prob.yData))  # array of indices
		elapsed = 0;
        # retrieve statistics from full version
        fLoss = logistic_loss(prob, wFull, bFull)
        zMask = abs.(wFull) .< 1e-10
        # error metrics
        dsOut = []; dsFun = []; dsSol = []
		for s = 1:div(K, length(prob.yData))  # full passes
            @info("Running pass: $s")
			shuffle!(sInd)  # shuffle indices
			# run a full iteration over the dataset
			for idx in sInd
                elapsed += 1  # update elapsed iters count
                λi = λs(elapsed)  # update step
				wGrad[:], bGrad = _grad(prob, wCurr, bCurr, idx)
				wCurr[:] = ℓ₁prox(wCurr - λi * wGrad, λi * prob.τ)
				bCurr -= λi * bGrad
				wAvg += (wCurr - wAvg) / elapsed
				bAvg += (bCurr - bAvg) / elapsed
			end
            # update stats
            push!(dsOut, norm(wAvg[zMask]))
            push!(dsFun, logistic_loss(prob, wAvg, bAvg))
            push!(dsSol, _distance(wAvg, bAvg, wFull, bFull))
		end
		# run remaining iters
		shuffle!(sInd)
		for idx in sInd
            elapsed += 1  # update elapsed iters count
            λi = λs(elapsed)  # update step
			wGrad[:], bGrad = _grad(prob, wCurr, bCurr, idx)
			wCurr[:] = ℓ₁prox(wCurr - λi * wGrad, λi * prob.τ)
			bCurr -= λi * bGrad
			wAvg += (wCurr - wAvg) / elapsed
			bAvg += (bCurr - bAvg) / elapsed
			(elapsed > K) && break
		end
        # update stats
        push!(dsOut, norm(wAvg[zMask]))
        push!(dsFun, logistic_loss(prob, wAvg, bAvg))
        push!(dsSol, _distance(wAvg, bAvg, wFull, bFull))
        return (wAvg, bAvg), dsFun .- fLoss, dsOut, dsSol
	end


	"""
		rda(prob::LsProb, K, γ, wFull, bFull, wCurr, bCurr) -> ((wCurr, bCurr), dsFun, dsOut, dsSol)

	Loop for regularized dual averaging, running for `K` iterations
    with a step size multiplier `γ` given a full-gradient guess `(wFull, bFull)`
    and an initial guess `(wCurr, bCurr)`.
    Return:
    - `wCurr, bCurr`: final iterates found
    - `dsFun`: function value gap
    - `dsOut`: norm on the complement of the support of `wFull`
	- `dsSol`: normalized distance to full solution `(wFull, bFull)`
	"""
	function rda(prob::RegProb, K, γ,
				 wFull::Array{Float64, 1}, bFull::Float64,
				 wCurr::Array{Float64, 1}, bCurr::Float64)
		# stepsize schedule
        λs = (i -> sqrt(i) / γ)
		# average iterates
		wAvg = fill(0.0, length(wCurr)); bAvg = bCurr
		# average gradient
		wGradAvg = fill(0.0, length(wCurr)); bGradAvg = 0.0
		m = length(prob.yData)
		sInd = collect(1:length(prob.yData))  # array of indices
		k = 0  # init iter. counter
        # retrieve statistics from full version
        fLoss = logistic_loss(prob, wFull, bFull)
        zMask = abs.(wFull) .< 1e-10
        # error metrics
        dsOut = []; dsFun = []; dsSol = []
		for s = 1:div(K, length(prob.yData))  # full passes
			@info("Running pass: $s")
			shuffle!(sInd)
			# update stats every 10 full passes
			if (s % 10 == 1)
				push!(dsOut, norm(wAvg[zMask]))
				push!(dsFun, logistic_loss(prob, wAvg, bAvg))
				push!(dsSol, _distance(wAvg, bAvg, wFull, bFull))
			end
			# run a full iteration over the dataset
			for idx in sInd
				k += 1  # update elapsed iters count
				λi = λs(k)
				wGrad, bGrad = _grad(prob, wCurr, bCurr, idx)
				wGradAvg = ((k - 1) / k) * wGradAvg + (wGrad / k)
				bGradAvg = ((k - 1) / k) * bGradAvg + (bGrad / k)
				wCurr[:] = ℓ₁prox(-wGradAvg * λi, λi * prob.τ)
				bCurr = -λi * bGradAvg
				# update average iterates
				wAvg += (wCurr - wAvg) / k
				bAvg += (bCurr - bAvg) / k
			end
        end
		# update stats
		push!(dsOut, norm(wAvg[zMask]))
		push!(dsFun, logistic_loss(prob, wAvg, bAvg))
		push!(dsSol, _distance(wAvg, bAvg, wFull, bFull))
		# run remaining iters
		shuffle!(sInd)
		for idx in sInd
			k += 1  # update elapsed iters count
			λi = λs(k)
			wGrad, bGrad = _grad(prob, wCurr, bCurr, idx)
			wGradAvg = ((k - 1) / k) * wGradAvg + (wGrad / k)
			bGradAvg = ((k - 1) / k) * bGradAvg + (bGrad / k)
			wCurr[:] = ℓ₁prox(-wGradAvg * λi, λi * prob.τ)
			bCurr = -λi * bGradAvg
			# update average iterates
			wAvg += (wCurr - wAvg) / k
			bAvg += (bCurr - bAvg) / k
			(k > K) && break
		end
		# update stats (final)
		push!(dsOut, norm(wAvg[zMask]))
		push!(dsFun, logistic_loss(prob, wAvg, bAvg))
		push!(dsSol, _distance(wAvg, bAvg, wFull, bFull))
		return (wAvg, bAvg), dsFun .- fLoss, dsOut, dsSol
	end


	"""
		logistic_loss(prob::RegProb, wCurr, bCurr)

	Compute the logistic loss over the dataset given parameters `wCurr, bCurr`,
	including the ℓ₁ penalty.
	"""
	function logistic_loss(prob::RegProb, wCurr, bCurr)
		return mean(log.(1 .+ exp.(-prob.yData .* (prob.xData * wCurr .+ bCurr)))) +
			prob.τ * norm(wCurr, 1)
	end


	"""
		sProx(prob::RegProb, T, K, λ) -> ((wCurr, bCurr), dsFun, dsOut, dsSol)

	Apply the stochastic proximal gradient method to solve an ℓ₁-reg. least
	squares problem `prob`. Run for `T` iterations with inner iteration
	number given by `K`.
	Step size schedule is given by `λ`, which can be a constant or a callable.
	Return:
	- `wCurr, bCurr`: the estimates found by the algorithm
	- `dsFun`: loss function values, scaled by the loss found by the full proximal
	  gradient method
	- `dsOut`: the norm of the weight vector `wCurr` outside the support
	  weights found by the full gradient method
	- `dsSol`: the (normalized) distance of the stochastic solution to the full
	  gradient solution
	"""
	function sProx(prob::RegProb, T, K, λ)
		d = 784; wCurr = randn(d); normalize!(wCurr); bCurr = 0.0
		step = setup_step(λ)
		# get solution from full proximal gradient
		(wFull, bFull), errFull, zMask, nnzFull = ista(
			prob, copy(wCurr), bCurr, 20 * T)
		# error metrics
		dsOut = fill(0.0, T)
		dsFun = fill(0.0, T)
		dsSol = fill(0.0, T)
		for k = 1:T
			dsFun[k] = logistic_loss(prob, wCurr, bCurr)
			dsOut[k] = norm(wCurr[zMask])
			dsSol[k] = norm(wCurr - wFull) / norm(wFull)
			@info("sProx - iteration: $(k) - err: $(dsFun[k]) - dsOut: $(dsOut[k])")
			wCurr[:], bCurr = _proxGrad(prob, wCurr, bCurr, K, step(k))
		end
		return (wCurr, bCurr), dsFun .- errFull, dsOut, dsSol
	end


	"""
		sProx(prob::RegProb, T, K, λ,
		      wFull::Array{Float64, 1}, bFull::Float64,
			  wCurr::Array{Float64, 1}, bFull::Float64) -> ((wCurr, bCurr), dsFun, dsOut, dsSol)

	Apply the stochastic proximal gradient method to solve an ℓ₁-reg. least
	squares problem `prob`. Run for `T` iterations with inner iteration
	number given by `K`, given the iterates found by running full
	proximal gradient steps, `wFull` and `bFull`.
	Step size schedule is given by `λ`, which can be a constant or a callable.
	Return:
	- `wCurr, bCurr`: the estimates found by the algorithm
	- `dsFun`: the loss function value, scaled by the loss found by the full
	  proximal gradient method
	- `dsOut`: the norm of the weight vector `wCurr` outside the support of the
	  weight vector `wFull`
	- `dsSol`: the (normalized) distance of the stochastic solution to `wFull`
	"""
	function sProx(prob::RegProb, T, K, λ,
				   wFull::Array{Float64, 1}, bFull::Float64,
				   wCurr::Array{Float64, 1}, bCurr::Float64)
		fLoss = logistic_loss(prob, wFull, bFull)
		# compute mask of zeros
		zMask = (abs.(wFull) .< 1e-10)
		step = setup_step(λ)
		# error metrics
		dsOut = fill(0.0, T+1)
		dsFun = fill(0.0, T+1)
		dsSol = fill(0.0, T+1)
		for k = 1:T
			dsFun[k] = logistic_loss(prob, wCurr, bCurr) - fLoss
			dsOut[k] = norm(wCurr[zMask])
			dsSol[k] = _distance(wCurr, bCurr, wFull, bFull)
			@info("sProx - iteration: $(k) - err: $(dsFun[k]) - dist: $(dsOut[k])")
			wCurr[:], bCurr = _proxGrad(prob, wCurr, bCurr, K, step(k))
		end
		dsFun[T+1] = logistic_loss(prob, wCurr, bCurr) - fLoss
		dsOut[T+1] = norm(wCurr[zMask])
		dsSol[T+1] = _distance(wCurr, bCurr, wFull, bFull)
		return (wCurr, bCurr), dsFun, dsOut, dsSol
	end


	"""
		ista(prob, iters; η=nothing)

	Run the ISTA algorithm to solve an ℓ₁-reg. logistic regression problem
	`prob` for at most `iters` iterations with a constant step size `η`.
	"""
	function ista(prob::RegProb, iters::Int; η=nothing, ϵ=1e-5)
		d = 784;  # MNIST
		wCurr = randn(d); normalize!(wCurr); bCurr = 0.0
		return ista(prob, wCurr, bCurr, iters, η=η, ϵ=ϵ)
	end


	#= termRatio: compute the numerical ratio for convergence checks =#
	_termRatio(prob, wNext, wCurr, bNext, bCurr) = begin
		nDiff = norm(vcat(wNext - wCurr, bNext - bCurr))
		return nDiff / (prob.τ * min(norm(wNext), norm(wCurr)))
	end


	"""
		ista(prob::RegProb, wCurr::Array{Float64, 1}, bCurr::Float64,
			 iters::Int; η=nothing, ϵ=1e-5)

	Run the ISTA algorithm starting from `(wCurr, bCurr)` for at most `iters`
	iterations. Optionally, set the step size `η` and the numerical tolerance
	`ϵ`.
	"""
	function ista(prob::RegProb, wCurr::Array{Float64, 1}, bCurr::Float64,
				  iters::Int; η=nothing, ϵ=1e-5)
		η = (η == nothing) ? 1 / SparseLogReg.getGradConstant(prob) : η
		wNext = copy(wCurr); bNext = bCurr
		for k = 1:iters
			wNext[:], bNext = _proxGradFull(prob, wCurr, bCurr, η)
			eps = _termRatio(prob, wNext, wCurr, bNext, bCurr)
			wCurr[:] = wNext[:]; bCurr = bNext
            if (k % 100 == 0)
                eps = _termRatio(prob, wNext, wCurr, bNext, bCurr)
                @info("(k, ϵ): $(k), $(eps)")
			    # break if reached numerical accuracy or if everything is zero
                ((eps <= ϵ) || isinf(eps)) && break
            end
		end
		# return iterates, final loss, mask of zeros, as well as number
		# of nonzeros
        return (wCurr, bCurr), logistic_loss(prob, wCurr, bCurr),
            (abs.(wCurr) .<= 1e-10), sum(abs.(wCurr) .> 1e-10)
	end


	"""
		fista(prob, iters; η=nothing)

	Run the FISTA algorithm to solve an ℓ₁-reg. logistic regression problem
	`prob` for at most `iters` iterations with a constant step size `η`.
	"""
	function fista(prob::RegProb, iters::Int; η=nothing, ϵ=1e-5)
		d = 784;  # MNIST
		wCurr = randn(d); normalize!(wCurr); bCurr = 0.0
		return fista(prob, wCurr, bCurr, iters, η=η, ϵ=ϵ)
	end


	"""
		fista(prob::RegProb, wCurr::Array{Float64, 1}, bCurr::Float64,
		      iters::Int; η=nothing, ϵ=1e-5)

	Run the FISTA algorithm starting from `(wCurr, bCurr)` for at most `iters`
	iterations. Optionally, set the step size `η` and the numerical tolerance
	`ϵ`.
	"""
	function fista(prob::RegProb, wCurr::Array{Float64, 1}, bCurr::Float64,
				   iters::Int; η=nothing, ϵ=1e-5)
		η = (η == nothing) ? 1 / SparseLogReg.getGradConstant(prob) : η
		wNext = copy(wCurr); bNext = 0.0; t = 1.0; tNew = 1.0
		yW = copy(wCurr); yB = 0.0
		for k = 1:iters
			# x_k = p_L(y_k)
			wNext[:], bNext = _proxGradFull(prob, yW, yB, η)
			tNew = (1 + sqrt(1 + 4 * t^2)) / 2
			yW[:] = wNext[:] + ((t - 1) / (tNew)) * (wNext[:] - wCurr[:])
			yB = bNext + ((t - 1) / tNew) * (bNext - bCurr)
            if (k % 100 == 0)
                eps = _termRatio(prob, wNext, wCurr, bNext, bCurr)
                @info("(k, ϵ): $(k), $(eps)")
			    # break if reached numerical accuracy or if everything is zero
                ((eps <= ϵ) || isinf(eps)) && break
            end
			# update "old" iterates
			wCurr[:] = wNext[:]; bCurr = bNext; t = tNew
		end
		return (wCurr, bCurr), logistic_loss(prob, wCurr, bCurr),
			(abs.(wCurr) .<= 1e-10), sum(abs.(wCurr) .> 1e-10)
	end

end
