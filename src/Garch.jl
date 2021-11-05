"""
Fit Garch model to financial data.
"""
module Garch

using Random, Turing, Distributed, MCMCChains, Distributions, Statistics, StatsBase, LinearAlgebra, Optim, NLSolversBase

export simulate, garch, garchZD, fitBayes, fitMLE

"""
    simulate(a, b, c; y₀=0.001, h₀=0.0, n=100, d=Normal(0., 1.))

Simulate Garch(1,1) process:
`rₜ = √hₜ ξₜ`
`hₜ = a + b yₜ² + c hₜ₋₁`

"""
function simulate(a, b, c; y₀=0.001, h₀=0.0, n=100, d=Normal(0., 1.))
    @assert c >= 0 && a >= 0 && b > 0 "a,c shoud be >=0, b > 0"
	h = fill(0., n)
	y = similar(h)
	y[1] = y₀
	ξ = rand(d, n - 1)
	for i = 2:n
		h[i] = c + a * y[i - 1] * y[i - 1] + b * h[i - 1]
		y[i] = ξ[i - 1] * √h[i]
	end
	return y, h
end

"""
    garchZD(y::Vector{Float64})

Model Zero drift Garch(1,1) process for Bayesian estimation.
"""
@model function garchZD(y)
	n = length(y)
	h = 0.0
	a ~ Uniform(0.0001, 1)
	b ~ Uniform(0.0, 1)
	y[1] ~ Normal(y[1], h)
	for i = 2:n
		h = a * y[i - 1] * y[i - 1] + b * h
		y[i] ~ Normal(0, √h)
	end
end

"""
    garch(y::Vector{Float64})

Model Garch(1,1) for Bayesian estimation.
"""
@model function garch(y)
	n = length(y)
	h = 0.01
	a ~ Uniform(0.0001, 1.0)
	b ~ Uniform(0.0001, 1.0)
	w ~ Uniform(0.0, 1.0)
	y[1] ~ Normal(y[1], h)
	for i = 2:n
		h = w + a * y[i - 1] * y[i - 1] + b * h
		y[i] ~ Normal(0, √h)
	end
end

"""
    fitMLE(x::Vector{Float64}; isZD=true)

Fit Garch(1,1) on `x` using Maximum Likelihood estimation.
"""
function fitMLE(x; isZD=true)
	model = isZD ? garchZD(x) : garch(x)
	out = optimize(model, MLE())
    m, s = coef(out), stderror(out)
	return (pars = m, serr = s)
end

"""
    fitBayes(x::Vector{Float64}; isZD=true, nchains=4, nsims=1000, isDistributed=false)

Fit Garch(1,1) on `x` using Bayesian MCMC.
"""
function fitBayes(x; isZD=true, nchains=4, nsims=1000, isDistributed=false)
	model = isZD ? garchZD(x) : garch(x)
	if isDistributed
		chn = sample(model, NUTS(0.65), MCMCDistributed(), nsims, nchains, progress=false)
	else
		chn = sample(model, sampler, nsims, progress=false)
	end
	return chn
end


end
