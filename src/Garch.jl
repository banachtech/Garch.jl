module Garch

using Random, Statistics, Optim, Parameters

export Std, GJR, fit!, simulate!, predict!

@with_kw mutable struct Std
    @deftype Float64
    σ = 0.01
    @assert σ > 0.0
    μ₁ = rand()
    @assert μ₁ > 0.0 && μ₁ < 1.0
    μ₂ = rand()
    @assert μ₂ > 0.0 && μ₂ < 1.0
end

@with_kw mutable struct GJR
    @deftype Float64
    σ = 0.01
    @assert σ > 0.0
    μ₁ = rand()
    @assert μ₁ > 0.0 && μ₁ < 1.0
    μ₂ = rand()
    @assert μ₂ > 0.0 && μ₂ < 1.0
    μ₃ = rand()
    @assert μ₃ > 0.0 && μ₃ < 1.0
    isdown::Bool = true
end

function nll(z, σ, rt)
    μ₁, μ₂ = exp.(-exp.(-z))
    ll = 0.0
    n = length(rt)
    σ² = σ * σ
    h = σ²
    μ = 1.0 - μ₂
    for t ∈ 2:n
        h = σ² + μ₁ * (μ₂ * h + μ * rt[t-1] * rt[t-1] - σ²)
        if h > 0.0
            ll += 0.5 * (log(h) + rt[t] * rt[t] / h)
        end
    end
    return ll
end

function nll(z, σ, rt, J)
    μ₁, μ₂, μ₃ = exp.(-exp.(-z))
    ll = 0.0
    n = length(rt)
    σ² = σ * σ
    h = σ²
    for t ∈ 2:n
        h = σ² + μ₁ * ((1.0 - μ₂ + 2.0 * (μ₂ - μ₃) * J[t-1]) * rt[t-1] * rt[t-1] + μ₃ * h - σ²)
        if h > 0.0
            ll += 0.5 * (log(h) + rt[t] * rt[t] / h)
        end
    end
    return ll
end


function fit!(model::GJR, x::Vector{Float64}; time_limit = 60.0, solver = NelderMead(), g_tol = 1.0e-6)
    @unpack σ, μ₁, μ₂, μ₃, isdown = model
    σ = std(x)
    J = 1.0 .* (isdown ? (x .< 0.0) : (x .> 0.0))
    p = [-log(-log(μ₁)), -log(-log(μ₂)), -log(-log(μ₃))]
    p0 = 4.0 .* rand(3)
    res = optimize(p -> nll(p, σ, x, J), p0, solver, Optim.Options(time_limit = time_limit, g_tol = g_tol))
    μ₁, μ₂ = exp.(-exp.(-res.minimizer))
    @pack! model = σ, μ₁, μ₂, μ₃, isdown
    return res
end


function fit!(model::Std, x::Vector{Float64}; time_limit = 60.0, solver = NelderMead(), g_tol = 1.0e-6)
    @unpack σ, μ₁, μ₂ = model
    σ = std(x)
    p = [-log(-log(μ₁)), -log(-log(μ₂))]
    p0 = 4.0 .* rand(2)
    res = optimize(p -> nll(p, σ, x), p0, solver, Optim.Options(time_limit = time_limit, g_tol = g_tol))
    μ₁, μ₂ = exp.(-exp.(-res.minimizer))
    @pack! model = σ, μ₁, μ₂
    return res
end


function simulate(model::Std, pathlen::Integer; r0 = 0.0)
    r = fill(0.0, pathlen + 1)
    r[1] = r0
    @unpack σ, μ₁, μ₂ = model
    σ² = σ * σ
    h = σ²
    μ = 1.0 - μ₂
    for t ∈ 1:pathlen
        h = max(0.0, σ² + μ₁ * (μ₂ * h + μ * r[t] * r[t] - σ²))
        r[t+1] = √h * randn()
    end
    return r[2:end]
end

function simulate(model::GJR, pathlen::Integer; r0 = 0.0)
    r = fill(0.0, pathlen + 1)
    r[1] = r0
    @unpack σ, μ₁, μ₂, μ₃, isdown = model
    σ² = σ * σ
    h = σ²
    for t ∈ 1:pathlen
        J = isdown ? (r[t] < 0.0) : (r[t] > 0.0)
        h = max(σ² + μ₁ * ((1.0 - μ₂ + 2.0 * (μ₂ - μ₃) * J) * r[t] * r[t] + μ₃ * h - σ²), 0.0)
        r[t+1] = √h * randn()
    end
    return r[2:end]
end

function predict(model::Union{GJR,Std}, paths::Integer, pathlen::Integer)
    vol = fill(0.0, paths)
    vol .= [std(simulate(model, pathlen, r0 = 0.0)) for i ∈ 1:paths]
    return vol
end

end