include("Utils.jl")

using Zygote
using Revise
using Trapz
using Roots
using Plots
using LaTeXStrings
using HypergeometricFunctions: _₂F₁, pFq
using LinearAlgebra: eigvals, cholesky, Diagonal
import Base.length, Base.collect, Base.values

using .Utils: geomspace

# SI units
Gₙ = 6.67408e-11  # m^3 s^-2 kg^-1
c = 299792458.  # m/s
MSun = 1.98855e30  # kg
pc = 3.08567758149137e16 # m
yr_to_s = 365.25 * 24 * 3600

ℳ(m₁, m₂) = (m₁ * m₂)^(3/5) / (m₁ + m₂)^(1/5)
r_isco(m₁) = 6 * Gₙ * m₁ / c^2
f_isco(m₁) = √(Gₙ * m₁ / r_isco(m₁)^3) / π
rₛ(m₁, ρₛ, γₛ) = ((3 - γₛ) * 0.2^(3 - γₛ) * m₁ / (2 * π * ρₛ))^(1/3)
ψᵥ(m₁::Float64, m₂::Float64) = 1/16 * (c^3 / (π * Gₙ * ℳ(m₁, m₂)))^(5/3)

abstract type Binary end

# Phase can be converted to the hypergeometric parametrization
abstract type HypableDress <: Binary end

values(system::T) where T <: Binary = (fn -> getfield(system, fn)).(fieldnames(T))
collect(system::T) where T <: Binary = values(system)
length(system::T) where T <: Binary = length(values(system))
ψᵥ(system::T) where T <: Binary = 1/16 * (c^3 / (π * Gₙ * ℳ(system)))^(5/3)

# Static dress
struct StaticDress <: HypableDress
    γₛ::Float64
    c_f::Float64
    ℳ::Float64
    Φ_c::Float64
    t̃_c::Float64
    dₗ_ι::Float64
end

# γₛ(sd::StaticDress) = sd.γₛ
# c_f(sd::StaticDress) = sd.c_f
ℳ(sd::StaticDress) = sd.ℳ
q(sd::StaticDress) = sd.q
intrinsics(StaticDress) = (:γₛ, :c_f, :ℳ)

function c_f(m₁, m₂, ρₛ, γₛ)
    # Λ = 3.0
    Λ = √(m₁ / m₂)  # TODO: sort this out!
    ξ = 0.58
    M = m₁ + m₂
    c_gw = 64 * Gₙ^3 * M * m₁ * m₂ / (5 * c^5)
    c_df = 8 * π * √(Gₙ) * m₂ * log(Λ) * ρₛ * rₛ(m₁, ρₛ, γₛ)^γₛ * ξ / (√(M) * m₁)
    return c_df / c_gw * (Gₙ * M / π^2)^((11 - 2 * γₛ) / 6)
end

function make_static_dress(m₁, m₂, ρₛ, γₛ, dₗ=1e8*pc, ι=0.0, Φ_c=0.0, t_c=0.0)
    @assert m₁ > m₂
    @assert m₂ > 0
    t̃_c = t_c + dₗ / c
    dₗ_ι = log((1 + cos(ι)^2) / (2 * dₗ))
    return StaticDress(γₛ, c_f(m₁, m₂, ρₛ, γₛ), ℳ(m₁, m₂), Φ_c, t̃_c, dₗ_ι)
end

# Converter
function hypify(sd::StaticDress)
    ϑ = 5 / (11 - 2 * sd.γₛ)
    fₜ = sd.c_f^(3 / (11 - 2 * sd.γₛ))
    return HypParams(ψᵥ(sd), ϑ, 0., 1., fₜ)
end

# Dynamic dress
struct DynamicDress <: HypableDress
    γₛ::Float64
    c_f::Float64
    ℳ::Float64
    q::Float64
    Φ_c::Float64
    t̃_c::Float64
    dₗ_ι::Float64
end

# ρₛ(dd::DynamicDress) = dd.ρₛ
# γₛ(dd::DynamicDress) = dd.γₛ
# c_f(dd::DynamicDress) = dd.c_f
ℳ(dd::DynamicDress) = dd.ℳ
q(dd::DynamicDress) = dd.q
m₁(dd::DynamicDress) = (1 + dd.q)^(1/5) / dd.q^(3/5) * dd.ℳ
m₂(dd::DynamicDress) = (1 + dd.q)^(1/5) * dd.q^(2/5) * dd.ℳ

# Get intrinsic parameters
function intrinsics(T::Type)
    if T == StaticDress
        return (:γₛ, :c_f, :ℳ)
    elseif T == DynamicDress
        return (:γₛ, :c_f, :ℳ, :q)
    end
end

function make_dynamic_dress(m₁, m₂, ρₛ, γₛ, dₗ=1e8 * pc, ι=0.0, Φ_c=0.0, t_c=0.0)
    sd = make_static_dress(m₁, m₂, ρₛ, γₛ, dₗ, ι, Φ_c, t_c)
    return DynamicDress(sd.γₛ, sd.c_f, ℳ(sd), m₂ / m₁, sd.Φ_c, sd.t̃_c, sd.dₗ_ι)
end

make_static_dress(dd::DynamicDress) = StaticDress(dd.γₛ, dd.c_f, ℳ(dd), dd.Φ_c, dd.t̃_c, dd.dₗ_ι)

# TODO: rederive with new waveform model
function f_b(m₁, m₂, γₛ)
    α₁ = 1.39191077
    α₂ = 0.443089063
    β = 7.03764788e3
    κ = -2.44956668
    δ = 0.839699908
    return β * (m₂ / MSun)^α₂ / (m₁ / MSun)^α₁ * (1 + κ * log10(γₛ) + δ)
end

# Converter
function hypify(dd::DynamicDress)
    f_eq = hypify(make_static_dress(dd)).fₜ
    fₜ = f_b(m₁(dd), m₂(dd), dd.γₛ)

    γₑ = 5/2  # TODO: CHANGE!
    
    ϑ = 5 / (2 * γₑ)
    λ = (11 - 2 * (dd.γₛ + γₑ)) / 3
    η = (5 + 2 * γₑ) / (2 * (8 - dd.γₛ)) * (f_eq / fₜ)^((11 - 2 * dd.γₛ) / 3)

    return HypParams(ψᵥ(dd), ϑ, λ, η, fₜ)
end

# Convenient waveform parametrization
struct HypParams
    ψᵥ::Float64
    ϑ::Float64
    λ::Float64
    η::Float64
    fₜ::Float64
end

function Φ_to_c_indef(f, hp::HypParams)
    x = f / hp.fₜ
    if hp.λ == 0 && hp.η == 1
        return hp.ψᵥ / f^(5/3) * _₂F₁(1, hp.ϑ, 1 + hp.ϑ, -x^(-5 / (3 * hp.ϑ)))
    else
        return hp.ψᵥ / f^(5/3) * (1 - hp.η * x^(-hp.λ) * (
            1 - _₂F₁(1, hp.ϑ, 1 + hp.ϑ, -x^(-5 / (3 * hp.ϑ)))
        ))
    end
end

function t_to_c_indef(f, hp::HypParams)
    x = f / hp.fₜ
    if hp.λ == 0 && hp.η == 1
        return 5 * hp.ψᵥ * _₂F₁(1, 8*hp.ϑ/5, 1 + 8*hp.ϑ/5, -x^(-5/(3*hp.ϑ))) / (
            16 * π * f^(8/3)
        )
    else
        # Can hit unsafe_gamma
        coeff = hp.ψᵥ * x^(-hp.λ) / (16 * π * (1 + hp.λ) * (8 + 3 * hp.λ) * f^(8/3))
        term₁ = 5 * (1 + hp.λ) * (8 + 3 * hp.λ) * x^hp.λ
        term₂ = 8 * hp.λ * (8 + 3 * hp.λ) * hp.η * _₂F₁(1, hp.ϑ, 1 + hp.ϑ, -x^(-5/(3*hp.ϑ)))
        term₃ = -40 * (1 + hp.λ) * hp.η * _₂F₁(
            1,
            -1/5 * hp.ϑ * (8 + 3 * hp.λ),
            1 - 1/5 * hp.ϑ * (8 + 3 * hp.λ),
            -x^(5/(3*hp.ϑ))
        )
        term₄ = -8 * hp.λ * hp.η * (3 + 3 * hp.λ + 5 * _₂F₁(
            1, 1/5 * hp.ϑ * (8 + 3 * hp.λ), 1 + 1/5 * hp.ϑ * (8 + 3 * hp.λ), -x^(-5/(3 * hp.ϑ))
        ))
        return coeff * (term₁ + term₂ + term₃ + term₄)
    end
end

Φ_to_c(f, f_c, hp::HypParams) = Φ_to_c_indef(f, hp) - Φ_to_c_indef(f_c, hp)
t_to_c(f, f_c, hp::HypParams) = t_to_c_indef(f, hp) - t_to_c_indef(f_c, hp)

function d²Φ_dt²(f, hp::HypParams)
    x = f / hp.fₜ
    if hp.λ == 0 && hp.η == 1
        return 12 * π^2 * f^(11/3) * (1 + x^(-5/(3*hp.ϑ))) / (5 * hp.ψᵥ)
    else
        return 12 * π^2 * f^(11/3) * (1 + x^(5/(3*hp.ϑ)) * x^hp.λ) / (
            hp.ψᵥ * (
                -5 * hp.η - 3 * hp.λ * hp.η + 5 * x^hp.λ + x^(5/(3*hp.ϑ)) * (5*x^hp.λ - 3*hp.λ*hp.η)
                + 3 * hp.λ * hp.η * (1 + x^(5/(3*hp.ϑ))) * _₂F₁(
                    1, hp.ϑ, 1 + hp.ϑ, -x^(5/(3*hp.ϑ))
                )
            )
        )
    end
end

# If possible, convert to HypParams
Φ_to_c(f, f_c, system::T) where T <: HypableDress = Φ_to_c(f, f_c, hypify(system))
t_to_c(f, f_c, system::T) where T <: HypableDress = t_to_c(f, f_c, hypify(system))
d²Φ_dt²(f, system::T) where T <: HypableDress = d²Φ_dt²(f, hypify(system))

# General waveform functions
Φ̃(f, f_c, system::T) where T <: Binary = 2 * π * f * t_to_c(f, f_c, system) - Φ_to_c(f, f_c, system)
Ψ(f, f_c, system::T) where T <: Binary = 2 * π * f * system.t̃_c - system.Φ_c - π/4 - Φ̃(f, f_c, system)

function amp₊(f, system::T) where T <: Binary
    h₀ = 1/2 * 4 * π^(2/3) * (Gₙ * ℳ(system))^(5/3) * f^(2/3) / c^4 * √(2 * π / d²Φ_dt²(f, system))
    return h₀ * exp(system.dₗ_ι)
end

# LISA noise curve
Sₙ_LISA(f) = 1 / f^14 * 1.80654e-17 * (0.000606151 + f^2) * (3.6864e-76 + 3.6e-37 * f^8 + 2.25e-30 * f^10 + 8.66941e-21 * f^14)

# trapz for matrix-values functions
function trapz_1d_mat(xs, ys)
    @assert length(xs) == length(ys)
    result = zeros(size(ys[1]))
    for i in range(1, stop=length(xs) - 1)
        result .+= (xs[i + 1] - xs[i]) * (ys[i + 1] + ys[i]) ./ 2
    end
    return result
end

###################
# Fisher analysis #
###################
# Factors for rescaling some parameter derivatives to relative ones (d/dx d/dlog(x))
rescalings(sd::StaticDress) = [sd.γₛ, sd.c_f, sd.ℳ, 1., 1., 1.]
rescalings(dd::DynamicDress) = [dd.γₛ, dd.c_f, dd.ℳ, dd.q, 1., 1., 1.]

function fim_integrand_num(f, f_c, system::T) where T <: Binary
    ∂amp₊ = collect(values(gradient(s -> amp₊(f, s), system)[1]))
    ∂amp₊[∂amp₊ .=== nothing] .= 0.
    ∂amp₊ = convert(Array{Float64}, ∂amp₊)

    ∂Ψ = collect(values(gradient(s -> Ψ(f, f_c, s), system)[1]))
    ∂Ψ[∂Ψ .=== nothing] .= 0.
    ∂Ψ = convert(Array{Float64}, ∂Ψ)

    # Convert to log derivatives for intrinsic parameters
    scales = rescalings(system)
    ∂amp₊ .*= scales
    ∂Ψ .*= scales
    
    return 4 * (∂amp₊ * ∂amp₊' + amp₊(f, system)^2 * ∂Ψ * ∂Ψ')
end

# From here on, I assume fₕ = f_c
function fim(fₗ, fₕ, f_c, system::T, n) where T <: Binary
    fs = geomspace(fₗ, fₕ, n)
    integrand(f) =  fim_integrand_num(f, f_c, system) / Sₙ_LISA(f)
    return trapz_1d_mat(fs, integrand.(fs))
end

function fim_uncertainties(fₗ, fₕ, f_c, system::T, n=1000) where T <: Binary
    Γ = fim(fₗ, fₕ, f_c, system, n)

    # Improve stability
    scales = sqrt(inv(Diagonal(Γ)))
    Γr = scales * Γ * scales
    Σ = scales * inv(Γr) * scales

    # Checks
    @assert all(eigvals(Γr) .> 0) eigvals(Γr)

    return sqrt.([Σ[i, i] for i in 1:size(Σ)[1]])
end

function sig_noise_ratio(fₗ, fₕ, system::T, n::Int=5000) where T <: Binary
    fs = geomspace(fₗ, fₕ, n)
    integrand(f) = amp₊(f, system)^2 / Sₙ_LISA(f)
    return √(4 * trapz(fs, integrand.(fs)))
end

#################
# Static checks #
#################
fₗ = 2.26e-2  # frequency 5 years before merger (HACKY ESTIMATE)
fₕ = 4.39701  # ISCO frequency (HACKY ESTIMATE)
sd = make_static_dress(1e3 * MSun, 1. * MSun, 226. * MSun/pc^3, 7/3)
sd = make_static_dress(1e3 * MSun, 1. * MSun, 1e-2 * MSun/pc^3, 2.4)
println(log10(fim_uncertainties(fₗ, fₕ, fₕ, sd, 100)[1]))
println(log10(fim_uncertainties(fₗ, fₕ, fₕ, sd, 100)[2]))
println(log10(fim_uncertainties(fₗ, fₕ, fₕ, sd, 100)[3]))
sig_noise_ratio(fₗ, fₕ, sd, 1000)  # agrees with David's notebook

##################
# Dynamic checks #
##################
fₗ = 2.26e-2  # frequency 5 years before merger (HACKY ESTIMATE)
fₕ = 4.39701  # ISCO frequency (HACKY ESTIMATE)
dd = make_dynamic_dress(1e3 * MSun, 1. * MSun, 226. * MSun/pc^3, 2.25001)#7/3)
println(log10(fim_uncertainties(fₗ, fₕ, fₕ, dd, 100)[1]))
println(log10(fim_uncertainties(fₗ, fₕ, fₕ, dd, 100)[2]))
println(log10(fim_uncertainties(fₗ, fₕ, fₕ, dd, 100)[3]))
println(log10(fim_uncertainties(fₗ, fₕ, fₕ, dd, 100)[4]))
sig_noise_ratio(fₗ, fₕ, dd, 1000)  # agrees with David's notebook


############
# Plotting #
############
function plot_static_errs(m₁, m₂)
    fₕ = f_c = f_isco(m₁)
    t_c = 5.0 * yr_to_s  # scales
    n_intrinsic = length(intrinsics(StaticDress))

    function fn(log10_ρₛ, γₛ)
        sd = make_static_dress(m₁, m₂, 10^log10_ρₛ * MSun/pc^3, γₛ)
        fₗ = find_zero(f -> t_to_c(f, f_c, sd) - t_c, (0.001 * f_c, f_c))
        return log10.(fim_uncertainties(fₗ, fₕ, fₕ, sd, 80)[1:n_intrinsic])
    end

    # Apply to grid
    log10_ρₛs = -2:0.4:3
    γₛs = 2.25:0.01:2.4
    uncertainties = zeros(length(γₛs), length(log10_ρₛs), n_intrinsic)
    print(size(uncertainties))
    for (i, log10_ρₛ) in enumerate(log10_ρₛs)
        for (j, γₛ) in enumerate(γₛs)
            uncertainties[j, i, :] = fn(log10_ρₛ, γₛ)
        end
    end
    
    plots = []
    for i in 1:size(uncertainties, 3)
        p = contour(log10_ρₛs, γₛs, uncertainties[:, :, i], fill=true, fillcolor=cgrad(:inferno, rev=true), size=(550, 500))
        xaxis!(L"\log_{10} \rho_s", (minimum(log10_ρₛs), maximum(log10_ρₛs)))
        yaxis!(L"\gamma_s", (minimum(γₛs), maximum(γₛs)))
        title!("$(fieldnames(StaticDress)[i])")
        push!(plots, p)
    end
    plot(plots...)
end

function plot_dynamic_errs(m₁, m₂)
    fₕ = f_c = f_isco(m₁)
    t_c = 5.0 * yr_to_s  # scales
    n_intrinsic = length(intrinsics(DynamicDress))

    function fn(log10_ρₛ, γₛ)
        sd = make_dynamic_dress(m₁, m₂, 10^log10_ρₛ * MSun/pc^3, γₛ)
        fₗ = find_zero(f -> t_to_c(f, f_c, sd) - t_c, (0.001 * f_c, f_c))
        return log10.(fim_uncertainties(fₗ, fₕ, fₕ, sd, 80)[1:n_intrinsic])
    end

    # Apply to grid
    log10_ρₛs = -2:0.4:3
    γₛs = 2.25001:0.01:2.4
    uncertainties = zeros(length(γₛs), length(log10_ρₛs), n_intrinsic)
    for (i, log10_ρₛ) in enumerate(log10_ρₛs)
        for (j, γₛ) in enumerate(γₛs)
            uncertainties[j, i, :] = fn(log10_ρₛ, γₛ)
        end
    end
    
    plots = []
    for i in 1:size(uncertainties, 3)
        p = contour(log10_ρₛs, γₛs, uncertainties[:, :, i], fill=true, fillcolor=cgrad(:inferno, rev=true), size=(550, 500))
        xaxis!(L"\log_{10} \rho_s", (minimum(log10_ρₛs), maximum(log10_ρₛs)))
        yaxis!(L"\gamma_s", (minimum(γₛs), maximum(γₛs)))
        title!("$(fieldnames(DynamicDress)[i])")
        push!(plots, p)
    end
    plot(plots...)
end

function plot_snr(m₁, m₂)
    log10_ρₛs = -2:0.2:3
    γₛs = 1.9:0.05:7/3

    fₕ = f_c = f_isco(m₁ * MSun)
    t_c = 5.0 * yr_to_s  # scales

    function fn(log10_ρₛ, γₛ)
        system = make_static_dress(m₁, m₂, 10^log10_ρₛ * MSun/pc^3, γₛ)
        fₗ = find_zero(f -> t_to_c(f, f_c, system) - t_c, (0.001 * f_c, f_c))
        return sig_noise_ratio(fₗ, fₕ, system)
    end

    heatmap(
        log10_ρₛs,
        γₛs,
        fn,
        # clim=(8.5, 9.1),
        size=(550, 500)
    )
    xaxis!(L"\log_{10} \rho_s")
    yaxis!(L"\gamma_s")
end

################
# Other checks #
################
function plot_Φ()
    m₁ = 1e3
    m₂ = 1.0
    γₛ = 7/3.
    ρₛ = 200.
    sd = make_static_dress(m₁, m₂, ρₛ, γₛ)
    dd = make_dynamic_dress(m₁, m₂, ρₛ, γₛ)

    f_c = f_isco(m₁ * MSun) * 100
    fs = geomspace(1e-4, f_c * 0.99, 500)
    
    Φₛ(f) = Φ_to_c(f, f_c, sd)
    Φ(f) = Φ_to_c(f, f_c, dd)
    Φᵥ(f) = ψᵥ(dd) ./ f.^(5/3) .- ψᵥ(dd) / f_c^(5/3)

    plot(fs, Φᵥ.(fs), label="V")
    plot!(fs, Φᵥ.(fs) .- Φₛ.(fs), legend=:bottomleft, label="V - S")
    plot!(fs, Φᵥ.(fs) .- Φ.(fs), legend=:bottomleft, label="V - D")

    xaxis!(L"f~\mathrm{[Hz]}", :log, (6e-4, 8e0))
    yaxis!(L"\mathrm{Phase~[rad]}", :log, (1e-4, 1e12))
end

function plot_amp₊()
    m₁ = 1e3
    m₂ = 1.0
    γₛ = 1.9
    ρₛ = 100.0
    sd = make_static_dress(m₁, m₂, ρₛ, γₛ)
    dd = make_dynamic_dress(m₁, m₂, ρₛ, γₛ)

    f_c = f_isco(m₁ * MSun) * 100
    fs = geomspace(1e-4, f_c * 0.999, 1000)
    
    plot(fs, (f -> amp₊(f, sd)).(fs) ./ fs.^(-7/6), legend=:bottomleft, label="S")
    plot!(fs, (f -> amp₊(f, dd)).(fs) ./ fs.^(-7/6), legend=:bottomleft, label="D")
    xaxis!(L"f~\mathrm{[Hz]}", :log)
    yaxis!("Amplitude", :log)
end

function plot_t_to_c()
    m₁ = 1e3
    m₂ = 1.0
    γₛ = 1.9
    ρₛ = 0.01
    sd = make_static_dress(m₁, m₂, ρₛ, γₛ)
    dd = make_dynamic_dress(m₁, m₂, ρₛ, γₛ)

    f_c = f_isco(m₁ * MSun)
    fs = geomspace(1e-6, f_c * 0.999, 500)
    
    t_to_cₛ(f) = t_to_c(f, f_c, sd)
    t_to_c_indefᵥ(f) = 5 * ψᵥ(sd) / (16 * π * f^(8/3))
    t_to_cᵥ(f) = t_to_c_indefᵥ(f) - t_to_c_indefᵥ(f_c)
    t_to_c_d(f) = t_to_c(f, f_c, dd)

    plot(fs, t_to_cᵥ.(fs), label="V")
    plot!(fs, t_to_cᵥ.(fs) - t_to_cₛ.(fs), label="V - S")
    plot!(fs, t_to_cᵥ.(fs) - t_to_c_d.(fs), label="V - D")
    xaxis!(L"f~\mathrm{[Hz]}", :log)
    yaxis!(L"t~[\mathrm{s}]", :log)
end