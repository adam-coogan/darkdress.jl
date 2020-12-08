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
abstract type HypableDress <: Binary end

values(system::T) where T <: Binary = (fn -> getfield(system, fn)).(fieldnames(T))
collect(system::T) where T <: Binary = values(system)
length(system::T) where T <: Binary = length(values(system))
ψᵥ(system::T) where T <: Binary = 1/16 * (c^3 / (π * Gₙ * ℳ(system)))^(5/3)

# Static dress
struct StaticDress <: HypableDress
    γₛ::Float64
    # f_eq::Float64
    c_f::Float64
    ℳ::Float64
    Φ_c::Float64
    t̃_c::Float64
    dₗ_ι::Float64
end

ℳ(system::StaticDress) = system.ℳ

# Units: [m₁] = [m₂] = M⊙, [ρₛ] = M⊙ / pc^3, [dL] = pc.
function make_static_dress(m₁, m₂, ρₛ, γₛ, dₗ=1e8, ι=0.0, Φ_c=0.0, t_c=0.0)
    m₁ *= MSun
    m₂ *=  MSun
    ρₛ *= MSun / pc^3
    dₗ *= pc
    t̃_c = t_c + dₗ / c

    a = 5/3
    b = 5 / (11 - 2 * γₛ)
    # Λ = 3.0
    Λ = √(m₁ / m₂)  # TODO: sort this out!
    ξ = 0.58
    M = m₁ + m₂
    c_gw = 64 * Gₙ^3 * M * m₁ * m₂ / (5 * c^5)
    c_df = 8 * π * √(Gₙ) * m₂ * log(Λ) * ρₛ * rₛ(m₁, ρₛ, γₛ)^γₛ * ξ / (√(M) * m₁)
    c_f = c_df / c_gw * (Gₙ * M / π^2)^((11 - 2 * γₛ) / 6)
    
    dₗ_ι = log((1 + cos(ι)^2) / (2 * dₗ))

    return StaticDress(γₛ, c_f, ℳ(m₁, m₂), Φ_c, t̃_c, dₗ_ι)
end

# Converter
function hypify(system::StaticDress)
    a = 5/3
    b = 5 / (11 - 2 * system.γₛ)
    fₜ = system.c_f^(b / a)
    println(fₜ)
    d = ψᵥ(system)# / fₜ^a
    return HypParams(a, b, 0.0, d, fₜ)
end

# Static dress
struct DynamicDress <: HypableDress
    m₁::Float64
    m₂::Float64
    ρₛ::Float64
    γₛ::Float64
    Φ_c::Float64
    t̃_c::Float64
    dₗ_ι::Float64
end

ℳ(system::DynamicDress) = ℳ(system.m₁, system.m₂)

# Units: [m₁] = [m₂] = M⊙, [ρₛ] = M⊙ / pc^3, [dL] = pc.
function make_dynamic_dress(m₁, m₂, ρₛ, γₛ, dₗ=1e8, ι=0.0, Φ_c=0.0, t_c=0.0)
    m₁ *= MSun
    m₂ *=  MSun
    ρₛ *= MSun / pc^3
    dₗ *= pc
    t̃_c = t_c + dₗ / c
    dₗ_ι = log((1 + cos(ι)^2) / (2 * dₗ))

    return DynamicDress(m₁, m₂, ρₛ, γₛ, Φ_c, t̃_c, dₗ_ι)
end

# Converter
function hypify(system::DynamicDress)
    # Static dress parameters
    aₛ = 5/3
    bₛ = 5 / (11 - 2 * system.γₛ)
    dₛ = ψᵥ(system.m₁, system.m₂)
    # Λ = √(system.m₁ / system.m₂)  # TODO: sort this out!
    Λ = 3.
    ξ = 0.58
    M = system.m₁ + system.m₂
    c_gw = 64 * Gₙ^3 * M * system.m₁ * system.m₂ / (5 * c^5)
    c_df = 8 * π * √(Gₙ) * system.m₂ * log(Λ) * system.ρₛ * rₛ(system.m₁, system.ρₛ, system.γₛ)^system.γₛ * ξ / (√(M) * system.m₁)
    c_f = c_df / c_gw * (Gₙ * M / π^2)^((11 - 2 * system.γₛ) / 6)
    f_eq = c_f^(bₛ / aₛ)  # this is what's needed

    c_d = 5/3
    f_b = 0.45  # TODO: CHANGE!
    γₑ = 5/2  # TODO: CHANGE!
    b_fn(b) = 2 * sin(b * π) / (1 - b) - 5 * π / (8 - system.γₛ) * f_eq^((11 - 2 * system.γₛ) / 3) / f_b^((11 - 2 * (system.γₛ + γₑ)) / 3)
    b_d = find_zero(b_fn, (-1., 1.))

    a_d = b_d * (11 - 2 * (system.γₛ + γₑ)) / (3 * (b_d - 1))
    d_d = ψᵥ(system) * sin(π * b_d) / (π * b_d) * f_b^a_d

    return HypParams(a_d, b_d, c_d, d_d, f_b)
end

# Convenient waveform parametrization
struct HypParams
    a::Float64
    b::Float64
    c::Float64
    d::Float64
    fₜ::Float64
end

function Φ_to_c_indef(f, hp::HypParams)
    x = f / hp.fₜ
    return hp.d * f^(-(hp.a + hp.c)) * _₂F₁(1, hp.b, 1 + hp.b, -x^(-hp.a / hp.b))
end

function t_to_c_indef(f, hp::HypParams)
    x = f / hp.fₜ
    if hp.c == 0
        coeff = hp.a * hp.b * hp.d / (2 * π * (hp.a * (hp.b - 1) + hp.b))
        hyp_term = _₂F₁(
            1,
            1 - hp.b * (1 + hp.a) / hp.a,
            2 - hp.b * (1 + hp.a) / hp.a,
            -x^(hp.a / hp.b)
        )
        return coeff * f^(-(1 + hp.a)) * x^(hp.a / hp.b) * hyp_term
    else
        coeff = hp.d * f^(-(1 + hp.a + hp.c)) / (2 * π * (1 + hp.c) * (1 + hp.a + hp.c))
        term₁ = hp.a * (1 + hp.c) * (1 - _₂F₁(1, -hp.b*(1+hp.a+hp.c)/hp.a, 1-hp.b*(1+hp.a+hp.c)/hp.a, -x^(hp.a/hp.b)))
        term₂ = hp.c * (1 + hp.a + hp.c) * pFq([1, hp.b], [1+hp.b], -x^(-hp.a/hp.b))
        term₃ = -hp.a * hp.c * pFq([1, hp.b*(1+hp.a+hp.c)/hp.a], [(hp.a+hp.b+hp.a*hp.b+hp.b*hp.c)/hp.a], -x^(-hp.a/hp.b))
        return coeff * (term₁ + term₂ + term₃)
    end
end

Φ_to_c(f, f_c, hp::HypParams) = Φ_to_c_indef(f, hp) - Φ_to_c_indef(f_c, hp)
t_to_c(f, f_c, hp::HypParams) = t_to_c_indef(f, hp) - t_to_c_indef(f_c, hp)

function d²Φ_dt²(f, hp::HypParams)
    x = f / hp.fₜ
    if hp.c == 0
        return 4 * π^2 * f^(2 + hp.a) * (1 + x^(-hp.a / hp.b)) / (hp.a * hp.d)
    else
        return 4 * π^2 * f^(2 + hp.a + hp.c) / (
            hp.a * hp.d / (1 + x^(-hp.a / hp.b))
            + hp.c * hp.d * _₂F₁(1, hp.b, 1 + hp.b, -x^(-hp.a / hp.b))
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

# Model-independent code
# trapz for matrix-values functions
function trapz_1d_mat(xs, ys)
    @assert length(xs) == length(ys)
    result = zeros(size(ys[1]))
    for i in range(1, stop=length(xs) - 1)
        result .+= (xs[i + 1] - xs[i]) * (ys[i + 1] + ys[i]) ./ 2
    end
    return result
end

function fim_integrand_num(f, f_c, system::T) where T <: Binary
    ∂amp₊ = collect(values(gradient(s -> amp₊(f, s), system)[1]))
    ∂amp₊[∂amp₊ .=== nothing] .= 0.
    ∂amp₊ = convert(Array{Float64}, ∂amp₊)

    ∂Ψ = collect(values(gradient(s -> Ψ(f, f_c, s), system)[1]))
    ∂Ψ[∂Ψ .=== nothing] .= 0.
    ∂Ψ = convert(Array{Float64}, ∂Ψ)

    # Convert to log derivatives for intrinsic parameters
    if T == StaticDress
        rescaling = [system.γₛ, system.c_f, system.ℳ, 1., 1., 1.]
    elseif T == DynamicDress
        rescaling = [system.m₁, system.m₂, system.ρₛ, system.γₛ, 1., 1., 1.]
    end
    ∂amp₊ .*= rescaling
    ∂Ψ .*= rescaling
    
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
    Γ = (Γ .+ Γ') ./ 2

    # Improve stability
    scales = sqrt(inv(Diagonal(Γ)))
    Γr = scales * Γ * scales
    Σ = scales * inv(Γr) * scales
    # println([Σ[i, i] for i in 1:size(Σ)[1]])
    # println(eigvals(Γr))

    # Checks
    @assert all(eigvals(Γr) .> 0) eigvals(Γr)

    return sqrt.([Σ[i, i] for i in 1:size(Σ)[1]])
end

function sig_noise_ratio(fₗ, fₕ, system::T, n::Int=5000) where T <: Binary
    fs = geomspace(fₗ, fₕ, n)
    integrand(f) = amp₊(f, system)^2 / Sₙ_LISA(f)
    return √(4 * trapz(fs, integrand.(fs)))
end
# Model-independent code ends

# Static checks
fₗ = 2.26e-2  # frequency 5 years before merger (HACKY ESTIMATE)
fₕ = 4.39701  # ISCO frequency (HACKY ESTIMATE)
sd = make_static_dress(1e3, 1., 226., 7/3)
println(log10(fim_uncertainties(fₗ, fₕ, fₕ, sd, 100)[1]))
println(log10(fim_uncertainties(fₗ, fₕ, fₕ, sd, 100)[2]))
println(log10(fim_uncertainties(fₗ, fₕ, fₕ, sd, 100)[3]))
sig_noise_ratio(fₗ, fₕ, sd, 1000)  # agrees with David's notebook

# Dynamic checks
fₗ = 2.26e-2  # frequency 5 years before merger (HACKY ESTIMATE)
fₕ = 4.39701  # ISCO frequency (HACKY ESTIMATE)
dd = make_dynamic_dress(1e3, 1., 226., 7/3)
println(log10(fim_uncertainties(fₗ, fₕ, fₕ, dd, 100)[1]))
println(log10(fim_uncertainties(fₗ, fₕ, fₕ, dd, 100)[2]))
println(log10(fim_uncertainties(fₗ, fₕ, fₕ, dd, 100)[3]))
sig_noise_ratio(fₗ, fₕ, dd, 1000)  # agrees with David's notebook

# Other checks
function plot_snr(m₁, m₂)
    log10_ρₛs = -2:0.2:3
    γₛs = 1.9:0.02:7/3

    fₕ = f_c = f_isco(m₁ * MSun)
    t_c = 5.0 * yr_to_s  # scales

    function fn(log10_ρₛ, γₛ)
        system = make_static_dress(m₁, m₂, 10^log10_ρₛ, γₛ)
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

function plot_uncertainties(m₁, m₂, i)
    log10_ρₛs = -2:0.5:3
    γₛs = 2.25:0.03:2.4

    fₕ = f_c = f_isco(m₁ * MSun)
    t_c = 5.0 * yr_to_s  # scales

    function fn(log10_ρₛ, γₛ)
        system = make_static_dress(m₁, m₂, 10.0^log10_ρₛ, γₛ)
        fₗ = find_zero(f -> t_to_c(f, f_c, system) - t_c, (0.001 * f_c, f_c))
        return log10(fim_uncertainties(fₗ, fₕ, fₕ, system, 80)[i])
    end

    # heatmap(log10_ρₛs, γₛs, fn, fillcolor=cgrad(:inferno, rev=true), size=(550, 500))
    contour(log10_ρₛs, γₛs, fn, fill=true, fillcolor=cgrad(:inferno, rev=true), size=(550, 500))
    xaxis!(L"\log_{10} \rho_s", (minimum(log10_ρₛs), maximum(log10_ρₛs)))
    yaxis!(L"\gamma_s", (minimum(γₛs), maximum(γₛs)))
    title!("$(fieldnames(StaticDress)[i])")
end

function plot_Φ()
    m₁ = 1e3
    m₂ = 1.0
    γₛ = 7/3.
    ρₛ = 0.01
    sd = make_static_dress(m₁, m₂, ρₛ, γₛ)
    sd_alt = make_static_dress(m₁, m₂, ρₛ, 2.8)
    dd = make_dynamic_dress(m₁, m₂, ρₛ, γₛ)

    f_c = f_isco(m₁ * MSun)
    fs = geomspace(1e-4, f_c * 0.9999, 500)
    
    Φₛ(f) = Φ_to_c(f, f_c, sd)
    Φₛ_alt(f) = Φ_to_c(f, f_c, sd_alt)
    Φ(f) = Φ_to_c(f, f_c, dd)
    Φᵥ(f) = ψᵥ(dd) ./ f.^(5/3) .- ψᵥ(dd) / f_c^(5/3)

    plot(fs, Φᵥ.(fs), label="V")
    plot!(fs, Φᵥ.(fs) .- Φₛ.(fs), legend=:bottomleft, label="S")
    plot!(fs, Φᵥ.(fs) .- Φₛ_alt.(fs), legend=:bottomleft, label="S alt")
    plot!(fs, Φᵥ.(fs) .- Φ.(fs), legend=:bottomleft, label="D")

    xaxis!(L"f~\mathrm{[Hz]}", :log)
    yaxis!(L"\mathrm{Phase~[rad]}", :log)
    title!("Dephasing")
end

function plot_amp₊()
    m₁ = 1e3
    m₂ = 1.0
    γₛ = 1.9
    ρₛ = 100.0
    system = make_static_dress(m₁, m₂, ρₛ, γₛ)

    f_c = f_isco(m₁ * MSun)
    fs = geomspace(1e-4, f_c * 0.999, 1000)
    
    plot(fs, (f -> amp₊(f, system)).(fs) ./ fs.^(-7/6), legend=:bottomleft)
    xaxis!(L"f~\mathrm{[Hz]}", :log)
    yaxis!("Amplitude", :log)
    title!("Amplitude")
end

function plot_t_to_c()
    m₁ = 1e3
    m₂ = 1.0
    γₛ = 1.9
    ρₛ = 0.01
    system = make_static_dress(m₁, m₂, ρₛ, γₛ)

    f_c = f_isco(m₁ * MSun)
    fs = geomspace(1e-4, f_c * 0.999, 1000)
    
    t_to_cₛ(f) = t_to_c(f, f_c, system)
    t_to_c_indefᵥ(f) = 5 * ψᵥ(system) / (16 * π * f^(8/3))
    t_to_cᵥ(f) = t_to_c_indefᵥ(f) - t_to_c_indefᵥ(f_c)

    println(t_to_cᵥ.(fs) - t_to_cₛ.(fs))
    # plot(fs, t_to_cᵥ.(fs), label=L"t_V")
    plot(fs, t_to_cₛ.(fs) .* fs.^(8/3), label=L"t_S", legend=:bottomleft)
    xaxis!(L"f~\mathrm{[Hz]}", :log)
    # yaxis!(L"t~[\mathrm{s}]", :log)
    title!("Time to merge")
end