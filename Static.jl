include("Utils.jl")

using Zygote
using Revise
using Trapz
using Roots
using Plots
using LaTeXStrings
using HypergeometricFunctions: _₂F₁
using LinearAlgebra: eigvals, cholesky, Diagonal
import Base.length, Base.collect, Base.values

using .Utils: geomspace

# SI units
Gₙ = 6.67408e-11  # m^3 s^-2 kg^-1
c = 299792458.  # m/s
MSun = 1.98855e30  # kg
pc = 3.08567758149137e16 # m

ℳ(m₁, m₂) = (m₁ * m₂)^(3/5) / (m₁ + m₂)^(1/5)
r_isco(m₁) = 6 * Gₙ * m₁ / c^2
f_isco(m₁) = √(Gₙ * m₁ / r_isco(m₁)^3) / π

# trapz for matrix-values functions
function trapz_1d_mat(xs, ys)
    @assert length(xs) == length(ys)
    result = zeros(size(ys[1]))
    for i in range(1, stop=length(xs) - 1)
        result .+= (xs[i + 1] - xs[i]) * (ys[i + 1] + ys[i]) ./ 2
    end
    return result
end

abstract type Binary end

struct StaticDress <: Binary
    m₁::Float64
    m₂::Float64
    ρₛ::Float64
    γₛ::Float64
    Φ_c::Float64
    t̃_c::Float64
    dₗ_ι::Float64
end

"""
Builds a StaticDress. Units: [m₁] = [m₂] = M⊙, [ρₛ] = M⊙ / pc^3, [dL] = pc.
"""
function make_StaticDress(ρₛ, γₛ, m₁=1e3, m₂=1.0, dₗ=1e8, ι=0.)
    m₁ *= MSun
    m₂ *=  MSun
    ρₛ *= MSun / pc^3
    dₗ *= pc
    Φ_c = 0.
    t̃_c = 0. + dₗ / c
    return StaticDress(m₁, m₂, ρₛ, γₛ, Φ_c, t̃_c, log((1 + cos(ι)^2) / (2*dₗ)))
end

values(system::Binary) = (fn -> getfield(system, fn)).(fieldnames(StaticDress))
collect(system::Binary) = values(system)
length(system::Binary) = length(values(system))

ℳ(system::StaticDress) = ℳ(system.m₁, system.m₂)
r_isco(system::StaticDress) = r_isco(system.m₁)
f_isco(system::StaticDress) = f_isco(system.m₁)
ψᵥ(system::StaticDress) = 1/16 * (c^3 / (π * Gₙ * ℳ(system)))^(5/3)
rₛ(system::StaticDress) = ((3 - system.γₛ) * 0.2^(3 - system.γₛ) * system.m₁ / (2 * π * system.ρₛ))^(1/3)
Λ(system::StaticDress) = 3.0
ξ(system::StaticDress) = 0.58
M(system::StaticDress) = system.m₁ + system.m₂
c_gw(system::StaticDress) = 64 * Gₙ^3 * M(system) * system.m₁ * system.m₂ / (5 * c^5)
c_df(system::StaticDress) = 8 * π * √(Gₙ) * system.m₂ * log(Λ(system)) * system.ρₛ * rₛ(system)^system.γₛ * ξ(system) / (√(M(system)) * system.m₁)
c_f(system::StaticDress) = c_df(system) / c_gw(system) * (Gₙ * M(system) / π^2)^((11 - 2*system.γₛ) / 6)

# Computing the phase
function Φ_indef(f, system::StaticDress)
    b = 5 / (11 - 2 * system.γₛ)
    z = -c_f(system) * f^((2 * system.γₛ - 11) / 3)
    return ψᵥ(system) / f^(5/3) * _₂F₁(1, b, 1 + b, z)
end

function t_to_c_indef(f, system::StaticDress)
    b = 8 / (11 - 2 * system.γₛ)
    z = -c_f(system) * f^((2 * system.γₛ - 11) / 3)
    return 5 * ψᵥ(system) / (16 * π * f^(8/3)) * _₂F₁(1, b, 1 + b, z)
end

Φ_to_c(f, f_c, system::StaticDress) = Φ_indef(f, system) - Φ_indef(f_c, system)
t_to_c(f, f_c, system::StaticDress) = t_to_c_indef(f, system) - t_to_c_indef(f_c, system)
Φ̃(f, f_c, system::Binary) = 2 * π * f * t_to_c(f, f_c, system) - Φ_to_c(f, f_c, system)
Ψ(f, f_c, system::Binary) = 2 * π * f * system.t̃_c - system.Φ_c - π/4 - Φ̃(f, f_c, system)

function amp₊(f, system::StaticDress)
    d²Φ_dt² = 12 * π^2 * (f^(11/3) + c_f(system) * f^(2 * system.γₛ / 3)) / (5 * ψᵥ(system))
    h₀ = 1/2 * 4 * π^(2/3) * (Gₙ * ℳ(system))^(5/3) * f^(2/3) / c^4 * √(2 * π / d²Φ_dt²)
    return h₀ * exp(system.dₗ_ι)
end

# Model-independent code begins
function fim_integrand_num(f, f_c, system::Binary)
    ∂amp₊ = collect(values(gradient(s -> amp₊(f, s), system)[1]))
    ∂amp₊[∂amp₊ .=== nothing] .= 0.
    ∂amp₊ = convert(Array{Float64}, ∂amp₊)

    ∂Ψ = collect(values(gradient(s -> Ψ(f, f_c, s), system)[1]))
    ∂Ψ[∂Ψ .=== nothing] .= 0.
    ∂Ψ = convert(Array{Float64}, ∂Ψ)

    # Convert to log derivatives for intrinsic parameters
    rescaling = [system.m₁, system.m₂, system.ρₛ, system.γₛ, 1., 1., 1.]
    ∂amp₊ .*= rescaling
    ∂Ψ .*= rescaling
    
    return 4 * (∂amp₊ * ∂amp₊' + amp₊(f, system)^2 * ∂Ψ * ∂Ψ')
end

# Specialize to LISA
Sₙ_LISA(f) = 1 / f^14 * 1.80654e-17 * (0.000606151 + f^2) * (3.6864e-76 + 3.6e-37 * f^8 + 2.25e-30 * f^10 + 8.66941e-21 * f^14)

# From here on, I assume fₕ = f_c
function fim(fₗ, fₕ, f_c, system::Binary, n)
    fs = geomspace(fₗ, fₕ, n)
    integrand(f) =  fim_integrand_num(f, f_c, system) / Sₙ_LISA(f)
    return trapz_1d_mat(fs, integrand.(fs))
end

function fim_uncertainties(fₗ, fₕ, f_c, system::Binary, n=1000)
    Γ = fim(fₗ, fₕ, f_c, system, n)
    Γ = (Γ .+ Γ') ./ 2

    # Improve stability
    scales = sqrt(inv(Diagonal(Γ)))
    Γr = scales * Γ * scales
    Σ = scales * inv(Γr) * scales

    # Checks
    @assert all(eigvals(Γr) .> 0) eigvals(Γr)

    return sqrt.([Σ[i, i] for i in 1:size(Σ)[1]])
end

function sig_noise_ratio(fₗ, fₕ, system::Binary, n::Int=5000)
    fs = geomspace(fₗ, fₕ, n)
    integrand(f) = amp₊(f, system)^2 / Sₙ_LISA(f)
    return √(4 * trapz(fs, integrand.(fs)))
end
# Model-independent code ends

fₗ = 2.26e-2  # frequency 5 years before merger (HACKY ESTIMATE)
fₕ = 4.39701  # ISCO frequency (HACKY ESTIMATE)
system = make_StaticDress(226., 7/3)
println(log10(fim_uncertainties(fₗ, fₕ, fₕ, system, 100)[1]))
println(log10(fim_uncertainties(fₗ, fₕ, fₕ, system, 100)[2]))
println(log10(fim_uncertainties(fₗ, fₕ, fₕ, system, 100)[3]))
sig_noise_ratio(fₗ, fₕ, system, 1000)  # agrees with David's notebook

function plot_snr()
    log10_ρₛs = -2:0.2:3
    γₛs = 1.9:0.02:7/3

    fₕ = f_c = f_isco(1e3 * MSun)
    t_c = 5.0 * 365 * 24 * 3600  # scales
    fₗ = find_zero(f -> t_to_c(f, f_c, system) - t_c, (0.001 * f_c, f_c))

    heatmap(
        log10_ρₛs,
        γₛs,
        (log10_ρₛ, γₛ) -> sig_noise_ratio(fₗ, fₕ, f_c, make_EdaDress(10^log10_ρₛ, γₛ)),
        clim=(8.5, 9.1),
        size=(550, 500)
    )
    xaxis!(L"\log_{10} \rho_s")
    yaxis!(L"\gamma_s")
end

function plot_uncertainties(i)
    log10_ρₛs = -2:0.3:3
    γₛs = 1.9:0.03:7/3

    fₕ = f_c = f_isco(1e3 * MSun)
    t_c = 5.0 * 365 * 24 * 3600  # scales
    fₗ = find_zero(f -> t_to_c(f, f_c, system) - t_c, (0.001 * f_c, f_c))

    fn(log10_ρₛ, γₛ) = log10(fim_uncertainties(fₗ, fₕ, fₕ, make_EdaDress(10.0^log10_ρₛ, γₛ), 40)[i])

    # heatmap(log10_ρₛs, γₛs, fn, fillcolor=cgrad(:inferno, rev=true), size=(550, 500))
    contour(log10_ρₛs, γₛs, fn, fill=true, fillcolor=cgrad(:inferno, rev=true), size=(550, 500))
    xaxis!(L"\log_{10} \rho_s", (minimum(log10_ρₛs), maximum(log10_ρₛs)))
    yaxis!(L"\gamma_s", (minimum(γₛs), maximum(γₛs)))
    title!("$(fieldnames(EdaDress)[i])")
end