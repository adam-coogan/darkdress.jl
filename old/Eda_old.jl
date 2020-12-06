include("Utils.jl")
include("Systems.jl")
include("Detectors.jl")

using Zygote
using Revise
using Trapz
using Plots
using LaTeXStrings
using HypergeometricFunctions: _₂F₁
using LinearAlgebra: eigvals, cholesky, Diagonal
import Base.length, Base.collect, Base.values

using .Utils: geomspace
using .Detectors: Detector, LISA, Sₙ
using .Systems: Binary

"""
Use SI units for simplicity
"""
Gₙ = 6.67408e-11  # m^3 s^-2 kg^-1
c = 299792458.  # m/s
MSun = 1.98855e30  # kg
pc = 3.08567758149137e16 # m

function trapz_1d_mat(xs, ys)
    @assert length(xs) == length(ys)
    result = zeros(size(ys[1]))
    for i in range(1, stop=length(xs) - 1)
        result .+= (xs[i + 1] - xs[i]) * (ys[i + 1] + ys[i]) ./ 2
    end
    return result
end

struct EdaDress <: Binary
    ℳ::AbstractFloat  # kg
    𝒜::AbstractFloat  # units?
    α::AbstractFloat  # 1
    cₑ::AbstractFloat  # units?
    Φ_c::AbstractFloat
    t̃_c::AbstractFloat
end

values(system::EdaDress) = [system.ℳ, system.𝒜, system.α, system.cₑ, system.Φ_c, system.t̃_c]
collect(system::EdaDress) = values(system)
length(::EdaDress) = 6
ℳ(m₁, m₂) = (m₁ * m₂)^(3/5) / (m₁ + m₂)^(1/5)
# r_isco(system::Binary) = 6 * Gₙ * system.m₁ / c^2
# f_isco(system::Binary) = √(Gₙ * system.m₁ / r_isco(system)^3) / π

function Φ̃_indef(f′, f, system::EdaDress)
    coeff = 10/3 * (8 * π * Gₙ * system.ℳ / c^3)^(-5/3)
    term_1 = -3 * f * f′^(1 - 2 * system.α / 3) * Gₙ^(-11/6 + system.α/3) * π^(11/3 - 2 * system.α/3) * _₂F₁(
        1,
        (3 - 2 * system.α) / (11 - 2 * system.α),
        (14 - 4 * system.α) / (11 - 2 * system.α),
        -f′^((11 - 2*system.α) / 3) * Gₙ^(-11/6+system.α/3) * π^(11/3-2*system.α/3) / (4 * system.cₑ)
    ) / (4 * system.cₑ * (3 - 2 * system.α))
    term_2 = -3 * f′^(2-2*system.α/3) * Gₙ^(-11/6+system.α/3) * π^(11/3-2*system.α/3) * _₂F₁(
        1,
        (6 - 2 * system.α) / (11 - 2 * system.α),
        (17 - 4 * system.α) / (11 - 2 * system.α),
        -f′^((11 - 2 * system.α) / 3) * Gₙ^(-11/6+system.α/3) * π^(11/3 - 2 * system.α/3) / (4 * system.cₑ)
    ) / (8 * system.cₑ * (system.α - 3))
    return coeff * (term_1 + term_2)
end

Φ̃(f, system::EdaDress) = Φ̃_indef(f, f, system) - Φ̃_indef(5, f, system)  # Φ̃_indef(f_isco(system), f, system)
δ̃(f, system::EdaDress) = (Gₙ / (π * f)^2)^(1. - system.α / 3)  # units
L(f, system::EdaDress) = 1. + 4 * system.cₑ * δ̃(f, system)^((11 - 2 * system.α) / (6 - 2 * system.α))  # units!
amp₊(f, system::EdaDress) = system.𝒜 * f^(-7/6) * L(f, system)^(-1/2)
Ψ(f, system::EdaDress) = 2 * π * f * system.t̃_c -system.Φ_c - π/4 - Φ̃(f, system)

function fim_integrand_num(f, system::EdaDress)
    ∂amp₊ = collect(values(gradient(s -> amp₊(f, s), system)[1]))
    ∂amp₊[∂amp₊ .=== nothing] .= 0.
    ∂amp₊ = convert(Array{Float64}, ∂amp₊)

    ∂Ψ = collect(values(gradient(s -> Ψ(f, s), system)[1]))
    ∂Ψ[∂Ψ .=== nothing] .= 0.
    ∂Ψ = convert(Array{Float64}, ∂Ψ)

    # # Convert to log derivatives for intrinsic parameters
    # rescaling = [1., 1., system.ℳ, system.𝒜, system.α, system.cₑ]
    # ∂amp₊ .*= rescaling
    # ∂Ψ .*= rescaling
    
    return 4 * (∂amp₊ * ∂amp₊' + amp₊(f, system)^2 * ∂Ψ * ∂Ψ')
end

# Specialize to LISA
Sₙ_LISA(f) = 1 / f^14 * 1.80654e-17 * (0.000606151 + f^2) * (3.6864e-76 + 3.6e-37 * f^8 + 2.25e-30 * f^10 + 8.66941e-21 * f^14)

function fim(fₗ, fₕ, system::EdaDress, n)
    fs = geomspace(fₗ, fₕ, n)
    nums = map(f -> fim_integrand_num(f, system), fs)
    Sₙs = Sₙ_LISA.(fs)
    return trapz_1d_mat(fs, nums ./ Sₙs)
end

function fim_uncertainties(fₗ, fₕ, system::EdaDress, n=1000)
    Γ = fim(fₗ, fₕ, system, n)
    Γ = (Γ .+ Γ') ./ 2

    # Improve stability
    scales = sqrt(inv(Diagonal(Γ)))
    Γr = scales * Γ * scales
    Σ = scales * inv(Γr) * scales

    # Checks
    # println("evs: ", eigvals(Γr))
    @assert all(eigvals(Γr) .> 0) eigvals(Γr)

    return sqrt.([Σ[i, i] for i in 1:size(Σ)[1]])
end

function sig_noise_ratio(fₗ, fₕ, system::EdaDress, n::Int=5000)
    fs = geomspace(fₗ, fₕ, n)
    amp²(f) = amp₊(f, system)^2
    amp²s = map(amp², fs)
    Sₙs = Sₙ_LISA.(fs)
    return sqrt.(4 * trapz(fs, amp²s ./ Sₙs))
end

"""
Builds an EdaDress. Units: [m₁] = [m₂] = M⊙, [ρₛ] = M⊙ / pc^3, [dL] = pc.
"""
function dd_factory(ρₛ, γₛ, m₁=1e3, m₂=1.0, dL=1e8, ι=0.)
    m₁ *= MSun
    m₂ *=  MSun
    ρₛ *= MSun / pc^3
    dL *= pc
    ι = 0.
    Φ_c = 0.
    t̃_c = 0. + dL / c

    # Jump through Eda+'s hoops
    𝒜 = √(5/24) / π^(2/3) * c / dL * (Gₙ * ℳ(m₁, m₂) / c^3)^(5/6) * (1 + cos(ι)^2) / 2
    rₛ = 0.54 * pc
    r_min = 6 * Gₙ * m₁ / c^2
    m_dm = 4 * π * rₛ^γₛ * ρₛ * r_min^(3 - γₛ) / (3 - γₛ)
    M_eff = m₁ - m_dm
    F = r_min^(γₛ - 3) * Gₙ * m_dm
    ϵ = F / (Gₙ * M_eff)
    Λ = exp(3)
    c_df = 8 * π * Gₙ^2 * m₂ * ρₛ * rₛ^γₛ * log(Λ) * (Gₙ * M_eff)^(-3/2) * ϵ^((2 * γₛ - 3) / (6 - 2 * γₛ))
    c_gw = 256 / 5 * (Gₙ * m₂ / c^3) * (Gₙ * M_eff / c)^2 * ϵ^(4 / (3 - γₛ))
    c̃ = c_df / c_gw
    cₑ = M_eff^((11 - 2 * γₛ) / 6) * c̃ * ϵ^((11 - 2 * γₛ) / (6 - 2 * γₛ))

    return EdaDress(ℳ(m₁, m₂), 𝒜, γₛ, cₑ, Φ_c, t̃_c)
end

fₗ = 2.26e-2  # frequency 5 years before merger (HACKY ESTIMATE)
fₕ = 4.39701  # ISCO frequency (HACKY ESTIMATE)
system = dd_factory(226., 7/3)
println("log10(Δℳ / ℳ) = ", log10.(fim_uncertainties(fₗ, fₕ, system, 100) ./ values(system))[1])
println("log10(Δγₛ / γₛ) = ", log10.(fim_uncertainties(fₗ, fₕ, system, 100) ./ values(system))[3])
println(log10.(fim_uncertainties(fₗ, fₕ, system, 10) ./ values(system))[1:end-2])
sig_noise_ratio(fₗ, fₕ, system, 1000)  # agrees with David's notebook

function plot_snr()
    log10_ρₛs = -2:0.2:3
    γₛs = 1.9:0.02:7/3

    fₗ = 2.26e-2  # frequency 5 years before merger (HACKY ESTIMATE)
    fₕ = 4.39701  # ISCO frequency (HACKY ESTIMATE)

    heatmap(
        log10_ρₛs,
        γₛs,
        (log10_ρₛ, γₛ) -> sig_noise_ratio(fₗ, fₕ, dd_factory(10^log10_ρₛ, γₛ)),
        clim=(8.7, 9.2)
    )
    xaxis!(L"\log_{10} \rho_s")
    yaxis!(L"\gamma_s")
end

function plot_Δγₛ()
    log10_ρₛs = -2:0.5:3
    γₛs = 1.9:0.05:7/3

    fₗ = 2.26e-2  # frequency 5 years before merger (HACKY ESTIMATE)
    fₕ = 4.39701  # ISCO frequency (HACKY ESTIMATE)

    get_Δγₛ(log10_ρₛ, γₛ) = log10(fim_uncertainties(fₗ, fₕ, dd_factory(10.0^log10_ρₛ, γₛ), 1000)[3] / γₛ)

    heatmap(log10_ρₛs, γₛs, get_Δγₛ)#, clim=(8.7, 9.2))
    xaxis!(L"\log_{10} \rho_s")
    yaxis!(L"\gamma_s")
end