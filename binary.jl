import Base: length
using HypergeometricFunctions: _₂F₁
using SpecialFunctions: beta_inc
using Roots

# Basic functions
ℳ(m₁, m₂) = (m₁ * m₂)^(3/5) / (m₁ + m₂)^(1/5)
m₁(ℳ, q) = (1 + q)^(1/5) / q^(3/5) * ℳ
m₂(ℳ, q) = (1 + q)^(1/5) * q^(2/5) * ℳ
r_isco(m₁) = 6 * Gₙ * m₁ / c^2
f_isco(m₁) = √(Gₙ * m₁ / r_isco(m₁)^3) / π
rₛ(m₁, ρₛ, γₛ) = ((3 - γₛ) * 0.2^(3 - γₛ) * m₁ / (2 * π * ρₛ))^(1/3)
ψᵥ(m₁, m₂) = 1/16 * (c^3 / (π * Gₙ * ℳ(m₁, m₂)))^(5/3)
ξ(γₛ) = beta_inc(γₛ - 1/2, 3/2, 1/2)[2]

function c_f(m₁, m₂, ρₛ, γₛ)
    Λ = √(m₁ / m₂)
    M = m₁ + m₂
    c_gw = 64 * Gₙ^3 * M * m₁ * m₂ / (5 * c^5)
    c_df = 8 * π * √(Gₙ) * m₂ * log(Λ) * ρₛ * rₛ(m₁, ρₛ, γₛ)^γₛ * ξ(γₛ) / (√(M) * m₁)
    return c_df / c_gw * (Gₙ * M / π^2)^((11 - 2 * γₛ) / 6)
end

"""
Dynamic dress break frequency. Fit using HaloFeedback data from 5 years before
merger.
"""
function f_b(m₁, m₂, γₛ)
    β  = 0.8162599280541165
    α₁ = 1.441237217113085
    α₂ = 0.4511442198433961
    ρ  = -0.49709119294335674
    γᵣ = 1.4395688575650551

    return β * (m₁ / (1e3 * MSun))^(-α₁) * (m₂ / MSun)^α₂ * (1 + ρ * log(γₛ / γᵣ))
end

# Different density profile parametrization
function ρ₁_to_ρₛ(m₁, ρ₁, γₛ, r_ref)
    m̃₁ = (3 - γₛ) * 0.2^(3 - γₛ) * m₁ / (2 * π)
    return (ρ₁ * r_ref^γₛ / m̃₁^(γₛ / 3))^(1 / (1 - γₛ / 3))
end

# Parent type for all binaries
abstract type Binary{T <: Real} end

# Can be represented in the hypergeometric parametrization
abstract type HypableDress{T <: Real} <: Binary{T} end

"""
Vacuum binary
"""
struct VacuumBinary{T} <: Binary{T}
    ℳ::T
    Φ_c::T
    t̃_c::T
    dₗ_ι::T
end

"""
Static dress
"""
struct StaticDress{T} <: HypableDress{T}
    γₛ::T
    c_f::T
    ℳ::T
    Φ_c::T
    t̃_c::T
    dₗ_ι::T
end

"""
Dynamic dress
"""
struct DynamicDress{T} <: HypableDress{T}
    γₛ::T
    c_f::T
    ℳ::T
    q::T
    Φ_c::T
    t̃_c::T
    dₗ_ι::T
end

"""
Hypergeometric waveform parametrization
"""
struct HypParams{T <: Real}
    ψᵥ::T
    ϑ::T
    λ::T
    η::T
    fₜ::T
end

"""
General functions
"""
ψᵥ(system::B) where B <: Binary = 1/16 * (c^3 / (π * Gₙ * ℳ(system)))^(5/3)
length(::B) where B <: Binary = length(fieldnames(B))

# Intrinsic binary parameters
intrinsics(::Type{StaticDress}) = (:γₛ, :c_f, :ℳ)
intrinsics(::Type{DynamicDress}) = (:γₛ, :c_f, :ℳ, :q)

# Frequency at time t [s] before coalescence at frequency f_c [Hz], [Hz].
function f_of_t_to_c(t, f_c, system::B) where B <: Binary
    return find_zero(f -> t_to_c(f, f_c, system) - t, (0.0001 * f_c, f_c))
end

"""
Phase functions. Every `Binary` type must implement `Φ_to_c_indef`,
`t_to_c_indef` and `d²Φ_dt²`.
"""
Φ_to_c(f, f_c, system::B) where B <: Binary = Φ_to_c_indef(f, system) - Φ_to_c_indef(f_c, system)
t_to_c(f, f_c, system::B) where B <: Binary = t_to_c_indef(f, system) - t_to_c_indef(f_c, system)

# If possible, convert system to HypParams
Φ_to_c_indef(f, system::B) where B <: HypableDress{T} where T <: Real = Φ_to_c_indef(f, convert(HypParams{T}, system))
t_to_c_indef(f, system::B) where B <: HypableDress{T} where T <: Real = t_to_c_indef(f, convert(HypParams{T}, system))
d²Φ_dt²(f, system::B) where B <: HypableDress{T} where T <: Real = d²Φ_dt²(f, convert(HypParams{T}, system))

# Phase of Fourier transform of waveform in stationary phase approximation
Φ̃(f, f_c, system::B) where B <: Binary = 2 * π * f * t_to_c(f, f_c, system) - Φ_to_c(f, f_c, system)
Ψ(f, f_c, system::B) where B <: Binary = 2 * π * f * system.t̃_c - system.Φ_c - π/4 - Φ̃(f, f_c, system)

# Amplitude functions
function amp₊(f, system::B) where B <: Binary
    h₀ = 1/2 * 4 * π^(2/3) * (Gₙ * ℳ(system))^(5/3) * f^(2/3) / c^4 * √(2 * π / d²Φ_dt²(f, system))
    return h₀ * exp(system.dₗ_ι)
end

"""
Getters
"""
ℳ(system::B) where B <: Binary = system.ℳ
q(dd::DynamicDress) = dd.q
m₁(dd::DynamicDress) = m₁(dd.ℳ, dd.q)
m₂(dd::DynamicDress) = m₂(dd.ℳ, dd.q)

"""
Phase functions
"""
Φ_to_c_indef(f, vb::VacuumBinary) = ψᵥ(vb) / f^(5/3)
t_to_c_indef(f, vb::VacuumBinary) = 5 * ψᵥ(vb) / (16 * π * f^(8/3))
d²Φ_dt²(f, system::VacuumBinary) = 12 * π^2 * f^(11/3) / (5 * ψᵥ(system))

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

function d²Φ_dt²(f, hp::HypParams)
    x = f / hp.fₜ
    if hp.λ == 0 && hp.η == 1
        return 12 * π^2 * f^(11/3) * (1 + x^(-5 / (3 * hp.ϑ))) / (5 * hp.ψᵥ)
    else
        return 12 * π^2 * f^(11/3) * x^hp.λ * (1 + x^(5 / (3 * hp.ϑ))) / (
            hp.ψᵥ * (
                5 * x^hp.λ - 5 * hp.η - 3 * hp.η * hp.λ + x^(5 / (3 * hp.ϑ)) * (5 * x^hp.λ - 3 * hp.η * hp.λ)
                + 3 * (1 + x^(5 / (3 * hp.ϑ))) * hp.η * hp.λ * _₂F₁(1, hp.ϑ, 1 + hp.ϑ, -x^(-5 / (3 * hp.ϑ)))
            )
        )
    end
end

"""
Factories
"""
function make_vacuum_binary(m₁, m₂, dₗ=1e8*pc, ι=0.0, Φ_c=0.0, t_c=0.0)
    @assert m₁ > m₂
    @assert m₂ > 0
    t̃_c = t_c + dₗ / c
    dₗ_ι = log((1 + cos(ι)^2) / (2 * dₗ))
    return VacuumBinary(ℳ(m₁, m₂), Φ_c, t̃_c, dₗ_ι)
end

function make_dress(::Type{StaticDress}, m₁, m₂, ρₛ, γₛ, dₗ=1e8*pc, ι=0.0, Φ_c=0.0, t_c=0.0)
    @assert m₁ > m₂
    @assert m₂ > 0
    t̃_c = t_c + dₗ / c
    dₗ_ι = log((1 + cos(ι)^2) / (2 * dₗ))
    return StaticDress(γₛ, c_f(m₁, m₂, ρₛ, γₛ), ℳ(m₁, m₂), Φ_c, t̃_c, dₗ_ι)
end

function make_dress(::Type{DynamicDress}, m₁, m₂, ρₛ, γₛ, dₗ=1e8*pc, ι=0.0, Φ_c=0.0, t_c=0.0)
    sd = make_dress(StaticDress, m₁, m₂, ρₛ, γₛ, dₗ, ι, Φ_c, t_c)
    return DynamicDress(sd.γₛ, sd.c_f, ℳ(sd), m₂ / m₁, sd.Φ_c, sd.t̃_c, sd.dₗ_ι)
end

"""
Converters
"""
# Binary -> Array
Base.convert(::Type{Array{T,1}}, vb::VacuumBinary{T}) where T <: Real = [vb.ℳ, vb.Φ_c, vb.t̃_c, vb.dₗ_ι]
Base.convert(::Type{Array{T,1}}, sd::StaticDress{T}) where T <: Real = [sd.γₛ, sd.c_f, sd.ℳ, sd.Φ_c, sd.t̃_c, sd.dₗ_ι]
Base.convert(::Type{Array{T,1}}, dd::DynamicDress{T}) where T <: Real = [dd.γₛ, dd.c_f, dd.ℳ, dd.q, dd.Φ_c, dd.t̃_c, dd.dₗ_ι]
Base.convert(::Type{Array{T,1}}, hp::HypParams{T}) where T <: Real = [hp.ψᵥ, hp.ϑ, hp.λ, hp.η, hp.fₜ]

# DynamicDress -> StaticDress
Base.convert(
    ::Type{StaticDress{T}}, dd::DynamicDress{T}
) where T <: Real = StaticDress{T}(dd.γₛ, dd.c_f, ℳ(dd), dd.Φ_c, dd.t̃_c, dd.dₗ_ι)

# StaticDress -> HypParams
function Base.convert(::Type{HypParams{T}}, sd::StaticDress{T}) where T <: Real
    ϑ = 5 / (11 - 2 * sd.γₛ)
    fₜ = sd.c_f^(3 / (11 - 2 * sd.γₛ))
    return HypParams{T}(ψᵥ(sd), ϑ, 0., 1., fₜ)
end

# DynamicDress -> HypParams
function Base.convert(::Type{HypParams{T}}, dd::DynamicDress{T}) where T <: Real
    f_eq = convert(HypParams{T}, convert(StaticDress{T}, dd)).fₜ  # StaticDress -> HypParams
    # Compute new parameters
    fₜ = f_b(m₁(dd), m₂(dd), dd.γₛ)
    γₑ = 5/2  # TODO: CHANGE!
    ϑ = 5 / (2 * γₑ)
    λ = (11 - 2 * (dd.γₛ + γₑ)) / 3
    η = (5 + 2 * γₑ) / (2 * (8 - dd.γₛ)) * (f_eq / fₜ)^((11 - 2 * dd.γₛ) / 3)
    return HypParams{T}(ψᵥ(dd), ϑ, λ, η, fₜ)
end