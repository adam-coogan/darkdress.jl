import Base: values, collect, length
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

# Different density profile parametrization
function ρ₁_to_ρₛ(m₁, ρ₁, γₛ, r_ref)
    m̃₁ = (3 - γₛ) * 0.2^(3 - γₛ) * m₁ / (2 * π)
    return (ρ₁ * r_ref^γₛ / m̃₁^(γₛ / 3))^(1 / (1 - γₛ / 3))
end

# Parent type for all binaries
abstract type Binary end

# Can be represented in the hypergeometric parametrization
abstract type HypableDress <: Binary end

# General functions
ψᵥ(system::T) where T <: Binary = 1/16 * (c^3 / (π * Gₙ * ℳ(system)))^(5/3)
values(system::T) where T <: Binary = (fn -> getfield(system, fn)).(fieldnames(T))
collect(system::T) where T <: Binary = values(system)
length(system::T) where T <: Binary = length(values(system))

"""
Frequency at time t [s] before coalescence at frequency f_c [Hz], [Hz].
"""
function f_of_t_to_c(t, f_c, system::T) where T <: Binary
    return find_zero(f -> t_to_c(f, f_c, system) - t, (0.001 * f_c, f_c))
end

# Phase functions
Φ̃(f, f_c, system::T) where T <: Binary = 2 * π * f * t_to_c(f, f_c, system) - Φ_to_c(f, f_c, system)
Ψ(f, f_c, system::T) where T <: Binary = 2 * π * f * system.t̃_c - system.Φ_c - π/4 - Φ̃(f, f_c, system)

# Amplitude functions
function amp₊(f, system::T) where T <: Binary
    h₀ = 1/2 * 4 * π^(2/3) * (Gₙ * ℳ(system))^(5/3) * f^(2/3) / c^4 * √(2 * π / d²Φ_dt²(f, system))
    return h₀ * exp(system.dₗ_ι)
end

"""
Vacuum binary
"""
# mutable
struct VacuumBinary <: Binary
    ℳ
    Φ_c
    t̃_c
    dₗ_ι
end

# Getters
ℳ(vb::VacuumBinary) = vb.ℳ

# Factory
function make_vacuum_binary(m₁, m₂, dₗ=1e8*pc, ι=0.0, Φ_c=0.0, t_c=0.0)
    @assert m₁ > m₂
    @assert m₂ > 0
    t̃_c = t_c + dₗ / c
    dₗ_ι = log((1 + cos(ι)^2) / (2 * dₗ))
    return VacuumBinary(ℳ(m₁, m₂), Φ_c, t̃_c, dₗ_ι)
end

# Waveform functions
Φ_to_c_indef(f, vb::VacuumBinary) = ψᵥ(vb) / f^(5/3)
Φ_to_c(f, f_c, vb::VacuumBinary) = Φ_to_c_indef(f, vb) - Φ_to_c_indef(f_c, vb)

t_to_c_indef(f, vb::VacuumBinary) = 5 * ψᵥ(vb) / (16 * π * f^(8/3))
t_to_c(f, f_c, vb::VacuumBinary) = t_to_c_indef(f, vb) - t_to_c_indef(f_c, vb)

d²Φ_dt²(f, vb::VacuumBinary) = 12 * π^2 * f^(11/3) / (5 * ψᵥ(vb))

"""
Static dress
"""
# mutable
struct StaticDress <: HypableDress
    γₛ
    c_f
    ℳ
    Φ_c
    t̃_c
    dₗ_ι
end

# Getters
# c_f(sd::StaticDress) = sd.c_f
ℳ(sd::StaticDress) = sd.ℳ
q(sd::StaticDress) = sd.q

function c_f(m₁, m₂, ρₛ, γₛ)
    Λ = √(m₁ / m₂)
    M = m₁ + m₂
    c_gw = 64 * Gₙ^3 * M * m₁ * m₂ / (5 * c^5)
    c_df = 8 * π * √(Gₙ) * m₂ * log(Λ) * ρₛ * rₛ(m₁, ρₛ, γₛ)^γₛ * ξ(γₛ) / (√(M) * m₁)
    return c_df / c_gw * (Gₙ * M / π^2)^((11 - 2 * γₛ) / 6)
end

# StaticDress factories
function make_dress(::Type{StaticDress}, m₁, m₂, ρₛ, γₛ, dₗ=1e8*pc, ι=0.0, Φ_c=0.0, t_c=0.0)
    @assert m₁ > m₂
    @assert m₂ > 0
    t̃_c = t_c + dₗ / c
    dₗ_ι = log((1 + cos(ι)^2) / (2 * dₗ))
    return StaticDress(γₛ, c_f(m₁, m₂, ρₛ, γₛ), ℳ(m₁, m₂), Φ_c, t̃_c, dₗ_ι)
end

function make_dress(::Type{StaticDress}, values::AbstractArray)
    @assert size(values) == size(StaticDress)
    return StaticDress(values...)
end

"""
Dynamic dress
"""
# mutable
struct DynamicDress <: HypableDress
    γₛ
    c_f
    ℳ
    q
    Φ_c
    t̃_c
    dₗ_ι
end

# TODO: rederive with new waveform model!
function f_b(m₁, m₂, γₛ)
    α₁ = 1.39191077
    α₂ = 0.443089063
    β = 7.03764788e3
    κ = -2.44956668
    δ = 0.839699908
    return β * (m₂ / MSun)^α₂ / (m₁ / MSun)^α₁ * (1 + κ * log10(γₛ) + δ)
end

# Getters
# c_f(dd::DynamicDress) = dd.c_f
ℳ(dd::DynamicDress) = dd.ℳ
q(dd::DynamicDress) = dd.q
m₁(dd::DynamicDress) = m₁(dd.ℳ, dd.q)
m₂(dd::DynamicDress) = m₂(dd.ℳ, dd.q)

# Converters
# DynamicDress -> StaticDress
Base.convert(::Type{StaticDress}, dd::DynamicDress) = StaticDress(dd.γₛ, dd.c_f, ℳ(dd), dd.Φ_c, dd.t̃_c, dd.dₗ_ι)

# DynamicDress factory
function make_dress(::Type{DynamicDress}, m₁, m₂, ρₛ, γₛ, dₗ=1e8*pc, ι=0.0, Φ_c=0.0, t_c=0.0)
    sd = make_dress(StaticDress, m₁, m₂, ρₛ, γₛ, dₗ, ι, Φ_c, t_c)
    DynamicDress(sd.γₛ, sd.c_f, ℳ(sd), m₂ / m₁, sd.Φ_c, sd.t̃_c, sd.dₗ_ι)
end

# Intrinsic binary parameters
intrinsics(::Type{StaticDress}) = (:γₛ, :c_f, :ℳ)
intrinsics(::Type{DynamicDress}) = (:γₛ, :c_f, :ℳ, :q)

"""
Hypergeometric waveform parametrization
"""
# mutable
struct HypParams
    ψᵥ
    ϑ
    λ
    η
    fₜ
end

# StaticDress -> HypParams
function Base.convert(::Type{HypParams}, sd::StaticDress)
    ϑ = 5 / (11 - 2 * sd.γₛ)
    fₜ = sd.c_f^(3 / (11 - 2 * sd.γₛ))
    return HypParams(ψᵥ(sd), ϑ, 0., 1., fₜ)
end

# DynamicDress -> HypParams
function Base.convert(::Type{HypParams}, dd::DynamicDress)
    f_eq = convert(HypParams, convert(StaticDress, dd)).fₜ  # StaticDress -> HypParams
    # Compute new parameters
    fₜ = f_b(m₁(dd), m₂(dd), dd.γₛ)
    γₑ = 5/2  # TODO: CHANGE!
    ϑ = 5 / (2 * γₑ)
    λ = (11 - 2 * (dd.γₛ + γₑ)) / 3
    η = (5 + 2 * γₑ) / (2 * (8 - dd.γₛ)) * (f_eq / fₜ)^((11 - 2 * dd.γₛ) / 3)
    return HypParams(ψᵥ(dd), ϑ, λ, η, fₜ)
end

# Phase to coalescence plus constant
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

# Time to coalescence plus constant
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

"""
Phase until coalescence at frequency f_c [Hz], [rad].
"""
Φ_to_c(f, f_c, hp::HypParams) = Φ_to_c_indef(f, hp) - Φ_to_c_indef(f_c, hp)

"""
Time until coalescence at frequency f_c [Hz], [s].
"""
t_to_c(f, f_c, hp::HypParams) = t_to_c_indef(f, hp) - t_to_c_indef(f_c, hp)

"""
Second time derivative of phase as a function of frequency [rad / s^2].
"""
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

# If possible, convert system to HypParams
Φ_to_c(f, f_c, system::T) where T <: HypableDress = Φ_to_c(f, f_c, convert(HypParams, system))
t_to_c(f, f_c, system::T) where T <: HypableDress = t_to_c(f, f_c, convert(HypParams,system))
d²Φ_dt²(f, system::T) where T <: HypableDress = d²Φ_dt²(f, convert(HypParams,system))