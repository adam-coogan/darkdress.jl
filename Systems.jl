module Systems

using Zygote
using HypergeometricFunctions: _₂F₁
using SpecialFunctions: gamma
using ..Utils: Gₙ, c, pc_to_km, km_to_pc, heaviside

# FloatOrAbsArr = Union{AbstractFloat, AbstractArray{AbstractFloat, 1}}

import Base.length
export Binary, VacuumBinary, DarkDress, amp₊, ampₓ, length, ΔΦ_to_c

abstract type Binary end
abstract type DarkDress <: Binary end

struct VacuumBinary <: Binary
    ℳ::AbstractFloat
end

struct DynamicDress <: DarkDress
    m₁::AbstractFloat
    m₂::AbstractFloat
    ρₛ::AbstractFloat
    γₛ::AbstractFloat
end

struct StaticDress <: DarkDress
    m₁::AbstractFloat
    m₂::AbstractFloat
    ρₛ::AbstractFloat
    γₛ::AbstractFloat
end

VacuumBinary(m₁, m₂) = VacuumBinary((m₁ * m₂)^(3/5) / (m₁ + m₂)^(1/5))
VacuumBinary(system::DarkDress) = VacuumBinary(system.m₁, system.m₂)

length(system::VacuumBinary) = 1
length(system::DarkDress) = 4

"""
Chirp mass [Mₒ]
"""
ℳ(system::VacuumBinary) = system.ℳ
ℳ(system::Binary) = (system.m₁ * system.m₂)^(3/5) / (system.m₁ + system.m₂)^(1/5)  # fallback

M_tot(system::Binary) = system.m₁ + system.m₂

"""
Factor for computing vacuum phase
"""
ψᵥ(system::Binary) = 1 / 16 * (c^3 / (π * Gₙ * pc_to_km * ℳ(system)))^(5/3)

"""
'Intrinsic' part of the strain amplitude [pc].
"""
function h₀(f, system::Binary)
    # Time domain
    amp_td = 4 * π^(2/3) * Gₙ^(5/3) * ℳ(system)^(5/3) * f^(2/3) / c^4 / km_to_pc^(2/3)  # pc
    # Fourier domain
    return 1 / 2 * amp_td * √(2 * π / d²Φ_dt²(f, system))  # s
end

"""
Amplitude of h₊ [1].
"""
amp₊(f, d, ι, system::Binary) = h₀(f, system) * 1 / d * (1 + cos(ι)^2) / 2

"""
Amplitude of hₓ [1].
"""
ampₓ(f, d, ι, system::Binary) = h₀(f, system) * 1 / d * cos(ι)

"""
Phase of waveform in Fourier domain [1].
"""
function Ψ(f, d, Φ_c, t_c, system::Binary)
    t = t_c - t_to_c(f, system)
    Φ = Φ_c - Φ_to_c(f, system)
    return 2 * π * f * (t + d * pc_to_km / c) - Φ - π / 4
end

h₊(f, d, ι, Φ_c, t_c, system::Binary) = exp(1im * Ψ(f, d, Φ_c, t_c, system)) * amp₊(f, d, ι, system)

hₓ(f, d, ι, Φ_c, t_c, system::Binary) = exp(1im * Ψ(f, d, Φ_c, t_c, system)) * ampₓ(f, d, ι, system)

"""
ISCO radius [pc].
"""
r_isco(system::Binary) = 6 * Gₙ * system.m₁ / c^2

"""
ISCO frequency [Hz].
"""
f_isco(system::Binary) = √(Gₙ * km_to_pc^2 * system.m₁ / r_isco(system)^3) / π

### Subtypes should implement the following functions
"""
Frequency [Hz] at which the system will merge in time t [s].
"""
f_to_c(t, system::VacuumBinary) = (5 * ψᵥ(system) / (π * t))^(3/8) / (2 * √(2))

"""
Phase remaining to coalescence [1].
"""
Φ_to_c(f, system::VacuumBinary) = ψᵥ(system) / f^(5/3)

"""
Time remaining until coalescence [s].
"""
t_to_c(f, system::VacuumBinary) = 5 * ψᵥ(system) / (16 * π * f^(8/3))

"""
Second derivative of phase [Hz^2].

TODO: can probably compute with autograd!
"""
d²Φ_dt²(f, system::VacuumBinary) = 12 * π^2 * f^(11/3) / (5 * ψᵥ(system))

# Dark dress
ξ(system::DarkDress) = 0.58
rₛ(system::DarkDress) = ((3 - system.γₛ) * (0.2^(3.0 - system.γₛ)) * system.m₁ / (2 * π * system.ρₛ))^(1/3)

function c_gw(system::DarkDress)  # pc^3 km / s → km^4 / s
    return 64 * (Gₙ * pc_to_km)^3 * M_tot(system) * system.m₁ * system.m₂ / (5 * c^5)
end

function c_df(system::DarkDress)  # pc^(γₛ-5/2) km / s → km^(γₛ-3/2) / s
    return (
        8
        * π
        * sqrt(Gₙ * pc_to_km)
        * system.m₂
        * (system.ρₛ / pc_to_km^3)
        * ξ(system)
        * log(Λ(system)
        * (rₛ(system) * pc_to_km)^system.γₛ
    ) / sqrt(M_tot(system)) * system.m₁)
end

# Static dress
# TODO: UNITS!!!
function Φ_to_c_indef(f, system::StaticDress)
    return (
        4
        * π^(2 * (1 - system.γₛ / 3))
        * f^(2 * (1 - system.γₛ / 3))
        * (Gₙ * pc_to_km)^(-1 / 2 + system.γₛ / 3)
        * M_tot(system)^(-1 / 2 + system.γₛ / 3)
        / (6 * c_df(system) - 2 * c_df(system) * system.γₛ)
    ) * _₂F₁(
        1,                                           # a
        (6 - 2 * system.γₛ) / (11 - 2 * system.γₛ),   # b
        (17 - 4 * system.γₛ) / (11 - 2 * system.γₛ),  # c
        -c_gw(system)                                # z
        * f^((11 - 2 * system.γₛ) / 3)
        * (Gₙ * pc_to_km)^(-11 / 6 + system.γₛ / 3)
        * M_tot(system)^(-11 / 6 + system.γₛ / 3)
        * π^(11 / 3 - 2 * system.γₛ / 3)
        / c_df(system)
    )
end

function Φ_to_c(f, system::StaticDress)
    Φ_to_c_indef(f_isco, system) - Φ_to_c_indef(f, system)
end

function d²Φ_dt²(f, system::StaticDress)
    return (
        3 * π^(11 / 3) * c_gw(system) * f^(11 / 3) / (Gₙ * pc_to_km * M_tot(system))^(4 / 3)
        + (
            3 * c_df(system) * f^(2 * system.γₛ / 3) * (Gₙ * pc_to_km)^(1 / 2 - system.γₛ / 3)
            * M_tot(system) * π^(2 * system.γₛ / 3)
        )
    )
end

function t_to_c_indef(f, system::StaticDress)
    return (
        2
        * f^(1 - 2 * system.γₛ / 3)
        * (Gₙ * pc_to_km)^(-1 / 2 + system.γₛ / 3)
        * M_tot(system)^(-1 / 2 + system.γₛ / 3)
        * π^(1 - 2 * system.γₛ / 3)
        / (3 * c_df(system) - 2 * system.γₛ * c_df(system))
        * _₂F₁(
            1,                                           # a
            (3 - 2 * system.γₛ) / (11 - 2 * system.γₛ),   # b
            (14 - 4 * system.γₛ) / (11 - 2 * system.γₛ),  # c
            -c_gw(system)                                # z
            * f^(11 / 3 - 2 * system.γₛ / 3)
            * (Gₙ * pc_to_km)^(-11 / 6 + system.γₛ / 3)
            * M_tot(system)^(-11 / 6 + system.γₛ / 3)
            * π^(11 / 3 - 2 * system.γₛ / 3)
            / c_df(system)
        )
    )
end

function t_to_c(f, system::StaticDress)
    t_to_c_indef(f_isco, system) - t_to_c_indef(f, system)
end

# Dynamic dress
γₑ(system::DynamicDress) = 5/2.

function f_b(system::DynamicDress)
    _a₁ = 1.39191077
    _a₂ = 0.443089063
    _b = 4730.50412
    _c = -1.58268208
    _d = 1.73695145
    return _b * system.m₂^_a₂ / system.m₁^ _a₁ * (1. + _c * log(system.γₛ) + _d)
end

γₗ(system::DynamicDress) = (8 - (system.γₛ + γₑ(system))) * 2 / 3
γₕ(system::DynamicDress) = (8 - system.γₛ) * 2 / 3

Λ(system::DynamicDress) = √(system.m₁ / system.m₂)

function ΔΦ_to_c_norm(system::DynamicDress)
    a_un = (
        2
        * c_df(system)
        * π^((system.γₛ - 8) * 2 / 3)
        * (Gₙ * pc_to_km * M_tot(system))^((19 - 2 * system.γₛ) / 6)
    ) / (c_gw(system)^2 * (8 - system.γₛ))
    
    return a_un / f_b(system)^γₕ(system)
end

ΔΦ_to_c(f, system::DynamicDress) = ΔΦ_to_c_norm(system) / (
    (f / f_b(system))^γₗ(system)
    * (1. + (f / f_b(system))^(γₕ(system) - γₗ(system)))
)

Φ_to_c(f, system::DynamicDress) = Φ_to_c(f, VacuumBinary(system)) .- ΔΦ_to_c(f, system)

function d²Φ_dt²(f, system::DynamicDress)
    ratio = f / f_b(system)
    return -(
        12
        * π^2
        * f^(11 / 3)
        * (ratio^γₕ(system) + ratio^γₗ(system))^2
        / (
            3
            * ΔΦ_to_c_norm(system)
            * f^(5 / 3)
            * (ratio^γₕ(system) * γₕ(system) + ratio^γₗ(system) * γₗ(system))
            - 5 * (ratio^γₕ(system) + ratio^γₗ(system))^2 * ψᵥ(system)
        )
    )
end

function t_to_c(f, system::DynamicDress)
    b = (1 + γₗ(system)) / (γₕ(system) - γₗ(system))
    x = (f / f_b(system))^(γₕ(system) - γₗ(system))
    tᵥ = t_to_c(f, VacuumBinary(system))
    norm = ΔΦ_to_c_norm(system)
    Δt = heaviside(1e3-x) * norm / (2 * π * f_b(system) * (1 + γₗ(system))) * (
        (1 + γₗ(system)) / ((1 + x) * x^b) + gamma(1 - b) * gamma(1 + b) - _₂F₁(1, -b, 1 - b, -x) / x^b
    )
    Δt += heaviside(x-1e3) * norm * γₕ(system) / (2 * π * f_b(system) * (1 + γₕ(system))) / (f / f_b(system))^(γₕ(system)+1)
    return tᵥ - Δt
end

end  # module Systems
